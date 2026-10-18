"""Transcribe Tibetan audio from GCS with the Google Gen AI SDK on Vertex AI.

Recommended usage strategy for this repository:
1. First try whole mixed-language audio only if you want a quick feasibility test.
2. If mixed Han/Tibetan results are unstable, keep the current VAD/LID pipeline
   and send only Tibetan segments to this function.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
import httpx
from functools import lru_cache
from pathlib import Path
from typing import Any

from google import genai
from google.cloud import storage
from google.genai import errors as genai_errors
from google.genai import types


# Default multimodal model on Vertex AI for this repository as of March 23, 2026.
# Callers may override this with another compatible model ID, such as
# "gemini-1.5-pro-preview-0409", if they need to reproduce older behavior.
DEFAULT_MODEL_NAME = "gemini-3.1-pro-preview"
DEFAULT_RETRY_ATTEMPTS = 5
DEFAULT_RETRY_INITIAL_DELAY_SEC = 5.0
DEFAULT_RETRY_BACKOFF_MULTIPLIER = 2.0
DEFAULT_RETRY_MAX_DELAY_SEC = 60.0
DEFAULT_MIN_REQUEST_INTERVAL_SEC = 2.0


_VERTEX_REQUEST_LOCK = threading.Lock()
_LAST_VERTEX_REQUEST_AT = 0.0


# The model is instructed to return a JSON array of segments so downstream code
# can work directly with Python lists after json.loads(...).
SEGMENT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "start_time": {
                "type": "STRING",
                "description": "Segment start timestamp in MM:SS.sss or HH:MM:SS.sss format relative to the start of this audio clip.",
            },
            "end_time": {
                "type": "STRING",
                "description": "Segment end timestamp in MM:SS.sss or HH:MM:SS.sss format relative to the start of this audio clip.",
            },
            "tibetan_text": {
                "type": "STRING",
                "description": "Complete Tibetan Unicode transcription in standard Uchen orthography. Do not use Wylie.",
            },
            "translation_zh": {
                "type": "STRING",
                "description": "Faithful literal Mandarin Chinese translation, as close to the Tibetan wording as possible.",
            },
        },
        "required": ["start_time", "end_time", "tibetan_text", "translation_zh"],
    },
}

PASS1_TIBETAN_ONLY_SCHEMA: dict[str, Any] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "start_time": {
                "type": "STRING",
                "description": "Segment start timestamp in MM:SS.sss or HH:MM:SS.sss format relative to the start of this audio clip.",
            },
            "end_time": {
                "type": "STRING",
                "description": "Segment end timestamp in MM:SS.sss or HH:MM:SS.sss format relative to the start of this audio clip.",
            },
            "tibetan_text": {
                "type": "STRING",
                "description": "Complete Tibetan Unicode transcription in standard Uchen orthography. Do not use Wylie.",
            },
        },
        "required": ["start_time", "end_time", "tibetan_text"],
    },
}

PASS2_TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "index": {
                "type": "INTEGER",
                "description": "The 0-based index of the segment from the input array.",
            },
            "tibetan_text": {
                "type": "STRING",
                "description": "Refined Tibetan Unicode text (fix obvious homophone typos against scripture if applicable, otherwise keep identical).",
            },
            "translation_zh": {
                "type": "STRING",
                "description": "Faithful, literal Traditional Chinese translation of this Tibetan segment.",
            },
        },
        "required": ["index", "tibetan_text", "translation_zh"],
    },
}

DEFAULT_LAMRIM_GLOSSARY = """
常見《菩提道次第廣論》(Lamrim Chenmo) 藏漢術語與正字法參考（請優先採用標準藏文正字法，避免同音誤寫）：
- ཀུན་སློང་ (動機/發心)、ཀུན་སྤྱོད་ (行為/行持)、བྱང་ཆུབ་ཀྱི་སེམས་ (菩提心)、སྙིང་རྗེ་ (悲心)、ངེས་འབྱུང་ (出離心)
- འཆི་བ་མི་རྟག་པ་ (死無常/念死無常)、མ་བསྒོམས་པའི་ཉེས་དམིགས་ (未修念死之過患)、བསྒོམས་པའི་ཕན་ཡོན་ (修習念死之勝利)
- རྩ་བ་གསུམ་ (三根本)、རྒྱུ་མཚན་དགུ་ (九因相)、ཐག་བཅད་པ་གསུམ་ (三決定)、ས་བཅད་ (科判)
- ངེས་པར་འཆི་བ་ (決定死)、ནམ་འཆི་ངེས་མེད་ (死無定期)、འཆི་བའི་ཚེ་ཆོས་མ་གཏོགས་གཞན་གང་གིས་ཀྱང་མི་ཕན་པ་ (死時除法而外餘皆無益)
- འཛམ་བུའི་གླིང་ (贍部洲)、ཚེ་ལ་ངེས་པ་མེད་པ་ (壽無定準)、འཆི་རྐྱེན་ཤིན་ཏུ་མང་བ་ (死緣極多)、གསོན་རྐྱེན་ཉུང་བ་ (活緣極少)
- ལུས་ཤིན་ཏུ་ཉམ་ཆུང་བ་ (身極微弱)、ཆུ་བུར་བཞིན་ (如水沫/水泡)、འབྱུང་བ་བཞི་ (四大)、གནོད་བྱ་གནོད་བྱེད་ (所害與能害)
- འདུས་བྱས་ (有為法/造作)、རྒྱུ་རྐྱེན་ (因緣)、ཕྱི་བཤོལ་ (推延/拖延)、ལེ་ལོ་ (懈怠)、ཞེ་དམ་མང་དུ་བྱ་ (應多立誓)
- འཆི་བདག་སུ་དང་མི་བཤེས་པ། །གློ་བུར་དག་ཏུ་འབབ་འགྱུར་བས། །སང་དག་འཆི་འམ་མ་ཤེས་པས། །དམ་པའི་ཆོས་ལ་མྱུར་དུ་བརྩོན། ། (死主悉無親，忽爾而降臨，莫知明日死，應疾修正法。注意：འཆི་བདག་ 是「死主」，切勿誤聽為同音的 ཕྱི་དགྲ་「外敵」；མི་བཤེས་པ་ 是「悉無親/不親近」，切勿誤寫為 མི་བཤིག་པ་)
- འདི་སང་དེ་རིང་འདི་བྱ་ཞེས། །བྱ་བ་མི་ལ་བཟང་པོ་མིན། །རྣམ་ཤེས་འཕོས་ནས་མེད་འགྱུར་བའི། །སང་དེ་གདོན་མི་ཟ་བར་འོང་། ། (此明後作此，是說非賢人，汝當何日無，其明日定有)
- ཚེ་འདི་ (今生/現世)、འཇིག་རྟེན་ཕྱི་མ་ / སྐྱེ་བ་ཕྱི་མ་ (後世/來世)、ཐར་པ་ (解脫)、ཐམས་ཅད་མཁྱེན་པ་ (一切智/一切種智)
""".strip()


def _build_context_section(reference_text: str = "", previous_context: str = "") -> str:
    sections: list[str] = [DEFAULT_LAMRIM_GLOSSARY]
    if reference_text and reference_text.strip():
        sections.append(f"當次課程《廣論》原文與進度參考：\n{reference_text.strip()}")
    if previous_context and previous_context.strip():
        sections.append(
            "前一段剛結束的藏文語境（僅供上下文連貫與專有名詞參考，請勿重複轉寫前一段內容）：\n"
            f"{previous_context.strip()}"
        )
    return "\n\n".join(sections)


def _build_pass1_tibetan_prompt(
    attempt_mixed_audio: bool,
    reference_text: str = "",
    previous_context: str = "",
) -> str:
    """Create the Pass 1 prompt focused strictly on complete Tibetan Unicode transcription."""

    if attempt_mixed_audio:
        audio_scope = (
            "The audio may contain mostly Tibetan speech with brief Mandarin Chinese at the very beginning or end. "
            "Transcribe ALL Tibetan speech segments completely and ignore any Mandarin Chinese speech."
        )
    else:
        audio_scope = (
            "The audio contains Tibetan Buddhist dharma teaching (Ü-Tsang / Lhasa / monastic Geshe style). "
            "Transcribe EVERY spoken Tibetan sentence from 00:00 to the very end of the audio clip without omitting anything. "
            "If there is a brief Mandarin Chinese interpreter voice at the boundary, ignore the Mandarin words and transcribe all Tibetan speech."
        )

    context_block = _build_context_section(reference_text=reference_text, previous_context=previous_context)

    return f"""
You are an expert Tibetan linguist and Tibetan Buddhist (Lamrim Chenmo) transcription specialist.

Task:
- {audio_scope}
- Output Tibetan text in proper, standard Tibetan Unicode script (Uchen orthography).
- When the speaker quotes classical Tibetan scripture (ལུང་) or Lamrim root text, use the exact classical orthography.
- When the speaker gives colloquial commentary (འགྲེལ་བཤད་), faithfully transcribe every spoken Tibetan clause and particle (e.g., རེད་པཱ།, བྱས་ཙང་, དེ་སོང་ཙང་, ཨ་ནི་, ཟེར་དུས་).
- Do NOT skip, summarize, or truncate any part of the Tibetan speech. Cover the entire audio duration from start to finish.
- Do NOT repeat the same phrase in a loop.
- Never output Wylie transliteration.
- Split the transcription into natural sentence/clause segments (typically 3 to 8 seconds each) with accurate timestamps relative to the start of this audio clip.

Domain & Orthography Reference:
{context_block}

Output requirements:
- Return valid JSON only (an array of objects with keys: start_time, end_time, tibetan_text).
- If there is truly no Tibetan speech in the clip, return [].
""".strip()


def _build_pass2_translation_prompt(
    segments: list[dict[str, Any]],
    reference_text: str = "",
    previous_context: str = "",
) -> str:
    """Create the Pass 2 prompt for Lhasa phonetic ASR orthography restoration and Tibetan-to-Chinese translation."""

    context_block = _build_context_section(reference_text=reference_text, previous_context=previous_context)
    items_for_prompt = [
        {"index": i, "tibetan_text": seg.get("tibetan_text") or seg.get("text") or ""}
        for i, seg in enumerate(segments)
    ]
    input_json = json.dumps(items_for_prompt, ensure_ascii=False, indent=2)

    return f"""
You are an expert Tibetan linguist and translator of Tibetan Buddhist philosophy (specifically Tsongkhapa's Lamrim Chenmo 《菩提道次第廣論》) and Lhasa-dialect (Ü-Tsang) monastic oral commentary.

The input JSON array comes from Pass 1 acoustic Tibetan speech recognition listening to a Gelug Geshe speaking in Lhasa dialect. Because Lhasa dialect has silent prefixes/suffixes and many homophones, the Pass 1 transcript may contain phonetic spellings (e.g., `ཀི་` instead of genitive `ཀྱི་`/`གྱི་`/`གི་`, `གོ་བར་` instead of `གོ་འཕང་`, `མར་སེམས་ཅན་` instead of `མ་སེམས་ཅན་`, `ཕྱི་དགྲ་` instead of `འཆི་བདག་`, or missing shad `།`).

Each segment object may contain two complementary acoustic transcripts of the exact same audio slice:
- `tibetan_text`: Output from a Whisper encoder-decoder model (good syllable structure, but may substitute homophones like `སྐུ་ངོ་` for `ཀུན་སློང་`, `བྲལ་` for `འཕྲད་`, or hallucinate extra words at the very end of a slice).
- `ctc_tibetan_text` (optional): Output from OpenPecha's 1.28GB `op-wav2vec2-ft-v17` frame-synchronous CTC acoustic model. Because CTC decodes frame-by-frame without an autoregressive language decoder, it **never hallucinates extra clauses and never loops**, and often captures the exact Lhasa phonemes (`ཀུན་སློང་ཀུན་སྤྱོད་ཀི་རིམ་པ་`, `རྒྱུན་ལྡན་ནས་`, `འགྲོ་བ་མི་`, `ཆོས་དང་འཕྲད་`) even when Whisper misses them.

Task:
1. **Dual-Acoustic ModelFusion & Orthography Restoration (`tibetan_text`)**:
   - Read the ENTIRE sequence of segments first to understand the full sentence flow, grammar, and whether each clause is a scriptural citation (`ལུང་`) or colloquial explanation (`འགྲེལ་བཤད་`).
   - Cross-reference `tibetan_text` (Whisper) and `ctc_tibetan_text` (Wav2Vec2-CTC) when both are present: use `ctc_tibetan_text` to verify what phonemes were actually spoken (especially for Dharma terms and to trim any unvoiced tail hallucination at the end of a Whisper slice), and combine their strengths to reconstruct the exact spoken Tibetan sentence.
   - Restore phonetic Lhasa ASR spellings or homophone errors into standard, grammatically correct Tibetan Unicode orthography (`tibetan_text`), while preserving all spoken content and genuine colloquial particles (`ཨ་ནི་`, `ཨེ་ནས་`, `བྱས་ཙང་`, `དེ་སོང་ཙང་`, `རེད་པཱ།`, `གསུངས་བསྡད་ཀྱི་ཡོད་རེད།`).
   - Remove any accidental stuttering or trailing repeated words produced by the acoustic model.
2. **Faithful Traditional Chinese Translation (`translation_zh`)**:
   - Translate each restored `tibetan_text` segment into faithful, literal Traditional Mandarin Chinese (`translation_zh`).
   - Bypass any on-site interpreter paraphrase: translate directly from the Tibetan wording and grammar, while using accurate Buddhist terminology (e.g., 法尊法師《廣論》譯詞) when classical terms or root verses are cited.
3. **Strict 1-to-1 Alignment**:
   - Every single input item (matching `index` 0 to {len(segments) - 1}) MUST be present in your output array.

Domain & Orthography Reference:
{context_block}

Input Tibetan Segments JSON:
{input_json}

Output requirements:
- Return valid JSON only: an array of objects with keys `index`, `tibetan_text`, and `translation_zh`.
""".strip()


def _build_transcription_prompt(
    attempt_mixed_audio: bool,
    reference_text: str = "",
    previous_context: str = "",
) -> str:
    """Create the single-pass multimodal prompt sent alongside the audio Part."""

    if attempt_mixed_audio:
        audio_scope = (
            "The audio may contain both Mandarin Chinese and Tibetan speech. "
            "Transcribe only the Tibetan speech segments and ignore non-Tibetan speech."
        )
    else:
        audio_scope = (
            "The audio contains Tibetan speech. Transcribe all spoken Tibetan from the beginning to the very end of the clip. "
            "Ignore any brief Mandarin Chinese words at the clip boundaries."
        )

    context_block = _build_context_section(reference_text=reference_text, previous_context=previous_context)

    return f"""
You are an expert Tibetan linguist and Tibetan speech transcription specialist.

Task:
- {audio_scope}
- Output Tibetan text in proper Tibetan Unicode script.
- Also provide a faithful Mandarin Chinese translation for each Tibetan segment.
- Bypass any interpreter embellishment, smoothing, doctrinal paraphrase, or added explanation.
- Translate directly from the Tibetan pronunciation and preserve the wording as literally as possible.
- The Chinese translation must be a loyal literal translation, not a polished summary or explanatory rewrite.
- Do NOT repeat the same phrase in a loop.
- Never output Wylie transliteration.
- Never output explanations, Markdown, speaker notes, or commentary.
- Use audio timestamps to split the output into natural segments (3 to 8 seconds each).

Domain & Orthography Reference:
{context_block}

Output requirements:
- Return valid JSON only.
- The top-level JSON value must be an array.
- Each array item must be an object with exactly these keys:
  - start_time
  - end_time
  - tibetan_text
  - translation_zh
- start_time and end_time must use MM:SS.sss or HH:MM:SS.sss.
- tibetan_text must contain only the Tibetan Unicode transcription for that segment.
- translation_zh must contain the Mandarin Chinese literal translation for that same Tibetan segment.

If there is no Tibetan speech, return an empty JSON array: [].
""".strip()


def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Split a GCS URI into bucket and object path."""

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected a GCS URI starting with 'gs://', got: {gcs_uri}")

    remainder = gcs_uri[5:]
    bucket_name, _, blob_name = remainder.partition("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"Malformed GCS URI: {gcs_uri}")
    return bucket_name, blob_name


def build_gcs_uri(prefix: str, object_name: str) -> str:
    """Join a GCS prefix and object name into a normalized GCS URI."""

    normalized_prefix = prefix.rstrip("/")
    normalized_object = object_name.lstrip("/")
    if not normalized_prefix.startswith("gs://"):
        raise ValueError(f"GCS prefix must start with 'gs://': {prefix}")
    return f"{normalized_prefix}/{normalized_object}"


def upload_bytes_to_gcs(
    *,
    data: bytes,
    gcs_uri: str,
    content_type: str,
) -> str:
    """Upload in-memory bytes to GCS and return the destination URI."""

    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return gcs_uri


def upload_file_to_gcs(
    *,
    local_path: str,
    gcs_uri: str,
    content_type: str,
) -> str:
    """Upload a local file to GCS and return the destination URI."""

    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path, content_type=content_type)
    return gcs_uri


def delete_gcs_object(gcs_uri: str) -> None:
    """Delete a staged GCS object if it exists."""

    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete(if_generation_match=None)


def seconds_to_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.sss for prompt- and JSON-friendly output."""

    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    if millis:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def timestamp_to_seconds(value: str | float | int) -> float:
    """Parse SS(.sss), MM:SS(.sss), or HH:MM:SS(.sss) into seconds."""

    if isinstance(value, (int, float)):
        return float(value)

    text = (value or "").strip()
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Invalid timestamp: {value}")


def make_staging_object_name(
    *,
    source_name: str,
    suffix: str,
    extension: str,
) -> str:
    """Create a deterministic but readable staging object name."""

    safe_source = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source_name).stem).strip("_") or "audio"
    safe_suffix = re.sub(r"[^A-Za-z0-9._-]+", "_", suffix).strip("_") or "part"
    safe_extension = extension.lstrip(".") or "wav"
    return f"{safe_source}/{safe_suffix}.{safe_extension}"


def _strip_json_fences(raw_text: str) -> str:
    """Remove Markdown code fences and return the JSON payload text."""

    text = (raw_text or "").strip()
    if not text:
        return text

    # Common model formatting: ```json ... ```
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    # Fallback: extract from the first JSON-looking character onward.
    first_array = text.find("[")
    first_object = text.find("{")
    starts = [idx for idx in (first_array, first_object) if idx != -1]
    if starts:
        return text[min(starts):].strip()

    return text


def _normalize_segment_payload(payload: Any) -> list[dict[str, str]]:
    """Validate and normalize a structured segment payload."""

    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        payload = payload["segments"]
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array, got: {type(payload).__name__}")

    normalized: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Segment #{index} is not an object: {item!r}")

        start_time = str(item.get("start_time", "")).strip()
        end_time = str(item.get("end_time", "")).strip()
        tibetan_text = str(item.get("tibetan_text", item.get("text", ""))).strip()
        translation_zh = str(item.get("translation_zh", "")).strip()

        if not start_time or not end_time or not tibetan_text:
            raise ValueError(f"Segment #{index} is missing required fields: {item!r}")

        normalized.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "text": tibetan_text,
                "tibetan_text": tibetan_text,
                "translation_zh": translation_zh,
            }
        )

    return normalized


def _parse_raw_json_array(response_text: str, strict: bool = False) -> list[dict[str, Any]]:
    """Parse a JSON array (or object with 'segments' list), with optional partial-JSON salvage."""

    cleaned = _strip_json_fences(response_text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        if strict:
            raise ValueError(f"Malformed JSON: {exc}") from exc
        decoder = json.JSONDecoder()
        stripped = cleaned.lstrip()
        if stripped.startswith("["):
            items: list[dict[str, Any]] = []
            idx = cleaned.find("[") + 1
            while idx < len(cleaned):
                while idx < len(cleaned) and cleaned[idx].isspace():
                    idx += 1
                if idx >= len(cleaned) or cleaned[idx] == "]":
                    break
                try:
                    item, end = decoder.raw_decode(cleaned, idx)
                except json.JSONDecodeError:
                    break
                if isinstance(item, dict):
                    items.append(item)
                idx = end
                while idx < len(cleaned) and cleaned[idx].isspace():
                    idx += 1
                if idx < len(cleaned) and cleaned[idx] == ",":
                    idx += 1
                    continue
                if idx < len(cleaned) and cleaned[idx] == "]":
                    break
            if items:
                payload = items
            else:
                raise ValueError(f"Model response was not valid JSON: {cleaned[:500]}") from exc
        else:
            raise ValueError(f"Model response was not valid JSON: {cleaned[:500]}") from exc

    if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
        payload = payload["segments"]
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array, got: {type(payload).__name__}")
    return payload


def _parse_segment_json(response_text: str, strict: bool = False) -> list[dict[str, str]]:
    """Parse and validate the model response as a list of segment dicts."""

    payload = _parse_raw_json_array(response_text, strict=strict)
    return _normalize_segment_payload(payload)


def _detect_repetition_loop(segments: list[dict[str, str]], max_consecutive: int = 3) -> bool:
    """Detect whether the model fell into a repetition hallucination loop."""

    if not segments:
        return False

    consecutive_bo = 1
    consecutive_zh = 1
    prev_bo = ""
    prev_zh = ""

    for seg in segments:
        bo = re.sub(r"[\s་།༎༔]+", "", seg.get("tibetan_text") or seg.get("text") or "")
        zh = re.sub(r"\s+", "", seg.get("translation_zh") or "")

        if bo and len(bo) >= 4:
            if bo == prev_bo:
                consecutive_bo += 1
                if consecutive_bo >= max_consecutive:
                    return True
            else:
                consecutive_bo = 1
                prev_bo = bo

        if zh and len(zh) >= 4:
            if zh == prev_zh:
                consecutive_zh += 1
                if consecutive_zh >= max_consecutive:
                    return True
            else:
                consecutive_zh = 1
                prev_zh = zh

        # Also check internal substring repetition inside a single segment (e.g. phrase repeated 5x)
        raw_bo = (seg.get("tibetan_text") or seg.get("text") or "").strip()
        if len(raw_bo) >= 40 and re.search(r"(.{8,30}?)\1{3,}", raw_bo):
            return True

    # Check if >= 6 segments and more than half have identical Tibetan text
    if len(segments) >= 6:
        bo_Counter: dict[str, int] = {}
        for seg in segments:
            bo_norm = re.sub(r"[\s་།༎༔]+", "", seg.get("tibetan_text") or seg.get("text") or "")
            if len(bo_norm) >= 4:
                bo_Counter[bo_norm] = bo_Counter.get(bo_norm, 0) + 1
        if bo_Counter and max(bo_Counter.values()) >= max(4, int(len(segments) * 0.5)):
            return True

    return False


def _deduplicate_repeated_segments(segments: list[dict[str, str]]) -> list[dict[str, str]]:
    """Collapse consecutive duplicate segments produced by model repetition loops."""

    if len(segments) <= 1:
        return segments

    deduped: list[dict[str, str]] = []
    for seg in segments:
        bo_norm = re.sub(r"[\s་།༎༔]+", "", seg.get("tibetan_text") or seg.get("text") or "")
        if deduped and bo_norm and len(bo_norm) >= 4:
            prev_bo_norm = re.sub(
                r"[\s་།༎༔]+", "", deduped[-1].get("tibetan_text") or deduped[-1].get("text") or ""
            )
            if bo_norm == prev_bo_norm:
                deduped[-1]["end_time"] = seg.get("end_time", deduped[-1]["end_time"])
                continue
        deduped.append(dict(seg))
    return deduped


def _throttle_vertex_requests(min_request_interval_sec: float) -> None:
    """Enforce a minimum spacing between Vertex requests in this process."""

    if min_request_interval_sec <= 0:
        return

    global _LAST_VERTEX_REQUEST_AT
    with _VERTEX_REQUEST_LOCK:
        now = time.monotonic()
        wait_for = (_LAST_VERTEX_REQUEST_AT + min_request_interval_sec) - now
        if wait_for > 0:
            time.sleep(wait_for)
        _LAST_VERTEX_REQUEST_AT = time.monotonic()


@lru_cache(maxsize=None)
def _get_vertex_client(project: str, location: str, api_key: str, timeout_sec: float | None = None) -> genai.Client:
    """Create and cache a Google Gen AI SDK client for Vertex AI."""

    if timeout_sec is not None:
        http_options = types.HttpOptions(api_version="v1", httpxClient=httpx.Client(timeout=float(timeout_sec)))
    else:
        http_options = types.HttpOptions(api_version="v1")

    client_kwargs: dict[str, Any] = {
        "vertexai": True,
        "http_options": http_options,
    }
    if api_key:
        client_kwargs["api_key"] = api_key
    else:
        client_kwargs["project"] = project
        client_kwargs["location"] = location
    return genai.Client(**client_kwargs)


def translate_tibetan_segments_to_zh(
    *,
    segments: list[dict[str, Any]],
    project: str,
    location: str = "global",
    api_key: str = "",
    model_name: str = DEFAULT_MODEL_NAME,
    reference_text: str = "",
    previous_context: str = "",
    batch_size: int = 25,
    thinking_level: str = "HIGH",
    temperature: float = 0.1,
    max_output_tokens: int = 65535,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_initial_delay_sec: float = DEFAULT_RETRY_INITIAL_DELAY_SEC,
    retry_backoff_multiplier: float = DEFAULT_RETRY_BACKOFF_MULTIPLIER,
    retry_max_delay_sec: float = DEFAULT_RETRY_MAX_DELAY_SEC,
    min_request_interval_sec: float = DEFAULT_MIN_REQUEST_INTERVAL_SEC,
    timeout_sec: float = 300.0,
) -> list[dict[str, Any]]:
    """Pass 2: Contextual refinement and Traditional Chinese translation of Tibetan segments.

    Groups contiguous Tibetan segments into batches so the model can see full sentences,
    scriptural citations (ལུང་), and cross-segment context before translating into Chinese.
    Updates each segment's `tibetan_text`, `text`, and `translation_zh` in place and returns
    the updated list.
    """

    if not segments:
        return segments

    if not api_key and not project:
        raise ValueError("Either project or api_key must be provided.")

    client = _get_vertex_client(project, location, api_key, timeout_sec)
    normalized_thinking_level = (thinking_level or "HIGH").strip().upper()
    attempts = max(1, int(retry_attempts))
    rolling_context = (previous_context or "").strip()

    for batch_start in range(0, len(segments), max(1, batch_size)):
        batch = segments[batch_start : batch_start + max(1, batch_size)]
        payload_for_prompt = []
        for idx, seg in enumerate(batch):
            item_payload: dict[str, Any] = {
                "index": idx,
                "start_time": str(seg.get("start_time", seg.get("start", ""))),
                "end_time": str(seg.get("end_time", seg.get("end", ""))),
                "tibetan_text": str(seg.get("tibetan_text") or seg.get("text") or ""),
            }
            ctc_txt = str(seg.get("ctc_tibetan_text") or "").strip()
            if ctc_txt:
                item_payload["ctc_tibetan_text"] = ctc_txt
            payload_for_prompt.append(item_payload)
        prompt = _build_pass2_translation_prompt(
            segments=payload_for_prompt,
            reference_text=reference_text,
            previous_context=rolling_context,
        )
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=PASS2_TRANSLATION_SCHEMA,
            thinking_config=types.ThinkingConfig(
                thinking_level=normalized_thinking_level,
            ),
        )

        translated_items: list[dict[str, Any]] = []
        for attempt in range(1, attempts + 1):
            try:
                _throttle_vertex_requests(float(min_request_interval_sec))
                current_temp = temperature if attempt == 1 else min(0.8, temperature + 0.15 * (attempt - 1))
                generation_config.temperature = current_temp

                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=generation_config,
                )
                parsed_payload = getattr(response, "parsed", None)
                if isinstance(parsed_payload, list) and parsed_payload:
                    translated_items = [item for item in parsed_payload if isinstance(item, dict)]
                    break

                response_text = getattr(response, "text", None) or ""
                if not response_text.strip():
                    if attempt < attempts:
                        continue
                    break

                translated_items = _parse_raw_json_array(response_text, strict=(attempt < attempts))
                if translated_items:
                    break
            except Exception as exc:
                if attempt < attempts:
                    delay = min(
                        float(retry_initial_delay_sec) * (float(retry_backoff_multiplier) ** (attempt - 1)),
                        float(retry_max_delay_sec),
                    )
                    print(
                        f"[WARN] Pass 2 translation failed on attempt {attempt}/{attempts}: {exc}. "
                        f"Retrying in {delay:.1f}s."
                    )
                    time.sleep(delay)
                    continue
                print(f"[WARN] Pass 2 translation failed after {attempts} attempts: {exc}")

        # Map translated items back to batch segments
        by_index: dict[int, dict[str, Any]] = {}
        for pos, item in enumerate(translated_items):
            if not isinstance(item, dict):
                continue
            raw_idx = item.get("index", pos)
            try:
                idx_val = int(raw_idx)
            except (TypeError, ValueError):
                idx_val = pos
            by_index[idx_val] = item

        for idx, seg in enumerate(batch):
            matched = by_index.get(idx)
            if not matched and idx < len(translated_items) and isinstance(translated_items[idx], dict):
                matched = translated_items[idx]
            if not matched:
                continue

            refined_bo = str(matched.get("tibetan_text") or matched.get("refined_tibetan_text") or "").strip()
            trans_zh = str(matched.get("translation_zh", "")).strip()
            if refined_bo:
                seg["tibetan_text"] = refined_bo
                seg["text"] = refined_bo
            if trans_zh:
                seg["translation_zh"] = trans_zh

        # Update rolling context with the last 4 segments of this batch
        tail_lines = []
        for seg in batch[-4:]:
            bo_txt = str(seg.get("tibetan_text") or seg.get("text") or "").strip()
            zh_txt = str(seg.get("translation_zh") or "").strip()
            if bo_txt:
                tail_lines.append(f"- 藏文：{bo_txt}" + (f" ｜ 中譯：{zh_txt}" if zh_txt else ""))
        if tail_lines:
            rolling_context = "\n".join(tail_lines)

    return segments


def transcribe_tibetan_audio_from_gcs(
    *,
    gcs_uri: str,
    project: str,
    location: str,
    api_key: str = "",
    mime_type: str,
    model_name: str = DEFAULT_MODEL_NAME,
    attempt_mixed_audio: bool = False,
    two_pass: bool = False,
    pass1_only: bool = False,
    reference_text: str = "",
    previous_context: str = "",
    thinking_level: str = "HIGH",
    temperature: float = 0.0,
    max_output_tokens: int = 65535,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_initial_delay_sec: float = DEFAULT_RETRY_INITIAL_DELAY_SEC,
    retry_backoff_multiplier: float = DEFAULT_RETRY_BACKOFF_MULTIPLIER,
    retry_max_delay_sec: float = DEFAULT_RETRY_MAX_DELAY_SEC,
    min_request_interval_sec: float = DEFAULT_MIN_REQUEST_INTERVAL_SEC,
    timeout_sec: float = 600.0,
) -> list[dict[str, str]]:
    """Transcribe Tibetan audio stored in GCS with Vertex AI.

    Args:
        gcs_uri: The GCS URI to the audio object, for example
            "gs://my-bucket/audio/tibetan_clip.mp3".
        project: Google Cloud project ID for Vertex AI.
        location: Vertex AI location, for example "global" or "us-central1".
            Required for standard Vertex mode; ignored when api_key is used.
        api_key: Optional Google Cloud API key for Vertex AI in express mode.
        mime_type: MIME type for the audio, such as "audio/mpeg" or "audio/wav".
        model_name: Gemini model ID. Defaults to the latest stable multimodal
            model on Vertex AI.
        attempt_mixed_audio: If True, prompt the model to extract only Tibetan
            speech from a mixed Han/Tibetan recording. If False, assume the
            audio is already Tibetan-only.
        two_pass: If True (and pass1_only is False), first transcribe pure
            Tibetan text (Pass 1) and then run contextual Tibetan-to-Chinese
            refinement and translation (Pass 2).
        pass1_only: If True, run only Pass 1 (pure Tibetan transcription without
            Chinese translation) so the caller can batch Pass 2 across multiple
            audio blocks for broader paragraph context.
        reference_text: Optional Lamrim / scripture text or glossary for this
            lecture to anchor homophones and Buddhist technical terms.
        previous_context: Optional preceding Tibetan/Chinese lines from the
            immediately prior audio block for cross-block continuity.
        thinking_level: Rough reasoning level for Gemini 3 models. Use "HIGH"
            for deeper reasoning on harder audio understanding tasks.
        temperature: Generation temperature. Use 0.0 for the most deterministic
            transcription behavior.
        max_output_tokens: Maximum output token budget for the JSON transcript.
        retry_attempts: Maximum number of attempts for retryable API failures.
        retry_initial_delay_sec: Initial delay before retrying a retryable error.
        retry_backoff_multiplier: Exponential backoff multiplier between retries.
        retry_max_delay_sec: Upper bound for retry delays.
        min_request_interval_sec: Minimum spacing between Vertex requests in this
            Python process. Increase this if you frequently hit 429 errors.

    Returns:
        A list of JSON-like dicts containing start_time, end_time, text,
        tibetan_text, and translation_zh.

    Raises:
        RuntimeError: For quota, API, or malformed model response failures.
        ValueError: For clearly invalid function inputs.
    """

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected a GCS URI starting with 'gs://', got: {gcs_uri}")

    if not api_key and not project:
        raise ValueError("Either project or api_key must be provided.")

    use_pass1_schema = bool(pass1_only or two_pass)
    client = _get_vertex_client(project, location, api_key, timeout_sec)
    audio_part = types.Part.from_uri(file_uri=gcs_uri, mime_type=mime_type)
    if use_pass1_schema:
        prompt = _build_pass1_tibetan_prompt(
            attempt_mixed_audio=attempt_mixed_audio,
            reference_text=reference_text,
            previous_context=previous_context,
        )
        schema = PASS1_TIBETAN_ONLY_SCHEMA
    else:
        prompt = _build_transcription_prompt(
            attempt_mixed_audio=attempt_mixed_audio,
            reference_text=reference_text,
            previous_context=previous_context,
        )
        schema = SEGMENT_RESPONSE_SCHEMA

    normalized_thinking_level = (thinking_level or "HIGH").strip().upper()
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                audio_part,
            ],
        )
    ]
    generation_config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=schema,
        audio_timestamp=True,
        thinking_config=types.ThinkingConfig(
            thinking_level=normalized_thinking_level,
        ),
    )

    attempts = max(1, int(retry_attempts))
    response = None
    last_api_error: Exception | None = None
    final_segments: list[dict[str, str]] = []

    for attempt in range(1, attempts + 1):
        try:
            _throttle_vertex_requests(float(min_request_interval_sec))

            # Increase temperature on retries to break hallucination loops
            current_temp = temperature if attempt == 1 else min(1.0, temperature + 0.2 * (attempt - 1))
            generation_config.temperature = current_temp

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config,
            )

            parsed_segments: list[dict[str, str]] | None = None
            parsed_payload = getattr(response, "parsed", None)
            if parsed_payload not in (None, ""):
                try:
                    parsed_segments = _normalize_segment_payload(parsed_payload)
                except ValueError:
                    parsed_segments = None

            if parsed_segments is None:
                response_text = getattr(response, "text", None) or ""
                if not response_text.strip():
                    if attempt < attempts:
                        print(
                            f"[WARN] Vertex AI returned an empty response on attempt {attempt}/{attempts} "
                            f"for {gcs_uri}. Retrying..."
                        )
                        continue
                    print(f"[WARN] Vertex AI returned an empty response for {gcs_uri}. Defaulting to empty segment list.")
                    return []

                try:
                    parsed_segments = _parse_segment_json(response_text, strict=True)
                except ValueError as parse_exc:
                    if attempt < attempts:
                        delay = min(
                            float(retry_initial_delay_sec) * (float(retry_backoff_multiplier) ** (attempt - 1)),
                            float(retry_max_delay_sec),
                        )
                        print(
                            f"[WARN] Vertex JSON parse failed (hallucination) on attempt {attempt}/{attempts} "
                            f"for {gcs_uri}. Retrying with temp {current_temp:.2f}. Error: {parse_exc}"
                        )
                        time.sleep(delay)
                        continue
                    print(
                        f"[WARN] Vertex AI returned a malformed payload for {gcs_uri} after {attempts} attempts. "
                        "Trying to salvage partial JSON."
                    )
                    try:
                        parsed_segments = _parse_segment_json(response_text, strict=False)
                    except ValueError:
                        return []

            if _detect_repetition_loop(parsed_segments):
                if attempt < attempts:
                    delay = min(
                        float(retry_initial_delay_sec) * (float(retry_backoff_multiplier) ** (attempt - 1)),
                        float(retry_max_delay_sec),
                    )
                    print(
                        f"[WARN] Repetition hallucination loop detected in valid JSON on attempt {attempt}/{attempts} "
                        f"for {gcs_uri}. Retrying with higher temperature."
                    )
                    time.sleep(delay)
                    continue
                print(
                    f"[WARN] Repetition loop persisted on final attempt for {gcs_uri}. "
                    "Collapsing consecutive duplicate segments."
                )
                parsed_segments = _deduplicate_repeated_segments(parsed_segments)

            final_segments = parsed_segments
            break

        except genai_errors.APIError as exc:
            last_api_error = exc
            code = getattr(exc, "code", None)
            message = getattr(exc, "message", None) or str(exc)
            retryable = code in {429, 500, 503}
            if retryable and attempt < attempts:
                delay = min(
                    float(retry_initial_delay_sec) * (float(retry_backoff_multiplier) ** (attempt - 1)),
                    float(retry_max_delay_sec),
                )
                print(
                    f"[WARN] Vertex request failed with {code} on attempt {attempt}/{attempts} "
                    f"for {gcs_uri}. Retrying in {delay:.1f}s."
                )
                time.sleep(delay)
                continue
            if code == 429:
                raise RuntimeError(
                    "Vertex AI quota/rate-limit error while transcribing audio. "
                    f"Tried {attempts} times with backoff but still received 429. "
                    "Retry later, slow the request rate, or request more quota."
                ) from exc
            raise RuntimeError(f"Vertex AI API call failed ({code}): {message}") from exc
        except Exception as exc:
            if attempt < attempts and ("timeout" in str(type(exc)).lower() or "timeout" in str(exc).lower()):
                delay = min(
                    float(retry_initial_delay_sec) * (float(retry_backoff_multiplier) ** (attempt - 1)),
                    float(retry_max_delay_sec),
                )
                print(
                    f"[WARN] Vertex request failed with timeout on attempt {attempt}/{attempts} "
                    f"for {gcs_uri}. Retrying in {delay:.1f}s. Exact Error: {type(exc).__name__}: {exc}"
                )
                time.sleep(delay)
                continue
            raise RuntimeError(f"Vertex AI request failed: {exc}") from exc

    if response is None and not final_segments:
        raise RuntimeError(f"Vertex AI request failed after {attempts} attempts: {last_api_error}")

    if two_pass and not pass1_only and final_segments:
        translate_tibetan_segments_to_zh(
            segments=final_segments,
            project=project,
            location=location,
            api_key=api_key,
            model_name=model_name,
            reference_text=reference_text,
            previous_context=previous_context,
            thinking_level=thinking_level,
            retry_attempts=retry_attempts,
            retry_initial_delay_sec=retry_initial_delay_sec,
            retry_backoff_multiplier=retry_backoff_multiplier,
            retry_max_delay_sec=retry_max_delay_sec,
            min_request_interval_sec=min_request_interval_sec,
            timeout_sec=timeout_sec,
        )

    return final_segments


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build a simple CLI for manual verification."""

    parser = argparse.ArgumentParser(
        description="Transcribe Tibetan audio from GCS with Vertex AI Gemini via google-genai."
    )
    parser.add_argument("--gcs-uri", required=True, help="GCS URI to the audio file.")
    parser.add_argument("--project", default="", help="Google Cloud project ID.")
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional Google Cloud API key for Vertex AI express mode.",
    )
    parser.add_argument(
        "--location",
        default="global",
        help="Vertex AI location, for example global or us-central1.",
    )
    parser.add_argument(
        "--mime-type",
        default="audio/mpeg",
        help="Audio MIME type, for example audio/mpeg or audio/wav.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Gemini model ID to use on Vertex AI.",
    )
    parser.add_argument(
        "--thinking-level",
        default="HIGH",
        help="Gemini 3 thinking level, for example LOW or HIGH.",
    )
    parser.add_argument(
        "--attempt-mixed-audio",
        action="store_true",
        help="Try to extract only Tibetan speech from a mixed Han/Tibetan recording.",
    )
    parser.add_argument(
        "--two-pass",
        action="store_true",
        help="Use Two-Pass pipeline (Pass 1 Tibetan transcription -> Pass 2 contextual Chinese translation).",
    )
    parser.add_argument(
        "--reference-text",
        default="",
        help="Optional Lamrim / Buddhist scripture reference text or glossary.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help="How many times to retry retryable Vertex errors such as 429.",
    )
    parser.add_argument(
        "--retry-initial-delay-sec",
        type=float,
        default=DEFAULT_RETRY_INITIAL_DELAY_SEC,
        help="Initial backoff delay in seconds for retryable errors.",
    )
    parser.add_argument(
        "--retry-backoff-multiplier",
        type=float,
        default=DEFAULT_RETRY_BACKOFF_MULTIPLIER,
        help="Exponential multiplier applied between retries.",
    )
    parser.add_argument(
        "--retry-max-delay-sec",
        type=float,
        default=DEFAULT_RETRY_MAX_DELAY_SEC,
        help="Maximum backoff delay in seconds.",
    )
    parser.add_argument(
        "--min-request-interval-sec",
        type=float,
        default=DEFAULT_MIN_REQUEST_INTERVAL_SEC,
        help="Minimum spacing between Vertex requests in this process.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if not args.project and not args.api_key:
        raise SystemExit("Either --project or --api-key must be provided.")
    segments = transcribe_tibetan_audio_from_gcs(
        gcs_uri=args.gcs_uri,
        project=args.project,
        location=args.location,
        api_key=args.api_key,
        mime_type=args.mime_type,
        model_name=args.model_name,
        attempt_mixed_audio=args.attempt_mixed_audio,
        two_pass=args.two_pass,
        reference_text=args.reference_text,
        thinking_level=args.thinking_level,
        retry_attempts=args.retry_attempts,
        retry_initial_delay_sec=args.retry_initial_delay_sec,
        retry_backoff_multiplier=args.retry_backoff_multiplier,
        retry_max_delay_sec=args.retry_max_delay_sec,
        min_request_interval_sec=args.min_request_interval_sec,
    )
    print(json.dumps(segments, ensure_ascii=False, indent=2))
