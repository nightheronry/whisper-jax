"""Vertex AI / Gemini client for Tibetan STT orthography restoration and Chinese translation."""

from __future__ import annotations

import json
import re
import threading
import time
from functools import lru_cache
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from src.config import get_vertex_config

_VERTEX_REQUEST_LOCK = threading.Lock()
_LAST_VERTEX_REQUEST_AT = 0.0

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


def _throttle_vertex_requests(min_request_interval_sec: float) -> None:
    """Enforces minimum spacing between Vertex requests in this process."""
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
def get_vertex_client(project: str, location: str, api_key: str, timeout_sec: float | None = None) -> genai.Client:
    """Creates and caches a Google Gen AI SDK client for Vertex AI."""
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


def _strip_json_fences(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return text
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    first_array = text.find("[")
    first_object = text.find("{")
    starts = [idx for idx in (first_array, first_object) if idx != -1]
    if starts:
        return text[min(starts):].strip()
    return text


def _parse_raw_json_array(response_text: str, strict: bool = False) -> list[dict[str, Any]]:
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


def _build_pass2_translation_prompt(
    segments: list[dict[str, Any]],
    reference_text: str = "",
    previous_context: str = "",
) -> str:
    context_block = _build_context_section(reference_text=reference_text, previous_context=previous_context)
    input_json = json.dumps(segments, ensure_ascii=False, indent=2)

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


def translate_tibetan_segments(
    segments: list[dict[str, Any]],
    reference_text: str = "",
    previous_context: str = "",
    batch_size: int = 25,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Pass 2: Contextual refinement and Traditional Chinese translation of Tibetan segments.

    If Vertex AI credentials are not configured, logs a message and returns the segments untouched.
    """
    if not segments:
        return segments

    cfg = config or get_vertex_config()
    project = cfg.get("project", "")
    api_key = cfg.get("api_key", "")
    location = cfg.get("location", "global")
    model_name = cfg.get("model_name", "gemini-3.1-pro-preview")
    thinking_level = cfg.get("thinking_level", "HIGH")
    retry_attempts = cfg.get("retry_attempts", 5)
    retry_initial_delay_sec = cfg.get("retry_initial_delay_sec", 5.0)
    retry_backoff_multiplier = cfg.get("retry_backoff_multiplier", 2.0)
    retry_max_delay_sec = cfg.get("retry_max_delay_sec", 60.0)
    min_request_interval_sec = cfg.get("min_request_interval_sec", 2.0)
    max_output_tokens = cfg.get("max_output_tokens", 65535)

    if not api_key and not project:
        print("[INFO] No Vertex AI project or api_key configured. Skipping Pass 2 translation.")
        return segments

    try:
        client = get_vertex_client(project, location, api_key, timeout_sec=300.0)
    except Exception as e:
        print(f"[WARN] Failed to initialize Vertex AI client ({e}). Skipping Pass 2 translation.")
        return segments

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
            temperature=0.1,
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
                current_temp = 0.1 if attempt == 1 else min(0.8, 0.1 + 0.15 * (attempt - 1))
                generation_config.temperature = current_temp

                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=generation_config,
                )
                parsed_payload = getattr(response, "parsed", None)
                if isinstance(parsed_payload, list) and parsed_payload:
                    translated_items = [item for item in parsed_payload if isinstance(item, dict)]
                else:
                    response_text = getattr(response, "text", None) or ""
                    if response_text.strip():
                        translated_items = _parse_raw_json_array(response_text, strict=(attempt < attempts))

                if translated_items:
                    um = getattr(response, "usage_metadata", None)
                    if um:
                        p_tok = getattr(um, "prompt_token_count", 0) or 0
                        c_tok = getattr(um, "candidates_token_count", 0) or 0
                        t_tok = getattr(um, "thoughts_token_count", 0) or 0
                        print(
                            f"[Vertex AI] Translated {len(translated_items)} segment(s) "
                            f"(prompt: {p_tok}, output: {c_tok}, thoughts: {t_tok} tokens)"
                        )
                    break
            except Exception as exc:
                if attempt < attempts:
                    delay = min(
                        float(retry_initial_delay_sec) * (float(retry_backoff_multiplier) ** (attempt - 1)),
                        float(retry_max_delay_sec),
                    )
                    print(
                        f"[WARN] Pass 2 translation attempt {attempt}/{attempts} failed: {exc}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue
                print(f"[WARN] Pass 2 translation failed after {attempts} attempts: {exc}")

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

        tail_lines = []
        for seg in batch[-4:]:
            bo_txt = str(seg.get("tibetan_text") or seg.get("text") or "").strip()
            zh_txt = str(seg.get("translation_zh") or "").strip()
            if bo_txt:
                tail_lines.append(f"- 藏文：{bo_txt}" + (f" ｜ 中譯：{zh_txt}" if zh_txt else ""))
        if tail_lines:
            rolling_context = "\n".join(tail_lines)

    return segments
