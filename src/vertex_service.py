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

拉薩口語講經語用與音訊忠實原則（必讀）：
1. **語音忠實優先（嚴禁為了湊合口譯而捏造藏文）**：現場口譯常作白話意譯或總結性闡釋（例如把「錯失勸勉福緣」概括說成「淪為口頭上」）。恢復 `tibetan_text` 時**必須嚴格聽從實際音訊發音**（例如確實聽到 `སྐལ་བ་ཆད་པ་རེད` 就保留 `སྐལ་བ་ཆད་པ་རེད`，絕不強行改成未聽到的詞），在漢譯時再貼合前後語境信實表達（如「在自己身上等於辜負／錯失了這份勸勉福緣」）。
2. **指示代詞忠實與嚴禁腦補人物**：拉薩口語 `སྔོན་མ་དེ་གས་`、`དེ་གས་`（前面那些／那些項目），指代前文剛提過的次第或道理（如共下、共中、上士道基礎次第），**嚴禁擅自改寫為指人代詞 `དེ་ཚོས་`，也嚴禁憑空腦補「（大成就者）」等原文未說出的人物主詞**。
3. **受格 `-ར` 與承接主語**：如 `རང་སོ་སོར་བསྐུལ་མ་ཐུབ་པ་རེད།`（受格「對自己」），主語是前句的勸勉／法義，應譯為「（那些勸勉）沒能策勵／勸勉到自己」，不可錯改格助詞或誤譯為「自己沒能被勸動」。
4. **佛門敬仰詞 `སྐྱབས་སུ་མཆི་`**：在口語評論聽眾對發菩提心聖者的崇敬心態時（如「覺得那些發菩提心者真了不起，我皈依他們」），必須譯為「我皈依／我頂禮（崇敬）」，切勿誤譯為求救「救護我！」。
5. **強調詞 `ཡེ་`（如 `ཡེ་སྐྱེ་གི་རེད་ད།`）極性原則**：口語 `ཡེ་` 本身只表「完全／極度」，**本身不是否定詞**！只有當該句確實出現否定助詞（`མ་`、`མི་`、`མེད་`、`མིན་`）時才譯為否定；若句尾為肯定助動詞（`སྐྱེ་གི་རེད་ད།`），絕對不可腦補加上「不」字！
6. **高危音韻誤聽辨析**：
   - `དོག་དོག་དེ་མར་ཆུང་དུ་འགྲོ་`（心胸狹隘逐漸縮小）vs `གྲོག་པོ་དེ་མར་མཆོང་`（跳下深淵）。
   - `ཆགས་པ་འཕེལ་རྒྱུ་སྐྱེ་གི་རེད་`（貪著更加增長生起）vs `ཕྱིར་སྐྱོག་རྒྱག་`（退卻）。
   - `འཁྲིས་དེ་` / `གོང་འོག་` / `ཐད་`（在法本頁面上下旁邊找經文位置）vs `རྟགས་`（理證）vs `ཁྲིད་དེབ་`（教材）。
7. **口語承前省略補註**：當講者複述前一句假設並跳過感嘆詞（如「明明有能力思惟卻不去用，若就這樣死去（真是太可惜、太可怕），這才恰好是念死之心」）時，若缺省會導致語意顛倒（變成「不思考而死反而生起念死心」），應在譯文中以全形括號 `（……）` 適度補出承前省略之判斷語氣，使因果合乎法義與原意。
8. **結尾設問與指代**：`ཕ་འདྲ་ཞིག་... བྱེད་རྒྱུ་ཡོད་རེད་` 若緊接正面榜樣（如智慧長者），意為「未來就能成為那樣」，切勿誤翻成負面反問「未來還能做什麼呢？」。
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
                "description": "Restored standard Tibetan Unicode text matching what was actually spoken.",
            },
            "translation_zh": {
                "type": "STRING",
                "description": "Faithful, doctrinally accurate Traditional Chinese translation of this Tibetan segment.",
            },
        },
        "required": ["index", "tibetan_text", "translation_zh"],
    },
}

PASS3_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "seg_index": {
                "type": "INTEGER",
                "description": "Global segment index that shows a suspected error.",
            },
            "issue": {
                "type": "STRING",
                "description": "Concise description of the polarity flip, homophone hallucination, or contextual contradiction.",
            },
        },
        "required": ["seg_index", "issue"],
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


def _build_context_section(
    reference_text: str = "",
    previous_context: str = "",
    zh_anchor_text: str = "",
    next_context: str = "",
) -> str:
    sections: list[str] = [DEFAULT_LAMRIM_GLOSSARY]
    if reference_text and reference_text.strip():
        sections.append(f"當次課程《廣論》原文與進度參考：\n{reference_text.strip()}")
    if previous_context and previous_context.strip():
        sections.append(
            "前一段剛結束的藏文語境（僅供上下文連貫與指代參考，請勿重複轉寫前一段內容）：\n"
            f"{previous_context.strip()}"
        )
    if zh_anchor_text and zh_anchor_text.strip():
        sections.append(
            "緊接在本段格西開示之後的【現場中文口譯員同步口譯紀錄】（極重要之消歧錨點）：\n"
            f"{zh_anchor_text.strip()}"
        )
    if next_context and next_context.strip():
        sections.append(
            "緊接在後的下段語境預覽（僅供判斷句子是否延續到下一段，請勿提前轉寫）：\n"
            f"{next_context.strip()}"
        )
    return "\n\n".join(sections)


def _build_pass2_translation_prompt(
    segments: list[dict[str, Any]],
    reference_text: str = "",
    previous_context: str = "",
    zh_anchor_text: str = "",
    next_context: str = "",
    has_audio: bool = False,
) -> str:
    context_block = _build_context_section(
        reference_text=reference_text,
        previous_context=previous_context,
        zh_anchor_text=zh_anchor_text,
        next_context=next_context,
    )
    input_json = json.dumps(segments, ensure_ascii=False, indent=2)

    audio_instruction = (
        "- **PRIMARY GROUND TRUTH AUDIO ATTACHED**: The exact spoken Tibetan MP3 for this turn is attached! "
        "Listen directly to the Geshe's Lhasa pronunciation and intonation (question vs assertion, pauses, and root phonemes) to resolve any conflict between ASR drafts. Never invent words you do not hear.\n"
        if has_audio
        else ""
    )
    anchor_instruction = (
        "- **INTERPRETER DISAMBIGUATION ANCHOR**: An on-site Mandarin interpreter transcript immediately follows this Tibetan turn. "
        "**DO NOT copy the interpreter's paraphrase AND DO NOT fabricate Tibetan words you do not hear in the audio merely to match the interpreter's wording**; translate strictly from the Tibetan grammar and actual spoken words. "
        "HOWEVER, whenever the Tibetan ASR drafts have homophone ambiguity (e.g. `ཆུང་དུ་འགྲོ་` vs `མཆོང་`, `འཁྲིས་དེ་` vs `ཁྲིད་དེབ་`), "
        "polarity ambiguity (positive growth vs negative decrease), or speaker action context (e.g. searching pages in the Lamrim text vs citing logical proofs), "
        "you MUST cross-check the interpreter's meaning and never produce a Chinese translation whose causal polarity contradicts what the interpreter heard live!\n"
        if zh_anchor_text.strip()
        else ""
    )

    return f"""
You are an expert Tibetan linguist and translator of Tibetan Buddhist philosophy (specifically Tsongkhapa's Lamrim Chenmo 《菩提道次第廣論》) and Lhasa-dialect (Ü-Tsang) monastic oral commentary.

The input JSON array comes from Pass 1 acoustic Tibetan speech recognition listening to a Gelug Geshe speaking in Lhasa dialect. Because Lhasa dialect has silent prefixes/suffixes and many homophones, Pass 1 drafts may contain phonetic spellings (e.g., `ཀི་` instead of genitive `ཀྱི་`/`གྱི་`/`གི་`, `གོ་བར་` instead of `གོ་འཕང་`, `ཕྱི་དགྲ་` instead of `འཆི་བདག་`).

Each segment object provides complementary acoustic drafts for the exact timestamp slice:
- `whisper_tibetan_text` (or `tibetan_text`): Output from fine-tuned Whisper (strong grammar and clause coherence, but may substitute homophones or hallucinate/repeat at silence boundaries).
- `ctc_beam_text` (`ctc_tibetan_text`): Output from Wav2Vec2-CTC + 3-gram BoKenlm beam search (good syllable boundary hints, but the 3-gram LM can occasionally force an unnatural noun/verb).
- `ctc_greedy_text` (optional): Raw frame-synchronous Wav2Vec2-CTC phonemes without LM bias (often preserves the exact root stem phoneme even when both Whisper and KenLM miss it).

NOTE: Neither Whisper nor CTC is infallible — NEVER blindly favor CTC over Whisper or vice versa! Evaluate all drafts objectively against:
{audio_instruction}{anchor_instruction}- Overall doctrinal logic, sentence flow across ALL segments in this turn, and the Lamrim reference/pragmatics rules below.

Task:
1. **Holistic Turn Comprehension & Orthography Restoration (`tibetan_text`)**:
   - Read the ENTIRE sequence of segments (and listen to the attached audio if present) BEFORE editing any single segment, so you grasp the full argument, metaphors, whether the Geshe is flipping pages looking for a quote (`ང་ནི་རྙེད་ཀྱི་མི་འདུག`), or repeating a rhetorical condition.
   - Restore phonetic spellings and homophone mishearings into standard, grammatically accurate Tibetan Unicode orthography (`tibetan_text`), preserving genuine colloquial discourse markers (`ཨ་ནི་`, `ཨེ་ནས་`, `བྱས་ཙང་`, `རེད་པཱ།`).
   - Remove accidental ASR stuttering or trailing noise loops.
2. **Doctrinally Faithful Traditional Chinese Translation (`translation_zh`)**:
   - Translate each restored `tibetan_text` segment into accurate, natural Traditional Mandarin Chinese (`translation_zh`).
   - Use standard Buddhist terminology (法尊法師《廣論》譯詞) for root texts and technical terms.
   - Respect the Lhasa Pragmatics rules below (especially `སྐྱབས་སུ་མཆི་`, `ཡེ་` polarity, and parenthetical completion `（……）` for obvious oral ellipses where omission flips causality).
3. **Strict 1-to-1 Alignment**:
   - Every single input item (matching `index` 0 to {len(segments) - 1}) MUST be present in your output array.

Domain, Pragmatics & Context Reference:
{context_block}

Input Tibetan Segments JSON:
{input_json}

Output requirements:
- Return valid JSON only: an array of objects with keys `index`, `tibetan_text`, and `translation_zh`.
""".strip()


def _record_usage(response: Any, token_tracker: dict[str, int] | None, label: str = "Pass 2") -> None:
    um = getattr(response, "usage_metadata", None)
    if not um:
        return
    p_tok = int(getattr(um, "prompt_token_count", 0) or 0)
    c_tok = int(getattr(um, "candidates_token_count", 0) or 0)
    t_tok = int(getattr(um, "thoughts_token_count", 0) or 0)
    audio_tok = 0
    for det in (getattr(um, "prompt_tokens_details", None) or []):
        mod = str(getattr(det, "modality", "")).upper()
        if "AUDIO" in mod:
            audio_tok += int(getattr(det, "token_count", 0) or 0)
    if token_tracker is not None:
        token_tracker["prompt_tokens"] = token_tracker.get("prompt_tokens", 0) + p_tok
        token_tracker["audio_tokens"] = token_tracker.get("audio_tokens", 0) + audio_tok
        token_tracker["output_tokens"] = token_tracker.get("output_tokens", 0) + c_tok
        token_tracker["thoughts_tokens"] = token_tracker.get("thoughts_tokens", 0) + t_tok
        token_tracker["requests"] = token_tracker.get("requests", 0) + 1
    audio_note = f", audio: {audio_tok}" if audio_tok else ""
    print(f"[Vertex AI {label}] prompt: {p_tok}{audio_note}, output: {c_tok}, thoughts: {t_tok} tokens")


def estimate_vertex_cost_ntd(token_tracker: dict[str, int], usd_to_ntd: float = 32.5) -> dict[str, float]:
    """Estimates USD and NTD cost from accumulated token counts (Gemini 3.1 Pro Preview rates)."""
    in_tok = float(token_tracker.get("prompt_tokens", 0))
    out_tok = float(token_tracker.get("output_tokens", 0) + token_tracker.get("thoughts_tokens", 0))
    usd = (in_tok / 1_000_000.0) * 2.00 + (out_tok / 1_000_000.0) * 12.00
    return {"usd": round(usd, 4), "ntd": round(usd * usd_to_ntd, 2)}


def _call_vertex_json(
    client: Any,
    model_name: str,
    parts: list[types.Part],
    schema: dict[str, Any],
    thinking_level: str,
    max_output_tokens: int,
    cfg: dict[str, Any],
    label: str = "Pass 2",
    token_tracker: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Shared retry + throttling wrapper for structured JSON Vertex AI calls."""
    attempts = max(1, int(cfg.get("retry_attempts", 5)))
    retry_initial = float(cfg.get("retry_initial_delay_sec", 5.0))
    retry_mult = float(cfg.get("retry_backoff_multiplier", 2.0))
    retry_max = float(cfg.get("retry_max_delay_sec", 60.0))
    min_interval = float(cfg.get("min_request_interval_sec", 2.0))

    contents = [types.Content(role="user", parts=parts)]
    generation_config = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
    )

    for attempt in range(1, attempts + 1):
        try:
            _throttle_vertex_requests(min_interval)
            generation_config.temperature = 0.1 if attempt == 1 else min(0.8, 0.1 + 0.15 * (attempt - 1))
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config,
            )
            items: list[dict[str, Any]] = []
            parsed_payload = getattr(response, "parsed", None)
            if isinstance(parsed_payload, list):
                items = [item for item in parsed_payload if isinstance(item, dict)]
            else:
                response_text = getattr(response, "text", None) or ""
                if response_text.strip():
                    items = _parse_raw_json_array(response_text, strict=(attempt < attempts))

            _record_usage(response, token_tracker, label=label)
            if items or parsed_payload == []:
                return items
        except Exception as exc:
            if attempt < attempts:
                delay = min(retry_initial * (retry_mult ** (attempt - 1)), retry_max)
                print(f"[WARN] {label} attempt {attempt}/{attempts} failed: {exc}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
            print(f"[WARN] {label} failed after {attempts} attempts: {exc}")
    return []


def translate_tibetan_segments(
    segments: list[dict[str, Any]],
    reference_text: str = "",
    previous_context: str = "",
    batch_size: int = 50,
    config: dict[str, Any] | None = None,
    zh_anchor_text: str = "",
    next_context: str = "",
    audio_bytes: bytes | None = None,
    audio_mime_type: str = "audio/mpeg",
    token_tracker: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Pass 2: Audio-grounded & interpreter-anchored Tibetan refinement and Chinese translation."""
    if not segments:
        return segments

    cfg = config or get_vertex_config()
    project = cfg.get("project", "")
    api_key = cfg.get("api_key", "")
    location = cfg.get("location", "global")
    model_name = cfg.get("model_name", "gemini-3.1-pro-preview")
    thinking_level = (cfg.get("thinking_level") or "LOW").strip().upper()
    max_output_tokens = int(cfg.get("max_output_tokens", 65535))

    if not api_key and not project:
        print("[INFO] No Vertex AI project or api_key configured. Skipping Pass 2 translation.")
        return segments

    try:
        client = get_vertex_client(project, location, api_key, timeout_sec=300.0)
    except Exception as e:
        print(f"[WARN] Failed to initialize Vertex AI client ({e}). Skipping Pass 2 translation.")
        return segments

    rolling_context = (previous_context or "").strip()

    for batch_start in range(0, len(segments), max(1, batch_size)):
        batch = segments[batch_start : batch_start + max(1, batch_size)]
        payload_for_prompt = []
        for idx, seg in enumerate(batch):
            whisper_txt = str(seg.get("whisper_tibetan_text") or seg.get("tibetan_text") or seg.get("text") or "")
            item_payload: dict[str, Any] = {
                "index": idx,
                "start_time": str(seg.get("start_time", seg.get("start", ""))),
                "end_time": str(seg.get("end_time", seg.get("end", ""))),
                "whisper_tibetan_text": whisper_txt,
            }
            ctc_txt = str(seg.get("ctc_tibetan_text") or "").strip()
            if ctc_txt:
                item_payload["ctc_beam_text"] = ctc_txt
            ctc_greedy = str(seg.get("ctc_greedy_text") or "").strip()
            if ctc_greedy and ctc_greedy != ctc_txt:
                item_payload["ctc_greedy_text"] = ctc_greedy
            payload_for_prompt.append(item_payload)

        # Pass audio only if the whole batch fits within one request or audio corresponds directly to this batch
        use_audio_here = bool(audio_bytes) and len(segments) <= max(1, batch_size)
        prompt = _build_pass2_translation_prompt(
            segments=payload_for_prompt,
            reference_text=reference_text,
            previous_context=rolling_context,
            zh_anchor_text=zh_anchor_text,
            next_context=next_context if (batch_start + batch_size >= len(segments)) else "",
            has_audio=use_audio_here,
        )
        parts: list[types.Part] = []
        if use_audio_here and audio_bytes:
            parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime_type))
        parts.append(types.Part.from_text(text=prompt))

        translated_items = _call_vertex_json(
            client=client,
            model_name=model_name,
            parts=parts,
            schema=PASS2_TRANSLATION_SCHEMA,
            thinking_level=thinking_level,
            max_output_tokens=max_output_tokens,
            cfg=cfg,
            label=f"Pass 2 ({len(batch)} segs)",
            token_tracker=token_tracker,
        )

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


def audit_and_refine_transcript(
    bo_rows: list[dict[str, Any]],
    turn_anchors: dict[int, str] | None = None,
    reference_text: str = "",
    audio_extractor_fn: Any = None,
    config: dict[str, Any] | None = None,
    token_tracker: dict[str, int] | None = None,
    max_flagged: int = 12,
) -> list[dict[str, Any]]:
    """Pass 3: Full-session doctrinal consistency, polarity, and homophone audit.

    1) Scans the complete translated lecture alongside live Mandarin anchors to spot
       logical/causal polarity flips, absurd ASR hallucinations, or corrupted citations.
    2) For each flagged region, extracts a generous ±8s audio window and re-evaluates
       directly from audio with HIGH thinking to produce final corrected text.
    Returns a list of audit diagnostic records describing every fix applied.
    """
    if not bo_rows:
        return []

    cfg = config or get_vertex_config()
    project = cfg.get("project", "")
    api_key = cfg.get("api_key", "")
    location = cfg.get("location", "global")
    model_name = cfg.get("model_name", "gemini-3.1-pro-preview")
    max_output_tokens = int(cfg.get("max_output_tokens", 65535))

    if not api_key and not project:
        return []

    try:
        client = get_vertex_client(project, location, api_key, timeout_sec=300.0)
    except Exception as e:
        print(f"[WARN] Pass 3 audit skipped ({e})")
        return []

    # Build compact full-lecture view grouped by turn_index / chunk_index
    lines: list[str] = []
    last_turn = None
    anchors = turn_anchors or {}
    for idx, r in enumerate(bo_rows):
        t_idx = r.get("turn_index", r.get("chunk_index"))
        if t_idx != last_turn:
            last_turn = t_idx
            lines.append(f"\n--- Turn {t_idx} ---")
            anchor = anchors.get(t_idx, "").strip()
            if anchor:
                lines.append(f"[現場口譯大意]: {anchor[:2500]}")
        st = float(r.get("start", 0.0))
        ed = float(r.get("end", 0.0))
        lines.append(
            f"[{idx}] ({st:.1f}-{ed:.1f}s) BO: {r.get('text_bo', '')} || ZH: {r.get('text_zh', '')}"
        )

    audit_prompt = f"""
You are a senior Tibetan Buddhist doctrinal editor auditing a complete translated Lamrim lecture.
Below is the full sequence of Tibetan segments (`BO`), their Traditional Chinese translations (`ZH`), and the live Mandarin interpreter's summary (`[現場口譯大意]`) for each speaking turn.

Identify ONLY severe errors (up to {max_flagged} items) in these four categories:
1. **Causal / Polarity / Conditional Reversal**: Chinese translation states the exact opposite of Buddhist logic, contradicts itself across adjacent clauses (e.g. "if without this foundation, the previous ones exist, on the foundation of prior mind-training..." caused by mishearing `གཞི་ལ་མ་སོང་ན` vs `གཞི་ནས་`), or contradicts the live Mandarin interpreter anchor (e.g. "pain makes attachment decrease" vs interpreter "makes attachment grow").
2. **Pronoun / Subject Hallucination or Grammar Flip**: E.g. translating impersonal `སྔོན་མ་དེ་གས་` ("those preceding stages/points") as fabricated people like "（大成就者）", or flipping dative `རང་སོ་སོར་` ("could not urge ourselves") into wrong agent `རང་སོ་སོས་`.
3. **Absurd ASR Homophone Hallucination**: Nonsensical physical/modern words inconsistent with surrounding sentences (e.g., "jumping into a ravine" right before "mind becomes broad", "taxi driver", "textbook/logical proof" when the Geshe is merely looking for a page in the scripture).
4. **Colloquial Idiom Misread / Corrupted Citation**: E.g., translating `སྐྱབས་སུ་མཆི་` as "Save/rescue me!", misreading a positive role-model comparison as a skeptical question, or mangled verse lines (e.g.《四百論》).

If a segment is already accurate and coherent, DO NOT flag it.
Return a JSON array of `{{"seg_index": <int>, "issue": "<short explanation>"}}`. Return `[]` if everything is sound.

Reference Context:
{reference_text[:2000]}

Full Lecture Transcript:
{''.join(l + chr(10) for l in lines)}
""".strip()

    flagged = _call_vertex_json(
        client=client,
        model_name=model_name,
        parts=[types.Part.from_text(text=audit_prompt)],
        schema=PASS3_AUDIT_SCHEMA,
        thinking_level="HIGH",
        max_output_tokens=max_output_tokens,
        cfg=cfg,
        label="Pass 3 Audit",
        token_tracker=token_tracker,
    )

    valid_flagged = []
    for item in flagged[:max_flagged]:
        try:
            sidx = int(item.get("seg_index"))
        except (TypeError, ValueError):
            continue
        if 0 <= sidx < len(bo_rows):
            valid_flagged.append({"seg_index": sidx, "issue": str(item.get("issue", "")).strip()})

    if not valid_flagged:
        print("[Pass 3 Audit] No severe contradictions or hallucinations flagged.")
        return []

    print(f"[Pass 3 Audit] Flagged {len(valid_flagged)} segment(s) for targeted HIGH-thinking audio review.")
    if audio_extractor_fn is None:
        return valid_flagged

    # Group nearby flagged indices (same turn, within 2 indices and 30s) into focused review windows
    valid_flagged.sort(key=lambda x: x["seg_index"])
    groups: list[list[dict[str, Any]]] = []
    for item in valid_flagged:
        sidx = item["seg_index"]
        if not groups:
            groups.append([item])
            continue
        prev_idx = groups[-1][-1]["seg_index"]
        same_turn = (
            bo_rows[sidx].get("turn_index", bo_rows[sidx].get("chunk_index"))
            == bo_rows[prev_idx].get("turn_index", bo_rows[prev_idx].get("chunk_index"))
        )
        if same_turn and (sidx - prev_idx) <= 2 and (float(bo_rows[sidx]["end"]) - float(bo_rows[prev_idx]["start"])) <= 30.0:
            groups[-1].append(item)
        else:
            groups.append([item])

    audit_records: list[dict[str, Any]] = []
    for grp in groups:
        base_turn = bo_rows[grp[0]["seg_index"]].get("turn_index", bo_rows[grp[0]["seg_index"]].get("chunk_index"))
        min_idx = grp[0]["seg_index"]
        for _ in range(2):
            if min_idx > 0:
                cand = bo_rows[min_idx - 1]
                if cand.get("turn_index", cand.get("chunk_index")) == base_turn and (float(bo_rows[min_idx]["start"]) - float(cand["end"])) <= 15.0:
                    min_idx -= 1
                else:
                    break
        max_idx = grp[-1]["seg_index"]
        for _ in range(2):
            if max_idx + 1 < len(bo_rows):
                cand = bo_rows[max_idx + 1]
                if cand.get("turn_index", cand.get("chunk_index")) == base_turn and (float(cand["start"]) - float(bo_rows[max_idx]["end"])) <= 15.0:
                    max_idx += 1
                else:
                    break
        target_indices = {g["seg_index"] for g in grp}
        win_start = max(0.0, float(bo_rows[min_idx]["start"]) - 4.0)
        win_end = float(bo_rows[max_idx]["end"]) + 4.0

        clip_bytes = audio_extractor_fn(win_start, win_end)
        t_idx = bo_rows[grp[0]["seg_index"]].get("turn_index", bo_rows[grp[0]["seg_index"]].get("chunk_index"))
        anchor = anchors.get(t_idx, "").strip()

        sub_items = []
        for local_i, g_i in enumerate(range(min_idx, max_idx + 1)):
            r = bo_rows[g_i]
            sub_items.append(
                {
                    "index": local_i,
                    "global_index": g_i,
                    "flagged": g_i in target_indices,
                    "start_time": f"{int(r['start']//60):02d}:{int(r['start']%60):02d}",
                    "end_time": f"{int(r['end']//60):02d}:{int(r['end']%60):02d}",
                    "whisper_tibetan_text": r.get("whisper_bo") or r.get("text_bo", ""),
                    "ctc_beam_text": r.get("ctc_bo", ""),
                    "ctc_greedy_text": r.get("ctc_greedy_bo", ""),
                    "current_tibetan_text": r.get("text_bo", ""),
                    "current_translation_zh": r.get("text_zh", ""),
                }
            )

        issues_desc = "; ".join(f"[{g['seg_index']}] {g['issue']}" for g in grp)
        review_prompt = f"""
You are performing a HIGH-precision audio verification pass on flagged segments from a Tibetan Dharma lecture.
Audit Issues Flagged: {issues_desc}

Listen directly to the attached audio clip ({win_start:.1f}s - {win_end:.1f}s), compare the raw ASR drafts (`whisper_tibetan_text`, `ctc_beam_text`, `ctc_greedy_text`) and the live interpreter summary below, and fix ALL misheard Tibetan words and reversed/erroneous Chinese translations in this window.
CRITICAL RULES FOR AUDIO FIX:
- Stay 100% faithful to the actual spoken Tibetan syllables in the audio (never invent Tibetan words you did not hear just to mimic the interpreter's Chinese phrasing).
- Keep `སྔོན་མ་དེ་གས་` ("those preceding stages/points") as `དེ་གས་` if spoken; NEVER replace `དེ་གས་` with `དེ་ཚོས་` or invent human subjects like `（大成就者）`.
- Keep items that are already accurate unchanged. Return ALL {len(sub_items)} items (index 0 to {len(sub_items)-1}).

{DEFAULT_LAMRIM_GLOSSARY}

[現場口譯大意參考]:
{anchor[:2500] if anchor else "(無)"}

Segments to Verify:
{json.dumps(sub_items, ensure_ascii=False, indent=2)}
""".strip()

        parts = []
        if clip_bytes:
            parts.append(types.Part.from_bytes(data=clip_bytes, mime_type="audio/mpeg"))
        parts.append(types.Part.from_text(text=review_prompt))

        fixed_items = _call_vertex_json(
            client=client,
            model_name=model_name,
            parts=parts,
            schema=PASS2_TRANSLATION_SCHEMA,
            thinking_level="HIGH",
            max_output_tokens=max_output_tokens,
            cfg=cfg,
            label=f"Pass 3 Audio Fix ({win_start:.0f}-{win_end:.0f}s)",
            token_tracker=token_tracker,
        )

        by_local = {}
        for pos, item in enumerate(fixed_items):
            if isinstance(item, dict):
                try:
                    li = int(item.get("index", pos))
                except (TypeError, ValueError):
                    li = pos
                by_local[li] = item

        for local_i, g_i in enumerate(range(min_idx, max_idx + 1)):
            matched = by_local.get(local_i)
            if not matched:
                continue
            new_bo = str(matched.get("tibetan_text") or "").strip()
            new_zh = str(matched.get("translation_zh") or "").strip()
            old_bo = bo_rows[g_i].get("text_bo", "")
            old_zh = bo_rows[g_i].get("text_zh", "")
            if (new_bo and new_bo != old_bo) or (new_zh and new_zh != old_zh):
                if new_bo:
                    bo_rows[g_i]["text_bo"] = new_bo
                if new_zh:
                    bo_rows[g_i]["text_zh"] = new_zh
                    bo_rows[g_i]["text"] = new_zh
                bo_rows[g_i]["audited"] = True
                audit_records.append(
                    {
                        "seg_index": g_i,
                        "start": bo_rows[g_i]["start"],
                        "end": bo_rows[g_i]["end"],
                        "flagged_reason": next((g["issue"] for g in grp if g["seg_index"] == g_i), "neighbor_context"),
                        "old_bo": old_bo,
                        "new_bo": bo_rows[g_i]["text_bo"],
                        "old_zh": old_zh,
                        "new_zh": bo_rows[g_i]["text_zh"],
                    }
                )

    return audit_records
