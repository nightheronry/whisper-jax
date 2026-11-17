"""End-to-end batch transcription pipeline coordinator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import numpy as np
import pyewts
import torch
import whisperx

from src.audio import (
    build_lid_blocks,
    detect_block_language_zh_bo,
    encode_audio_slice_mp3_bytes,
    group_timeline_into_turns,
    merge_for_asr,
    parse_audio_cut_path,
    run_vad_segments,
    slice_audio,
    smooth_langs_binary,
    split_audio_at_pauses,
    split_vad_on_lang_shift,
)
from src.config import (
    SAMPLE_RATE,
    get_default_compute_type,
    get_default_device,
    get_vertex_config,
    resolve_lamrim_reference,
)
from src.exporters import (
    export_all_formats,
    load_json,
    save_json,
)
from src.models import (
    MPSWhisperPipeline,
    clean_wylie_repetitions,
    decode_wav2vec2_ctc_beam,
    load_all_models,
    transcribe_zh_anchor_text,
    truncate_by_ctc_syllables,
)
from src.vertex_service import (
    audit_and_refine_transcript,
    estimate_vertex_cost_ntd,
    translate_tibetan_segments,
)


def checkpoint_path_for_audio(audio_path: str | Path, checkpoint_dir: str | Path) -> Path:
    chunk_info = parse_audio_cut_path(audio_path)
    stem = Path(audio_path).stem
    filename = f"{chunk_info['chunk_index']:04d}_{stem}.json"
    return Path(checkpoint_dir) / filename


def save_chunk_checkpoint(result: dict[str, Any], checkpoint_dir: str | Path) -> Path:
    p = checkpoint_path_for_audio(result["audio_path"], checkpoint_dir)
    save_json(p, result)
    return p


def load_chunk_checkpoint(audio_path: str | Path, checkpoint_dir: str | Path) -> tuple[dict[str, Any] | None, Path]:
    p = checkpoint_path_for_audio(audio_path, checkpoint_dir)
    if not p.exists():
        return None, p
    try:
        return load_json(p), p
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint {p}: {e}")
        return None, p


def write_checkpoint_manifest(chunk_results: list[dict[str, Any]], checkpoint_dir: str | Path) -> Path:
    manifest_path = Path(checkpoint_dir) / "manifest.json"
    payload = {
        "total_chunks": len(chunk_results),
        "chunk_summaries": [res.get("summary", {}) for res in chunk_results],
    }
    save_json(manifest_path, payload)
    return manifest_path


def pad_bo_blocks(
    bo_blocks: list[dict[str, Any]],
    all_blocks: list[dict[str, Any]],
    audio_duration_sec: float,
    pad_start_sec: float = 0.5,
    pad_end_sec: float = 0.6,
) -> list[dict[str, Any]]:
    """Adds safety padding (0.5s~0.6s) to Tibetan speech blocks without invading Chinese blocks."""
    if not bo_blocks:
        return []
    zh_blocks = sorted([b for b in all_blocks if b.get("lang") == "zh"], key=lambda x: x["start"])
    padded = []
    for idx, b in enumerate(bo_blocks):
        st = float(b["start"])
        ed = float(b["end"])
        min_st = 0.0
        max_ed = float(audio_duration_sec)
        for zb in zh_blocks:
            if float(zb["end"]) <= st:
                min_st = max(min_st, float(zb["end"]))
            if float(zb["start"]) >= ed:
                max_ed = min(max_ed, float(zb["start"]))
                break
        if idx > 0:
            min_st = max(min_st, float(padded[-1]["end"]))
        if idx + 1 < len(bo_blocks):
            max_ed = min(max_ed, float(bo_blocks[idx + 1]["start"]))

        new_st = max(min_st, st - pad_start_sec)
        new_ed = min(max_ed, ed + pad_end_sec)
        item = b.copy()
        item["start"] = round(new_st, 3)
        item["end"] = round(new_ed, 3)
        padded.append(item)
    return padded


def build_transcript_row(
    start: float | int,
    end: float | int,
    lang: str,
    text: str = "",
    text_bo: str = "",
    text_zh: str = "",
    source: str = "",
) -> dict[str, Any]:
    text = (text or "").strip()
    text_bo = (text_bo or "").strip()
    text_zh = (text_zh or "").strip()
    if lang == "zh" and not text_zh:
        text_zh = text
    if lang == "bo" and not text:
        text = text_zh or text_bo
    return {
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "lang": lang,
        "source": source,
        "text": text,
        "text_bo": text_bo,
        "text_zh": text_zh,
    }


def shift_time_rows(
    rows: list[dict[str, Any]],
    offset_sec: float,
    extra_fields: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    shifted = []
    for row in rows:
        item = row.copy()
        if "start" in item:
            item["start"] = round(float(item["start"]) + offset_sec, 3)
        if "end" in item:
            item["end"] = round(float(item["end"]) + offset_sec, 3)
        if extra_fields:
            item.update(extra_fields)
        shifted.append(item)
    return shifted


def format_rolling_context(rows: list[dict[str, Any]], max_items: int = 4) -> str:
    lines = []
    for r in rows[-max_items:]:
        bo_txt = (r.get("text_bo") or r.get("tibetan_text") or "").strip()
        zh_txt = (r.get("text_zh") or r.get("translation_zh") or "").strip()
        if bo_txt:
            lines.append(f"- 藏文：{bo_txt}" + (f" ｜ 中譯：{zh_txt}" if zh_txt else ""))
    return "\n".join(lines)


def transcribe_zh_blocks(
    audio: np.ndarray,
    zh_blocks: list[dict[str, Any]],
    asr_models: dict[str, Any],
    align_models: dict[str, Any],
    device: str,
) -> list[dict[str, Any]]:
    """Transcribes Mandarin interpreter blocks and aligns word timestamps."""
    zh_model = asr_models.get("zh")
    if zh_model is None or not zh_blocks:
        return []

    rows = []
    for block in zh_blocks:
        s1 = int(block["start"] * SAMPLE_RATE)
        s2 = int(block["end"] * SAMPLE_RATE)
        audio_chunk = audio[s1:s2]
        if len(audio_chunk) < int(0.4 * SAMPLE_RATE):
            continue

        try:
            seg_iter, _ = zh_model.model.transcribe(
                audio_chunk,
                language="zh",
                beam_size=5,
                condition_on_previous_text=False,
            )
            base_segments = [{"start": float(s.start), "end": float(s.end), "text": (s.text or "").strip()} for s in seg_iter]
        except Exception as e:
            print(f"[WARN] Chinese transcription error: {e}")
            continue

        aligned_segments = base_segments
        if align_models.get("zh") is not None:
            model_a, metadata = align_models["zh"]
            try:
                align_dev = device if device != "mps" else "cpu"
                aligned = whisperx.align(
                    base_segments,
                    model_a,
                    metadata,
                    audio_chunk,
                    align_dev,
                    return_char_alignments=False,
                )
                aligned_segments = aligned.get("segments", base_segments)
            except Exception:
                aligned_segments = base_segments

        for s in aligned_segments:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            st = round(block["start"] + float(s["start"]), 3)
            en = round(block["start"] + float(s["end"]), 3)
            if en <= st:
                continue
            rows.append(
                build_transcript_row(
                    st,
                    en,
                    "zh",
                    text=text,
                    text_zh=text,
                    source="whisperx",
                )
            )

    rows.sort(key=lambda x: (x["start"], x["end"]))
    return rows


def extract_bo_pass1_segments(
    audio: np.ndarray,
    bo_blocks: list[dict[str, Any]],
    asr_models: dict[str, Any],
    align_models: dict[str, Any],
    device: str,
    bo_slice_sec: float = 9.0,
) -> list[dict[str, Any]]:
    """Runs Pass 1 Whisper + Wav2Vec2 CTC (beam & greedy) on padded Tibetan blocks."""
    if not bo_blocks:
        return []

    ewts = pyewts.pyewts()
    bo_whisper = (asr_models or {}).get("bo")
    bo_align_tuple = (align_models or {}).get("bo")
    segments_for_pass2: list[dict[str, Any]] = []

    for block in bo_blocks:
        blk_st = float(block["start"])
        blk_ed = float(block["end"])
        s1 = int(blk_st * SAMPLE_RATE)
        s2 = int(blk_ed * SAMPLE_RATE)
        blk_audio = audio[s1:s2]
        if len(blk_audio) < int(0.4 * SAMPLE_RATE):
            continue

        pause_slices = split_audio_at_pauses(
            blk_audio,
            sample_rate=SAMPLE_RATE,
            target_sec=bo_slice_sec,
            min_sec=6.0,
            max_sec=11.0,
            tail_merge_sec=1.8,
        )
        sub_chunks = [blk_audio[s_off:end_off] for s_off, end_off in pause_slices]
        if isinstance(bo_whisper, MPSWhisperPipeline):
            batch_raw_txts = bo_whisper.transcribe_batch(sub_chunks)
        else:
            batch_raw_txts = None

        for idx_s, (s_off, end_off) in enumerate(pause_slices):
            sub_chunk = sub_chunks[idx_s]
            st_sec = round(blk_st + s_off / float(SAMPLE_RATE), 3)
            ed_sec = round(blk_st + end_off / float(SAMPLE_RATE), 3)

            # (A) Wav2Vec2 CTC Beam + Greedy
            ctc_beam, ctc_greedy = "", ""
            if bo_align_tuple:
                try:
                    res = decode_wav2vec2_ctc_beam(
                        sub_chunk,
                        bo_align_tuple,
                        device=device,
                        return_tuple=True,
                    )
                    ctc_beam, ctc_greedy = res
                except Exception as e:
                    print(f"[WARN] Wav2Vec2 CTC beam search failed ({e})")

            # (B) Whisper
            whisper_uni = ""
            if bo_whisper is not None:
                if batch_raw_txts is not None:
                    raw_txt = (batch_raw_txts[idx_s] or "").strip()
                else:
                    try:
                        seg_iter, _ = bo_whisper.model.transcribe(
                            sub_chunk,
                            language="bo",
                            beam_size=5,
                            condition_on_previous_text=False,
                            repetition_penalty=1.35,
                            no_repeat_ngram_size=3,
                        )
                        raw_txt = " ".join((s.text or "").strip() for s in seg_iter).strip()
                    except Exception:
                        raw_txt = ""

                if raw_txt:
                    cleaned = clean_wylie_repetitions(raw_txt) if raw_txt.isascii() else raw_txt
                    if cleaned.isascii():
                        try:
                            whisper_uni = ewts.toUnicode(cleaned).strip()
                        except Exception:
                            whisper_uni = cleaned
                    else:
                        whisper_uni = cleaned

            whisper_gated = truncate_by_ctc_syllables(whisper_uni, ctc_beam) if (whisper_uni and ctc_beam) else whisper_uni
            primary_bo = whisper_gated or ctc_beam or ctc_greedy
            if not primary_bo:
                continue

            segments_for_pass2.append(
                {
                    "start": st_sec,
                    "end": ed_sec,
                    "start_time": f"{int(st_sec // 60):02d}:{int(st_sec % 60):02d}",
                    "end_time": f"{int(ed_sec // 60):02d}:{int(ed_sec % 60):02d}",
                    "whisper_tibetan_text": whisper_gated,
                    "tibetan_text": primary_bo,
                    "ctc_tibetan_text": ctc_beam,
                    "ctc_greedy_text": ctc_greedy,
                    "text": primary_bo,
                    "translation_zh": "",
                }
            )

    return segments_for_pass2


def transcribe_bo_hybrid(
    audio: np.ndarray,
    bo_blocks: list[dict[str, Any]],
    asr_models: dict[str, Any],
    align_models: dict[str, Any],
    device: str,
    reference_text: str = "",
    rolling_context: str = "",
    vertex_config: dict[str, Any] | None = None,
    bo_slice_sec: float = 9.0,
    zh_anchor_text: str = "",
    next_context: str = "",
    audio_bytes: bytes | None = None,
    token_tracker: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Pass 1 Dual-Acoustic ModelFusion (Whisper + BoKenlm CTC) + Pass 2 Vertex AI Translation."""
    segments_for_pass2 = extract_bo_pass1_segments(
        audio=audio,
        bo_blocks=bo_blocks,
        asr_models=asr_models,
        align_models=align_models,
        device=device,
        bo_slice_sec=bo_slice_sec,
    )
    if not segments_for_pass2:
        return [], rolling_context

    # Pass 2: Vertex AI Gemini contextual refinement and translation
    translate_tibetan_segments(
        segments=segments_for_pass2,
        reference_text=reference_text,
        previous_context=rolling_context,
        config=vertex_config,
        zh_anchor_text=zh_anchor_text,
        next_context=next_context,
        audio_bytes=audio_bytes,
        token_tracker=token_tracker,
    )

    out_rows = []
    for seg in segments_for_pass2:
        text_bo = (seg.get("tibetan_text") or seg.get("text") or "").strip()
        text_zh = (seg.get("translation_zh") or "").strip()
        row = build_transcript_row(
            seg["start"],
            seg["end"],
            "bo",
            text=text_zh or text_bo,
            text_bo=text_bo,
            text_zh=text_zh,
            source="hybrid_openpecha_vertex",
        )
        row["whisper_bo"] = seg.get("whisper_tibetan_text", "")
        row["ctc_bo"] = seg.get("ctc_tibetan_text", "")
        row["ctc_greedy_bo"] = seg.get("ctc_greedy_text", "")
        out_rows.append(row)

    out_rows.sort(key=lambda x: (x["start"], x["end"]))
    new_context = format_rolling_context(out_rows) if out_rows else rolling_context
    return out_rows, new_context


def process_audio_chunk(
    audio_path: str | Path,
    models: dict[str, Any],
    device: str,
    transcribe_zh: bool = True,
    reference_text: str = "",
    rolling_context: str = "",
    vertex_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Processes an individual 5-minute sliced chunk through the full VAD -> LID -> ASR -> Pass 2 pipeline."""
    chunk_info = parse_audio_cut_path(audio_path)
    audio = whisperx.load_audio(str(audio_path))
    audio_duration_sec = len(audio) / float(SAMPLE_RATE)

    asr_models = models["asr_models"]
    align_models = models["align_models"]
    zh_model = asr_models["zh"]

    # 1. Fine-grained VAD
    raw_vad = run_vad_segments(
        audio,
        zh_model.vad_model,
        chunk_size=8,
        vad_onset=0.45,
        vad_offset=0.30,
    )

    # 2. Split on language shift
    vad_segments = split_vad_on_lang_shift(
        zh_model,
        audio,
        raw_vad,
        min_split_sec=6.0,
        sub_win_sec=3.0,
        sub_hop_sec=1.5,
    )

    # 3. High-recall LID + smoothing
    lid_blocks = build_lid_blocks(vad_segments, min_lid_sec=1.8, max_gap=0.3, max_block_sec=8.0)
    seg_lang = ["other"] * len(vad_segments)
    seg_conf = [0.0] * len(vad_segments)

    for block in lid_blocks:
        lang, conf, _ = detect_block_language_zh_bo(zh_model, audio, block["start"], block["end"])
        for idx in block["idxs"]:
            seg_lang[idx] = lang
            seg_conf[idx] = conf

    seg_lang = smooth_langs_binary(
        seg_lang,
        seg_conf=seg_conf,
        vad_segments=vad_segments,
        default_lang="bo",
    )

    timeline_rows = []
    for i, (seg, lang) in enumerate(zip(vad_segments, seg_lang)):
        timeline_rows.append(
            {
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "duration": round(seg["end"] - seg["start"], 3),
                "lang": lang,
                "lid_conf": round(seg_conf[i], 3),
            }
        )

    # 4. Merge blocks for ASR
    local_asr_blocks = merge_for_asr(
        vad_segments,
        seg_lang,
        merge_gap=0.6,
        min_asr_sec=0.8,
        max_block_sec=40.0,
        bo_merge_gap=0.8,
        bo_min_asr_sec=0.25,
        bo_max_block_sec=10.0,
    )
    zh_asr_blocks = [row for row in local_asr_blocks if row["lang"] == "zh"]
    bo_asr_blocks = pad_bo_blocks(
        [row for row in local_asr_blocks if row["lang"] == "bo"],
        local_asr_blocks,
        audio_duration_sec=audio_duration_sec,
        pad_start_sec=0.5,
        pad_end_sec=0.6,
    )

    transcript_rows: list[dict[str, Any]] = []

    # 5. Chinese ASR (if requested)
    if transcribe_zh and zh_asr_blocks:
        transcript_rows.extend(
            transcribe_zh_blocks(audio, zh_asr_blocks, asr_models, align_models, device)
        )

    # 6. Tibetan Pass 1 + Pass 2
    updated_context = rolling_context
    if bo_asr_blocks:
        bo_rows, updated_context = transcribe_bo_hybrid(
            audio,
            bo_asr_blocks,
            asr_models,
            align_models,
            device,
            reference_text=reference_text,
            rolling_context=rolling_context,
            vertex_config=vertex_config,
        )
        transcript_rows.extend(bo_rows)

    transcript_rows.sort(key=lambda x: (x["start"], x["end"]))

    # Shift time relative to full audio
    extra_fields = {
        "chunk_index": chunk_info["chunk_index"],
        "audio_file": Path(audio_path).name,
    }
    shifted_timeline = shift_time_rows(timeline_rows, chunk_info["start_sec"], extra_fields)
    shifted_blocks = shift_time_rows(local_asr_blocks, chunk_info["start_sec"], extra_fields)
    shifted_transcripts = shift_time_rows(transcript_rows, chunk_info["start_sec"], extra_fields)

    summary = {
        "chunk_index": chunk_info["chunk_index"],
        "audio_file": Path(audio_path).name,
        "chunk_start": chunk_info["start_sec"],
        "chunk_end": chunk_info["end_sec"],
        "vad_segments": len(vad_segments),
        "asr_blocks": len(local_asr_blocks),
        "transcript_rows": len(transcript_rows),
        "zh_segments": sum(1 for row in timeline_rows if row["lang"] == "zh"),
        "bo_segments": sum(1 for row in timeline_rows if row["lang"] == "bo"),
        "transcribe_zh": transcribe_zh,
    }

    result = dict(chunk_info)
    result["audio_path"] = str(audio_path)
    result["summary"] = summary
    result["timeline_rows"] = shifted_timeline
    result["asr_blocks"] = shifted_blocks
    result["transcript_rows"] = shifted_transcripts
    result["updated_rolling_context"] = updated_context
    return result


def _detect_global_vad_and_turns(
    audio: np.ndarray,
    zh_model: Any,
    max_bo_turn_sec: float = 280.0,
) -> dict[str, Any]:
    """Runs VAD + LID across the full audio and groups segments into natural speaker turns."""
    raw_vad = run_vad_segments(
        audio,
        zh_model.vad_model,
        chunk_size=8,
        vad_onset=0.45,
        vad_offset=0.30,
    )
    vad_segments = split_vad_on_lang_shift(
        zh_model,
        audio,
        raw_vad,
        min_split_sec=6.0,
        sub_win_sec=3.0,
        sub_hop_sec=1.5,
    )
    lid_blocks = build_lid_blocks(vad_segments, min_lid_sec=1.8, max_gap=0.3, max_block_sec=8.0)
    seg_lang = ["other"] * len(vad_segments)
    seg_conf = [0.0] * len(vad_segments)

    for block in lid_blocks:
        lang, conf, _ = detect_block_language_zh_bo(zh_model, audio, block["start"], block["end"])
        for idx in block["idxs"]:
            seg_lang[idx] = lang
            seg_conf[idx] = conf

    seg_lang = smooth_langs_binary(
        seg_lang,
        seg_conf=seg_conf,
        vad_segments=vad_segments,
        default_lang="bo",
        max_flanked_zh_sec=12.0,
    )

    timeline_rows = []
    for i, (seg, lang) in enumerate(zip(vad_segments, seg_lang)):
        timeline_rows.append(
            {
                "chunk_index": 0,
                "audio_file": "full_audio",
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "duration": round(seg["end"] - seg["start"], 3),
                "lang": lang,
                "source": "vad_lid",
                "lid_conf": round(seg_conf[i], 3),
            }
        )

    turns = group_timeline_into_turns(
        vad_segments,
        seg_lang,
        max_bo_turn_sec=max_bo_turn_sec,
    )
    asr_blocks = [
        {
            "chunk_index": t["turn_index"],
            "audio_file": f"turn_{t['turn_index']:04d}.mp3",
            "start": round(t["start"], 3),
            "end": round(t["end"], 3),
            "lang": t["lang"],
        }
        for t in turns
    ]
    return {
        "timeline_rows": timeline_rows,
        "turns": turns,
        "asr_blocks": asr_blocks,
    }


def _pad_turn_bounds(
    turns: list[dict[str, Any]],
    turn_pos: int,
    audio_duration_sec: float,
    pad_start_sec: float = 0.5,
    pad_end_sec: float = 0.6,
) -> tuple[float, float]:
    """Safely pads a turn boundary without intruding into neighboring turns."""
    t = turns[turn_pos]
    min_st = float(turns[turn_pos - 1]["end"]) if turn_pos > 0 else 0.0
    max_ed = float(turns[turn_pos + 1]["start"]) if (turn_pos + 1 < len(turns)) else float(audio_duration_sec)
    st = max(min_st, float(t["start"]) - pad_start_sec)
    ed = min(max_ed, float(t["end"]) + pad_end_sec)
    return round(st, 3), round(ed, 3)


def _run_turn_aware_pipeline(
    audio_path: Path,
    out_dir: Path,
    mode: str,
    resolved_ref: str,
    target_device: str,
    compute_type: str,
    transcribe_zh: bool,
    force_reprocess: bool,
    pass2_only: bool,
    v_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Turn-aware, audio-grounded, interpreter-anchored pipeline with Pass 3 audit."""
    turn_chk_dir = out_dir / "turn_checkpoints"
    turn_chk_dir.mkdir(parents=True, exist_ok=True)
    timeline_cache = turn_chk_dir / "global_timeline.json"

    audio_grounding = str(v_cfg.get("audio_grounding", "turn")).strip().lower()
    use_zh_anchor = bool(v_cfg.get("use_zh_anchor", True))
    enable_audit = bool(v_cfg.get("enable_pass3_audit", True))
    v_enabled = bool(v_cfg.get("enabled", True))

    print("[1/5] Loading full audio for turn-aware analysis...")
    audio = whisperx.load_audio(str(audio_path))
    audio_duration_sec = len(audio) / float(SAMPLE_RATE)

    models: dict[str, Any] | None = None
    need_pass1 = force_reprocess or (not pass2_only) or (not timeline_cache.exists())

    if timeline_cache.exists() and not force_reprocess:
        print(f"[2/5] Loaded cached global VAD/LID timeline from {timeline_cache.name}")
        global_data = load_json(timeline_cache)
    else:
        print("[2/5] Loading models and building global VAD/LID speaker turns...")
        models = load_all_models(
            device=target_device,
            compute_type=compute_type,
            use_mps_for_bo=(target_device == "mps"),
            transcribe_zh=transcribe_zh,
            load_zh_anchor=use_zh_anchor,
        )
        global_data = _detect_global_vad_and_turns(audio, models["asr_models"]["zh"])
        save_json(timeline_cache, global_data)

    timeline_rows = global_data["timeline_rows"]
    turns = global_data["turns"]
    asr_blocks = global_data["asr_blocks"]
    print(f"Detected {len(turns)} speaker turn(s) ({sum(1 for t in turns if t['lang']=='bo')} BO, {sum(1 for t in turns if t['lang']=='zh')} ZH).")

    # Pass 1: Extract raw BO (Whisper + CTC beam/greedy) and ZH (formal or fast anchor) per turn
    pass1_by_turn: dict[int, dict[str, Any]] = {}
    missing_pass1 = False
    for t in turns:
        p1_path = turn_chk_dir / f"pass1_turn_{t['turn_index']:04d}.json"
        if p1_path.exists() and not force_reprocess:
            pass1_by_turn[t["turn_index"]] = load_json(p1_path)
        else:
            missing_pass1 = True

    if missing_pass1:
        if models is None:
            print("[2b/5] Loading local ASR models for missing Pass 1 turns...")
            models = load_all_models(
                device=target_device,
                compute_type=compute_type,
                use_mps_for_bo=(target_device == "mps"),
                transcribe_zh=transcribe_zh,
                load_zh_anchor=use_zh_anchor,
            )
        asr_models = models["asr_models"]
        align_models = models["align_models"]
        zh_anchor_model = asr_models.get("zh_anchor") or asr_models.get("zh")

        print("[3/5] Running Pass 1 local acoustic transcription (BO dual-ASR + ZH anchor)...")
        for pos, t in enumerate(turns):
            t_idx = t["turn_index"]
            if t_idx in pass1_by_turn:
                continue
            p1_path = turn_chk_dir / f"pass1_turn_{t_idx:04d}.json"
            if t["lang"] == "zh":
                zh_rows: list[dict[str, Any]] = []
                zh_anchor = ""
                if transcribe_zh:
                    zh_rows = transcribe_zh_blocks(
                        audio,
                        [{"start": t["start"], "end": t["end"], "lang": "zh"}],
                        asr_models,
                        align_models,
                        target_device,
                    )
                    for r in zh_rows:
                        r["chunk_index"] = t_idx
                        r["turn_index"] = t_idx
                        r["audio_file"] = f"turn_{t_idx:04d}.mp3"
                    zh_anchor = " ".join((r.get("text_zh") or "").strip() for r in zh_rows).strip()
                elif use_zh_anchor:
                    s1, s2 = int(t["start"] * SAMPLE_RATE), int(t["end"] * SAMPLE_RATE)
                    zh_anchor = transcribe_zh_anchor_text(zh_anchor_model, audio[s1:s2])

                p1_data = {
                    "turn_index": t_idx,
                    "lang": "zh",
                    "start": t["start"],
                    "end": t["end"],
                    "zh_rows": zh_rows,
                    "zh_anchor": zh_anchor,
                }
            else:
                pad_st, pad_ed = _pad_turn_bounds(turns, pos, audio_duration_sec)
                # Group segments inside this BO turn on pauses > 2.2s so long silences aren't padded
                inner_blocks = []
                for seg in t["segments"]:
                    if inner_blocks and (seg["start"] - inner_blocks[-1]["end"]) <= 2.2:
                        inner_blocks[-1]["end"] = seg["end"]
                    else:
                        inner_blocks.append({"start": seg["start"], "end": seg["end"], "lang": "bo"})
                if inner_blocks:
                    inner_blocks[0]["start"] = min(inner_blocks[0]["start"], pad_st)
                    inner_blocks[-1]["end"] = max(inner_blocks[-1]["end"], pad_ed)

                bo_raw_segs = extract_bo_pass1_segments(
                    audio=audio,
                    bo_blocks=inner_blocks,
                    asr_models=asr_models,
                    align_models=align_models,
                    device=target_device,
                    bo_slice_sec=9.0,
                )
                p1_data = {
                    "turn_index": t_idx,
                    "lang": "bo",
                    "start": t["start"],
                    "end": t["end"],
                    "pad_start": pad_st,
                    "pad_end": pad_ed,
                    "raw_segments": bo_raw_segs,
                }
            save_json(p1_path, p1_data)
            pass1_by_turn[t_idx] = p1_data
    else:
        print("[3/5] Loaded all Pass 1 acoustic drafts from turn_checkpoints/ (skipped Whisper/CTC).")

    # Pass 2: Audio-Grounded + Interpreter-Anchored Turn Translation
    print("[4/5] Running Pass 2 (Audio-Grounded + Interpreter-Anchored Turn Translation)...")
    token_tracker: dict[str, int] = {
        "prompt_tokens": 0,
        "audio_tokens": 0,
        "output_tokens": 0,
        "thoughts_tokens": 0,
        "requests": 0,
    }
    bo_turn_indices = [t["turn_index"] for t in turns if t["lang"] == "bo"]
    turn_anchors: dict[int, str] = {}
    all_transcripts: list[dict[str, Any]] = []
    rolling_context = ""

    for t in turns:
        t_idx = t["turn_index"]
        p1_data = pass1_by_turn[t_idx]
        if t["lang"] == "zh":
            if transcribe_zh:
                all_transcripts.extend(p1_data.get("zh_rows", []))
            continue

        # Pair with immediately following ZH turn anchor
        paired_zh_idx = t.get("paired_zh_turn_index")
        zh_anchor = ""
        if paired_zh_idx is not None and paired_zh_idx in pass1_by_turn:
            zh_anchor = (pass1_by_turn[paired_zh_idx].get("zh_anchor") or "").strip()
        turn_anchors[t_idx] = zh_anchor

        p2_path = turn_chk_dir / f"pass2_turn_{t_idx:04d}.json"
        if p2_path.exists() and not force_reprocess and not pass2_only:
            p2_data = load_json(p2_path)
            bo_rows = p2_data.get("bo_rows", [])
            print(f"--- Turn {t_idx} ({t['start']:.1f}-{t['end']:.1f}s) [cached Pass 2: {len(bo_rows)} segs] ---")
            all_transcripts.extend(bo_rows)
            if bo_rows:
                rolling_context = format_rolling_context(bo_rows)
            continue

        raw_segs = [s.copy() for s in p1_data.get("raw_segments", [])]
        if not raw_segs:
            continue

        # Lookahead preview from next BO turn's first 2 raw segments
        next_context = ""
        try:
            pos_bo = bo_turn_indices.index(t_idx)
            if pos_bo + 1 < len(bo_turn_indices):
                nxt_p1 = pass1_by_turn.get(bo_turn_indices[pos_bo + 1], {})
                nxt_segs = nxt_p1.get("raw_segments", [])[:2]
                if nxt_segs:
                    next_context = "\n".join(
                        f"- ({s['start_time']}) {s.get('whisper_tibetan_text') or s.get('tibetan_text', '')}"
                        for s in nxt_segs
                    )
        except ValueError:
            pass

        clip_bytes = None
        if v_enabled and audio_grounding == "turn":
            pad_st = float(p1_data.get("pad_start", t["start"]))
            pad_ed = float(p1_data.get("pad_end", t["end"]))
            s1, s2 = int(pad_st * SAMPLE_RATE), int(pad_ed * SAMPLE_RATE)
            if s2 > s1:
                clip_bytes = encode_audio_slice_mp3_bytes(audio[s1:s2], sample_rate=SAMPLE_RATE)

        print(
            f"\n--- Turn {t_idx} ({t['start']:.1f}-{t['end']:.1f}s, {len(raw_segs)} segs"
            + (f", audio={len(clip_bytes)//1024}KB" if clip_bytes else "")
            + (f", zh_anchor={len(zh_anchor)}ch" if zh_anchor else ", no zh_anchor")
            + ") ---"
        )

        if v_enabled:
            translate_tibetan_segments(
                segments=raw_segs,
                reference_text=resolved_ref,
                previous_context=rolling_context,
                batch_size=max(len(raw_segs), 50),
                config=v_cfg,
                zh_anchor_text=zh_anchor,
                next_context=next_context,
                audio_bytes=clip_bytes,
                token_tracker=token_tracker,
            )

        bo_rows = []
        for seg in raw_segs:
            text_bo = (seg.get("tibetan_text") or seg.get("text") or "").strip()
            text_zh = (seg.get("translation_zh") or "").strip()
            row = build_transcript_row(
                seg["start"],
                seg["end"],
                "bo",
                text=text_zh or text_bo,
                text_bo=text_bo,
                text_zh=text_zh,
                source="hybrid_openpecha_vertex",
            )
            row["chunk_index"] = t_idx
            row["turn_index"] = t_idx
            row["audio_file"] = f"turn_{t_idx:04d}.mp3"
            row["whisper_bo"] = seg.get("whisper_tibetan_text", "")
            row["ctc_bo"] = seg.get("ctc_tibetan_text", "")
            row["ctc_greedy_bo"] = seg.get("ctc_greedy_text", "")
            bo_rows.append(row)

        save_json(p2_path, {"turn_index": t_idx, "bo_rows": bo_rows})
        all_transcripts.extend(bo_rows)
        if bo_rows:
            rolling_context = format_rolling_context(bo_rows)

    all_transcripts.sort(key=lambda x: (x["start"], x["end"]))

    # Pass 3: Full-session doctrinal consistency & polarity audit
    audit_records: list[dict[str, Any]] = []
    if v_enabled and enable_audit:
        bo_only_rows = [r for r in all_transcripts if r.get("lang") == "bo"]
        if bo_only_rows:
            print("\n[4b/5] Running Pass 3 full-lecture doctrinal & polarity audit...")
            def _extract_mp3(st_s: float, ed_s: float) -> bytes:
                i1 = max(0, int(st_s * SAMPLE_RATE))
                i2 = min(len(audio), int(ed_s * SAMPLE_RATE))
                if i2 <= i1:
                    return b""
                return encode_audio_slice_mp3_bytes(audio[i1:i2], sample_rate=SAMPLE_RATE)

            audit_records = audit_and_refine_transcript(
                bo_rows=bo_only_rows,
                turn_anchors=turn_anchors,
                reference_text=resolved_ref,
                audio_extractor_fn=_extract_mp3,
                config=v_cfg,
                token_tracker=token_tracker,
            )
            if audit_records:
                print(f"[Pass 3 Audit] Applied {len(audit_records)} audio-verified refinement(s).")

    # Save diagnostic report + cost summary
    cost_est = estimate_vertex_cost_ntd(token_tracker)
    diag_path = out_dir / "diagnostic_report.json"
    save_json(
        diag_path,
        {
            "audio_file": audio_path.name,
            "mode": mode,
            "vertex_model": v_cfg.get("model_name"),
            "thinking_level": v_cfg.get("thinking_level"),
            "audio_grounding": audio_grounding,
            "use_zh_anchor": use_zh_anchor,
            "enable_pass3_audit": enable_audit,
            "token_usage": token_tracker,
            "estimated_cost": cost_est,
            "pass3_audit_fixes": audit_records,
        },
    )

    # Export all final artifacts (strip internal Pass 1 debug fields so JSON matches web UI schema)
    print("\n[5/5] Exporting final artifacts...")
    debug_keys = {"whisper_bo", "ctc_bo", "ctc_greedy_bo", "turn_index", "audited"}
    export_rows = [{k: v for k, v in r.items() if k not in debug_keys} for r in all_transcripts]
    artifacts = export_all_formats(
        output_dir=out_dir,
        timeline_rows=timeline_rows,
        asr_blocks=asr_blocks,
        transcript_rows_raw=export_rows,
        convert_s2t=True,
    )
    artifacts["diagnostic_report"] = diag_path

    print("\n==================================================")
    print("Pipeline Complete! Output Artifacts:")
    for k, p in artifacts.items():
        print(f" - {k}: {p}")
    print(
        f"Vertex AI Usage: {token_tracker['requests']} requests | "
        f"prompt={token_tracker['prompt_tokens']} (audio={token_tracker['audio_tokens']}), "
        f"output={token_tracker['output_tokens']}, thoughts={token_tracker['thoughts_tokens']} tokens | "
        f"Est. Cost: ${cost_est['usd']:.4f} USD (~NT$ {cost_est['ntd']:.2f})"
    )
    print("==================================================")

    return {
        "output_dir": out_dir,
        "artifacts": artifacts,
        "token_usage": token_tracker,
        "estimated_cost": cost_est,
        "audit_fixes": audit_records,
    }


def run_pipeline(
    audio_file: str | Path,
    output_dir: str | Path | None = None,
    mode: str = "bo_only",
    lamrim_page: str | None = "1",
    reference_text: str | None = None,
    device: str | None = None,
    chunk_duration_min: int = 5,
    force_reprocess: bool = False,
    vertex_config: dict[str, Any] | None = None,
    pass2_only: bool = False,
) -> dict[str, Any]:
    """Main pipeline entry point for end-to-end transcription and translation."""
    audio_path = Path(audio_file).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    stem = audio_path.stem
    out_dir = Path(output_dir) if output_dir else audio_path.parent / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    target_device = device or get_default_device()
    compute_type = get_default_compute_type(target_device)
    transcribe_zh = mode.lower() == "bilingual"

    resolved_ref = resolve_lamrim_reference(reference_text if reference_text is not None else lamrim_page)
    v_cfg = vertex_config or get_vertex_config()

    print(f"==================================================")
    print(f"Audio File:        {audio_path.name}")
    print(f"Output Directory:  {out_dir}")
    print(f"Mode:              {mode} (transcribe_zh={transcribe_zh})")
    print(f"Device:            {target_device} ({compute_type})")
    print(f"Turn Chunking:     {v_cfg.get('turn_chunking', True)} (audio_grounding={v_cfg.get('audio_grounding', 'turn')}, zh_anchor={v_cfg.get('use_zh_anchor', True)}, audit={v_cfg.get('enable_pass3_audit', True)})")
    print(f"Vertex AI:         {v_cfg.get('model_name')} (thinking={v_cfg.get('thinking_level')})")
    print(f"Lamrim Reference:  {len(resolved_ref)} characters loaded")
    print(f"==================================================")

    if v_cfg.get("turn_chunking", True):
        return _run_turn_aware_pipeline(
            audio_path=audio_path,
            out_dir=out_dir,
            mode=mode,
            resolved_ref=resolved_ref,
            target_device=target_device,
            compute_type=compute_type,
            transcribe_zh=transcribe_zh,
            force_reprocess=force_reprocess,
            pass2_only=pass2_only,
            v_cfg=v_cfg,
        )

    checkpoint_dir = out_dir / "chunk_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    audio_cuts_dir = out_dir / "audio_cuts"
    audio_cuts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Slice audio
    print("[1/4] Slicing audio with ffmpeg...")
    sliced_paths = slice_audio(audio_path, audio_cuts_dir, trim_duration_min=chunk_duration_min)
    print(f"Generated {len(sliced_paths)} chunk(s).")

    # 2. Load Models
    print("[2/4] Loading models (Whisper, Wav2Vec2 CTC, BoKenlm, VAD)...")
    models = load_all_models(
        device=target_device,
        compute_type=compute_type,
        use_mps_for_bo=(target_device == "mps"),
        transcribe_zh=transcribe_zh,
    )

    # 3. Process Chunks
    print("[3/4] Processing chunks...")
    chunk_results: list[dict[str, Any]] = []
    rolling_context = ""

    for idx, cut_file in enumerate(sliced_paths, 1):
        cut_name = Path(cut_file).name
        print(f"\n--- Chunk {idx}/{len(sliced_paths)}: {cut_name} ---")

        if not force_reprocess:
            cached, chk_p = load_chunk_checkpoint(cut_file, checkpoint_dir)
            if cached is not None:
                print(f"Loaded existing checkpoint from {chk_p.name}")
                chunk_results.append(cached)
                rolling_context = cached.get("updated_rolling_context", rolling_context)
                continue

        result = process_audio_chunk(
            cut_file,
            models=models,
            device=target_device,
            transcribe_zh=transcribe_zh,
            reference_text=resolved_ref,
            rolling_context=rolling_context,
            vertex_config=v_cfg,
        )
        save_chunk_checkpoint(result, checkpoint_dir)
        rolling_context = result.get("updated_rolling_context", rolling_context)
        chunk_results.append(result)

    write_checkpoint_manifest(chunk_results, checkpoint_dir)

    # 4. Consolidate and Export
    print("\n[4/4] Flattening and exporting final artifacts...")
    all_timeline: list[dict[str, Any]] = []
    all_asr_blocks: list[dict[str, Any]] = []
    all_transcripts: list[dict[str, Any]] = []

    for cr in chunk_results:
        all_timeline.extend(cr.get("timeline_rows", []))
        all_asr_blocks.extend(cr.get("asr_blocks", []))
        all_transcripts.extend(cr.get("transcript_rows", []))

    all_timeline.sort(key=lambda x: (x["start"], x["end"]))
    all_asr_blocks.sort(key=lambda x: (x["start"], x["end"]))
    all_transcripts.sort(key=lambda x: (x["start"], x["end"]))

    artifacts = export_all_formats(
        output_dir=out_dir,
        timeline_rows=all_timeline,
        asr_blocks=all_asr_blocks,
        transcript_rows_raw=all_transcripts,
        convert_s2t=True,
    )

    print("\n==================================================")
    print("Pipeline Complete! Output Artifacts:")
    for k, p in artifacts.items():
        print(f" - {k}: {p}")
    print("==================================================")

    return {
        "output_dir": out_dir,
        "artifacts": artifacts,
        "chunk_results": chunk_results,
    }
