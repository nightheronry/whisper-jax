"""Audio processing, slicing, VAD, and pause-based segment splitting."""

import math
import os
import subprocess
from pathlib import Path
from typing import Any
import ffmpeg
import numpy as np
import torch
from whisperx.audio import N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram
from whisperx.vads import Pyannote, Vad


def parse_audio_cut_path(path: str | Path) -> dict[str, Any]:
    stem = Path(path).stem
    parts = stem.split("_")
    chunk_index = int(parts[0]) if parts and parts[0].isdigit() else 0
    start_sec = float(parts[1]) if len(parts) > 1 else 0.0
    end_sec = float(parts[2]) if len(parts) > 2 else 0.0
    return {
        "chunk_index": chunk_index,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": max(0.0, end_sec - start_sec),
    }


def slice_audio(
    input_audio: str | Path,
    output_dir: str | Path,
    trim_duration_min: int = 5,
) -> list[str]:
    """Slices input audio into chunks of trim_duration_min using ffmpeg.

    Cleans up old sliced chunks in output_dir before creating new ones.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale cuts
    for f in output_dir.glob("*.mp3"):
        try:
            f.unlink()
        except OSError:
            pass

    trim_duration = 60 * trim_duration_min
    probe = ffmpeg.probe(str(input_audio))
    stream = next((s for s in probe["streams"] if s["codec_type"] == "audio"), None)
    if stream is None:
        raise ValueError(f"No audio stream found in {input_audio}")

    duration = math.floor(float(stream["duration"]))
    num_cuts = math.ceil(duration / trim_duration)

    audio_input = ffmpeg.input(str(input_audio))
    generated_cuts = []
    flag = 0

    for cut in range(num_cuts):
        end = min(duration, flag + trim_duration)
        out_file = output_dir / f"{cut}_{flag}_{end}.mp3"
        audio_cut = audio_input.audio.filter("atrim", start=flag, end=end)
        audio_output = ffmpeg.output(audio_cut, str(out_file))
        ffmpeg.run(audio_output, overwrite_output=True, quiet=True)
        generated_cuts.append(str(out_file))
        flag += trim_duration

    # Sort strictly by (chunk_index, start, end)
    generated_cuts.sort(
        key=lambda x: tuple(int(v) for v in Path(x).stem.split("_")[:3] if v.isdigit())
    )
    return generated_cuts


def run_vad_segments(
    audio: np.ndarray,
    vad_model: Any,
    chunk_size: int = 8,
    vad_onset: float = 0.45,
    vad_offset: float = 0.30,
) -> list[dict[str, float]]:
    """Fine-grained VAD segmentation."""
    if issubclass(type(vad_model), Vad):
        waveform = vad_model.preprocess_audio(audio)
        merge_chunks = vad_model.merge_chunks
    else:
        waveform = Pyannote.preprocess_audio(audio)
        merge_chunks = Pyannote.merge_chunks

    vad_scores = vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
    merged = merge_chunks(vad_scores, chunk_size, onset=vad_onset, offset=vad_offset)

    segs = []
    for s in merged:
        start = float(s["start"])
        end = float(s["end"])
        if end - start >= 0.2:
            segs.append({"start": start, "end": end})
    return segs


def _lang_probs(pipeline: Any, chunk: np.ndarray) -> list[tuple[str, float]]:
    n_mels = pipeline.model.feat_kwargs.get("feature_size") or 80
    mel = log_mel_spectrogram(
        chunk[:N_SAMPLES],
        n_mels=n_mels,
        padding=max(0, N_SAMPLES - len(chunk)),
    )
    enc = pipeline.model.encode(mel)
    return [(tok[2:-2], float(p)) for tok, p in pipeline.model.model.detect_language(enc)[0]]


def detect_zh_bo(
    pipeline: Any,
    chunk: np.ndarray,
    win_sec: float = 4.5,
    hop_sec: float = 4.0,
    min_win_sec: float = 0.4,
    zh_low_th: float = 0.97,
    other_min_th: float = 0.03,
) -> tuple[str, float, dict[str, float]]:
    """High-recall Tibetan vs Chinese language identifier."""
    votes = {"zh": 0.0, "bo": 0.0, "other": 0.0}
    win = int(win_sec * SAMPLE_RATE)
    hop = int(hop_sec * SAMPLE_RATE)
    min_len = int(min_win_sec * SAMPLE_RATE)

    if len(chunk) <= int(6.5 * SAMPLE_RATE):
        ranges = [(0, len(chunk))]
    else:
        ranges = [(i, min(i + win, len(chunk))) for i in range(0, len(chunk) - win + 1, hop)]
        if ranges and ranges[-1][1] < len(chunk) and (len(chunk) - ranges[-1][0]) >= min_len:
            ranges.append((max(0, len(chunk) - win), len(chunk)))

    for a, b in ranges:
        sub = chunk[a:b]
        if len(sub) < min_len:
            continue
        for lang, prob in _lang_probs(pipeline, sub):
            if lang.startswith("zh"):
                votes["zh"] += prob
            elif lang.startswith("bo"):
                votes["bo"] += prob
            else:
                votes["other"] += prob

    total = votes["zh"] + votes["bo"] + votes["other"]
    if total == 0:
        return "other", 0.0, votes

    zh_r = votes["zh"] / total
    bo_r = votes["bo"] / total
    other_r = votes["other"] / total

    # Protect Tibetan recall
    if bo_r >= 0.02 or (zh_r < zh_low_th and (other_r + bo_r) >= other_min_th):
        return "bo", max(bo_r, 1.0 - zh_r), votes

    if bo_r >= zh_r:
        return "bo", bo_r, votes
    return "zh", zh_r, votes


def split_vad_segments_on_lang_change(
    lid_pipeline: Any,
    audio: np.ndarray,
    vad_segments: list[dict[str, float]],
    min_split_sec: float = 6.0,
    sub_win_sec: float = 3.0,
    sub_hop_sec: float = 1.5,
) -> list[dict[str, float]]:
    """Splits long VAD segments if language switches mid-segment."""
    refined = []
    for seg in vad_segments:
        st, ed = float(seg["start"]), float(seg["end"])
        dur = ed - st
        if dur < min_split_sec:
            refined.append({"start": st, "end": ed})
            continue

        mid = 0.5 * (st + ed)
        l_first, _, _ = detect_zh_bo(lid_pipeline, audio[int(st * SAMPLE_RATE) : int(mid * SAMPLE_RATE)])
        l_second, _, _ = detect_zh_bo(lid_pipeline, audio[int(mid * SAMPLE_RATE) : int(ed * SAMPLE_RATE)])
        if l_first == l_second and l_first in {"zh", "bo"}:
            refined.append({"start": st, "end": ed})
            continue

        sub_labels = []
        cur = st
        while cur + sub_win_sec <= ed + 1e-4:
            c_st = cur
            c_ed = min(cur + sub_win_sec, ed)
            c_audio = audio[int(c_st * SAMPLE_RATE) : int(c_ed * SAMPLE_RATE)]
            c_lang, _, _ = detect_zh_bo(lid_pipeline, c_audio)
            sub_labels.append({"start": c_st, "end": c_ed, "lang": c_lang})
            cur += sub_hop_sec

        if not sub_labels:
            refined.append({"start": st, "end": ed})
            continue

        split_pts = []
        for i in range(len(sub_labels) - 1):
            if sub_labels[i]["lang"] != sub_labels[i + 1]["lang"]:
                boundary = 0.5 * (sub_labels[i]["end"] + sub_labels[i + 1]["start"])
                split_pts.append(boundary)

        if not split_pts:
            refined.append({"start": st, "end": ed})
            continue

        cur_st = st
        for pt in split_pts:
            if pt - cur_st >= 1.0 and ed - pt >= 1.0:
                refined.append({"start": cur_st, "end": pt})
                cur_st = pt
        refined.append({"start": cur_st, "end": ed})

    return refined


split_vad_on_lang_shift = split_vad_segments_on_lang_change


def split_audio_at_pauses(
    audio_chunk: np.ndarray,
    sample_rate: int = 16000,
    target_sec: float = 9.0,
    min_sec: float = 6.0,
    max_sec: float = 11.0,
    tail_merge_sec: float = 1.8,
    window_ms: int = 80,
) -> list[tuple[int, int]]:
    """Splits an audio chunk at the minimum RMS energy pause points between [min_sec, max_sec].

    Avoids cutting in the middle of a spoken Tibetan syllable.
    Returns list of (start_sample, end_sample).
    """
    total = len(audio_chunk)
    max_samples = int(max_sec * sample_rate)
    if total <= max_samples:
        return [(0, total)]

    min_samples = int(min_sec * sample_rate)
    target_samples = int(target_sec * sample_rate)
    tail_merge_samples = int(tail_merge_sec * sample_rate)
    win_len = max(160, int(sample_rate * window_ms / 1000.0))
    hop_len = max(80, win_len // 2)

    slices = []
    offset = 0
    while offset < total:
        rem = total - offset
        if rem <= max_samples:
            slices.append((offset, total))
            break

        search_start = offset + min_samples
        search_end = min(total, offset + max_samples)
        region = audio_chunk[search_start:search_end]

        if len(region) >= win_len:
            best_pos = offset + target_samples
            best_score = float("inf")
            for i in range(0, len(region) - win_len + 1, hop_len):
                frame = region[i : i + win_len]
                rms = float(np.sqrt(np.mean(frame * frame) + 1e-12))
                cand_pos = search_start + i + win_len // 2
                dist_penalty = 0.002 * abs(cand_pos - (offset + target_samples)) / float(sample_rate)
                score = rms + dist_penalty
                if score < best_score:
                    best_score = score
                    best_pos = cand_pos
            cut = best_pos
        else:
            cut = min(total, offset + target_samples)

        if total - cut < tail_merge_samples:
            cut = total
        slices.append((offset, cut))
        offset = cut

    return slices


def build_lid_blocks(
    vad_segments: list[dict[str, Any]],
    min_lid_sec: float = 1.8,
    max_gap: float = 0.35,
    max_block_sec: float = 8.0,
) -> list[dict[str, Any]]:
    """Groups short adjacent VAD segments into larger blocks for stable language identification."""
    blocks = []
    i, n = 0, len(vad_segments)
    while i < n:
        st = vad_segments[i]["start"]
        ed = vad_segments[i]["end"]
        idxs = [i]
        j = i + 1
        while j < n:
            gap = vad_segments[j]["start"] - ed
            new_ed = vad_segments[j]["end"]
            if gap <= max_gap and (new_ed - st) <= max_block_sec and (ed - st) < min_lid_sec:
                ed = new_ed
                idxs.append(j)
                j += 1
            else:
                break
        blocks.append({"start": st, "end": ed, "idxs": idxs})
        i = j
    return blocks


def detect_block_language_zh_bo(
    lid_pipeline: Any,
    audio: np.ndarray,
    start: float,
    end: float,
) -> tuple[str, float, dict[str, float]]:
    s1, s2 = int(start * SAMPLE_RATE), int(end * SAMPLE_RATE)
    chunk = audio[s1:s2]
    if len(chunk) < int(0.3 * SAMPLE_RATE):
        return "other", 0.0, {"zh": 0.0, "bo": 0.0}
    return detect_zh_bo(lid_pipeline, chunk)


def smooth_langs_binary(
    seg_lang: list[str],
    seg_conf: list[float] | None = None,
    vad_segments: list[dict[str, Any]] | None = None,
    default_lang: str = "bo",
    max_zh_island_sec: float = 8.0,
    zh_strict_conf: float = 0.992,
    max_flanked_zh_sec: float = 12.0,
) -> list[str]:
    """Smooths language tags and fixes brief pseudo-Chinese hallucination islands flanked by Tibetan."""
    out = seg_lang[:]
    n = len(out)
    if n == 0:
        return out

    for i in range(n):
        if out[i] in {"zh", "bo"}:
            continue
        left = next((out[j] for j in range(i - 1, -1, -1) if out[j] in {"zh", "bo"}), None)
        right = next((out[j] for j in range(i + 1, n) if out[j] in {"zh", "bo"}), None)
        if left == right and left is not None:
            out[i] = left
        elif left == "bo" or right == "bo":
            out[i] = "bo"

    for i in range(n):
        if out[i] not in {"zh", "bo"}:
            out[i] = out[i - 1] if i > 0 else default_lang

    if seg_conf is not None and vad_segments is not None:
        for i in range(n):
            dur = float(vad_segments[i]["end"]) - float(vad_segments[i]["start"])
            conf = float(seg_conf[i])
            if out[i] == "zh" and dur <= max_zh_island_sec and conf < zh_strict_conf:
                left_bo = (i > 0 and out[i - 1] == "bo")
                right_bo = (i + 1 < n and out[i + 1] == "bo")
                if left_bo or right_bo:
                    out[i] = "bo"
            elif out[i] == "bo" and dur <= 2.5 and conf < 0.60 and 0 < i < n - 1:
                if out[i - 1] == "zh" and out[i + 1] == "zh":
                    if float(seg_conf[i - 1]) >= 0.995 and float(seg_conf[i + 1]) >= 0.995:
                        out[i] = "zh"

    # Absorb any short contiguous 'zh' island flanked on both sides by substantial 'bo' runs
    if vad_segments is not None and max_flanked_zh_sec > 0:
        i = 0
        while i < n:
            if out[i] != "zh":
                i += 1
                continue
            j = i
            while j < n and out[j] == "zh":
                j += 1
            island_dur = float(vad_segments[j - 1]["end"]) - float(vad_segments[i]["start"])
            has_left_bo = i > 0 and out[i - 1] == "bo"
            has_right_bo = j < n and out[j] == "bo"
            if has_left_bo and has_right_bo and island_dur <= max_flanked_zh_sec:
                for k in range(i, j):
                    out[k] = "bo"
            i = j

    return out


def group_timeline_into_turns(
    vad_segments: list[dict[str, Any]],
    seg_lang: list[str],
    max_bo_turn_sec: float = 280.0,
    min_pause_split_sec: float = 0.6,
) -> list[dict[str, Any]]:
    """Groups contiguous same-language VAD segments into natural speaker turns.

    In sequential Tibetan-Chinese teaching, a Geshe speaks a 2-4 minute 'bo' turn
    followed by an interpreter's 'zh' turn. Grouping by natural speaker turns
    eliminates mid-sentence hard cuts at arbitrary 5-minute boundaries and allows
    each 'bo' turn to be paired with its immediately following 'zh' interpreter turn.
    """
    if not vad_segments:
        return []

    raw_runs: list[dict[str, Any]] = []
    for seg, lang in zip(vad_segments, seg_lang):
        if lang not in {"bo", "zh"}:
            continue
        item = {"start": float(seg["start"]), "end": float(seg["end"]), "lang": lang}
        if raw_runs and raw_runs[-1]["lang"] == lang:
            raw_runs[-1]["end"] = item["end"]
            raw_runs[-1]["segments"].append(item)
        else:
            raw_runs.append(
                {
                    "lang": lang,
                    "start": item["start"],
                    "end": item["end"],
                    "segments": [item],
                }
            )

    # Split overly long continuous turns (e.g. solo lecture > max_bo_turn_sec) at widest pauses
    turns: list[dict[str, Any]] = []
    for run in raw_runs:
        dur = run["end"] - run["start"]
        if run["lang"] != "bo" or dur <= max_bo_turn_sec or len(run["segments"]) < 2:
            turns.append(run)
            continue

        segs = run["segments"]
        start_idx = 0
        while start_idx < len(segs):
            chunk_start_t = segs[start_idx]["start"]
            end_idx = start_idx + 1
            while end_idx < len(segs) and (segs[end_idx]["end"] - chunk_start_t) <= max_bo_turn_sec:
                end_idx += 1

            if end_idx >= len(segs):
                sub_segs = segs[start_idx:]
                turns.append(
                    {
                        "lang": "bo",
                        "start": sub_segs[0]["start"],
                        "end": sub_segs[-1]["end"],
                        "segments": sub_segs,
                    }
                )
                break

            # Look back from end_idx within [max_bo_turn_sec * 0.55, max_bo_turn_sec] for widest pause
            best_split = end_idx
            best_gap = -1.0
            for k in range(start_idx + 1, end_idx + 1):
                if (segs[k - 1]["end"] - chunk_start_t) < max_bo_turn_sec * 0.5:
                    continue
                gap = segs[k]["start"] - segs[k - 1]["end"]
                if gap >= min_pause_split_sec and gap > best_gap:
                    best_gap = gap
                    best_split = k
                elif best_gap < 0 and gap > best_gap:
                    best_gap = gap
                    best_split = k

            sub_segs = segs[start_idx:best_split]
            turns.append(
                {
                    "lang": "bo",
                    "start": sub_segs[0]["start"],
                    "end": sub_segs[-1]["end"],
                    "segments": sub_segs,
                }
            )
            start_idx = best_split

    # Index turns and link each 'bo' turn to the immediately following 'zh' turn (if any)
    for idx, turn in enumerate(turns):
        turn["turn_index"] = idx
        turn["paired_zh_turn_index"] = None
        if turn["lang"] == "bo":
            for nxt in range(idx + 1, len(turns)):
                if turns[nxt]["lang"] == "zh":
                    turn["paired_zh_turn_index"] = nxt
                    break
                if turns[nxt]["lang"] == "bo":
                    # Stop if another substantial bo turn appears before zh
                    if (turns[nxt]["end"] - turns[nxt]["start"]) >= 8.0:
                        break
    return turns


def encode_audio_slice_mp3_bytes(
    audio_array: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    bitrate: str = "48k",
) -> bytes:
    """Encodes a mono float32 waveform array into compact MP3 bytes in memory via ffmpeg."""
    pcm = np.asarray(audio_array, dtype=np.float32).tobytes()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(int(sample_rate)),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        "-f",
        "mp3",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=pcm, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return proc.stdout


def merge_for_asr(
    vad_segments: list[dict[str, Any]],
    seg_lang: list[str],
    merge_gap: float = 0.6,
    min_asr_sec: float = 0.8,
    max_block_sec: float = 40.0,
    bo_merge_gap: float = 0.8,
    bo_min_asr_sec: float = 0.25,
    bo_max_block_sec: float = 10.0,
) -> list[dict[str, Any]]:
    rows = []
    for s, lang in zip(vad_segments, seg_lang):
        if lang not in {"zh", "bo"}:
            continue
        rows.append({"start": float(s["start"]), "end": float(s["end"]), "lang": lang})

    if not rows:
        return []

    merged = [rows[0]]
    for cur in rows[1:]:
        prev = merged[-1]
        same_lang = cur["lang"] == prev["lang"]
        gap = cur["start"] - prev["end"]
        new_len = cur["end"] - prev["start"]
        cur_max_gap = bo_merge_gap if cur["lang"] == "bo" else merge_gap
        cur_max_len = bo_max_block_sec if cur["lang"] == "bo" else max_block_sec
        if same_lang and gap <= cur_max_gap and new_len <= cur_max_len:
            prev["end"] = cur["end"]
        else:
            merged.append(cur)

    final_blocks = []
    for m in merged:
        min_len = bo_min_asr_sec if m["lang"] == "bo" else min_asr_sec
        if (m["end"] - m["start"]) >= min_len:
            final_blocks.append(m)

    return final_blocks
