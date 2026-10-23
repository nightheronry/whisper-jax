"""Export transcript artifacts to CSV, JSON, SRT, and RTF formats."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import opencc
import pyewts


def sec_to_srt_time(sec: float | int) -> str:
    """Formats seconds as SRT timestamp HH:MM:SS,ms."""
    sec = max(0.0, float(sec))
    whole = int(sec)
    ms = int(round((sec - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0
    s = whole % 60
    m = (whole // 60) % 60
    h = whole // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def sec_to_doc_time(sec: float | int) -> str:
    """Formats seconds as H:MM:SS for RTF document headings."""
    total_sec = max(0, int(round(float(sec))))
    s = total_sec % 60
    m = (total_sec // 60) % 60
    h = total_sec // 3600
    return f"{h}:{m:02d}:{s:02d}"


def merge_text_piece(left: str | None, right: str | None, lang: str) -> str:
    left = (left or "").strip()
    right = (right or "").strip()
    if not left:
        return right
    if not right:
        return left
    sep = "" if lang in {"zh", "bo"} else " "
    return (left + sep + right).strip()


def merge_transcript_rows(
    rows: list[dict[str, Any]],
    zh_max_gap: float = 1.2,
    zh_max_block_sec: float = 90.0,
) -> list[dict[str, Any]]:
    """Merges consecutive Chinese transcript rows into natural paragraphs."""
    if not rows:
        return []
    out = [rows[0].copy()]
    for cur in rows[1:]:
        prev = out[-1]
        same_lang = cur.get("lang") == prev.get("lang")
        same_source = cur.get("source") == prev.get("source")
        same_chunk = cur.get("chunk_index") == prev.get("chunk_index") and cur.get("audio_file") == prev.get("audio_file")
        gap = cur["start"] - prev["end"]
        new_len = cur["end"] - prev["start"]
        should_merge_zh = (
            cur.get("lang") == "zh"
            and same_lang
            and same_source
            and same_chunk
            and gap <= zh_max_gap
            and new_len <= zh_max_block_sec
        )
        if should_merge_zh:
            prev["end"] = cur["end"]
            prev["text"] = merge_text_piece(prev.get("text"), cur.get("text"), "zh")
            prev["text_zh"] = merge_text_piece(prev.get("text_zh"), cur.get("text_zh"), "zh")
        else:
            out.append(cur.copy())
    return out


def merge_subtitle_rows(
    rows: list[dict[str, Any]],
    zh_max_gap: float = 0.5,
    zh_max_block_sec: float = 12.0,
    bo_max_gap: float = 0.8,
    bo_max_block_sec: float = 12.0,
) -> list[dict[str, Any]]:
    """Merges rows into shorter chunks suitable for video subtitles (.srt)."""
    if not rows:
        return []
    out = [rows[0].copy()]
    for cur in rows[1:]:
        prev = out[-1]
        lang = cur.get("lang")
        same_lang = lang == prev.get("lang")
        same_source = cur.get("source") == prev.get("source")
        same_chunk = cur.get("chunk_index") == prev.get("chunk_index") and cur.get("audio_file") == prev.get("audio_file")
        gap = cur["start"] - prev["end"]
        new_len = cur["end"] - prev["start"]

        if lang == "zh":
            allowed_gap = zh_max_gap
            allowed_len = zh_max_block_sec
        elif lang == "bo":
            allowed_gap = bo_max_gap
            allowed_len = bo_max_block_sec
        else:
            allowed_gap = -1.0
            allowed_len = -1.0

        if same_lang and same_source and same_chunk and gap <= allowed_gap and new_len <= allowed_len:
            prev["end"] = cur["end"]
            prev["text"] = merge_text_piece(prev.get("text"), cur.get("text"), str(lang))
            prev["text_bo"] = merge_text_piece(prev.get("text_bo"), cur.get("text_bo"), "bo")
            prev["text_zh"] = merge_text_piece(prev.get("text_zh"), cur.get("text_zh"), "zh")
        else:
            out.append(cur.copy())
    return out


def rtf_escape(text: str) -> str:
    """Escapes Unicode characters for RTF output."""
    out = []
    for ch in str(text or ""):
        code = ord(ch)
        if ch == "\\":
            out.append(r"\\")
        elif ch == "{":
            out.append(r"\{")
        elif ch == "}":
            out.append(r"\}")
        elif ch == "\n":
            out.append(r"\line ")
        elif 32 <= code <= 126:
            out.append(ch)
        else:
            signed = code if code <= 32767 else code - 65536
            out.append(f"\\u{signed}")
    return "".join(out)


def build_rtf_sections(
    rows: list[dict[str, Any]],
    zh_max_gap: float = 1.2,
    zh_max_block_sec: float = 90.0,
    bo_max_gap: float = 1.2,
    bo_max_block_sec: float = 90.0,
) -> list[dict[str, Any]]:
    """Groups rows into coherent timestamped sections for RTF document export."""
    sections = []
    current = None
    for row in rows:
        lang = row.get("lang")
        if lang not in {"zh", "bo"}:
            continue
        if lang == "bo":
            has_content = bool((row.get("text_bo") or "").strip() or (row.get("text_zh") or "").strip())
            allowed_gap = bo_max_gap
            allowed_len = bo_max_block_sec
        else:
            has_content = bool((row.get("text_zh") or row.get("text") or "").strip())
            allowed_gap = zh_max_gap
            allowed_len = zh_max_block_sec
        if not has_content:
            continue

        item = row.copy()
        if current is None:
            current = {
                "lang": lang,
                "source": item.get("source"),
                "chunk_index": item.get("chunk_index"),
                "audio_file": item.get("audio_file"),
                "start": item["start"],
                "end": item["end"],
                "rows": [item],
            }
            continue

        same_lang = lang == current["lang"]
        same_source = item.get("source") == current.get("source")
        same_chunk = (
            item.get("chunk_index") == current.get("chunk_index") and item.get("audio_file") == current.get("audio_file")
        )
        gap = float(item["start"]) - float(current["end"])
        new_len = float(item["end"]) - float(current["start"])

        if same_lang and same_source and same_chunk and gap <= allowed_gap and new_len <= allowed_len:
            current["end"] = item["end"]
            current["rows"].append(item)
        else:
            sections.append(current)
            current = {
                "lang": lang,
                "source": item.get("source"),
                "chunk_index": item.get("chunk_index"),
                "audio_file": item.get("audio_file"),
                "start": item["start"],
                "end": item["end"],
                "rows": [item],
            }

    if current is not None:
        sections.append(current)
    return sections


def write_csv(rows: list[dict[str, Any]], path: str | Path, fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_srt(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            text_bo = (r.get("text_bo") or "").strip()
            text_zh = (r.get("text_zh") or r.get("text") or "").strip()
            f.write(f"{i}\n")
            f.write(f"{sec_to_srt_time(r['start'])} --> {sec_to_srt_time(r['end'])}\n")
            if r.get("lang") == "bo":
                if text_bo:
                    f.write(f"[藏文] {text_bo}\n")
                if text_zh:
                    f.write(f"[中譯] {text_zh}\n")
            else:
                f.write(f"[漢語] {text_zh}\n")
            f.write("\n")


def write_rtf_sections(sections: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"{\rtf1\ansi\deff0\uc0",
        r"{\fonttbl{\f0\fnil Helvetica;}}",
        r"\f0\fs24",
    ]
    for section in sections:
        title = f"{sec_to_doc_time(section['start'])}-{sec_to_doc_time(section['end'])}"
        lines.append(rtf_escape(title) + r"\line")
        if section["lang"] == "bo":
            for idx, row in enumerate(section["rows"]):
                text_bo = (row.get("text_bo") or "").strip()
                text_zh = (row.get("text_zh") or "").strip()
                if text_bo:
                    lines.append(rtf_escape(f"藏文：{text_bo}") + r"\line")
                if text_zh:
                    lines.append(rtf_escape(f"漢譯：{text_zh}") + r"\line")
                if idx != len(section["rows"]) - 1:
                    lines.append(r"\line")
        else:
            for row in section["rows"]:
                text_zh = (row.get("text_zh") or row.get("text") or "").strip()
                if text_zh:
                    lines.append(rtf_escape(text_zh) + r"\line")
        lines.append(r"\line\line")
    lines.append("}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_unicode_transcript_rows(
    rows: list[dict[str, Any]],
    ewts_converter: pyewts.pyewts | None = None,
) -> list[dict[str, Any]]:
    converter = ewts_converter or pyewts.pyewts()
    unicode_rows = []
    for row in rows:
        item = row.copy()
        text_bo = (item.get("text_bo") or "").strip()
        if item.get("lang") == "bo" and text_bo and text_bo.isascii():
            try:
                item["text_bo"] = converter.toUnicode(text_bo)
            except Exception:
                item["text_bo"] = text_bo
        unicode_rows.append(item)
    return unicode_rows


def convert_rows_zh_to_traditional(
    rows: list[dict[str, Any]],
    zh_converter: opencc.OpenCC | None = None,
) -> list[dict[str, Any]]:
    converter = zh_converter or opencc.OpenCC("s2t")
    converted_rows = []
    for row in rows:
        item = row.copy()
        orig_text = item.get("text") or ""
        orig_text_zh = item.get("text_zh") or ""
        if orig_text_zh:
            item["text_zh"] = converter.convert(orig_text_zh)
        if item.get("lang") == "zh":
            target = orig_text if orig_text else item.get("text_zh", "")
            if target:
                item["text"] = converter.convert(target)
        elif item.get("lang") == "bo" and orig_text_zh and orig_text.strip() == orig_text_zh.strip():
            item["text"] = item["text_zh"]
        converted_rows.append(item)
    return converted_rows


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_all_formats(
    output_dir: str | Path,
    timeline_rows: list[dict[str, Any]],
    asr_blocks: list[dict[str, Any]],
    transcript_rows_raw: list[dict[str, Any]],
    convert_s2t: bool = True,
) -> dict[str, Path]:
    """Generates all standard pipeline output files in output_dir."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    # 1. Timeline CSV
    if timeline_rows:
        timeline_path = out_dir / "speech_lang_timeline.csv"
        write_csv(
            timeline_rows,
            timeline_path,
            ["chunk_index", "audio_file", "start", "end", "lang", "source", "duration"],
        )
        results["timeline_csv"] = timeline_path

    # 2. ASR Blocks JSON
    if asr_blocks:
        blocks_path = out_dir / "asr_blocks.json"
        save_json(blocks_path, asr_blocks)
        results["asr_blocks_json"] = blocks_path

    # Process transcript rows
    processed_rows = transcript_rows_raw
    if convert_s2t:
        processed_rows = convert_rows_zh_to_traditional(processed_rows)

    ewts_conv = pyewts.pyewts()
    processed_rows = build_unicode_transcript_rows(processed_rows, ewts_conv)

    # Paragraph-merged rows
    paragraph_rows = merge_transcript_rows(processed_rows)

    # 3. Bilingual transcript CSV
    csv_path = out_dir / "bilingual_transcript.csv"
    write_csv(
        paragraph_rows,
        csv_path,
        ["chunk_index", "audio_file", "start", "end", "lang", "source", "text_bo", "text_zh", "text"],
    )
    results["transcript_csv"] = csv_path

    # 4. Bilingual transcript JSON (full structure)
    json_path = out_dir / "bilingual_transcript.json"
    save_json(json_path, paragraph_rows)
    results["transcript_json"] = json_path

    # 5. Unicode transcript JSON
    unicode_json_path = out_dir / "bilingual_transcript_unicode.json"
    save_json(unicode_json_path, paragraph_rows)
    results["transcript_unicode_json"] = unicode_json_path

    # 6. Subtitles SRT
    srt_rows = merge_subtitle_rows(
        processed_rows,
        zh_max_gap=0.5,
        zh_max_block_sec=12.0,
        bo_max_gap=0.8,
        bo_max_block_sec=12.0,
    )
    srt_path = out_dir / "bilingual_transcript.srt"
    write_srt(srt_rows, srt_path)
    results["transcript_srt"] = srt_path

    # 7. RTF Document
    rtf_sections = build_rtf_sections(
        processed_rows,
        zh_max_gap=1.2,
        zh_max_block_sec=90.0,
        bo_max_gap=1.2,
        bo_max_block_sec=90.0,
    )
    rtf_path = out_dir / "bilingual_transcript.rtf"
    write_rtf_sections(rtf_sections, rtf_path)
    results["transcript_rtf"] = rtf_path

    return results
