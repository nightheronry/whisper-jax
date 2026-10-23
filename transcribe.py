#!/usr/bin/env python3
"""CLI Entry point for Tibetan Speech-to-Text & Translation Pipeline.

Usage:
  python transcribe.py 900121b_P001.mp3 --mode bo_only --lamrim-page 1
  python transcribe.py 900121b_P001.mp3 --mode bilingual --device mps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported cleanly
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.compat import preload_liblzma

# Preload liblzma before any other module loads kenlm
preload_liblzma()

from src.config import get_default_device, get_vertex_config
from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tibetan & Chinese Speech-to-Text with Pass 2 Translation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "audio",
        help="Path to input audio file (mp3, wav, m4a, flac, etc.)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (defaults to <audio_filename_stem>/)",
    )
    parser.add_argument(
        "--mode",
        choices=["bo_only", "bilingual"],
        default="bo_only",
        help="Transcription mode: 'bo_only' (transcribe Geshe Tibetan & translate to Chinese) or 'bilingual' (include on-site Chinese interpreter)",
    )
    parser.add_argument(
        "-p",
        "--lamrim-page",
        default="1",
        help="Lamrim Chenmo page number preset (e.g. 1) to anchor Dharma terms",
    )
    parser.add_argument(
        "-r",
        "--reference-text",
        default=None,
        help="Path to custom reference scripture text or raw text string",
    )
    parser.add_argument(
        "--device",
        default=get_default_device(),
        choices=["mps", "cuda", "cpu"],
        help="Compute device for inference",
    )
    parser.add_argument(
        "--chunk-duration",
        type=int,
        default=5,
        help="Slice duration in minutes for ffmpeg chunking",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-processing of chunks, ignoring existing chunk checkpoints",
    )
    parser.add_argument(
        "--no-vertex",
        action="store_true",
        help="Skip Pass 2 Vertex AI orthography restoration & translation",
    )
    parser.add_argument(
        "--vertex-model",
        default=None,
        help="Override default Gemini model ID (e.g. gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Google Cloud API key for Vertex AI express mode",
    )

    args = parser.parse_args()

    # Configure Vertex AI
    v_cfg = get_vertex_config()
    if args.no_vertex:
        v_cfg["enabled"] = False
    if args.vertex_model:
        v_cfg["model_name"] = args.vertex_model
    if args.api_key:
        v_cfg["api_key"] = args.api_key

    try:
        run_pipeline(
            audio_file=args.audio,
            output_dir=args.output_dir,
            mode=args.mode,
            lamrim_page=args.lamrim_page,
            reference_text=args.reference_text,
            device=args.device,
            chunk_duration_min=args.chunk_duration,
            force_reprocess=args.force,
            vertex_config=v_cfg,
        )
    except Exception as exc:
        print(f"\n[ERROR] Pipeline execution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
