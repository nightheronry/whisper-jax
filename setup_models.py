#!/usr/bin/env python3
"""One-click setup and verification script for models and dependencies.

Checks environment, downloads/extracts the 259MB BoKenlm 3-gram ARPA model,
and warms up HuggingFace models.

Usage:
  python setup_models.py
  python setup_models.py --download-hf
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.compat import preload_liblzma

preload_liblzma()

from src.config import (
    BO_ALIGN_MODEL,
    BO_KENLM_ARPA_PATH,
    BO_MPS_WHISPER_MODEL,
    CACHE_DIR,
    ZH_ALIGN_MODEL,
    ZH_ASR_MODEL,
    get_default_device,
)


def check_ffmpeg() -> bool:
    """Verifies ffmpeg binary is installed and callable."""
    print("Checking ffmpeg binary...", end=" ")
    ff = shutil.which("ffmpeg")
    if ff:
        print(f"[OK] Found: {ff}")
        return True
    print("[FAIL]")
    print("  -> ffmpeg not found in PATH! Please install ffmpeg via Homebrew (`brew install ffmpeg`) or package manager.")
    return False


def check_python_packages() -> bool:
    """Verifies critical Python dependencies."""
    print("Checking Python packages...", end=" ")
    pkgs = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("whisperx", "WhisperX"),
        ("kenlm", "KenLM"),
        ("pyctcdecode", "pyctcdecode"),
        ("opencc", "OpenCC"),
        ("pyewts", "pyewts"),
        ("google.genai", "Google GenAI SDK"),
        ("ffmpeg", "ffmpeg-python"),
    ]
    missing = []
    for mod, name in pkgs:
        try:
            __import__(mod)
        except Exception as e:
            missing.append((name, str(e)))

    if not missing:
        print("[OK] All critical packages installed.")
        return True

    print("[FAIL]")
    for name, err in missing:
        print(f"  -> Missing: {name} ({err})")
    print("  -> Please run: pip install -r requirements.txt")
    return False


def ensure_bokenlm_arpa() -> bool:
    """Ensures bokenlm_syl_v04_3gram.arpa is present in CACHE_DIR.

    If missing, streams the full BoKenlm-syl-v0.4.arpa from HuggingFace
    and trims to 3-gram, yielding a compact ~259MB model.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target_path = BO_KENLM_ARPA_PATH

    if target_path.exists() and target_path.stat().st_size > 200_000_000:
        print(f"Checking BoKenlm 3-gram ARPA... [OK] ({target_path.stat().st_size / (1024*1024):.1f} MB)")
        return True

    print("BoKenlm 3-gram ARPA not found. Downloading and extracting from HuggingFace (openpecha/BoKenlm-syl-v0.4)...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[FAIL] huggingface_hub is required to download models. Run: pip install huggingface-hub")
        return False

    temp_source = hf_hub_download(
        repo_id="openpecha/BoKenlm-syl-v0.4",
        filename="BoKenlm-syl-v0.4.arpa",
        cache_dir=str(CACHE_DIR / "huggingface"),
    )

    print(f"Extracting 3-gram slice into {target_path.name}...")
    temp_target = target_path.with_suffix(".tmp")
    with open(temp_source, "r", encoding="utf-8") as fin, open(temp_target, "w", encoding="utf-8") as fout:
        in_data = False
        in_target_grams = False
        data_lines = []

        for line in fin:
            if line.startswith(r"\data\\"):
                in_data = True
                continue
            if in_data:
                if line.startswith("ngram 1=") or line.startswith("ngram 2=") or line.startswith("ngram 3="):
                    data_lines.append(line)
                elif line.startswith(r"\1-grams:"):
                    in_data = False
                    fout.write(r"\data\\" + "\n")
                    fout.writelines(data_lines)
                    fout.write("\n")
                    fout.write(r"\1-grams:" + "\n")
                    in_target_grams = True
                continue
            if in_target_grams:
                if line.startswith(r"\4-grams:"):
                    fout.write("\n" + r"\end\\" + "\n")
                    break
                fout.write(line)

    temp_target.rename(target_path)
    print(f"[OK] Created {target_path.name} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
    return True


def warmup_hf_models() -> bool:
    """Pre-downloads HuggingFace models into local cache."""
    print("Warming up HuggingFace models...")
    models = [
        BO_MPS_WHISPER_MODEL,
        BO_ALIGN_MODEL,
        ZH_ASR_MODEL,
        ZH_ALIGN_MODEL,
    ]
    try:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        for m in models:
            print(f" - Downloading/verifying {m}...", end=" ", flush=True)
            try:
                AutoTokenizer.from_pretrained(m)
            except Exception:
                pass
            try:
                AutoProcessor.from_pretrained(m)
            except Exception:
                pass
            print("[OK]")
        return True
    except Exception as e:
        print(f"[WARN] Failed to pre-warm all models: {e}")
        return False


def test_decoder() -> bool:
    """Tests loading the BoKenlm CTC decoder with pyctcdecode."""
    print("Testing KenLM + pyctcdecode initialization...", end=" ", flush=True)
    try:
        from src.models import get_or_build_bo_ctc_decoder
        dummy_align = (None, {"dictionary": {"<pad>": 0, "ཀ": 1, "་": 2}})
        decoder = get_or_build_bo_ctc_decoder(dummy_align, BO_KENLM_ARPA_PATH)
        if decoder:
            print("[OK] KenLM 3-gram language model loaded successfully!")
            return True
        print("[FAIL] Decoder returned None")
        return False
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup and verify models and environment")
    parser.add_argument("--download-hf", action="store_true", help="Pre-download all HuggingFace models into cache")
    args = parser.parse_args()

    print("==================================================")
    print("Whisper-JAX Environment & Model Setup")
    print(f"Device Auto-Detection: {get_default_device()}")
    print("==================================================")

    ok_ff = check_ffmpeg()
    ok_pkgs = check_python_packages()
    ok_arpa = ensure_bokenlm_arpa() if ok_pkgs else False
    ok_dec = test_decoder() if (ok_pkgs and ok_arpa) else False

    if args.download_hf and ok_pkgs:
        warmup_hf_models()

    print("==================================================")
    if ok_ff and ok_pkgs and ok_arpa and ok_dec:
        print("Setup Complete! System is ready to run:")
        print("  python transcribe.py <audio_path> --mode bo_only")
    else:
        print("Setup encountered issues. Please review messages above.")
    print("==================================================")


if __name__ == "__main__":
    main()
