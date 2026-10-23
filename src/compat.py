"""Compatibility and platform dynamic library helpers."""

import ctypes
import ctypes.util
import glob
import os
import sys
import warnings

# Prevent OpenMP runtime collision crash on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Suppress known noisy torchcodec and pyannote warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"(pyannote|speechbrain|torchcodec|torchaudio)")

_LIBLZMA_LOADED = False


def preload_liblzma() -> bool:
    """Dynamically locates and preloads liblzma for kenlm on macOS and Linux.

    Avoids hardcoding specific user directories like /Users/pfhsu/miniforge3.
    """
    global _LIBLZMA_LOADED
    if _LIBLZMA_LOADED:
        return True

    if sys.platform != "darwin":
        _LIBLZMA_LOADED = True
        return True

    # 1. Try finding via system util
    lib_name = ctypes.util.find_library("lzma")
    if lib_name:
        try:
            ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
            _LIBLZMA_LOADED = True
            return True
        except Exception:
            pass

    # 2. Search dynamically in Python prefix (Conda / Venv) and Homebrew
    candidates = [
        *glob.glob(os.path.join(sys.prefix, "lib", "liblzma*.dylib")),
        *glob.glob(os.path.join(sys.prefix, "..", "..", "lib", "liblzma*.dylib")),
        *glob.glob(os.path.join(sys.prefix, "..", "..", "pkgs", "liblzma-*", "lib", "liblzma*.dylib")),
        "/opt/homebrew/opt/xz/lib/liblzma.dylib",
        "/opt/homebrew/lib/liblzma.dylib",
        "/usr/local/opt/xz/lib/liblzma.dylib",
        "/usr/local/lib/liblzma.dylib",
    ]

    for p in candidates:
        if os.path.exists(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                _LIBLZMA_LOADED = True
                return True
            except Exception:
                pass

    return False
