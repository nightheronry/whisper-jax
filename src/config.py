"""Global configuration, model registries, and reference texts."""

import os
from pathlib import Path
from typing import Any
import torch

try:
    from dotenv import load_dotenv
    # Load .env if present
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from src.compat import preload_liblzma

# Ensure liblzma is loaded before any subsequent kenlm import
preload_liblzma()

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
N_SAMPLES = 480000  # 30s @ 16kHz
CHUNK_DURATION_MIN = 5

# --- Model Registries ---
ZH_ASR_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
ZH_ALIGN_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
BO_MPS_WHISPER_MODEL = "openpecha/op-whisper_small-ft-v2"
BO_ALIGN_MODEL = "openpecha/op-wav2vec2-ft-v17"
BO_KENLM_ARPA_PATH = CACHE_DIR / "bokenlm_syl_v04_3gram.arpa"

# --- Device Detection ---
def get_default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_default_compute_type(device: str) -> str:
    if device == "cuda":
        return "float16"
    return "float32"


# --- Lamrim Chenmo Reference Presets ---
LAMRIM_PRESETS: dict[str, str] = {
    "1": """
本堂課進度：《菩提道次第廣論》卷一・第 1 頁（皈敬頌、造者殊勝、法殊勝、講聞軌理）
- 皈敬頌（མཆོད་བརྗོད）：南無姑如曼殊廓喀耶（ན་མོ་གུ་རུ་མཉྫུ་གྷོ་ཥཱ་ཡ། 頂禮上師妙吉祥音）
- 俱胝圓滿妙善所生身，成滿無邊眾生希願語，如實觀見無餘所知意，怙主無上釋迦尊前禮。
- 佛陀（སངས་རྒྱས）、薄伽梵（བཅོམ་ལྡན་འདས）、釋迦牟尼佛（ཤཱཀྱ་ཐུབ་པ）
- 文殊師利菩薩（འཇམ་དཔལ་དབྱངས）、慈氏彌勒菩薩（བྱམས་པ་མགོན་པོ）
- 龍樹菩薩（ཀླུ་སྒྲུབ）、聖勇/無著菩薩（ཐོགས་མེད）、月稱菩薩（ཟླ་བ་གྲགས་པ）、世親菩薩（དབྱིག་གཉེན）
- 阿底峽尊者（ཇོ་བོ་རྗེ་དཔལ་ལྡན་ཨ་ཏི་ཤ / མར་མེ་མཛད་དཔལ་ཡེ་ཤེས）、宗喀巴大師（རྗེ་ཙོང་ཁ་པ）
- 造者殊勝（མཛད་པ་པོ་ཆེ་བའི་ཡོན་ཏན）、法殊勝（ཆོས་ཀྱི་ཆེ་བའི་ཡོན་ཏན）、如何講聞相應正法（ཆོས་ཇི་ལྟར་ཉན་བཤད་བྱ་བའི་ཚུལ）
""".strip(),
    "79": """
本堂課進度：《菩提道次第廣論》共下士道・思惟死無常・思惟決定死（第 79~80 頁）
- 思惟壽命無增每日減少，中間生存之際全無不減壽量之時。
- 悉被老病使者所牽，唯為死故導令前行（ན་བ་དང་རྒ་བའི་ཕོ་ཉས་ཁྲིད་ནས། འཆི་བའི་ཆེད་དུ་འགྲོ་བ་ཁོ་ནར་ཟད་པ）。
- 譬如從諸高峰墮時，未至地前空墜之際，不應歡樂（དཔེར་ན་བྲག་མཐོན་པོའི་རྩེ་མོ་ནས་ལྷུང་བ་ས་ལ་མ་ལྷུང་བའི་བར་སྣང་ལ་འགྲོ་བའི་སྐབས་དེ་ལ་དགའ་བ་མི་རིགས་པ་བཞིན）。
- འཆི་བདག་ (死主閻羅)、འཆི་བ་མི་རྟག་པ་ (死無常)、ཚེ་ལ་ངེས་པ་མེད་པ་ (壽無定準)、བག་མེད་ (放逸)
""".strip(),
    "80": """
本堂課進度：《菩提道次第廣論》共下士道・思惟死無常・思惟決定死（第 80 頁）
- 思惟壽命無增每日減少，中間生存之際全無不減壽量之時。
- 首從入胎，即無剎那而能安住，唯是趣向他世而行，故於中間生存之際，悉被老病使者所牽，唯為死故導令前行（ན་བ་དང་རྒ་བའི་ཕོ་ཉས་ཁྲིད་ནས། འཆི་བའི་ཆེད་དུ་འགྲོ་བ་ཁོ་ནར་ཟད་པ）。
- 故不應計於存活際，不趣後世安住歡喜，譬如從諸高峰墮時，未至地前空墜之際，不應歡樂（དཔེར་ན་བྲག་མཐོན་པོའི་རྩེ་མོ་ནས་ལྷུང་བ་ས་ལ་མ་ལྷུང་བའི་བར་སྣང་ལ་འགྲོ་བའི་སྐབས་དེ་ལ་དགའ་བ་མི་རིགས་པ་བཞིན）。
- 《四百頌釋》（བཞི་བརྒྱ་པའི་འགྲེལ་པ）引經：人中勇識如初夜，安住世間胎胞中，彼從此後日日中，全無暫息趣死前。
- 《破四倒論》（ཕྱིན་ཅི་ལོག་བཞི་བཟློག་པའི་བསྟན་བཅོས）：如從險峰墮地壞，豈於此空受安樂？從生為死常奔馳，有情於中豈得樂？
- 《本生論》（སྐྱེས་རབས）：現有老病死作害，大勢怨敵無能遮，定赴他世苦惱處，誰有心知思愛此？（སེམས་དང་ལྡན་པའི་སྐྱེ་བོ་ཤེས་ལྡན་ཅིག་ཡིན་ན...）
- འཆི་བདག་ (死主閻羅)、འཆི་བ་མི་རྟག་པ་ (死無常)、ཚེ་ལ་ངེས་པ་མེད་པ་ (壽無定準)、བག་མེད་ (放逸)、སྙིང་རྗེ་ (悲憫)
""".strip(),
    "81": """
本堂課進度：《菩提道次第廣論》共下士道・思惟死無常・思惟決定死（第 81 頁）
- 《本生論》（སྐྱེས་རབས）：現有老病死作害，大勢怨敵無能遮，定赴他世苦惱處，誰有心知思愛此？
- 智者辨別慧：若知如是無悲愍，未來定死，智者誰復放逸？
- འཆི་བདག་ (死主閻羅)、འཆི་བ་མི་རྟག་པ་ (死無常)、སྐྱེས་རབས (本生論)、བག་མེད་ (放逸)
""".strip(),
}


def resolve_lamrim_reference(ref_arg: str | None) -> str:
    """Resolves reference text from a page preset number, file path, or custom string."""
    if not ref_arg:
        return LAMRIM_PRESETS["1"]
    cleaned = str(ref_arg).strip().lower().lstrip("p").lstrip("0") or "0"
    if cleaned in LAMRIM_PRESETS:
        return LAMRIM_PRESETS[cleaned]
    if str(ref_arg).strip() in LAMRIM_PRESETS:
        return LAMRIM_PRESETS[str(ref_arg).strip()]
    try:
        p = Path(ref_arg)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8").strip()
    except OSError:
        # Long raw text strings are not valid file names; treat as raw reference text
        pass
    return str(ref_arg).strip()


# --- Vertex AI / Gemini Configuration ---
def get_vertex_config() -> dict[str, Any]:
    project = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0883795355")
    location = os.getenv("VERTEX_LOCATION", "global")
    api_key = os.getenv("GOOGLE_CLOUD_API_KEY", "")
    model_name = os.getenv("VERTEX_MODEL_NAME", "gemini-3.1-pro-preview")
    staging_gcs = os.getenv("VERTEX_STAGING_GCS_PREFIX", "gs://webuddhist/vertex-staging")
    thinking_level = os.getenv("VERTEX_THINKING_LEVEL", "LOW")

    return {
        "enabled": True,
        "mode": "hybrid_openpecha_vertex",
        "two_pass": True,
        "project": project,
        "location": location,
        "api_key": api_key,
        "model_name": model_name,
        "staging_gcs_prefix": staging_gcs,
        "thinking_level": thinking_level,
        "max_output_tokens": int(os.getenv("VERTEX_MAX_OUTPUT_TOKENS", "65535")),
        "retry_attempts": int(os.getenv("VERTEX_RETRY_ATTEMPTS", "5")),
        "retry_initial_delay_sec": float(os.getenv("VERTEX_RETRY_INITIAL_DELAY_SEC", "5")),
        "retry_backoff_multiplier": float(os.getenv("VERTEX_RETRY_BACKOFF_MULTIPLIER", "2")),
        "retry_max_delay_sec": float(os.getenv("VERTEX_RETRY_MAX_DELAY_SEC", "60")),
        "min_request_interval_sec": float(os.getenv("VERTEX_REQUEST_PAUSE_SEC", "2")),
    }
