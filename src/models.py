"""ASR and Alignment Model loaders (Whisper, Wav2Vec2 CTC, BoKenlm)."""

import os
from pathlib import Path
from typing import Any
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.generation import GenerationMixin
import whisperx

from src.compat import preload_liblzma
from src.config import (
    BO_ALIGN_MODEL,
    BO_KENLM_ARPA_PATH,
    BO_MPS_WHISPER_MODEL,
    SAMPLE_RATE,
    ZH_ALIGN_MODEL,
    ZH_ASR_MODEL,
)

preload_liblzma()

_BO_CTC_BEAM_DECODER_CACHE: dict[str, Any] = {}


class MPSWhisperPipeline:
    """Runs batch inference for Tibetan Whisper on Apple Silicon MPS (or CUDA/CPU).

    Uses float32 on MPS to avoid SDPA cross-attention overflow and custom prompt IDs.
    """

    def __init__(self, model_id: str = BO_MPS_WHISPER_MODEL, device: str = "mps", batch_size: int = 6):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self._gen_fn = GenerationMixin.generate
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.model.generation_config.forced_decoder_ids = None
        self.model.config.forced_decoder_ids = None

        tokenizer = self.processor.tokenizer
        sot_seq = tokenizer.prefix_tokens
        bo_token_id = tokenizer.convert_tokens_to_ids("<|bo|>")
        transcribe_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")

        if bo_token_id is not None and transcribe_id is not None and notimestamps_id is not None:
            self.prompt_ids = [tokenizer.convert_tokens_to_ids("<|startoftranscript|>"), bo_token_id, transcribe_id, notimestamps_id]
        else:
            self.prompt_ids = sot_seq

    def transcribe_batch(self, audio_chunks: list[np.ndarray]) -> list[str]:
        if not audio_chunks:
            return []
        results = []
        for i in range(0, len(audio_chunks), self.batch_size):
            batch = audio_chunks[i : i + self.batch_size]
            inputs = self.processor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            feats = inputs.input_features.to(self.device, dtype=torch.float32)
            dec_ids = torch.tensor([self.prompt_ids] * len(batch), dtype=torch.long, device=self.device)

            with torch.no_grad():
                gen_ids = self._gen_fn(
                    self.model,
                    inputs=feats,
                    decoder_input_ids=dec_ids,
                    max_new_tokens=96,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.35,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0,
                )
            if self.device == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
            decoded = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
            for txt in decoded:
                t = (txt or "").strip()
                if t and t.isascii():
                    t = clean_wylie_repetitions(t)
                results.append(t)
        return results


def load_unigram_set_from_arpa(arpa_path: str | Path) -> set[str]:
    """Reads 1-grams from an ARPA file."""
    unigrams = set()
    in_1grams = False
    with open(arpa_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "\\1-grams:":
                in_1grams = True
                continue
            if stripped.startswith("\\") and stripped.endswith(":") and in_1grams:
                break
            if in_1grams and stripped:
                parts = stripped.split("\t")
                if len(parts) >= 2 and parts[1].strip():
                    word = parts[1].strip()
                    if not word.startswith("<") and not word.endswith(">"):
                        unigrams.add(word)
    return unigrams


def get_or_build_bo_ctc_decoder(bo_align_tuple: Any, arpa_path: str | Path | None = None) -> Any:
    """Builds and caches pyctcdecode + BoKenlm-syl-v0.4 beam search decoder."""
    if not bo_align_tuple:
        return None
    if arpa_path is None:
        arpa_path = BO_KENLM_ARPA_PATH

    arpa_str = str(Path(arpa_path).resolve())
    if not os.path.exists(arpa_str):
        return None
    if arpa_str in _BO_CTC_BEAM_DECODER_CACHE:
        return _BO_CTC_BEAM_DECODER_CACHE[arpa_str]

    from pyctcdecode import build_ctcdecoder

    _, align_meta = bo_align_tuple
    dictionary = align_meta["dictionary"]
    labels = [None] * len(dictionary)
    for ch, idx in dictionary.items():
        try:
            labels[int(idx)] = str(ch)
        except (ValueError, TypeError):
            labels[int(ch)] = str(idx)

    valid_chars = {c for c in labels if len(c) == 1 and c != " "}
    raw_unigrams = load_unigram_set_from_arpa(arpa_str)
    clean_unigrams = [u for u in raw_unigrams if u and all(c in valid_chars for c in u)]

    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=arpa_str,
        unigrams=clean_unigrams,
        alpha=0.5,
        beta=1.0,
    )
    _BO_CTC_BEAM_DECODER_CACHE[arpa_str] = decoder
    return decoder


def clean_wylie_repetitions(text: str) -> str:
    """Removes accidental tail repetition loops produced by Wylie acoustic models."""
    text = (text or "").strip()
    if not text:
        return ""
    words = text.split()
    if len(words) <= 2:
        return text
    changed = True
    while changed:
        changed = False
        for n in range(min(8, len(words) // 2), 0, -1):
            i = 0
            out = []
            while i < len(words):
                gram = words[i : i + n]
                if i + 2 * n <= len(words) and words[i + n : i + 2 * n] == gram:
                    out.extend(gram)
                    i += n
                    while i + n <= len(words) and words[i : i + n] == gram:
                        i += n
                    changed = True
                else:
                    out.append(words[i])
                    i += 1
            words = out
    return " ".join(words)


def truncate_by_ctc_syllables(whisper_uni: str, ctc_uni: str, ratio: float = 1.18, extra_syl: int = 2) -> str:
    """Uses frame-synchronous CTC syllable count to truncate Whisper tail hallucinations."""
    if not whisper_uni or not ctc_uni:
        return whisper_uni
    ctc_syls = [s for s in ctc_uni.replace("།", "་").split("་") if s.strip()]
    w_syls = [s for s in whisper_uni.replace("།", "་").split("་") if s.strip()]
    max_syls = max(1, int(len(ctc_syls) * ratio) + extra_syl)
    if len(w_syls) > max_syls:
        return "་".join(w_syls[:max_syls]) + "་"
    return whisper_uni


def decode_wav2vec2_ctc_beam(
    audio_segment: np.ndarray,
    bo_align_tuple: Any,
    ctc_decoder: Any = None,
    beam_width: int = 32,
    device: str = "cpu",
    return_tuple: bool = False,
) -> str | tuple[str, str]:
    """Wav2Vec2 CTC Forward + Beam Search with BoKenlm Language Model."""
    if not bo_align_tuple or len(audio_segment) < 400:
        return ("", "") if return_tuple else ""

    if ctc_decoder is None:
        ctc_decoder = get_or_build_bo_ctc_decoder(bo_align_tuple)

    align_model, align_meta = bo_align_tuple
    dictionary = align_meta.get("dictionary", {})
    id_to_char = {v: k for k, v in dictionary.items()}

    wf = torch.from_numpy(audio_segment).float().unsqueeze(0)
    wf = (wf - wf.mean()) / torch.sqrt(wf.var() + 1e-7)
    dev = "mps" if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
    wf = wf.to(dev)

    with torch.no_grad():
        logits_tensor = align_model(wf).logits[0]
        logits = logits_tensor.cpu().numpy()

    # Greedy decode
    pred_ids = np.argmax(logits, axis=-1).tolist()
    collapsed = []
    prev = None
    for pid in pred_ids:
        if pid != prev:
            ch = id_to_char.get(pid, "")
            if ch and ch.lower() not in {"[pad]", "<pad>", "<s>", "</s>", "[unk]", "<unk>", "|"}:
                collapsed.append("་" if ch == " " else ch)
        prev = pid
    greedy_txt = "".join(collapsed).replace("་་", "་").strip("་ ")

    # KenLM Beam Search decode
    if ctc_decoder is not None:
        try:
            beam_raw = ctc_decoder.decode(logits, beam_width=beam_width)
            beam_txt = "་".join(w for w in beam_raw.split(" ") if w.strip())
            if beam_txt and not beam_txt.endswith(("་", "།")):
                beam_txt += "་"
        except Exception:
            beam_txt = greedy_txt
    else:
        beam_txt = greedy_txt

    if return_tuple:
        return beam_txt, greedy_txt
    return beam_txt or greedy_txt


def load_all_models(
    device: str,
    compute_type: str,
    transcribe_zh: bool = False,
    use_mps_for_bo: bool = True,
) -> dict[str, Any]:
    """Loads and returns all requested ASR, Alignment, and VAD models."""
    print(f"Loading models on device={device} (compute_type={compute_type})...")

    # 1. Chinese ASR (always used for LID, and for ASR if transcribe_zh=True)
    zh_model = whisperx.load_model(
        ZH_ASR_MODEL,
        device if device != "mps" else "cpu",
        compute_type=compute_type if device != "mps" else "float32",
        language="zh",
    )
    asr_models = {"zh": zh_model}

    # 2. Tibetan ASR
    if use_mps_for_bo and device == "mps":
        bo_model = MPSWhisperPipeline(BO_MPS_WHISPER_MODEL, device="mps")
    else:
        bo_model = whisperx.load_model(
            BO_MPS_WHISPER_MODEL,
            device if device != "mps" else "cpu",
            compute_type=compute_type if device != "mps" else "float32",
            language="bo",
        )
    asr_models["bo"] = bo_model

    # 3. Align models
    align_models = {"zh": None, "bo": None}
    if transcribe_zh:
        try:
            align_models["zh"] = whisperx.load_align_model("zh", device if device != "mps" else "cpu", model_name=ZH_ALIGN_MODEL)
        except Exception as e:
            print(f"[WARN] Chinese align model failed: {e}")

    try:
        align_dev = "mps" if (device == "mps" and torch.backends.mps.is_available()) else "cpu"
        align_models["bo"] = whisperx.load_align_model("bo", align_dev, model_name=BO_ALIGN_MODEL)
    except Exception as e:
        print(f"[WARN] Tibetan align model failed: {e}")

    # 4. VAD model (already initialized within zh_model pipeline)
    vad_model = getattr(zh_model, "vad_model", None)

    return {
        "asr_models": asr_models,
        "align_models": align_models,
        "vad_model": vad_model,
    }
