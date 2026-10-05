"""Tahap 3 alur (arti): terjemahan Jawa (Latin) <-> Indonesia dengan NLLB-200-distilled-600M, lokal di CPU.

Bukan bagian repo OCR (src/): tahap 2-4 hidup di web/. Model dari HuggingFace (facebook/nllb-200-distilled-600M,
CC-BY-NC 4.0: hanya non-komersial). Tidak ada data yang dikirim ke layanan luar. Satu model melayani dua arah:
bawaannya Jawa -> Indonesia (tahap arti, batch dan Demo); arah sebaliknya dipakai halaman Terjemahan.
"""

import os
import threading
from functools import lru_cache

MODEL = os.environ.get("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
SRC, TGT = "jav_Latn", "ind_Latn"
# Kode bahasa yang dipakai web -> kode NLLB. Bahasa Jawa selalu beraksara Latin di sini; alih aksara dikerjakan web.
LANGUAGES = {"jv": "jav_Latn", "id": "ind_Latn"}
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _load():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(int(os.environ.get("NLLB_THREADS", "4")))  # sisakan CPU untuk training
    tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang=SRC)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
    return tokenizer, model


def translate(texts: list[str], batch_size: int = 16, max_new_tokens: int = 96, src: str = SRC, tgt: str = TGT) -> list[str]:
    """Terjemahkan daftar teks dari bahasa `src` ke `tgt` (kode NLLB); teks kosong menghasilkan string kosong."""
    import torch

    tokenizer, model = _load()
    out: list[str] = [""] * len(texts)
    todo = [(i, t.strip()) for i, t in enumerate(texts) if t and t.strip()]
    target = tokenizer.convert_tokens_to_ids(tgt)
    with _lock, torch.inference_mode():
        tokenizer.src_lang = src  # dibaca tokenizer saat menyandi; diatur di dalam kunci karena satu tokenizer dipakai bersama
        for start in range(0, len(todo), batch_size):
            chunk = todo[start: start + batch_size]
            enc = tokenizer([t for _, t in chunk], return_tensors="pt", padding=True, truncation=True, max_length=128)
            gen = model.generate(**enc, forced_bos_token_id=target, max_new_tokens=max_new_tokens, num_beams=4)
            for (i, _), text in zip(chunk, tokenizer.batch_decode(gen, skip_special_tokens=True)):
                out[i] = text
    return out
