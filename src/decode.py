"""Decode CTC dan CER.

Model mengeluarkan urutan VISUAL; decode meng-collapse CTC lalu mengembalikan
string ke urutan LOGIS. CER selalu dihitung pada urutan logis ternormalisasi NFC.
"""

from typing import Sequence

import torch
from rapidfuzz.distance import Levenshtein

from src.tokenizer import BLANK, Tokenizer, nfc


def ctc_collapse(ids: Sequence[int]) -> list[int]:
    """Gabungkan indeks berulang bersebelahan, lalu buang blank."""
    out, prev = [], None
    for i in ids:
        if i != prev and i != BLANK:
            out.append(i)
        prev = i
    return out


def greedy_decode(logits: torch.Tensor, lengths: torch.Tensor, tokenizer: Tokenizer) -> list[str]:
    """logits (B, T, C) -> string urutan logis."""
    best = logits.argmax(-1).cpu()
    return [
        tokenizer.to_logical(tokenizer.decode(ctc_collapse(seq[: int(n)].tolist())))
        for seq, n in zip(best, lengths)
    ]


def cer(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """CER tingkat korpus: total edit distance / total panjang referensi."""
    edits = total = 0
    for ref, hyp in zip(references, hypotheses, strict=True):
        ref, hyp = nfc(ref), nfc(hyp)
        edits += Levenshtein.distance(ref, hyp)
        total += len(ref)
    return edits / max(1, total)


def ned(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    """NED tingkat korpus: total edit distance / total max(panjang referensi, panjang hipotesis) per baris.

    Selalu 0-1, sedangkan CER tidak berbatas atas (sisipan dibagi panjang referensi saja; keluaran kosong = 1).
    Pembagi dijumlah dulu seperti `cer`; kompetisi ICDAR 2019 (LSVT, ReCTS, ArT) memakai rata-rata rasio per baris.
    """
    edits = total = 0
    for ref, hyp in zip(references, hypotheses, strict=True):
        ref, hyp = nfc(ref), nfc(hyp)
        edits += Levenshtein.distance(ref, hyp)
        total += max(len(ref), len(hyp))
    return edits / max(1, total)
