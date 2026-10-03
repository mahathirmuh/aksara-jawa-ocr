"""Beam search awalan CTC dengan LM karakter (shallow fusion) — kondisi "kamus" ablasi koreksi pasca-OCR.

skor(awalan) = log P_ctc(awalan) + alpha * log P_lm(awalan) + beta * panjang

Modul terpisah dari jalur greedy (`src/decode.py`, `src/evaluate.py`): training dan ablasi Fase 5
mengimpor modul-modul itu, jadi jalur resmi tidak berubah. Keluaran berupa indeks urutan VISUAL;
kembalikan ke urutan logis dengan `tokenizer.to_logical(tokenizer.decode(ids))`.
"""

import math
from typing import Sequence

import numpy as np
import torch

from src.charlm import EOS, CharLM
from src.dataset import to_tensor
from src.decode import ctc_collapse
from src.infer import add_margin, normalize_polarity
from src.tokenizer import BLANK

NEG_INF = -math.inf


def logaddexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


@torch.no_grad()
def line_log_probs(img, model, device: str | torch.device = "cpu", pad_ratio: float = 0.0) -> np.ndarray:
    """Log-probabilitas (T, C) satu baris; praproses identik dengan `src.infer.predict`."""
    x = to_tensor(add_margin(normalize_polarity(img), pad_ratio)).unsqueeze(0).to(device)
    return model(x)[0].float().log_softmax(-1).cpu().numpy()


def greedy_ids(log_probs: np.ndarray) -> list[int]:
    return ctc_collapse(log_probs.argmax(-1).tolist())


def prefix_beam_search(
    log_probs: np.ndarray,
    charset: Sequence[str],
    lm: CharLM | None = None,
    alpha: float = 0.0,
    beta: float = 0.0,
    beam_width: int = 16,
    top_k: int = 8,
    token_min_logp: float = -12.0,
) -> list[int]:
    """log_probs (T, C) dengan blank = indeks 0; `charset[i - 1]` = karakter indeks i."""
    use_lm = lm is not None and alpha != 0.0
    lm_cache: dict[tuple, float] = {(): 0.0}

    def history(ids: Sequence[int]) -> str:
        return "".join(charset[i - 1] for i in (ids[-(lm.order - 1):] if lm.order > 1 else ()))

    def lm_logp(prefix: tuple) -> float:
        value = lm_cache.get(prefix)
        if value is None:
            value = lm_logp(prefix[:-1]) + lm.logprob(history(prefix[:-1]), charset[prefix[-1] - 1])
            lm_cache[prefix] = value
        return value

    def score(prefix: tuple, probs: Sequence[float], final: bool = False) -> float:
        s = logaddexp(probs[0], probs[1]) + beta * len(prefix)
        if use_lm:
            s += alpha * lm_logp(prefix)
            if final:
                s += alpha * lm.logprob(history(prefix), EOS)
        return s

    beams: dict[tuple, list[float]] = {(): [0.0, NEG_INF]}  # awalan -> [log p_blank, log p_nonblank]
    for row in log_probs:
        candidates = [int(k) for k in np.argsort(row)[::-1][:top_k] if k != BLANK and row[k] >= token_min_logp]
        blank = float(row[BLANK])
        nxt: dict[tuple, list[float]] = {}

        def add(prefix: tuple, slot: int, value: float) -> None:
            entry = nxt.get(prefix)
            if entry is None:
                entry = nxt[prefix] = [NEG_INF, NEG_INF]
            entry[slot] = logaddexp(entry[slot], value)

        for prefix, (pb, pnb) in beams.items():
            total = logaddexp(pb, pnb)
            add(prefix, 0, total + blank)
            last = prefix[-1] if prefix else None
            for k in candidates:
                p = float(row[k])
                if k == last:
                    add(prefix, 1, pnb + p)  # ulangan tanpa blank: tetap satu karakter
                    add(prefix + (k,), 1, pb + p)  # dipisah blank: karakter baru
                else:
                    add(prefix + (k,), 1, total + p)
        beams = dict(sorted(nxt.items(), key=lambda item: score(*item), reverse=True)[:beam_width])
    return list(max(beams.items(), key=lambda item: score(*item, final=True))[0])
