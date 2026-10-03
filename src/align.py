"""Penjelasan keluaran untuk tampilan: beda per suku kata, kolom citra per suku kata, skor kandidat.

Dipakai `scripts/export_results.py` (kontrak data web) dan `src/serve.py` (demo). Tidak dipakai
training maupun jalur gerbang resmi (`src/evaluate.py`).
"""

import math

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from torch.nn import functional as F

from src.charlm import EOS, CharLM
from src.dataset import DOWNSAMPLE, H
from src.tokenizer import BLANK, Tokenizer, logical_syllables, nfc


def line_cer(reference: str, hypothesis: str) -> float:
    """CER satu baris, NFC, urutan logis."""
    reference, hypothesis = nfc(reference), nfc(hypothesis)
    return Levenshtein.distance(reference, hypothesis) / max(1, len(reference))


def syllable_diff(reference: str, hypothesis: str) -> list[dict]:
    """Segmen hipotesis per suku kata logis.

    Status: ok (sama), sub (beda), ins (tidak ada di label), sp (spasi tidak ada di label), dan
    del (suku kata label yang tidak dibaca, disisipkan di posisinya). Suku kata dipakai sebagai
    satuan supaya pewarnaan tidak memecah pasangan (C + pangkon + C tetap satu suku kata).
    """
    ref_all, hyp_all = logical_syllables(nfc(reference)), logical_syllables(nfc(hypothesis))
    # Spasi dijajarkan terpisah: kalau ikut, "ꦏꦸ -> spasi" bisa sama murahnya dengan "ꦏꦸ -> ꦏꦶ".
    ref = [s for s in ref_all if s != " "]
    hyp = [s for s in hyp_all if s != " "]
    ref_space_after, count = set(), 0
    for s in ref_all:
        if s == " ":
            ref_space_after.add(count - 1)
        else:
            count += 1

    status = ["ok"] * len(hyp)
    aligned = list(range(len(hyp)))  # indeks suku kata label yang sejajar dengan hyp[j]
    deleted: dict[int, list[str]] = {}
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref, hyp):
        for j in range(j1, j2):
            if tag == "equal":
                aligned[j] = i1 + (j - j1)
            elif tag == "replace":
                status[j], aligned[j] = "sub", i1 + min(j - j1, i2 - i1 - 1)
            elif tag == "insert":
                status[j], aligned[j] = "ins", i1 - 1
        if tag == "delete" or (tag == "replace" and i2 - i1 > j2 - j1):
            start = i1 if tag == "delete" else i1 + (j2 - j1)
            deleted.setdefault(j2, []).extend(ref[start:i2])

    segments, j = [], 0
    for syllable in hyp_all:
        if syllable == " ":
            ok = j > 0 and aligned[j - 1] in ref_space_after
            segments.append({"t": " ", "s": "ok" if ok else "sp"})
            continue
        if j in deleted:
            segments.append({"t": "".join(deleted.pop(j)), "s": "del"})
        segments.append({"t": syllable, "s": status[j]})
        j += 1
    if len(hyp) in deleted:
        segments.append({"t": "".join(deleted[len(hyp)]), "s": "del"})
    return segments


def greedy_path(log_probs: np.ndarray) -> tuple[list[int], list[list[int]]]:
    """Token greedy (urutan visual) dan rentang frame [awal, akhir] tiap token."""
    ids, spans, prev = [], [], BLANK
    for t, k in enumerate(log_probs.argmax(-1).tolist()):
        if k != BLANK and k != prev:
            ids.append(k)
            spans.append([t, t])
        elif k != BLANK and k == prev:
            spans[-1][1] = t
        prev = k
    return ids, spans


def syllable_spans(log_probs: np.ndarray, tokenizer: Tokenizer, image_width: int, image_height: int,
                   pad_ratio: float = 0.0) -> list[list[float]] | None:
    """[x0, x1] (fraksi lebar citra ASLI) untuk setiap suku kata logis keluaran greedy.

    Urutan visual hanya memindahkan tanda di dalam satu suku kata, jadi batas suku kata sama di
    kedua urutan. Bila tidak sama (seharusnya tidak terjadi), kembalikan None alih-alih menebak.
    """
    ids, frames = greedy_path(log_probs)
    visual = tokenizer.decode(ids)
    logical = tokenizer.to_logical(visual)
    margin = round(image_height * pad_ratio)
    padded_w, padded_h = image_width + 2 * margin, image_height + 2 * margin
    width96 = max(DOWNSAMPLE, round(padded_w * H / padded_h))
    out, pos = [], 0
    for syllable in logical_syllables(logical):
        if sorted(visual[pos: pos + len(syllable)]) != sorted(syllable):
            return None
        span = frames[pos: pos + len(syllable)]
        pos += len(syllable)
        # Kolom pada citra 96 px -> piksel citra berpadding -> fraksi citra asli.
        x0 = (min(f[0] for f in span) * DOWNSAMPLE / width96 * padded_w - margin) / image_width
        x1 = ((max(f[1] for f in span) + 1) * DOWNSAMPLE / width96 * padded_w - margin) / image_width
        out.append([round(min(max(x0, 0.0), 1.0), 4), round(min(max(x1, 0.0), 1.0), 4)])
    return out


def ctc_logp(log_probs: np.ndarray, text: str, tokenizer: Tokenizer) -> float:
    """log P_CTC(teks | citra): seberapa didukung piksel teks ini, untuk kandidat dari sumber apa pun."""
    target = tokenizer.encode(tokenizer.to_visual(nfc(text)))
    lp = torch.from_numpy(log_probs).float().unsqueeze(1)  # (T, 1, C)
    if not target:
        return float(lp[:, 0, BLANK].sum())
    if len(target) > lp.shape[0]:
        return -math.inf
    loss = F.ctc_loss(lp, torch.tensor([target]), torch.tensor([lp.shape[0]]), torch.tensor([len(target)]),
                      blank=BLANK, reduction="sum", zero_infinity=False)
    return -float(loss)


def lm_logp(text: str, tokenizer: Tokenizer, lm: CharLM) -> float:
    """log P_LM teks (urutan visual, termasuk akhir baris), sama dengan skor LM di beam search."""
    visual = tokenizer.to_visual(nfc(text))
    total = 0.0
    for i, ch in enumerate(visual):
        total += lm.logprob(visual[max(0, i - lm.order + 1): i] if lm.order > 1 else "", ch)
    return total + lm.logprob(visual[-(lm.order - 1):] if lm.order > 1 and visual else "", EOS)
