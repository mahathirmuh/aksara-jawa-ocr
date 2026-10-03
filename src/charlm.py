"""LM karakter n-gram (Witten-Bell) untuk beam search CTC — kondisi "kamus" ablasi koreksi pasca-OCR.

LM bekerja pada urutan VISUAL, yaitu ruang token keluaran CTC: satu token = satu karakter. Dibangun
HANYA dari split train. 87% label nyata ditulis tanpa spasi, jadi yang dipakai LM karakter, bukan
kamus kata; `--drop-space-prob` meniru `src.train --drop-space-prob`.

  python -m src.charlm --order 5 --lines 300000 --drop-space-prob 0.5
"""

import argparse
import math
import pickle
import random
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.tokenizer import TOKENIZER_PATH, Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits"
LM_DIR = ROOT / "data" / "charlm"
BOS, EOS = "\x02", "\x03"


class CharLM:
    def __init__(self, order: int, vocab: Iterable[str], grams: list[dict], contexts: list[dict],
                 meta: dict | None = None):
        self.order = order
        self.vocab = tuple(vocab)  # tanpa EOS
        self.grams = grams  # grams[k]: k-gram -> jumlah, k = 1..order
        self.contexts = contexts  # contexts[k]: riwayat k-1 karakter -> (total, jenis lanjutan)
        self.unigram_total = sum(grams[1].values())
        self.meta = meta or {}

    @classmethod
    def build(cls, lines: Iterable[str], order: int, vocab: Iterable[str], meta: dict | None = None) -> "CharLM":
        if order < 1:
            raise ValueError("order minimal 1")
        vocab = tuple(vocab)
        allowed = set(vocab) | {EOS}
        grams: list[Counter] = [Counter() for _ in range(order + 1)]
        pad = BOS * (order - 1)
        for line in lines:
            s = pad + line + EOS
            unknown = set(line) - allowed
            if unknown:
                raise ValueError(f"karakter di luar vocab: {sorted(f'U+{ord(c):04X}' for c in unknown)}")
            for k in range(1, order + 1):
                grams[k].update(s[i - k + 1 : i + 1] for i in range(order - 1, len(s)))
        contexts: list[dict] = [{} for _ in range(order + 1)]
        for k in range(2, order + 1):
            ctx: dict[str, tuple[int, int]] = {}
            for gram, n in grams[k].items():
                total, types = ctx.get(gram[:-1], (0, 0))
                ctx[gram[:-1]] = (total + n, types + 1)
            contexts[k] = ctx
        return cls(order, vocab, [dict(g) for g in grams], contexts, meta)

    def prob(self, history: str, ch: str) -> float:
        """P(ch | history), history = teks visual sebelumnya; ch boleh EOS. Jumlah atas vocab+EOS = 1."""
        p = (self.grams[1].get(ch, 0) + 1) / (self.unigram_total + len(self.vocab) + 1)
        if self.order == 1:
            return p
        padded = (BOS * (self.order - 1) + history)[-(self.order - 1):]
        for k in range(2, self.order + 1):
            h = padded[len(padded) - (k - 1):]
            total, types = self.contexts[k].get(h, (0, 0))
            if total:
                p = (self.grams[k].get(h + ch, 0) + types * p) / (total + types)
        return p

    def logprob(self, history: str, ch: str) -> float:
        return math.log(self.prob(history, ch))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"order": self.order, "vocab": self.vocab, "grams": self.grams,
                   "contexts": self.contexts, "meta": self.meta}
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "CharLM":
        with Path(path).open("rb") as f:
            d = pickle.load(f)
        return cls(d["order"], d["vocab"], d["grams"], d["contexts"], d["meta"])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--lines", type=int, default=300_000, help="baris train acak (seed); 0 = semua")
    parser.add_argument("--drop-space-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    tokenizer = Tokenizer.load(TOKENIZER_PATH)
    lines = (SPLITS_DIR / "train.txt").read_text(encoding="utf-8").splitlines()
    rng = random.Random(args.seed)
    rng.shuffle(lines)
    if args.lines:
        lines = lines[: args.lines]
    visual = [
        tokenizer.to_visual(line.replace(" ", "") if rng.random() < args.drop_space_prob else line)
        for line in lines
    ]
    started = time.time()
    meta = {"split": "train", "lines": len(lines), "drop_space_prob": args.drop_space_prob, "seed": args.seed}
    lm = CharLM.build(visual, args.order, tokenizer.charset, meta)
    out = Path(args.out) if args.out else LM_DIR / f"o{args.order}_n{len(lines)}_ds{args.drop_space_prob}.pkl"
    lm.save(out)
    sizes = ", ".join(f"{k}-gram {len(lm.grams[k]):,}" for k in range(1, lm.order + 1))
    print(f"LM order {lm.order} dari {len(lines):,} baris train ({time.time() - started:.0f} dtk): {sizes} -> {out}")


if __name__ == "__main__":
    main()
