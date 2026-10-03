"""Augmentasi teks karakter langka (Fase 5; hanya untuk data training).

Korpus hasil transliterasi hampir tidak memuat aksara murda, mahaprana, swara, pa cerek, nga
lelet, rerenggan, pada adeg-adeg, dan sebagainya: `src.corpus.inject_rare` hanya menjamin 100
baris per codepoint (0,011% baris train, ~11 kali pada `--train-lines 100000`), dan adeg-adeg
tidak pernah berada di awal baris, padahal cetakan memakainya untuk membuka paragraf. Uji buta
2026-09-14: adeg-adeg di tepi kiri potongan tidak pernah terbaca (0/10), aksara murda tertukar
dengan angka Jawa.

Karakter disisipkan pada teks SEBELUM dirender, jadi citra dan label tetap sama persis. Kebenaran
linguistik tidak diperlukan: model belajar bentuk glyph. Yang wajib: teks tetap well-formed, lolos
round-trip tokenizer, dan font yang merender punya glyph-nya. CarakanJawa tidak punya aksara O;
tanpa cek cmap, citra berisi kotak .notdef sementara label berisi O, dan tidak ada error.
"""

import random
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

import uharfbuzz as hb

from src.tokenizer import is_javanese, is_javanese_letter, is_mark, is_well_formed

RARE_MAX_LINE_FRACTION = 0.001  # codepoint yang muncul di < 0,1% baris dianggap langka
LINE_OPENERS = ("꧋",)  # pada adeg-adeg, pembuka paragraf di cetakan
N_COMMON_LETTERS = 20
MAX_ATTEMPTS = 8


def insert_codepoint(line: str, ch: str, common_letters: Sequence[str], rng) -> str | None:
    """Sisipkan satu codepoint ke baris; None bila tidak ada tempat yang cocok."""
    category = unicodedata.category(ch)
    if category == "Lo":  # aksara: gantikan satu aksara umum
        spots = [i for i, c in enumerate(line) if c in common_letters]
        if not spots:
            return None
        i = rng.choice(spots)
        return line[:i] + ch + line[i + 1 :]
    if category in ("Mn", "Mc"):  # sandhangan: tempelkan pada aksara yang belum bertanda
        spots = [
            i for i, c in enumerate(line)
            if is_javanese_letter(c) and (i + 1 == len(line) or not is_mark(line[i + 1]))
        ]
        if not spots:
            return None
        i = rng.choice(spots)
        return line[: i + 1] + ch + line[i + 1 :]
    # angka, pada, pangrangkep: sisipkan sebagai token di batas kata
    spots = [i for i, c in enumerate(line) if c == " "]
    if not spots:
        return None
    i = rng.choice(spots)
    return line[:i] + " " + ch + line[i:]


@lru_cache(maxsize=None)
def font_codepoints(font_path: str) -> frozenset[int]:
    """Codepoint yang ada di cmap font (dibaca HarfBuzz; tanpa fontTools)."""
    return frozenset(hb.Face(hb.Blob.from_file_path(font_path)).unicodes)


@dataclass(frozen=True)
class RareText:
    rare: tuple[str, ...]
    common_letters: tuple[str, ...]
    insert_prob: float = 0.0
    opener_prob: float = 0.0

    @classmethod
    def from_lines(cls, lines: Iterable[str], charset: Sequence[str], insert_prob: float = 0.0,
                   opener_prob: float = 0.0, max_line_fraction: float = RARE_MAX_LINE_FRACTION) -> "RareText":
        """Codepoint langka = codepoint aksara Jawa di charset yang muncul di sedikit baris `lines`."""
        lines = list(lines)
        per_line = Counter(ch for line in lines for ch in set(line))
        letters = Counter(ch for line in lines for ch in line if is_javanese_letter(ch))
        n = max(1, len(lines))
        rare = tuple(ch for ch in charset if is_javanese(ch) and per_line[ch] / n < max_line_fraction)
        common = tuple(ch for ch, _ in letters.most_common(N_COMMON_LETTERS))
        return cls(rare, common, insert_prob, opener_prob)

    @property
    def active(self) -> bool:
        return self.insert_prob > 0 or self.opener_prob > 0

    def __call__(self, text: str, rng: random.Random, font_path: str | Path,
                 round_trip: Callable[[str], bool]) -> str:
        """Teks dengan paling banyak satu codepoint langka dan satu pembuka baris tambahan.

        Probabilitas 0 tidak mengonsumsi angka acak. `round_trip(teks)` harus True bila tokenizer
        bisa mengembalikan urutan visualnya ke teks yang sama.
        """
        covered = font_codepoints(str(font_path))
        if self.insert_prob and rng.random() < self.insert_prob:
            choices = [ch for ch in self.rare if ord(ch) in covered]
            if choices:
                ch = rng.choice(choices)
                for _ in range(MAX_ATTEMPTS):
                    new = insert_codepoint(text, ch, self.common_letters, rng)
                    if new is not None and is_well_formed(new) and round_trip(new):
                        text = new
                        break
        if self.opener_prob and rng.random() < self.opener_prob:
            openers = [o for o in LINE_OPENERS if ord(o) in covered]
            if openers and not text.startswith(tuple(openers)):
                candidate = rng.choice(openers) + text
                # Sama dengan sisipan: tokenizer tanpa pembuka di charset akan gagal meng-encode label.
                if round_trip(candidate):
                    text = candidate
        return text
