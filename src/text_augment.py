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

import numpy as np
import uharfbuzz as hb
from PIL import Image, ImageDraw

from src.render import load_font
from src.tokenizer import is_javanese, is_javanese_letter, is_mark, is_well_formed

RARE_MAX_LINE_FRACTION = 0.001  # codepoint yang muncul di < 0,1% baris dianggap langka
LINE_OPENERS = ("꧋",)  # pada adeg-adeg, pembuka paragraf di cetakan
N_COMMON_LETTERS = 20
MAX_ATTEMPTS = 8
SIMILARITY_SIZE = 64  # ukuran render untuk membandingkan glyph
SIMILARITY_BASE = "ꦏ"  # sandhangan dibandingkan dalam keadaan menempel pada ka
INVISIBLE_MARK_PIXELS = 8  # ka + tanda berbeda <= sekian piksel dari ka polos: tandanya tidak tampak


def insert_codepoint(line: str, ch: str, common_letters: Sequence[str], rng, spaced: bool = True) -> str | None:
    """Sisipkan satu codepoint ke baris; None bila tidak ada tempat yang cocok.

    `spaced=False`: pada/pangrangkep ditempelkan ke kata sebelumnya seperti tanda baca korpus ("kata꧈ kata"),
    tanpa menambah spasi. Cetakan tidak mengapit pada dengan spasi, dan sisipan berspasi mengajari model
    mengeluarkan spasi di sekitar pada.
    """
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
    return line[:i] + (" " if spaced else "") + ch + line[i:]


@lru_cache(maxsize=None)
def font_codepoints(font_path: str) -> frozenset[int]:
    """Codepoint yang ada di cmap font (dibaca HarfBuzz; tanpa fontTools)."""
    return frozenset(hb.Face(hb.Blob.from_file_path(font_path)).unicodes)


def glyph_similarity(font_path: str, charset: Sequence[str]) -> dict[str, tuple[float, str]]:
    """Untuk tiap codepoint charset yang ada di font: IoU tertinggi terhadap glyph codepoint lain, dan lawannya.

    Tiap codepoint dirender sendirian di posisi yang sama lalu tintanya dibandingkan dengan codepoint sejenis.
    - Sandhangan dirender menempel pada ka dan dibandingkan TANPA tinta ka: ka sama di kedua sisi, dan kalau
      ikut dihitung semua tanda kecil tampak mirip (cecak vs tanpa tanda 0,96 di Noto). Tanda yang tidak
      menambah tinta sama sekali (ARDemak: panyangga) dianggap tidak tampak (IoU 1).
    - Pasangan aksara-angka tidak dibandingkan: glyph-nya memang kembar di banyak font (ga = angka satu,
      la = angka tujuh, E = angka enam), dan model membedakannya dari konteks (aksara di dalam kata, angka
      di dalam deret angka).
    Glyph tanpa tinta dianggap sama dengan apa pun (IoU 1).
    """
    covered = font_codepoints(font_path)
    chars = [ch for ch in charset if ord(ch) in covered]
    font = load_font(font_path, SIMILARITY_SIZE)
    size = SIMILARITY_SIZE

    def mask(text: str) -> np.ndarray:
        canvas = Image.new("L", (6 * size, 4 * size), 0)
        ImageDraw.Draw(canvas).text((2 * size, 2 * size), text, font=font, fill=255, anchor="ls")
        return (np.asarray(canvas) > 127).reshape(-1)

    marks = [is_mark(ch) for ch in chars]
    kinds = [unicodedata.category(ch) for ch in chars]
    base = mask(SIMILARITY_BASE).astype(np.float32)
    masks = np.stack([mask(SIMILARITY_BASE + ch if m else ch) for ch, m in zip(chars, marks)]).astype(np.float32)
    ink = masks.sum(axis=1)
    overlap = masks @ masks.T
    differ = ink[:, None] + ink[None, :] - 2 * overlap  # piksel yang hanya ada di salah satu
    base_ink = float(base.sum())
    unlike_base = ink + base_ink - 2 * (masks @ base)  # beda terhadap ka polos

    def similarity(i: int, j: int) -> float:
        if marks[i]:
            # Tinta tanda saja: jumlah tinta dikurangi tinta ka; ka yang tergeser tanda pre-base ikut saling hapus.
            own = ink[i] + ink[j] - 2 * base_ink
            return float((own - differ[i, j]) / (own + differ[i, j])) if own > 0 else float(differ[i, j] == 0)
        union = ink[i] + ink[j] - overlap[i, j]
        return float(overlap[i, j] / union) if union > 0 else 1.0

    best: dict[str, tuple[float, str]] = {}
    for i, ch in enumerate(chars):
        if ink[i] == 0:
            best[ch] = (1.0, "(kosong)")
            continue
        if marks[i] and unlike_base[i] <= INVISIBLE_MARK_PIXELS:
            best[ch] = (1.0, "(tanpa tanda)")
            continue
        rivals = [j for j in range(len(chars))
                  if j != i and marks[j] == marks[i] and {kinds[i], kinds[j]} != {"Lo", "Nd"}]
        scores = [(similarity(i, j), chars[j]) for j in rivals]
        best[ch] = max(scores, default=(0.0, ""))
    return best


@lru_cache(maxsize=None)
def similar_codepoints(font_path: str, charset: tuple[str, ...], threshold: float) -> frozenset[str]:
    """Codepoint yang glyph-nya di font ini nyaris sama dengan glyph codepoint sejenis (IoU >= threshold).

    Menyisipkannya memberi dua label untuk gambar yang sama: tolong = tarung dan panyangga tak tampak di
    ARDemak, rerenggan kiri = kanan di GumregahNew, pada windu = angka nol di Noto (dua-duanya token lepas,
    jadi konteks tidak membedakan), da mahaprana ~ dda (0,94) di abmAksaJawa. Setelah fase6_rare, model
    membaca dda cetakan sebagai da mahaprana 70 kali.
    """
    return frozenset(ch for ch, (iou, _) in glyph_similarity(font_path, charset).items() if iou >= threshold)


@dataclass(frozen=True)
class RareText:
    rare: tuple[str, ...]
    common_letters: tuple[str, ...]
    insert_prob: float = 0.0
    opener_prob: float = 0.0
    # Perbaikan sesudah fase6_rare; nilai bawaan = perilaku fase6_rare (dan scripts/eval_rare.py).
    charset: tuple[str, ...] = ()  # codepoint Jawa di charset tokenizer, pembanding kemiripan glyph
    max_similarity: float = 0.0  # > 0: lewati codepoint yang glyph-nya di font terpilih mirip glyph lain
    attach: bool = False  # True: pada/pangrangkep ditempelkan ke kata sebelumnya, tanpa spasi tambahan

    def __post_init__(self) -> None:
        if self.max_similarity > 0 and not self.charset:
            raise ValueError("max_similarity butuh charset pembanding; pakai RareText.from_lines")

    @classmethod
    def from_lines(cls, lines: Iterable[str], charset: Sequence[str], insert_prob: float = 0.0,
                   opener_prob: float = 0.0, max_line_fraction: float = RARE_MAX_LINE_FRACTION,
                   max_similarity: float = 0.0, attach: bool = False) -> "RareText":
        """Codepoint langka = codepoint aksara Jawa di charset yang muncul di sedikit baris `lines`."""
        lines = list(lines)
        per_line = Counter(ch for line in lines for ch in set(line))
        letters = Counter(ch for line in lines for ch in line if is_javanese_letter(ch))
        n = max(1, len(lines))
        javanese = tuple(ch for ch in charset if is_javanese(ch))
        rare = tuple(ch for ch in javanese if per_line[ch] / n < max_line_fraction)
        common = tuple(ch for ch, _ in letters.most_common(N_COMMON_LETTERS))
        return cls(rare, common, insert_prob, opener_prob, javanese, max_similarity, attach)

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
        similar = (similar_codepoints(str(font_path), self.charset, self.max_similarity)
                   if self.max_similarity else frozenset())
        if self.insert_prob and rng.random() < self.insert_prob:
            choices = [ch for ch in self.rare if ord(ch) in covered]
            # Pilih dulu, baru saring: jatah codepoint yang dilewati tidak pindah ke codepoint lain, jadi
            # codepoint yang tidak tersaring disisipkan sesering tanpa saringan.
            ch = rng.choice(choices) if choices else None
            if ch is not None and ch not in similar:
                for _ in range(MAX_ATTEMPTS):
                    new = insert_codepoint(text, ch, self.common_letters, rng, spaced=not self.attach)
                    if new is not None and is_well_formed(new) and round_trip(new):
                        text = new
                        break
        if self.opener_prob and rng.random() < self.opener_prob:
            openers = [o for o in LINE_OPENERS if ord(o) in covered and o not in similar]
            if openers and not text.startswith(tuple(openers)):
                candidate = rng.choice(openers) + text
                # Sama dengan sisipan: tokenizer tanpa pembuka di charset akan gagal meng-encode label.
                if round_trip(candidate):
                    text = candidate
        return text
