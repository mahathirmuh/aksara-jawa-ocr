"""Test fase 11 di src: opsi pecahan baris tetangga di SyntheticLines dan src.train (--neighbor-prob)."""

import random
from pathlib import Path

import numpy as np
import pytest

from src.augment import build_augment
from src.dataset import TRAIN_FONTS, SyntheticLines
from src.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
NOTO = str(TRAIN_FONTS[0])
LINES = ["ꦥꦤ꧀ꦕꦶ ꦫꦼꦒ ꦭꦼꦩꦃ", "ꦏꦸꦭ ꦧꦝꦺ ꦠꦶꦤ꧀ꦢꦏ꧀꧉", "ꦠꦿꦼꦱ꧀ꦤ ꦠꦲꦸꦤ꧀ ꧑꧙꧔꧕"]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.load(ROOT / "data" / "tokenizer.json")


def main_band(big, small) -> tuple[int, int] | None:
    """(geser tegak, tinggi) pita tinta `small` (baris bertinta pertama sampai terakhir) yang muncul utuh, piksel demi
    piksel, di `big`; None bila tidak ada atau lebarnya beda."""
    b, s = np.asarray(big.convert("L")), np.asarray(small.convert("L"))
    rows = np.where((s < 128).any(axis=1))[0]
    band = s[rows[0]:rows[-1] + 1]
    if b.shape[1] != s.shape[1]:
        return None
    return next(((y, len(band)) for y in range(b.shape[0] - len(band) + 1) if np.array_equal(b[y:y + len(band)], band)), None)


def test_neighbor_option_changes_only_the_image_and_is_off_by_default(tok):
    # Dengan augmentasi: undian tambahan sebelum augmentasi akan mengubah citra teraugmentasi (G2, run lama).
    kwargs = dict(deterministic=True, seed=3, track_prob=1.0, track_max=0.3, slant_prob=0.5, slant_max=0.6,
                  augment=build_augment("heavy"))
    plain = SyntheticLines(LINES, tok, [NOTO], **kwargs)
    off = SyntheticLines(LINES, tok, [NOTO], neighbor_prob=0.0, **kwargs)
    for i in range(len(LINES)):
        (a, ta), (b, tb) = plain.sample(i), off.sample(i)
        assert ta == tb and a.tobytes() == b.tobytes()  # peluang 0 tidak mengonsumsi angka acak
    # Dataset yang di-pickle versi sebelumnya tidak punya atributnya.
    legacy = SyntheticLines(LINES, tok, [NOTO], deterministic=True, seed=3)
    delattr(legacy, "neighbor_prob")
    assert legacy.sample(0)[1] == LINES[0]


def counting_rng(seed: str, calls: list) -> random.Random:
    """random.Random yang mencatat tiap panggilan random() ke `calls` (pilihan font dan ukuran memakai getrandbits)."""
    rng = random.Random(seed)
    original = rng.random
    rng.random = lambda: calls.append(1) or original()
    return rng


def test_options_at_zero_draw_no_random_numbers(tok):
    # Pembanding "plain" lawan "off" di atas memakai jalur kode yang sama, jadi undian yang selalu terjadi tidak terlihat
    # di sana. Di sini random() dihitung langsung: semua opsi mati = tidak ada undian random() sama sekali.
    for kwargs, expect_draws in (({}, False), ({"neighbor_prob": 0.0}, False), ({"neighbor_prob": 1.0}, True)):
        ds = SyntheticLines(LINES, tok, [NOTO], deterministic=True, seed=3, **kwargs)
        calls = []
        ds._rng = lambda idx, calls=calls: counting_rng(f"3:{idx}", calls)
        for i in range(len(LINES)):
            ds.sample(i)
        assert bool(calls) == expect_draws, kwargs


def test_neighbor_lines_are_other_training_lines_in_the_same_font_and_size(tok, monkeypatch):
    from src import dataset as ds_module

    rendered = []
    original = ds_module.render_line

    def recording(text, font, size, **kwargs):
        rendered.append((text, font, size))
        return original(text, font, size, **kwargs)

    monkeypatch.setattr(ds_module, "render_line", recording)
    lines = LINES * 5
    on = SyntheticLines(lines, tok, [NOTO, str(TRAIN_FONTS[1])], deterministic=True, seed=7, neighbor_prob=1.0)
    neighbors = []
    for i in range(len(lines)):
        rendered.clear()
        on.sample(i)
        (_, font, size), others = rendered[0], rendered[1:]
        assert all(f == font and s == size for _, f, s in others)  # font dan ukuran baris utama
        neighbors += [text for text, _, _ in others]
    assert neighbors and all(" " not in text for text in neighbors)  # tanpa spasi
    assert set(neighbors) <= {line.replace(" ", "") for line in lines}
    assert len(set(neighbors)) == len(LINES)  # diundi dari kumpulan baris, bukan baris tetap


def test_neighbor_fragments_sit_above_and_below_the_untouched_line(tok):
    kwargs = dict(deterministic=True, seed=5, track_prob=1.0, track_max=0.3)
    lines = LINES * 14
    plain = SyntheticLines(lines, tok, [NOTO], **kwargs)
    on = SyntheticLines(lines, tok, [NOTO], neighbor_prob=1.0, **kwargs)
    taller = unchanged = above = below = 0
    for i in range(len(lines)):
        (a, ta), (c, tc) = plain.sample(i), on.sample(i)
        assert tc == ta  # label tetap teks baris utama
        assert c.width == a.width and c.height >= a.height
        band = main_band(c, a)
        assert band is not None  # tinta baris utama tertempel utuh; pecahan hanya di luar tintanya
        y, height = band
        ink = np.asarray(c.convert("L")) < 128
        above += bool(ink[:y].any())
        below += bool(ink[y + height:].any())
        taller += c.height > a.height
        unchanged += c.convert("L").tobytes() == a.convert("L").tobytes()
    # Tiap sisi kosong dengan peluang 0,3: kadang tanpa pecahan sama sekali, sering di atas dan di bawah.
    assert taller >= 10 and above >= 10 and below >= 10 and 1 <= unchanged <= 12
    # Sekitar separuh baris mendapat pecahan.
    half = SyntheticLines([LINES[0]] * 40, tok, [NOTO], deterministic=True, seed=4, neighbor_prob=0.5)
    base = SyntheticLines([LINES[0]] * 40, tok, [NOTO], deterministic=True, seed=4)
    changed = [half.sample(i)[0].convert("L").tobytes() != base.sample(i)[0].convert("L").tobytes() for i in range(40)]
    assert 8 < sum(changed) < 32


def test_neighbor_option_defaults_to_off_and_reaches_only_the_training_lines(monkeypatch):
    from src import train
    from src.tokenizer import TOKENIZER_PATH

    assert train.parse_args(["--run", "x"]).neighbor_prob == 0.0
    monkeypatch.setattr(train, "read_split", lambda name: list(LINES) * 3)
    tok = Tokenizer.load(TOKENIZER_PATH)
    args = train.parse_args(["--run", "x", "--train-lines", "6", "--val-lines", "3", "--neighbor-prob", "0.5"])
    train_ds, val_ds = train.build_datasets(args, tok)
    assert train_ds.neighbor_prob == 0.5 and val_ds.neighbor_prob == 0.0  # validasi tidak berubah
