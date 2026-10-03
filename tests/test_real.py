"""Test pipeline data nyata — Fase 6. Citra uji dibuat di direktori sementara."""

from pathlib import Path

import pytest
import torch
from PIL import ImageOps

from src.augment import build_augment
from src.dataset import H, TRAIN_FONTS, SyntheticLines, collate
from src.real import ConcatLines, RealLine, RealLines, read_labels, split_by_source, validate, write_labels
from src.render import render_line
from src.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_JSON = ROOT / "data" / "tokenizer.json"

pytestmark = pytest.mark.skipif(not TOKENIZER_JSON.exists(), reason="tokenizer belum dibangun")

LINES = [
    "ꦧꦱ ꦗꦮ ꦲꦶꦏꦸ ꦧꦱ ꦏꦁ ꦢꦶꦲꦼꦁꦒꦺꦴ",
    "ꦏꦸꦛ ꦪꦺꦴꦒꦾꦏꦂꦠ ꦲꦤ ꦲꦶꦁ ꦠ꧀ꦭꦠꦃ",
    "ꦮꦺꦴꦁ꧈ ꦏꦺꦴꦮꦺ꧈ ꦚꦮꦶꦗꦶ꧈ ꦏꦿꦠꦺꦴꦤ꧀",
    "ꦲꦶꦁ ꦱꦶꦱꦶꦃ ꦭꦺꦴꦂ ꦮꦺꦠꦤ꧀ ꦲꦤ ꦏꦭꦶ꧇",
]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.load(TOKENIZER_JSON)


@pytest.fixture
def labels(tmp_path) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    augment = build_augment("heavy")
    import random

    for source in range(3):
        for i, text in enumerate(LINES):
            img = augment(render_line(text, TRAIN_FONTS[source % len(TRAIN_FONTS)], 48), random.Random(source * 10 + i))
            path = images / f"s{source}_{i}.png"
            img.save(path)
            rows.append(RealLine(path, text, f"papan_{source}", "outdoor"))
    labels_path = tmp_path / "labels.tsv"
    write_labels(rows, labels_path)
    return labels_path


def test_labels_roundtrip_and_validate(labels, tok):
    rows = read_labels(labels)
    assert len(rows) == 12
    assert rows[0].image_path.exists()
    assert validate(rows, tok) == []


def test_validate_catches_problems(labels, tok, tmp_path):
    rows = read_labels(labels)
    small = tmp_path / "small.png"
    render_line(LINES[0], TRAIN_FONTS[0], 12).save(small)
    bad = [
        RealLine(rows[0].image_path, "ꦏX", "a", ""),
        RealLine(tmp_path / "hilang.png", LINES[0], "b", ""),
        RealLine(small, LINES[0], "", ""),
    ]
    problems = " | ".join(validate(bad, tok))
    assert "U+0058" in problems
    assert "tidak ditemukan" in problems
    assert "tinggi" in problems
    assert "source_id kosong" in problems


def test_split_never_shares_a_source(labels):
    splits = split_by_source(read_labels(labels), min_test_lines=4, val_fraction=0.5)
    sources = {name: {r.source_id for r in part} for name, part in splits.items()}
    assert not (sources["train"] & sources["test"])
    assert not (sources["val"] & sources["test"])
    assert not (sources["train"] & sources["val"])
    assert sum(len(p) for p in splits.values()) == 12
    assert len(splits["test"]) >= 4


def test_real_lines_items_and_polarity(labels, tok, tmp_path):
    rows = read_labels(labels)
    ds = RealLines(rows[:1], tok, deterministic=True)
    x, y = ds[0]
    assert x.shape[:2] == (1, H)
    assert y.tolist() == tok.encode(tok.to_visual(LINES[0]))

    inverted = tmp_path / "inverted.png"
    from PIL import Image

    ImageOps.invert(Image.open(rows[0].image_path).convert("L")).save(inverted)
    x_inv, _ = RealLines([RealLine(inverted, LINES[0], "s", "")], tok)[0]
    # Setelah normalisasi polaritas, tinta tetap bernilai tinggi.
    assert torch.allclose(x, x_inv, atol=0.02)


def test_concat_lines_mixes_real_and_synthetic(labels, tok):
    real = RealLines(read_labels(labels), tok, deterministic=True)
    synthetic = SyntheticLines(LINES, tok, TRAIN_FONTS, deterministic=True)
    mixed = ConcatLines([(real, 2), (synthetic, 1)])
    assert len(mixed) == 2 * 12 + 4
    assert len(mixed.lines) == len(mixed)
    X, Y, in_lens, tgt_lens = collate([mixed[i] for i in (0, 12, 24)])
    assert X.shape[0] == 3


def test_add_margin_pads_with_white_background():
    from PIL import Image

    from src.infer import add_margin

    img = Image.new("L", (50, 20), 0)
    out = add_margin(img, 0.1)
    assert out.size == (54, 24)
    assert out.getpixel((0, 0)) == 255 and out.getpixel((2, 2)) == 0
    assert add_margin(img, 0.0) is img


def test_real_lines_pad_ratio_narrows_scaled_width(labels, tok):
    rows = read_labels(labels)[:1]
    plain = RealLines(rows, tok, deterministic=True)[0]
    padded = RealLines(rows, tok, deterministic=True, pad_ratio=0.1)[0]
    assert plain is not None and padded is not None
    # Margin menambah tinggi relatif lebih besar dari lebar; setelah skala ke H, lebar mengecil.
    assert padded[0].shape[-1] < plain[0].shape[-1]
    assert torch.equal(padded[1], plain[1])
