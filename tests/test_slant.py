"""Test fase 9 di src: render miring (italic) dan perbaikan font latih (pasangan tidak menumpuk, urutan bertabrakan)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import dataset as ds
from src.dataset import TRAIN_FONTS, SyntheticLines, oblique
from src.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
NOTO = str(TRAIN_FONTS[0])
LINES = ["ꦥꦤ꧀ꦕꦶ ꦫꦼꦒ ꦭꦼꦩꦃ", "ꦏꦸꦭ ꦧꦝꦺ ꦠꦶꦤ꧀ꦢꦏ꧀꧉", "ꦠꦿꦼꦱ꧀ꦤ ꦠꦲꦸꦤ꧀ ꧑꧙꧔꧕"]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.load(ROOT / "data" / "tokenizer.json")


def ink_columns(img: Image.Image, rows: slice) -> np.ndarray:
    return np.nonzero((np.asarray(img.convert("L"))[rows] < 128).any(axis=0))[0]


def test_oblique_leans_right_like_italic_print_and_keeps_all_ink():
    img = Image.new("L", (40, 30), 255)
    img.paste(0, (10, 0, 12, 30))  # batang tegak
    for shear in (0.2, 0.5):
        out = oblique(img, shear)
        assert out.size == (40 + round(shear * 30) + 1, 30)
        top, bottom = ink_columns(out, slice(0, 3)), ink_columns(out, slice(27, 30))
        assert top.mean() - bottom.mean() == pytest.approx(shear * 27, abs=2)  # atas bergeser ke kanan
        assert int((np.asarray(out) < 128).sum()) >= int((np.asarray(img) < 128).sum()) * 0.9  # tinta tidak terpotong
    assert np.array_equal(np.asarray(oblique(img, 0.0))[:, :40], np.asarray(img))


def test_slant_option_changes_only_the_image_and_is_off_by_default(tok):
    plain = SyntheticLines(LINES, tok, [NOTO], deterministic=True, seed=3, track_prob=1.0, track_max=0.3)
    off = SyntheticLines(LINES, tok, [NOTO], deterministic=True, seed=3, track_prob=1.0, track_max=0.3, slant_prob=0.0, slant_max=0.6)
    on = SyntheticLines(LINES, tok, [NOTO], deterministic=True, seed=3, track_prob=1.0, track_max=0.3, slant_prob=1.0, slant_max=0.6)
    for i in range(len(LINES)):
        (a, ta), (b, tb), (c, tc) = plain.sample(i), off.sample(i), on.sample(i)
        assert ta == tb == tc and a.tobytes() == b.tobytes()  # peluang 0 tidak mengonsumsi angka acak
        assert c.width > a.width and c.height == a.height  # miring: lebih lebar, label sama
    # Dataset yang di-pickle versi sebelumnya tidak punya atributnya.
    legacy = SyntheticLines(LINES, tok, [NOTO], deterministic=True, seed=3)
    for name in ("slant_prob", "slant_max", "avoid_conflicts"):
        delattr(legacy, name)
    assert legacy.sample(0)[1] == LINES[0]
    # Separuh baris dimiringkan; kemiringannya diundi dari 0 sampai slant_max.
    half = SyntheticLines([LINES[0]] * 40, tok, [NOTO], deterministic=True, seed=4, slant_prob=0.5, slant_max=0.6)
    base = SyntheticLines([LINES[0]] * 40, tok, [NOTO], deterministic=True, seed=4)
    widened = [half.sample(i)[0].width - base.sample(i)[0].width for i in range(40)]
    assert 8 < sum(w > 1 for w in widened) < 32
    assert max(widened) <= round(0.6 * base.sample(0)[0].height) + 1


def test_fonts_that_draw_a_sequence_like_another_are_not_used_for_lines_with_it(tok, monkeypatch):
    # Font palsu "konflik" (Noto) menggambar pangkon + lungsi seperti pangkon + lingsa: baris yang memuat urutan itu (juga
    # sesudah spasi dibuang) tidak dirender dengannya; baris lain dan run tanpa pilihan ini tidak berubah.
    tuladha = str(TRAIN_FONTS[1])
    monkeypatch.setattr(ds, "render_conflicts", lambda font: frozenset({"꧀꧉"}) if font == NOTO else frozenset())
    real_render, used = ds.render_line, []
    monkeypatch.setattr(ds, "render_line", lambda text, font, size, **kw: used.append(font) or real_render(text, font, size, **kw))
    lines = ["ꦏꦸꦭ ꦠꦶꦤ꧀ꦢꦏ꧀꧉", "ꦏꦸꦭ ꦠꦶꦤ꧀ꦢꦏ꧀ ꧉", "ꦏꦸꦭ ꦧꦝꦺ"] * 20

    def fonts(**kwargs):
        data = SyntheticLines(lines, tok, [NOTO, tuladha], deterministic=True, seed=1, **kwargs)
        used.clear()
        images = [data.sample(i)[0].tobytes() for i in range(len(lines))]
        return list(used), images

    loose_fonts, loose_images = fonts()
    fixed_fonts, fixed_images = fonts(avoid_conflicts=True)
    for i, line in enumerate(lines):
        if "꧀꧉" in line.replace(" ", ""):
            assert fixed_fonts[i] == tuladha
        else:
            assert (fixed_fonts[i], fixed_images[i]) == (loose_fonts[i], loose_images[i])
    # Undian pertama memang kadang memilih font konflik, jadi penggantian benar-benar teruji.
    assert any(f == NOTO for i, f in enumerate(loose_fonts) if "꧀꧉" in lines[i].replace(" ", ""))
    # Semua font berkonflik: tidak ada pengganti, font undian dipakai (tidak macet).
    monkeypatch.setattr(ds, "render_conflicts", lambda font: frozenset({"꧀꧉"}))
    stuck_fonts, _ = fonts(avoid_conflicts=True)
    assert stuck_fonts == loose_fonts


def test_shaping_check_and_conflicts_are_measured_by_rendering(monkeypatch):
    assert ds.shaping_ok(NOTO) and ds.shaping_ok(str(TRAIN_FONTS[1]))
    assert ds.render_conflicts(NOTO) == frozenset()
    # Render palsu: pasangan berjajar (tiga kali lebar ka) dan lungsi = lingsa.
    def fake(text, font, size, **kwargs):
        img = Image.new("L", (300, 40), 255)
        width = 30 * (3 if "꧀" in text else 1)
        img.paste(0, (10, 5, 10 + width, 35))
        return img
    ds.shaping_ok.cache_clear()
    ds.render_conflicts.cache_clear()
    monkeypatch.setattr(ds, "render_line", fake)
    try:
        assert ds.shaping_ok("palsu.ttf") is False
        assert ds.render_conflicts("palsu.ttf") == frozenset({"꧀꧉"})
    finally:
        ds.shaping_ok.cache_clear()
        ds.render_conflicts.cache_clear()


EXTRA = ROOT / "fonts" / "extra"


@pytest.mark.skipif(not (EXTRA / "NewKramawirya.ttf").exists(), reason="font tambahan tidak ada (tidak disebar)")
def test_known_font_defects_are_detected():
    assert ds.render_conflicts(str(EXTRA / "NewKramawirya.ttf")) == frozenset({"꧀꧉"})
    others = [p for p in EXTRA.glob("*.[to]tf") if p.name != "NewKramawirya.ttf"]
    assert all(ds.render_conflicts(str(p)) == frozenset() for p in others)
    # Pasangan BasaJan menumpuk di .venv (Pillow 11.3) tetapi tidak di .venv-xpu (Pillow 12.3): diukur, bukan diasumsikan.
    assert all(ds.shaping_ok(str(p)) for p in others)


def test_training_fonts_drop_fonts_whose_pasangan_do_not_stack(monkeypatch, tmp_path):
    from src import train

    for name in ("a.ttf", "b.ttf"):
        (tmp_path / name).write_bytes(Path(NOTO).read_bytes())
    monkeypatch.setattr(train, "shaping_ok", lambda font: Path(font).name != "b.ttf")
    plain = train.training_fonts(train.parse_args(["--run", "x", "--extra-fonts", str(tmp_path)]))
    fixed = train.training_fonts(train.parse_args(["--run", "x", "--extra-fonts", str(tmp_path), "--font-fixes"]))
    assert [p.name for p in plain][-2:] == ["a.ttf", "b.ttf"] and [p.name for p in fixed][-1] == "a.ttf"
    assert len(fixed) == len(plain) - 1
    monkeypatch.setattr(train, "shaping_ok", lambda font: False)
    with pytest.raises(ValueError, match="Semua font latih gagal cek shaping"):
        train.training_fonts(train.parse_args(["--run", "x", "--font-fixes"]))
    # start_record mencatat font yang benar-benar dipakai.
    monkeypatch.setattr(train, "shaping_ok", lambda font: Path(font).name != "b.ttf")
    record = train.start_record(train.parse_args(["--run", "x", "--extra-fonts", str(tmp_path), "--font-fixes"]), 0, 1, "cpu")
    assert "b.ttf" not in record["fonts"] and "a.ttf" in record["fonts"]


def test_new_options_default_to_off_and_reach_the_training_lines(monkeypatch):
    from src import train
    from src.tokenizer import TOKENIZER_PATH

    args = train.parse_args(["--run", "x"])
    assert (args.slant_prob, args.slant_max, args.font_fixes) == (0.0, 0.6, False)
    monkeypatch.setattr(train, "read_split", lambda name: list(LINES) * 3)
    tok = Tokenizer.load(TOKENIZER_PATH)
    args = train.parse_args(["--run", "x", "--train-lines", "6", "--val-lines", "3", "--slant-prob", "0.5",
                             "--slant-max", "0.4", "--font-fixes"])
    train_ds, val_ds = train.build_datasets(args, tok)
    assert (train_ds.slant_prob, train_ds.slant_max, train_ds.avoid_conflicts) == (0.5, 0.4, True)
    assert (val_ds.slant_prob, val_ds.avoid_conflicts) == (0.0, False)  # validasi tidak berubah
