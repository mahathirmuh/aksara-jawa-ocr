"""Test src/neighbors.py: pecahan baris tetangga ditempel di luar tinta baris utama, label tidak berubah."""

import random

import numpy as np
from PIL import Image

from src import neighbors as nb
from src.dataset import TRAIN_FONTS
from src.render import render_line

MAIN = "ꦲꦶꦁ ꦱꦶꦱꦶꦃ ꦭꦺꦴꦂ ꦮꦺꦠꦤ꧀"
ABOVE = "ꦏꦸꦭ ꦧꦝꦺ ꦠꦶꦤ꧀ꦢꦏ꧀ ꦥꦸꦤ꧀ꦏꦺ ꦱꦸꦏꦸ"  # suku dan pasangan: ada tinta di bawah garis dasar
BELOW = "ꦮꦶꦱ ꦗꦏ꧀ ꦩꦏꦶꦭ꧀ꦭꦶ ꦲꦺꦮꦺꦴꦤ꧀"  # wulu dan taling: ada tinta di atas


def ink(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L")) < nb.INK


def test_fragments_sit_outside_the_main_ink_with_a_gap():
    font = TRAIN_FONTS[0]
    main, above, below = (render_line(text, font, 56) for text in (MAIN, ABOVE, BELOW))
    out = nb.with_neighbors(main, above, below, random.Random(3))
    assert out.width == main.width and out.height > main.height
    # Baris utama ditempel utuh: tepat sekali kemunculannya, piksel demi piksel.
    m, o = ink(main), ink(out)
    offsets = [y for y in range(out.height - main.height + 1) if (o[y:y + main.height] & m).sum() == m.sum()]
    assert offsets, "baris utama tidak utuh"
    top = offsets[0]
    rows = np.where(m.any(axis=1))[0]
    main_top, main_end = top + rows[0], top + rows[-1] + 1
    # Ada tinta pecahan di atas dan di bawah tinta utama, dipisah baris kosong (tidak menimpa teks yang dilabeli).
    above_rows = np.where(o[:main_top].any(axis=1))[0]
    below_rows = np.where(o[main_end:].any(axis=1))[0]
    assert len(above_rows) and len(below_rows)
    assert above_rows[-1] < main_top - 1 and below_rows[0] > 0
    # Di dalam jalur tinta utama, tidak ada piksel tambahan.
    assert (o[main_top:main_end] == m[rows[0]:rows[-1] + 1]).all()


def test_band_takes_the_requested_share_of_the_ink():
    img = render_line(ABOVE, TRAIN_FONTS[0], 56)
    top, end = nb.ink_rows(img)
    for fraction in (0.1, 0.35):
        strip = nb.band(img, fraction, bottom=True)
        assert strip.width == img.width and strip.height == max(1, round((end - top) * fraction))
        assert (np.asarray(strip) == np.asarray(img)[end - strip.height:end]).all()
    head = nb.band(img, 0.2, bottom=False)
    assert (np.asarray(head) == np.asarray(img)[top:top + head.height]).all()
    assert nb.band(Image.new("L", (40, 20), 255), 0.3, True) is None and nb.ink_rows(Image.new("L", (5, 5), 255)) is None


def test_missing_or_blank_neighbors_leave_the_image_alone():
    main = render_line(MAIN, TRAIN_FONTS[1], 48)
    blank = Image.new("L", (300, 40), 255)
    for above, below in ((None, None), (blank, None), (None, blank), (blank, blank)):
        assert (np.asarray(nb.with_neighbors(main, above, below, random.Random(0))) == np.asarray(main)).all()
    assert nb.with_neighbors(blank, main, main, random.Random(0)) is not None  # citra tanpa tinta: apa adanya
    # Hanya satu sisi: tinggi bertambah di sisi itu saja.
    above = render_line(ABOVE, TRAIN_FONTS[1], 48)
    one = nb.with_neighbors(main, above, None, random.Random(1))
    rows = np.where(ink(one).any(axis=1))[0]
    assert ink(one)[rows[0]:].sum() > 0 and one.height >= main.height


def test_same_seed_same_picture_and_narrow_or_wide_neighbors_fit_the_width():
    font = TRAIN_FONTS[0]
    main = render_line(MAIN, font, 56)
    wide = render_line(ABOVE + " " + ABOVE + " " + ABOVE, font, 56)
    narrow = render_line("ꦏꦸ", font, 56)
    a = nb.with_neighbors(main, wide, narrow, random.Random(9))
    b = nb.with_neighbors(main, wide, narrow, random.Random(9))
    assert (np.asarray(a) == np.asarray(b)).all() and a.width == main.width
