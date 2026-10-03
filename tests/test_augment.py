"""Test augmentasi — Fase 5."""

import random

import numpy as np

from src.augment import PHYSICAL_ORDER, PRESET_OPS, Augment, build_augment, tight
from src.dataset import TRAIN_FONTS
from src.render import render_line

LINE = "ꦧꦱ ꦗꦮ ꦲꦶꦏꦸ ꦧꦱ ꦏꦁ ꦢꦶꦲꦼꦁꦒꦺꦴ"


class NoTrim(random.Random):
    """rng yang tidak pernah memangkas tinta."""

    def uniform(self, a, b):
        return a


def ink_rows(img):
    return np.where((np.asarray(img.convert("L")) < 128).any(axis=1))[0]


def test_tight_crop_puts_ink_on_top_and_bottom_edges():
    img = render_line(LINE, TRAIN_FONTS[0], 64)
    out = tight(img, NoTrim(0), heavy=False)
    rows = ink_rows(out)
    assert out.height < img.height
    assert rows[0] == 0 and rows[-1] == out.height - 1


def test_tight_trim_is_small():
    img = render_line(LINE, TRAIN_FONTS[0], 64)
    ink_height = ink_rows(img)[-1] - ink_rows(img)[0] + 1
    for seed in range(20):
        out = tight(img, random.Random(seed), heavy=True)
        assert out.height >= ink_height * 0.9


def test_presets_keep_margin_and_exclude_tight():
    assert "tight" in PHYSICAL_ORDER and "tight" not in PRESET_OPS
    assert "tight" not in build_augment("heavy").names
    assert "margin" in build_augment("heavy").names
    assert build_augment("blur+tight").names == ["tight", "blur"]


class Fixed(random.Random):
    """rng dengan random() tetap; uniform/choice tetap acak-deterministik."""

    def __init__(self, value):
        super().__init__(0)
        self.value = value

    def random(self):
        return self.value


def ink_pixels(img):
    return int((np.asarray(img.convert("L")) < 128).sum())


def test_heavy_preset_definition_is_frozen():
    assert build_augment("heavy").names == ["margin", "rotate", "elastic", "contrast", "texture", "blur", "noise", "jpeg"]
    assert build_augment("train").names == build_augment("heavy").names


def test_fase5_preset_uses_every_op():
    assert build_augment("fase5").names == PHYSICAL_ORDER
    assert build_augment("fase5").p == 0.5


def test_stroke_thickens_or_thins():
    img = render_line(LINE, TRAIN_FONTS[0], 64)
    base = ink_pixels(img)
    assert ink_pixels(Augment(["stroke"], 1.0, False)(img, Fixed(0.0))) > base
    assert ink_pixels(Augment(["stroke"], 1.0, False)(img, Fixed(0.99))) < base


def test_binarize_outputs_only_black_and_white():
    from src.augment import binarize, contrast

    img = contrast(render_line(LINE, TRAIN_FONTS[0], 64), random.Random(1), True)
    values = set(np.unique(np.asarray(binarize(img, random.Random(2), True))))
    assert values <= {0, 255} and len(values) == 2


def test_every_op_is_grayscale_and_deterministic():
    img = render_line(LINE, TRAIN_FONTS[0], 56)
    for name in PHYSICAL_ORDER:
        a = Augment([name], 1.0, True)(img, random.Random(3))
        b = Augment([name], 1.0, True)(img, random.Random(3))
        assert a.mode == "L", name
        assert a.tobytes() == b.tobytes(), name
