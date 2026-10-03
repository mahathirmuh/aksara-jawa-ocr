"""Test augmentasi teks karakter langka — Fase 5."""

import random
from pathlib import Path

import pytest

from src.dataset import TRAIN_FONTS
from src.render import render_line
from src.text_augment import LINE_OPENERS, RareText, font_codepoints, insert_codepoint
from src.tokenizer import Tokenizer, is_well_formed, nfc

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_JSON = ROOT / "data" / "tokenizer.json"
CARAKAN = ROOT / "fonts" / "extra" / "CarakanJawa.otf"
EXTRA_FONTS = sorted((ROOT / "fonts" / "extra").glob("*.[to]tf"))
NOTO = next(p for p in TRAIN_FONTS if p.name.startswith("NotoSansJavanese"))
LETTER_O = "ꦎ"

pytestmark = pytest.mark.skipif(not TOKENIZER_JSON.exists(), reason="tokenizer belum dibangun")

# Keluaran transliterator untuk kalimat Jawa nyata (sama dengan tests/test_dataset.py).
LINES = [
    "ꦧꦱ ꦗꦮ ꦲꦶꦏꦸ ꦧꦱ ꦏꦁ ꦢꦶꦲꦼꦁꦒꦺꦴ ꦢꦼꦤꦶꦁ ꦮꦺꦴꦁ ꦗꦮ꧉",
    "ꦏꦸꦛ ꦪꦺꦴꦒꦾꦏꦂꦠ ꦲꦤ ꦲꦶꦁ ꦠ꧀ꦭꦠꦃ ꦗꦮ ꦠꦼꦔꦃ꧈",
    "ꦮꦺꦴꦁ꧈ ꦏꦺꦴꦮꦺ꧈ ꦚꦮꦶꦗꦶ꧈ ꦏꦿꦠꦺꦴꦤ꧀ ꦥꦿꦝꦤ꧈ ꦱꦱ꧀ꦠꦿ",
    "ꦲꦶꦁ ꦱꦶꦱꦶꦃ ꦭꦺꦴꦂ ꦮꦺꦠꦤ꧀ ꧌ꦏꦶꦢꦸꦭ꧀꧍ ꦲꦤ ꦏꦭꦶ꧇",
]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.load(TOKENIZER_JSON)


@pytest.fixture(scope="module")
def check(tok):
    return lambda text: tok.to_logical(tok.to_visual(text)) == nfc(text)


def test_rare_set_comes_from_line_frequency(tok):
    rare = RareText.from_lines(LINES * 10, tok.charset).rare
    assert "ꦑ" in rare and "꧋" in rare  # ka murda, adeg-adeg: tidak ada di LINES
    assert "ꦏ" not in rare  # ka: umum
    assert all(ch in tok.charset for ch in rare)


def test_zero_probabilities_leave_text_and_rng_untouched(tok, check):
    augment = RareText.from_lines(LINES, tok.charset)
    rng, fresh = random.Random(5), random.Random(5)
    assert augment(LINES[0], rng, NOTO, check) == LINES[0]
    assert rng.random() == fresh.random()


def test_every_rare_codepoint_can_be_inserted_well_formed(tok, check):
    augment = RareText.from_lines(LINES, tok.charset, insert_prob=1.0)
    failed = []
    for ch in augment.rare:
        if not any(
            (new := insert_codepoint(LINES[seed % len(LINES)], ch, augment.common_letters, random.Random(seed)))
            and is_well_formed(new) and check(new)
            for seed in range(30)
        ):
            failed.append(f"U+{ord(ch):04X}")
    assert not failed


def test_insert_adds_a_rare_codepoint_and_keeps_labels_consistent(tok, check):
    augment = RareText.from_lines(LINES, tok.charset, insert_prob=1.0)
    changed = 0
    for seed in range(40):
        text = LINES[seed % len(LINES)]
        out = augment(text, random.Random(seed), NOTO, check)
        assert is_well_formed(out) and check(out)
        tok.encode(tok.to_visual(out))  # semua karakter ada di charset
        if out != text:
            changed += 1
            assert any(ch in augment.rare for ch in out)
    assert changed >= 30


def test_opener_prefixes_adeg_adeg_once(tok, check):
    augment = RareText.from_lines(LINES, tok.charset, opener_prob=1.0)
    out = augment(LINES[1], random.Random(0), NOTO, check)
    assert out == LINE_OPENERS[0] + LINES[1]
    assert augment(out, random.Random(1), NOTO, check) == out


@pytest.mark.skipif(not CARAKAN.exists(), reason="font tambahan tidak ada")
def test_font_without_glyph_never_receives_that_codepoint(tok, check):
    assert ord(LETTER_O) not in font_codepoints(str(CARAKAN))
    common = RareText.from_lines(LINES, tok.charset).common_letters
    augment = RareText((LETTER_O,), common, insert_prob=1.0)
    assert all(LETTER_O not in augment(LINES[0], random.Random(s), CARAKAN, check) for s in range(20))
    assert any(LETTER_O in augment(LINES[0], random.Random(s), NOTO, check) for s in range(20))


def test_augmented_text_renders_with_every_training_font(tok, check):
    augment = RareText.from_lines(LINES, tok.charset, insert_prob=1.0, opener_prob=1.0)
    for i, font in enumerate(list(TRAIN_FONTS) + EXTRA_FONTS):
        out = augment(LINES[i % len(LINES)], random.Random(i), font, check)
        assert all(ord(ch) in font_codepoints(str(font)) for ch in out if ch != " "), font.name
        assert render_line(out, font, 56).width > 0


def test_opener_outside_charset_is_skipped(tok, check):
    # Tokenizer tanpa adeg-adeg: pembuka tidak boleh menghasilkan label yang tidak bisa di-encode.
    augment = RareText.from_lines(LINES, tok.charset, opener_prob=1.0)
    no_opener = lambda text: LINE_OPENERS[0] not in text and check(text)  # noqa: E731
    assert augment(LINES[1], random.Random(0), NOTO, no_opener) == LINES[1]
