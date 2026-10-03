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


def test_attached_insert_adds_no_space(tok):
    # pada luhur (bukan aksara, bukan sandhangan): versi lama menyisipkan " ꧅" di depan spasi, jadi spasi bertambah.
    pada_luhur, line = "꧅", LINES[0]
    common = RareText.from_lines(LINES, tok.charset).common_letters
    spaced = insert_codepoint(line, pada_luhur, common, random.Random(0))
    attached = insert_codepoint(line, pada_luhur, common, random.Random(0), spaced=False)
    assert spaced.count(" ") == line.count(" ") + 1 and f" {pada_luhur} " in spaced
    assert attached.count(" ") == line.count(" ") and attached.replace(pada_luhur, "") == line
    assert f" {pada_luhur}" not in attached and f"{pada_luhur} " in attached


def test_glyph_similarity_compares_like_with_like(tok):
    from src.text_augment import glyph_similarity, similar_codepoints
    from src.tokenizer import is_javanese

    charset = tuple(ch for ch in tok.charset if is_javanese(ch))
    best = glyph_similarity(str(NOTO), charset)
    similar = similar_codepoints(str(NOTO), charset, 0.9)
    letter_e, ka, sa_murda, cecak, layar = "ꦌ", "ꦏ", "ꦯ", "ꦁ", "ꦂ"
    pada_windu, digit_zero, pada_madya, pada_luhur, adeg_adeg = "꧆", "꧐", "꧄", "꧅", "꧋"
    # Token lepas yang kembar: konteks tidak membedakan pada windu dari angka nol.
    assert best[pada_windu][0] >= 0.99 and best[pada_windu][1] == digit_zero
    assert {pada_windu, pada_madya, pada_luhur} <= similar
    # Aksara vs angka tidak dibandingkan: E dan angka enam satu glyph di Noto, tetapi konteks membedakannya.
    assert letter_e not in similar and best[letter_e][1] != "꧖"
    # Aksara yang jelas berbeda, dan pembuka baris, tidak tersaring.
    assert not {ka, sa_murda, adeg_adeg} & similar
    # Sandhangan dibandingkan tanpa tinta ka: tanda kecil yang terlihat jelas bukan "tanpa tanda".
    assert best[cecak][0] < 0.5 and best[layar][0] < 0.5 and not {cecak, layar} & similar


@pytest.mark.skipif(not (ROOT / "fonts" / "extra" / "ARDemak-Regular.ttf").exists(), reason="font tambahan tidak ada")
def test_invisible_and_duplicate_marks_are_similar(tok):
    from src.text_augment import glyph_similarity
    from src.tokenizer import is_javanese

    charset = tuple(ch for ch in tok.charset if is_javanese(ch))
    best = glyph_similarity(str(ROOT / "fonts" / "extra" / "ARDemak-Regular.ttf"), charset)
    panyangga, tolong, tarung = "ꦀ", "ꦵ", "ꦴ"
    assert best[panyangga] == (1.0, "(tanpa tanda)")  # ARDemak: panyangga tidak menambah tinta
    assert best[tolong][0] >= 0.99 and best[tolong][1] == tarung  # glyph yang sama


def test_filter_skips_similar_picks_without_boosting_the_rest(tok, check):
    from src.text_augment import similar_codepoints

    plain = RareText.from_lines(LINES, tok.charset, insert_prob=1.0)
    strict = RareText.from_lines(LINES, tok.charset, insert_prob=1.0, max_similarity=0.9)
    similar = similar_codepoints(str(NOTO), strict.charset, 0.9)
    assert similar & set(strict.rare)  # ada yang disaring, kalau tidak test ini kosong
    skipped = same = 0
    for seed in range(600):
        text = LINES[seed % len(LINES)]
        out_plain = plain(text, random.Random(seed), NOTO, check)
        out_strict = strict(text, random.Random(seed), NOTO, check)
        assert not set(out_strict) - set(text) & similar  # codepoint mirip tidak pernah disisipkan
        if out_strict == out_plain:
            same += 1  # pilihan yang sama, sisipan yang sama: jatah codepoint lain tidak berubah
        else:
            skipped += 1
            assert out_strict == text and (set(out_plain) - set(text)) & similar
    assert skipped > 0 and same > 400


def test_attach_with_filter_adds_no_space(tok, check):
    strict = RareText.from_lines(LINES, tok.charset, insert_prob=1.0, opener_prob=1.0, max_similarity=0.9, attach=True)
    changed = 0
    for seed in range(200):
        text = LINES[seed % len(LINES)]
        out = strict(text, random.Random(seed), NOTO, check)
        assert is_well_formed(out) and check(out)
        assert out.count(" ") == text.count(" ")
        changed += out != text
    assert changed >= 150


def test_max_similarity_requires_a_charset(tok):
    rare = RareText.from_lines(LINES, tok.charset)
    with pytest.raises(ValueError, match="charset"):
        RareText(rare.rare, rare.common_letters, 1.0, 0.0, max_similarity=0.9)


def test_default_rare_text_matches_fase6_behaviour(tok, check):
    # fase6_rare dan scripts/eval_rare.py memakai nilai bawaan. Nilai emas dari versi sebelum saringan ditambahkan
    # (commit bfe0631): 60 keluaran untuk (baris, seed) tetap.
    import hashlib

    augment = RareText.from_lines(LINES, tok.charset, insert_prob=1.0, opener_prob=1.0)
    positional = RareText(augment.rare, augment.common_letters, 1.0, 1.0)  # cara scripts/eval_rare.py
    outs = [augment(LINES[seed % len(LINES)], random.Random(seed), NOTO, check) for seed in range(60)]
    assert outs == [positional(LINES[seed % len(LINES)], random.Random(seed), NOTO, check) for seed in range(60)]
    digest = hashlib.sha256(chr(10).join(outs).encode("utf-8")).hexdigest()
    assert digest == "c83322f504294be827dd6918b851d1324c34de1c26a4f8ffcbdae759d3d92a2b"
