"""Test tokenizer — Fase 1.

Bagian atas memakai string uji kecil dan selalu jalan. Bagian bawah memakai
korpus penuh (data/splits) dan data/tokenizer.json; di-skip kalau belum dibangun.
"""

import random
from pathlib import Path

import pytest

from src.tokenizer import (
    BLANK,
    HarfBuzzOrder,
    Tokenizer,
    apply_permutation,
    default_font_paths,
    is_well_formed,
    logical_syllables,
    nfc,
    visual_syllables,
)

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_JSON = ROOT / "data" / "tokenizer.json"
SPLITS = ROOT / "data" / "splits"
FONTS = default_font_paths()

KA, PANGKON, TA, PA, MA, NA = "ꦏ", "꧀", "ꦠ", "ꦥ", "ꦩ", "ꦤ"
TALING, TARUNG, WULU, SUKU = "ꦺ", "ꦴ", "ꦶ", "ꦸ"
LAYAR, CECAK, WIGNYAN, CAKRA = "ꦂ", "ꦁ", "ꦃ", "ꦿ"
KERET, PENGKAL, PEPET, DIRGA = "ꦽ", "ꦾ", "ꦼ", "ꦻ"

SYLLABLES = {
    "ke": KA + TALING,
    "ko": KA + TALING + TARUNG,
    "kte": KA + PANGKON + TA + TALING,
    "kpe": KA + PANGKON + PA + TALING,
    "kre": KA + CAKRA + TALING,
    "kye": KA + PENGKAL + TALING,
    "keng": KA + TALING + CECAK,
    "kong": KA + TALING + TARUNG + CECAK,
    "ktme": KA + PANGKON + TA + PANGKON + MA + TALING,
    "kri": KA + KERET,
    "kru": KA + CAKRA + SUKU,
    "kah": KA + WIGNYAN,
    "kar": KA + LAYAR,
    "kê": KA + PEPET,
    "ktu": KA + PANGKON + TA + SUKU,
    "ké_dirga": KA + DIRGA,
    "kreng": KA + CAKRA + TALING + CECAK,
    "ki": KA + WULU,
}
LINES = [
    SYLLABLES["ke"] + TA + TALING,
    KA + " " + SYLLABLES["kte"] + NA + TALING + CECAK + " " + MA,
    SYLLABLES["kong"] + SYLLABLES["kreng"] + " " + SYLLABLES["ktme"] + PA,
    " ".join(SYLLABLES.values()),
]
SAMPLES = list(SYLLABLES.values()) + LINES

# Diamati dari HarfBuzz 14.4 pada Noto Sans Javanese, Tuladha Jejeg OT, dan
# Javanese Text — ini penjaga regresi, bukan aturan yang ditulis dari ingatan.
EXPECTED_VISUAL = {
    "ke": TALING + KA,
    "ko": TALING + KA + TARUNG,
    "kte": TALING + KA + PANGKON + TA,
    "kre": TALING + KA + CAKRA,
    "kong": TALING + KA + TARUNG + CECAK,
    "kru": KA + CAKRA + SUKU,
    "ké_dirga": DIRGA + KA,
}

needs_fonts = pytest.mark.skipif(not FONTS, reason="tidak ada font di fonts/")


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    if not FONTS:
        pytest.skip("tidak ada font di fonts/")
    tokenizer, _ = Tokenizer.build(SAMPLES, FONTS)
    return tokenizer


@needs_fonts
def test_blank_is_zero_and_charset_one_indexed(tok):
    ids = [i for s in SAMPLES for i in tok.encode(s)]
    assert BLANK == 0
    assert min(ids) == 1
    assert max(ids) == len(tok.charset)
    assert tok.n_classes == len(tok.charset) + 1


@needs_fonts
def test_decode_ignores_blank(tok):
    ids = tok.encode(SYLLABLES["kte"])
    with_blanks = [BLANK] + [x for i in ids for x in (i, BLANK)]
    assert tok.decode(with_blanks) == SYLLABLES["kte"]


@needs_fonts
@pytest.mark.parametrize("text", SAMPLES)
def test_encode_decode_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == nfc(text)


@needs_fonts
@pytest.mark.parametrize("name", EXPECTED_VISUAL)
def test_visual_order_matches_harfbuzz_observation(tok, name):
    assert tok.to_visual(SYLLABLES[name]) == EXPECTED_VISUAL[name]


@needs_fonts
@pytest.mark.parametrize("text", SAMPLES)
def test_reorder_roundtrip(tok, text):
    assert tok.to_logical(tok.to_visual(text)) == nfc(text)


@needs_fonts
@pytest.mark.parametrize("text", LINES)
def test_visual_syllables_align_with_logical(tok, text):
    expected = [tok.to_visual(s) for s in logical_syllables(text)]
    assert visual_syllables(tok.to_visual(text), tok.prebase) == expected


@needs_fonts
@pytest.mark.parametrize("text", LINES)
def test_syllable_table_matches_whole_line_shaping(tok, text):
    perm, _ = HarfBuzzOrder(FONTS).permutation(text)
    assert tok.to_visual(text) == apply_permutation(text, perm)


@needs_fonts
def test_taling_is_derived_prebase(tok):
    assert TALING in tok.prebase
    assert CAKRA not in tok.prebase


@needs_fonts
def test_save_load_roundtrip(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.charset == tok.charset
    for text in SAMPLES:
        assert loaded.encode(text) == tok.encode(text)
        assert loaded.to_visual(text) == tok.to_visual(text)
        assert loaded.to_logical(tok.to_visual(text)) == nfc(text)


def test_well_formed():
    assert is_well_formed(SYLLABLES["kte"])
    assert not is_well_formed(TALING + KA)
    assert not is_well_formed(KA + " " + TALING)


# --- korpus penuh --------------------------------------------------------

needs_corpus = pytest.mark.skipif(
    not (TOKENIZER_JSON.exists() and (SPLITS / "train.txt").exists()),
    reason="korpus/tokenizer belum dibangun: python -m src.corpus && python -m src.tokenizer",
)


@pytest.fixture(scope="module")
def corpus() -> list[str]:
    lines: list[str] = []
    for name in ("train", "val", "test"):
        lines += (SPLITS / f"{name}.txt").read_text(encoding="utf-8").splitlines()
    return lines


@pytest.fixture(scope="module")
def corpus_tok() -> Tokenizer:
    # Tanpa font: tabel harus sudah mencakup semua pola di korpus.
    return Tokenizer.load(TOKENIZER_JSON)


@needs_corpus
def test_corpus_has_enough_lines(corpus):
    assert len(corpus) >= 100_000


@needs_corpus
def test_corpus_encode_decode_roundtrip(corpus, corpus_tok):
    failures = [s for s in corpus if corpus_tok.decode(corpus_tok.encode(s)) != nfc(s)]
    print(f"\nencode/decode: {len(corpus) - len(failures):,}/{len(corpus):,} lolos")
    assert not failures, f"{len(failures)} baris gagal, contoh: {failures[:3]!r}"


@needs_corpus
def test_corpus_reorder_roundtrip(corpus, corpus_tok):
    failures = [s for s in corpus if corpus_tok.to_logical(corpus_tok.to_visual(s)) != nfc(s)]
    print(f"\nto_logical(to_visual): {len(corpus) - len(failures):,}/{len(corpus):,} lolos")
    assert not failures, f"{len(failures)} baris gagal, contoh: {failures[:3]!r}"


@needs_corpus
def test_corpus_nfc_is_noop_on_visual_order(corpus, corpus_tok):
    # encode() menormalisasi NFC; itu hanya aman untuk string visual kalau NFC
    # tidak mengubahnya.
    failures = [s for s in corpus if nfc(corpus_tok.to_visual(s)) != corpus_tok.to_visual(s)]
    assert not failures, f"{len(failures)} baris berubah oleh NFC"


@needs_corpus
def test_corpus_prebase_is_taling_family(corpus_tok):
    # Cakra dan suku sesekali ditaruh font sebelum aksara dasar (ligatur pada
    # pasangan), tapi bukan pada mayoritas kemunculannya.
    assert TALING in corpus_tok.prebase
    assert CAKRA not in corpus_tok.prebase and SUKU not in corpus_tok.prebase


@needs_corpus
def test_corpus_visual_order_agrees_with_whole_line_shaping(corpus, corpus_tok):
    # Urutan kanonis sengaja menyimpang dari HarfBuzz untuk tanda yang bertumpuk
    # di kolom yang sama (cakra pada pasangan); selebihnya harus sama.
    hb_order = HarfBuzzOrder(FONTS)
    sample = random.Random(0).sample(corpus, 2000)
    mismatches = [s for s in sample if corpus_tok.to_visual(s) != apply_permutation(s, hb_order.permutation(s)[0])]
    rate = 1 - len(mismatches) / len(sample)
    with_cakra = sum(CAKRA in s for s in mismatches)
    print(f"\nurutan kanonis vs shaping baris penuh: {rate:.2%} baris sama; {with_cakra}/{len(mismatches)} beda mengandung cakra")
    assert rate >= 0.95
    assert with_cakra == len(mismatches), f"beda tanpa cakra: {[s for s in mismatches if CAKRA not in s][:2]!r}"
