"""Test src/spelling.py: ejaan baku baris korpus sebelum dirender, dan pemakaiannya di SyntheticLines."""

import random
from pathlib import Path

import pytest

from src import dataset as ds
from src.dataset import TRAIN_FONTS, SyntheticLines
from src.spelling import PROBES, RULE_NAMES, standardize
from src.text_augment import font_codepoints
from src.tokenizer import Tokenizer, is_well_formed

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_JSON = ROOT / "data" / "tokenizer.json"
NOTO = str(TRAIN_FONTS[0])

# (Latin, ejaan korpus, ejaan baku) dari web/app/Support/AksaraWriter.php: fromLatin(kata, false) dan fromLatin(kata).
# Dicocokkan 2026-10-13 pada 4.609 kata kamus: standardize(korpus) == baku untuk semuanya.
PAIRS = [
    ("panci", "ꦥꦤ꧀ꦕꦶ", "ꦥꦚ꧀ꦕꦶ"),
    ("panjenengan", "ꦥꦤ꧀ꦗꦼꦤꦼꦔꦤ꧀", "ꦥꦚ꧀ꦗꦼꦤꦼꦔꦤ꧀"),
    ("rega", "ꦫꦼꦒ", "ꦉꦒ"),
    ("lemah", "ꦭꦼꦩꦃ", "ꦊꦩꦃ"),
    ("tresna", "ꦠꦿꦼꦱ꧀ꦤ", "ꦠꦽꦱ꧀ꦤ"),
    ("kreteg", "ꦏꦿꦼꦠꦼꦒ꧀", "ꦏꦽꦠꦼꦒ꧀"),
    ("pangreh", "ꦥꦁꦫꦺꦃ", "ꦥꦁꦫꦺꦃ"),          # taling, bukan pepet: tidak berubah
    ("taun 1945", "ꦠꦲꦸꦤ꧀ ꧑꧙꧔꧕", "ꦠꦲꦸꦤ꧀ ꧇꧑꧙꧔꧕꧇"),
    ("wulan januari", "ꦮꦸꦭꦤ꧀ ꦗꦤꦸꦲꦫꦶ", "ꦮꦸꦭꦤ꧀ ꦗꦤꦸꦲꦫꦶ"),  # n dan j di dua kata: tetap na
    ("kula", "ꦏꦸꦭ", "ꦏꦸꦭ"),
]


@pytest.mark.parametrize("latin, corpus, standard", PAIRS)
def test_standardize_matches_the_standard_spelling_of_the_script_writer(latin, corpus, standard):
    assert standardize(corpus) == standard
    assert standardize(standard) == standard  # ejaan yang sudah baku tidak diubah lagi


def test_rules_apply_only_where_the_standard_spelling_does():
    # Pasangan (sesudah pangkon) tetap ra/la berpepet: cerek dan lelet hanya untuk aksara yang berdiri sendiri.
    assert standardize("ꦏ꧀ꦭꦼ") == "ꦏ꧀ꦭꦼ" and standardize("ꦏ꧀ꦫꦼ") == "ꦏ꧀ꦫꦼ"
    # na di depan pangkon + aksara lain tidak menjadi nya.
    assert standardize("ꦥꦤ꧀ꦠ") == "ꦥꦤ꧀ꦠ"
    # Bilangan berpemisah diapit sekali; yang sudah berpangkat tidak diapit lagi.
    assert standardize("꧑꧐:꧓꧐ ꧒,꧕") == "꧇꧑꧐:꧓꧐꧇ ꧇꧒,꧕꧇"
    assert standardize("꧇꧑꧙꧔꧕꧇") == "꧇꧑꧙꧔꧕꧇"
    # Bilangan yang sudah dibuka atau ditutup pada pangkat tidak diapit lagi (tidak menjadi pada pangkat ganda).
    assert standardize("꧇꧑꧒") == "꧇꧑꧒" and standardize("꧑꧒꧇") == "꧑꧒꧇"
    text = "ꦥꦤ꧀ꦕꦶ ꦫꦼꦒ ꦭꦼꦩꦃ ꦠꦿꦼꦱ꧀ꦤ ꧗"
    assert standardize(text) == "ꦥꦚ꧀ꦕꦶ ꦉꦒ ꦊꦩꦃ ꦠꦽꦱ꧀ꦤ ꧇꧗꧇"
    # `only`: hanya aturan yang disebut.
    assert standardize(text, only={"keret"}) == "ꦥꦤ꧀ꦕꦶ ꦫꦼꦒ ꦭꦼꦩꦃ ꦠꦽꦱ꧀ꦤ ꧗"
    assert standardize(text, only=()) == text
    assert set(PROBES) == set(RULE_NAMES) - {"pangkat"}
    for name, (before, after) in PROBES.items():
        assert standardize(before, only={name}) == after, name


def test_rules_are_skipped_without_the_glyph_or_when_the_result_is_not_valid():
    text = "ꦥꦤ꧀ꦕꦶ ꦫꦼꦒ ꦭꦼꦩꦃ ꦠꦿꦼꦱ꧀ꦤ ꧗"
    every = frozenset(ord(ch) for ch in standardize(text))
    # Font tanpa pa cerek: aturan itu saja yang dilewati.
    assert standardize(text, every - {ord("ꦉ")}) == "ꦥꦚ꧀ꦕꦶ ꦫꦼꦒ ꦊꦩꦃ ꦠꦽꦱ꧀ꦤ ꧇꧗꧇"
    assert standardize(text, frozenset()) == text
    # Hasil yang ditolak (mis. tidak lolos round-trip tokenizer) dibuang per aturan, bukan seluruhnya.
    assert standardize(text, None, lambda new: "ꦊ" not in new) == "ꦥꦚ꧀ꦕꦶ ꦉꦒ ꦭꦼꦩꦃ ꦠꦽꦱ꧀ꦤ ꧇꧗꧇"
    assert standardize(text, None, lambda new: False) == text


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.load(TOKENIZER_JSON)


def test_standard_forms_are_well_formed_and_round_trip_the_tokenizer(tok):
    charset = set(tok.charset)
    for _, corpus, standard in PAIRS:
        assert is_well_formed(standard) and all(ch in charset for ch in standard)
        assert tok.to_logical(tok.to_visual(standard)) == standard
        assert tok.decode(tok.encode(tok.to_visual(standard))) == tok.to_visual(standard)


def test_training_fonts_have_the_glyphs_of_the_standard_spelling():
    for font in TRAIN_FONTS:
        covered = font_codepoints(str(font))
        assert all(ord(ch) in covered for ch in "ꦚꦉꦊꦽ꧇"), font.name


LINES = ["ꦥꦤ꧀ꦕꦶ ꦫꦼꦒ ꦭꦼꦩꦃ", "ꦠꦿꦼꦱ꧀ꦤ ꦠꦲꦸꦤ꧀ ꧑꧙꧔꧕", "ꦏꦸꦭ ꦧꦝꦺ ꦠꦶꦤ꧀ꦢꦏ꧀"]


def test_synthetic_lines_are_unchanged_without_the_option(tok):
    # Probabilitas 0 tidak mengonsumsi angka acak: citra dan teks sama persis dengan dataset tanpa pilihan itu, juga
    # bersama pilihan lain yang mengundi sesudahnya (jarak antar suku kata).
    for extra in ({}, {"track_prob": 1.0, "track_max": 0.3}, {"drop_space_prob": 0.5}):
        old = SyntheticLines(LINES, tok, [NOTO], deterministic=True, seed=3, **extra)
        new = SyntheticLines(LINES, tok, [NOTO], deterministic=True, seed=3, standard_prob=0.0, **extra)
        for i in range(len(LINES)):
            (a, text_a), (b, text_b) = old.sample(i), new.sample(i)
            assert text_a == text_b and a.tobytes() == b.tobytes()
    # Dataset yang di-pickle versi sebelumnya tidak punya atributnya.
    legacy = SyntheticLines(LINES, tok, [NOTO], deterministic=True, seed=3)
    del legacy.standard_prob
    assert legacy.sample(0)[1] == LINES[0]


def test_synthetic_lines_render_the_standard_spelling_and_label_it(tok, monkeypatch):
    always = SyntheticLines(LINES, tok, [NOTO], deterministic=True, seed=3, standard_prob=1.0)
    plain = SyntheticLines(LINES, tok, [NOTO], deterministic=True, seed=3)
    for i, line in enumerate(LINES):
        img, text = always.sample(i)
        assert text == standardize(line)
        x, y = always[i]
        assert tok.to_logical(tok.decode(y.tolist())) == text  # label = teks yang dirender
        if text != line:
            assert img.tobytes() != plain.sample(i)[0].tobytes()  # citranya memang berubah
    # Sebelum spasi dibuang: "n" penutup kata tidak menjadi nya di depan "c/j" pembuka kata berikutnya.
    joined = SyntheticLines(["ꦮꦸꦭꦤ꧀ ꦗꦤꦸꦲꦫꦶ"], tok, [NOTO], deterministic=True, seed=1, standard_prob=1.0, drop_space_prob=1.0)
    assert joined.sample(0)[1] == "ꦮꦸꦭꦤ꧀ꦗꦤꦸꦲꦫꦶ"
    # Separuh baris: keduanya muncul.
    half = SyntheticLines(LINES[:1] * 40, tok, [NOTO], deterministic=True, seed=5, standard_prob=0.5)
    seen = {half.sample(i)[1] for i in range(40)}
    assert seen == {LINES[0], standardize(LINES[0])}
    # Font tanpa glyph sebuah aturan: aturan itu dilewati untuk baris tersebut.
    from src import text_augment

    covered = font_codepoints(NOTO)
    monkeypatch.setattr(text_augment, "font_codepoints", lambda path: covered - {ord("ꦉ")})
    assert always.sample(0)[1] == "ꦥꦚ꧀ꦕꦶ ꦫꦼꦒ ꦊꦩꦃ"


def test_synthetic_lines_drop_a_rule_whose_result_does_not_round_trip(tok, monkeypatch):
    # Hasil aturan yang tidak lolos round-trip tokenizer (di sini: semua teks ber-pa cerek ditolak) dibuang per aturan;
    # aturan lain tetap dipakai, dan label tetap sama dengan teks yang dirender.
    always = SyntheticLines(LINES[:1], tok, [NOTO], deterministic=True, seed=3, standard_prob=1.0)
    original = SyntheticLines._round_trips
    monkeypatch.setattr(SyntheticLines, "_round_trips", lambda self, text: "ꦉ" not in text and original(self, text))
    assert always.sample(0)[1] == "ꦥꦚ꧀ꦕꦶ ꦫꦼꦒ ꦊꦩꦃ"
    monkeypatch.setattr(ds, "is_well_formed", lambda text: "ꦊ" not in text)  # nga lelet tidak well-formed: dilewati juga
    assert always.sample(0)[1] == "ꦥꦚ꧀ꦕꦶ ꦫꦼꦒ ꦭꦼꦩꦃ"


def test_fonts_that_render_both_spellings_alike_always_get_the_standard_one(tok, monkeypatch):
    # Font yang meligaturkan cakra + pepet menjadi keret (GumregahNew): satu gambar tidak boleh berlabel dua, jadi
    # aturan itu selalu dipakai begitu pilihannya hidup, juga pada baris yang undiannya tidak kena; aturan lain tidak.
    assert ds.spelling_twins(NOTO) == frozenset()  # Noto merender semua bentuk berbeda
    monkeypatch.setattr(ds, "spelling_twins", lambda font: frozenset({"keret"}))
    line = "ꦠꦿꦼꦱ꧀ꦤ ꦫꦼꦒ"
    never = SyntheticLines([line] * 30, tok, [NOTO], deterministic=True, seed=2, standard_prob=1e-9)
    assert {never.sample(i)[1] for i in range(30)} == {"ꦠꦽꦱ꧀ꦤ ꦫꦼꦒ"}
    # Tanpa pilihan itu (run lama), font kembar pun tidak diubah.
    off = SyntheticLines([line], tok, [NOTO], deterministic=True, seed=2)
    assert off.sample(0)[1] == line


def test_twins_are_found_by_comparing_renders(monkeypatch):
    from PIL import Image

    def fake_render(text, font, size, **kwargs):
        # Font palsu: keret dan cakra + pepet digambar sama, bentuk lain berbeda (lebar = fungsi teksnya).
        text = text.replace("ꦿꦼ", "ꦽ")
        img = Image.new("L", (40, 20), 255)
        for k, ch in enumerate(text):
            img.putpixel((k, ord(ch) % 20), 0)
        return img

    ds.spelling_twins.cache_clear()
    monkeypatch.setattr(ds, "render_line", fake_render)
    assert ds.spelling_twins("palsu.ttf") == frozenset({"keret"})
    ds.spelling_twins.cache_clear()


def test_language_model_lines_follow_the_same_rules(tmp_path, monkeypatch):
    from src import charlm

    splits = tmp_path / "splits"
    splits.mkdir()
    (splits / "train.txt").write_text("\n".join(LINES * 20) + "\n", encoding="utf-8")
    monkeypatch.setattr(charlm, "SPLITS_DIR", splits)
    monkeypatch.setattr(charlm, "LM_DIR", tmp_path)
    charlm.main(["--order", "3", "--lines", "0", "--drop-space-prob", "0.5"])
    charlm.main(["--order", "3", "--lines", "0", "--drop-space-prob", "0.5", "--standard-prob", "1.0"])
    old = charlm.CharLM.load(tmp_path / "o3_n60_ds0.5.pkl")
    new = charlm.CharLM.load(tmp_path / "o3_n60_ds0.5_std1.0.pkl")
    count = lambda lm, ch: lm.grams[1].get(ch, 0)  # noqa: E731
    assert count(old, "ꦉ") == 0 and count(old, "ꦚ") == 0 and count(old, "ꦽ") == 0
    assert count(new, "ꦉ") == 20 and count(new, "ꦚ") == 20 and count(new, "ꦽ") == 20 and count(new, "꧇") == 40
    assert new.meta["standard_prob"] == 1.0 and old.meta["standard_prob"] == 0.0
    # Tanpa pilihan itu, model bahasanya sama persis dengan sebelum pilihan itu ada (aliran acak spasi tidak bergeser).
    charlm.main(["--order", "3", "--lines", "0", "--drop-space-prob", "0.5", "--out", str(tmp_path / "lagi.pkl")])
    assert charlm.CharLM.load(tmp_path / "lagi.pkl").grams == old.grams
    half = tmp_path / "half.pkl"
    charlm.main(["--order", "3", "--lines", "0", "--drop-space-prob", "0.0", "--standard-prob", "0.5", "--out", str(half)])
    assert 0 < count(charlm.CharLM.load(half), "ꦉ") < 20
    assert random.Random("0:standard").random() != random.Random(0).random()
