"""Test scripts/eval_spacing.py: dosis-respons jarak antar suku kata, dengan data buatan dan pembaca palsu."""

import json
import random
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein

from src.dataset import TRAIN_FONTS
from src.decode import cer
from src.render import EmptyRender
from src.tokenizer import WINDOWS_JAVANESE_TEXT, Tokenizer, logical_syllables, nfc

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))  # scripts/ bukan paket

import eval_spacing  # noqa: E402
from compare_runs import alignment  # noqa: E402
from eval_spacing import (  # noqa: E402
    DEFAULT_CHECKPOINTS,
    NO_SPACE,
    SPACED,
    EvalError,
    ReadingCache,
    boundary_count,
    compare_pair,
    estimate_full,
    font_note,
    line_stats,
    resolve_checkpoints,
    select_lines,
    split_spaces,
    summarize_condition,
)

TOKENIZER_JSON = ROOT / "data" / "tokenizer.json"
NOTO = next(p for p in TRAIN_FONTS if p.name.startswith("NotoSansJavanese"))

KA, NA, TA, SA = "ꦏ", "ꦤ", "ꦠ", "ꦱ"
PANGKON, WULU, TALING, TARUNG = "꧀", "ꦶ", "ꦺ", "ꦴ"

# Keluaran transliterator untuk kalimat Jawa nyata (empat pertama sama dengan tests/test_eval_rare.py).
LINES = [
    "ꦧꦱ ꦗꦮ ꦲꦶꦏꦸ ꦧꦱ ꦏꦁ ꦢꦶꦲꦼꦁꦒꦺꦴ ꦢꦼꦤꦶꦁ ꦮꦺꦴꦁ ꦗꦮ꧉",
    "ꦏꦸꦛ ꦪꦺꦴꦒꦾꦏꦂꦠ ꦲꦤ ꦲꦶꦁ ꦠ꧀ꦭꦠꦃ ꦗꦮ ꦠꦼꦔꦃ꧈",
    "ꦮꦺꦴꦁ꧈ ꦏꦺꦴꦮꦺ꧈ ꦚꦮꦶꦗꦶ꧈ ꦏꦿꦠꦺꦴꦤ꧀ ꦥꦿꦝꦤ꧈ ꦱꦱ꧀ꦠꦿ",
    "ꦲꦶꦁ ꦱꦶꦱꦶꦃ ꦭꦺꦴꦂ ꦮꦺꦠꦤ꧀ ꧌ꦏꦶꦢꦸꦭ꧀꧍ ꦲꦤ ꦏꦭꦶ꧇",
    "ꦮꦺꦴꦁ ꦗꦮ ꦲꦤ ꦲꦶꦁ ꦏꦸꦛ",
    "ꦏꦭꦶ ꦲꦶꦏꦸ ꦲꦤ ꦲꦶꦁ ꦱꦶꦱꦶꦃ ꦮꦺꦠꦤ꧀",
]

needs_tokenizer = pytest.mark.skipif(not TOKENIZER_JSON.exists(), reason="tokenizer belum dibangun")


def spread(text: str) -> str:
    """Pembaca yang menulis spasi di setiap batas suku kata bukan-spasi: gejala G3 pada cetakan yang direnggangkan."""
    syllables = logical_syllables(text)
    out = []
    for k, syllable in enumerate(syllables):
        out.append(syllable)
        if k + 1 < len(syllables) and not syllable.isspace() and not syllables[k + 1].isspace():
            out.append(" ")
    return "".join(out)


# --- masukan -----------------------------------------------------------------------------------------------


def test_missing_checkpoints_are_skipped_not_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_spacing, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    snapshot = tmp_path / "checkpoints" / "run_b" / "last_snapshot.pt"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"x")
    other = tmp_path / "c.pt"
    other.write_bytes(b"x")
    found, skipped = resolve_checkpoints(["run_a", "run_b", f"lain={other}", f"hilang={tmp_path / 'tidak_ada.pt'}"])
    assert found == {"run_b": snapshot, "lain": other} and list(found) == ["run_b", "lain"]  # urutan = arah selisih
    assert [entry["name"] for entry in skipped] == ["run_a", "hilang"]
    assert skipped[1]["path"] == (tmp_path / "tidak_ada.pt").resolve().as_posix()
    with pytest.raises(EvalError, match="tidak ada satu pun checkpoint.*run_a.*run_x"):
        resolve_checkpoints(["run_a", "run_x"])
    with pytest.raises(EvalError, match="disebut dua kali"):
        resolve_checkpoints(["run_b", f"run_b={other}"])
    with pytest.raises(EvalError, match="bukan NAMA"):
        resolve_checkpoints([f"={other}"])
    # Pasangan pertama bawaan = efek jarak pada jumlah langkah yang sama (aturan analisis fase 7).
    assert DEFAULT_CHECKPOINTS[:2] == ("fase7_track", "fase7_ctrl")


def test_font_notes_say_how_the_font_relates_to_training():
    assert font_note(NOTO).startswith("font training (font inti)")
    if WINDOWS_JAVANESE_TEXT.exists():
        note = font_note(WINDOWS_JAVANESE_TEXT)
        assert "CarakanJawa" in note and "bukan font yang benar-benar baru" in note


# --- per baris ---------------------------------------------------------------------------------------------


def test_boundaries_are_between_adjacent_non_space_syllables():
    assert boundary_count("") == 0 and boundary_count(KA) == 0
    assert boundary_count(KA + NA + TA) == 2
    assert boundary_count(KA + WULU + NA + TALING + TARUNG) == 1  # sandhangan menempel pada aksaranya
    assert boundary_count(KA + PANGKON + NA + TA) == 1  # rantai pasangan = satu suku kata
    assert boundary_count(KA + NA + " " + TA + SA) == 2  # batas di kiri dan kanan spasi tidak dihitung
    assert boundary_count(" " + KA + NA + "  " + TA + " ") == 1  # spasi tepi dan spasi ganda
    # Membuang spasi menambah satu batas per spasi, kecuali bila pangkon bertemu aksara (menjadi pasangan).
    assert boundary_count(KA + NA + TA + SA) == 3
    assert boundary_count(KA + NA + PANGKON + " " + TA) == 1 == boundary_count(KA + NA + PANGKON + TA)
    rng = random.Random(0)
    for _ in range(300):
        text = "".join(rng.choice((KA, NA, TA, WULU, PANGKON)) for _ in range(rng.randint(1, 12)))
        assert boundary_count(text) == len(logical_syllables(text)) - 1  # tanpa spasi: setiap batas dihitung


def spaces(ref: str, hyp: str) -> tuple[int, int]:
    """(spasi palsu, spasi asli terbaca) satu baris."""
    stats = line_stats(ref, hyp)
    return stats["false_spaces"], stats["matched_spaces"]


def test_false_and_real_spaces_are_told_apart_by_the_gap_they_sit_in():
    assert split_spaces("") == ("", [0])
    assert split_spaces(" " + KA + "  " + NA + WULU) == (KA + NA + WULU, [1, 2, 0, 0])  # celah = sebelum tiap karakter

    # Referensi tanpa spasi (cetakan): setiap spasi keluaran palsu, dan hanya spasi yang salah.
    loose = line_stats(KA + NA + TA, KA + " " + NA + " " + TA)
    assert (loose["false_spaces"], loose["matched_spaces"], loose["ref_spaces"], loose["hyp_spaces"]) == (2, 0, 0, 2)
    assert (loose["boundaries"], loose["edits"], loose["edits_no_space"], loose["glyph_edits"]) == (2, 2, 0, 0)
    assert (loose["chars"], loose["glyph_chars"]) == (3, 3)

    ref = KA + NA + " " + TA + SA  # dua batas bukan-spasi, satu spasi asli
    assert spaces(ref, ref) == (0, 1)
    assert spaces(ref, KA + " " + NA + " " + TA + " " + SA) == (2, 1)  # spasi asli terbaca + dua palsu
    assert spaces(ref, KA + NA + TA + SA) == (0, 0)  # spasi asli hilang: recall turun, tidak ada yang palsu
    assert spaces(KA + " " + NA, KA + "  " + NA) == (1, 1)  # spasi ganda di celah kata: satu asli, satu palsu
    assert spaces(KA + NA + TA + " " + SA, KA + " " + NA + TA + SA) == (1, 0)  # spasi di celah lain bukan spasi asli
    assert spaces(KA + NA + TA, KA + " " + TA) == (1, 0)  # aksara terbaca spasi
    edge = line_stats(KA + NA, " " + KA + NA + " ")  # spasi tepi ikut dihitung: bisa melebihi satu per batas
    assert (edge["false_spaces"], edge["boundaries"]) == (2, 1)

    blind = line_stats(ref, KA + NA + TA + SA)
    assert (blind["edits"], blind["edits_no_space"], blind["glyph_edits"], blind["glyph_chars"]) == (1, 1, 0, 4)
    wrong = line_stats(ref, KA + NA + " " + TA + TA)
    assert (wrong["edits"], wrong["edits_no_space"], wrong["glyph_edits"]) == (1, 2, 1)


def test_a_space_one_letter_off_is_false_although_string_alignment_would_pair_it():
    def string_alignment_pairs(ref: str, hyp: str) -> int:
        return sum(1 for tag, i, _ in alignment(ref, hyp) if tag == "equal" and ref[i] == " ")

    # Celah kata tidak terbaca, spasi palsu satu aksara di kanannya (pola keluaran nyata pada teks berjarak).
    ref, shifted_right = KA + " " + NA + TA, KA + NA + " " + TA
    assert spaces(ref, shifted_right) == (1, 0)
    # Penjajaran string utuh seri (hapus spasi + sisip spasi vs sisip ꦤ + hapus ꦤ) dan memilih memasangkan spasinya,
    # tetapi hanya untuk geseran ke kanan: itulah sebabnya spasi dipasangkan per celah.
    assert string_alignment_pairs(ref, shifted_right) == 1
    assert string_alignment_pairs(KA + NA + " " + TA, KA + " " + NA + TA) == 0
    assert spaces(KA + NA + " " + TA, KA + " " + NA + TA) == (1, 0)  # geser kiri: sama-sama palsu

    # Aksara salah di samping celah tidak memindahkan celahnya.
    assert spaces(KA + " " + NA + TA, KA + " " + TA) == (0, 1)  # ꦤ hilang: celah kata tetap bertemu
    assert spaces(KA + " " + NA, KA + " " + SA) == (0, 1)  # ꦤ terbaca ꦱ
    assert spaces(KA + " " + NA, KA + SA + " " + NA) == (0, 1)  # aksara sisipan sebelum spasi
    assert spaces(KA + " " + NA, KA + " " + SA + NA) == (0, 1)  # aksara sisipan sesudah spasi
    assert spaces(KA + " " + NA + " " + TA, KA + " " + TA) == (0, 1)  # dua celah kata menyatu: satu pasangan saja


def test_spaces_pair_exactly_per_gap_when_the_letters_are_read_correctly():
    rng = random.Random(0)
    for _ in range(1000):
        glyphs = [rng.choice((KA, NA, TA, WULU)) for _ in range(rng.randint(0, 10))]

        def with_spaces() -> str:
            gaps = [" " * rng.choice((0, 0, 1, 2)) for _ in range(len(glyphs) + 1)]
            return gaps[0] + "".join(glyph + gap for glyph, gap in zip(glyphs, gaps[1:]))

        ref, hyp = with_spaces(), with_spaces()
        expected = sum(min(a, b) for a, b in zip(split_spaces(ref)[1], split_spaces(hyp)[1]))
        assert spaces(ref, hyp) == (hyp.count(" ") - expected, expected), (ref, hyp)
        assert line_stats(ref, hyp)["glyph_edits"] == 0
    for _ in range(1000):  # keluaran sembarang: jumlahnya tetap konsisten
        ref = "".join(rng.choice((KA, NA, TA, WULU, " ")) for _ in range(rng.randint(0, 10)))
        hyp = "".join(rng.choice((KA, NA, TA, WULU, " ")) for _ in range(rng.randint(0, 12)))
        stats = line_stats(ref, hyp)
        assert 0 <= stats["matched_spaces"] <= min(ref.count(" "), hyp.count(" ")), (ref, hyp)
        assert stats["false_spaces"] == hyp.count(" ") - stats["matched_spaces"], (ref, hyp)
        assert stats["edits"] == Levenshtein.distance(ref, hyp)
        assert stats["edits_no_space"] == Levenshtein.distance(ref, hyp.replace(" ", ""))
        assert stats["glyph_edits"] == Levenshtein.distance(ref.replace(" ", ""), hyp.replace(" ", ""))
        assert stats["glyph_chars"] == len(ref) - ref.count(" ") and stats["chars"] == len(ref)


def test_reader_that_spaces_every_boundary_scores_100_per_100_and_full_recall():
    for line in LINES:
        for ref in (line, line.replace(" ", "")):
            stats = line_stats(ref, spread(ref))
            assert stats["false_spaces"] == stats["boundaries"] > 0
            assert stats["matched_spaces"] == stats["ref_spaces"] and stats["glyph_edits"] == 0


# --- ringkasan ---------------------------------------------------------------------------------------------


PAIRS = [
    (KA + NA + " " + TA + SA, KA + " " + NA + " " + TA + SA),  # 2 batas; spasi asli terbaca, 1 palsu
    (KA + NA + TA, KA + NA + TA),  # 2 batas; persis
    (KA + " " + NA + " " + TA, KA + NA + " " + TA),  # 0 batas; 1 dari 2 spasi terbaca
    (KA + WULU + NA, KA + WULU + " " + SA),  # 1 batas; 1 palsu dan 1 aksara salah
]


def test_summary_is_micro_and_reproduces_the_official_cer():
    refs, hyps = [r for r, _ in PAIRS], [h for _, h in PAIRS]
    summary = summarize_condition(PAIRS, [line_stats(r, h) for r, h in PAIRS])
    assert (summary["lines"], summary["reference_chars"]) == (4, 16)
    assert (summary["boundaries"], summary["false_spaces"]) == (5, 2)
    assert summary["false_per_boundary"] == pytest.approx(2 / 5)  # = 40 spasi palsu per 100 batas
    assert (summary["ref_spaces"], summary["matched_spaces"], summary["hyp_spaces"]) == (3, 2, 4)
    assert summary["space_recall"] == pytest.approx(2 / 3)
    assert summary["cer"] == cer(refs, hyps)
    assert summary["cer_hyp_no_space"] == cer(refs, [h.replace(" ", "") for h in hyps])  # definisi src/evaluate.py
    assert summary["cer_space_insensitive"] == pytest.approx(1 / 13)  # hanya ꦤ -> ꦱ di baris terakhir
    assert summary["exact_line_accuracy"] == pytest.approx(1 / 4)

    # Bentuk tanpa-spasi: tidak ada spasi referensi, jadi recall tidak terdefinisi dan kedua CER kontrol sama.
    bare = [(r.replace(" ", ""), h) for r, h in PAIRS]
    summary = summarize_condition(bare, [line_stats(r, h) for r, h in bare])
    assert summary["space_recall"] is None and summary["ref_spaces"] == 0
    assert summary["false_spaces"] == summary["hyp_spaces"] == 4
    assert summary["cer_hyp_no_space"] == pytest.approx(summary["cer_space_insensitive"])


def test_summary_survives_output_that_stops_being_nfc_once_spaces_are_dropped():
    # Pangkon (kelas gabung 9) + spasi + cecak telu (7) adalah NFC, tetapi sesudah spasi dibuang keduanya bersebelahan
    # dan NFC menukarnya. src.evaluate.summarize menormalkan sesudah membuang spasi; jumlah per baris yang tidak
    # menormalkan menghentikan seluruh evaluasi dengan galat "tidak mereproduksi" (dan lagi di setiap jalan ulang,
    # karena bacaannya ada di cache) hanya karena satu aksara dasar tidak terbaca.
    pa, cecak_telu = "ꦥ", "꦳"
    hyp = KA + NA + PANGKON + " " + cecak_telu + WULU  # ꦥ tidak terbaca, cecak telu-nya tinggal
    assert nfc(hyp) == hyp and nfc(hyp.replace(" ", "")) != hyp.replace(" ", "")

    spaced = KA + NA + PANGKON + " " + pa + cecak_telu + WULU
    stats = line_stats(spaced, hyp)
    summary = summarize_condition([(spaced, hyp)], [stats])
    assert summary["cer"] == cer([spaced], [hyp]) == pytest.approx(1 / 7)
    assert summary["cer_hyp_no_space"] == cer([spaced], [hyp.replace(" ", "")]) == pytest.approx(4 / 7)
    assert summary["cer_space_insensitive"] == cer([spaced.replace(" ", "")], [hyp.replace(" ", "")])
    assert (stats["matched_spaces"], stats["false_spaces"]) == (1, 0)  # spasi tetap dipasangkan pada teks keluaran

    bare = spaced.replace(" ", "")  # bentuk tanpa-spasi: ꦥ terbaca spasi; kedua CER kontrol tetap sama
    stats = line_stats(bare, hyp)
    summary = summarize_condition([(bare, hyp)], [stats])
    assert summary["cer_hyp_no_space"] == cer([bare], [hyp.replace(" ", "")]) == summary["cer_space_insensitive"]
    assert (stats["matched_spaces"], stats["false_spaces"], stats["glyph_chars"]) == (0, 1, 6)

    extra = KA + NA + PANGKON + " " + cecak_telu + pa + WULU  # cecak telu tambahan sesudah celah kata
    plain = KA + NA + PANGKON + " " + pa + WULU
    summary = summarize_condition([(plain, extra)], [line_stats(plain, extra)])
    assert summary["cer_hyp_no_space"] == cer([plain], [extra.replace(" ", "")]) == pytest.approx(2 / 6)


def test_pair_difference_is_a_minus_b_per_condition_on_the_same_lines():
    refs = {NO_SPACE: [KA + NA + TA, NA + TA + SA + KA], SPACED: [KA + NA + " " + TA, NA + TA + " " + SA + KA]}

    def stats(reader) -> dict:
        return {(form, tracking): [line_stats(ref, reader(ref, tracking)) for ref in texts]
                for form, texts in refs.items() for tracking in (0.0, 0.3)}

    tight = stats(lambda ref, tracking: ref)  # dilatih dengan jarak: tidak terpengaruh
    loose = stats(lambda ref, tracking: spread(ref) if tracking else ref)  # renggang terbaca spasi
    rows = compare_pair(tight, loose, resamples=300, seed=0)
    assert [(row["form"], row["tracking"]) for row in rows] == list(tight)
    by = {(row["form"], row["tracking"]): row["metrics"] for row in rows}
    assert set(by[NO_SPACE, 0.3]) == {"false_per_boundary", "cer_space_insensitive"}  # tanpa spasi: tanpa recall
    assert set(by[SPACED, 0.3]) == {"false_per_boundary", "space_recall", "cer_space_insensitive"}
    false = by[NO_SPACE, 0.3]["false_per_boundary"]
    assert (false["a"], false["b"], false["diff"]) == (0.0, 1.0, -1.0)
    assert false["bootstrap"]["ci95"] == [-1.0, -1.0]  # setiap resample: A nol, B satu spasi per batas
    assert by[NO_SPACE, 0.0]["false_per_boundary"]["diff"] == 0
    assert by[NO_SPACE, 0.0]["false_per_boundary"]["bootstrap"]["ci95"] == [0.0, 0.0]
    assert by[SPACED, 0.3]["false_per_boundary"]["diff"] == -1.0
    assert by[SPACED, 0.3]["space_recall"]["diff"] == 0 and by[SPACED, 0.3]["space_recall"]["a"] == 1.0
    assert all(row["metrics"]["cer_space_insensitive"]["diff"] == 0 for row in rows)

    blind = stats(lambda ref, tracking: ref.replace(" ", ""))
    recall = {(row["form"], row["tracking"]): row["metrics"].get("space_recall")
              for row in compare_pair(tight, blind, resamples=0, seed=0)}
    assert recall[SPACED, 0.0]["diff"] == 1.0 and recall[SPACED, 0.0]["bootstrap"] is None
    assert recall[NO_SPACE, 0.0] is None


def test_full_run_estimate_extrapolates_measured_times_linearly_in_tracking():
    measured = [
        {"form": NO_SPACE, "tracking": 0.0, "render_ms": 2.0, "infer_ms": {"a": 100.0, "b": 300.0}},
        {"form": NO_SPACE, "tracking": 0.3, "render_ms": 5.0, "infer_ms": {"a": 160.0, "b": 360.0}},
        {"form": SPACED, "tracking": 0.0, "render_ms": 4.0, "infer_ms": {"a": 300.0, "b": None}},
        {"form": SPACED, "tracking": 0.3, "render_ms": None, "infer_ms": {"a": None, "b": None}},  # semua dari cache
    ]
    estimate = estimate_full(measured, lines=10, trackings=(0.0, 0.6), n_checkpoints=5)
    # tanpa-spasi: render 2 + 10t, inferensi 200 + 200t ms; berspasi: satu jarak terukur = konstan 4 dan 300 ms.
    assert estimate["render_s"] == pytest.approx(10 * (2 + 8 + 4 + 4) / 1000)
    assert estimate["infer_s"] == pytest.approx(10 * 5 * (200 + 320 + 300 + 300) / 1000)
    assert estimate["seconds"] == pytest.approx(estimate["render_s"] + estimate["infer_s"])
    assert estimate["extrapolated"] and estimate["checkpoints"] == 5
    assert estimate["flat"] and estimate["measured_trackings"] == [0.0, 0.3]  # berspasi: tanpa kemiringan
    assert not estimate_full(measured, 10, (0.0, 0.3), 5)["extrapolated"]
    measured[3] = {"form": SPACED, "tracking": 0.3, "render_ms": 7.0, "infer_ms": {"a": 420.0, "b": None}}
    sloped = estimate_full(measured, lines=10, trackings=(0.0, 0.6), n_checkpoints=5)
    assert not sloped["flat"]  # berspasi kini render 4 + 10t, inferensi 300 + 400t ms
    assert sloped["render_s"] == pytest.approx(10 * (2 + 8 + 4 + 10) / 1000)
    assert sloped["infer_s"] == pytest.approx(10 * 5 * (200 + 320 + 300 + 540) / 1000)
    assert estimate_full(measured[:2], 10, (0.0, 0.6), 5) is None  # bentuk berspasi tidak terukur
    assert estimate_full([], 10, (0.0, 0.6), 5) is None  # semua bacaan dari cache


# --- cache bacaan ------------------------------------------------------------------------------------------


def test_reading_cache_survives_restart_broken_lines_and_other_environments(tmp_path):
    path = tmp_path / "out" / "cache.jsonl"
    env = {"pillow": "1", "torch": "2"}
    cache = ReadingCache(path, env)
    key = cache.key("font", 64, 8, 0.3, KA + NA)
    assert cache.get(key, "ckpt") is None
    cache.put(key, {"ckpt": KA + " " + NA, "kosong": ""})
    assert cache.get(key, "ckpt") == KA + " " + NA and cache.get(key, "kosong") == ""  # keluaran kosong itu sah
    cache.put(cache.key("font", 64, 8, 0, KA), {"ckpt": KA})
    cache.close()

    again = ReadingCache(path, env)
    assert (again.loaded, again.foreign, again.broken) == (3, 0, 0)
    assert again.get(key, "ckpt") == KA + " " + NA and again.get(key, "kosong") == ""
    assert again.get(again.key("font", 64, 8, 0.0, KA), "ckpt") == KA  # --tracking 0 dan 0.0 sama
    assert again.get(again.key("font", 64, 8, 0.3, KA), "ckpt") is None  # jarak lain
    assert again.get(again.key("lain", 64, 8, 0.3, KA + NA), "ckpt") is None  # font lain
    assert again.get(again.key("font", 56, 8, 0.3, KA + NA), "ckpt") is None  # ukuran lain
    again.put(key, {"baru": NA})  # checkpoint yang menyusul: bacaan citra yang sama digabung
    again.close()

    with path.open("ab") as f:  # proses mati saat menulis: baris terakhir terpotong di tengah karakter
        f.write('{"v": 1, "env": {"pillow": "1", "torch": "2"}, "font": "font", "text": "ꦏ'.encode()[:-1])
    broken = ReadingCache(path, env)
    assert (broken.loaded, broken.broken) == (4, 1) and broken.get(key, "baru") == NA
    broken.put(key, {"lagi": TA})  # tidak tersambung ke baris yang terpotong
    broken.close()
    healed = ReadingCache(path, env)
    assert (healed.loaded, healed.broken) == (5, 1) and healed.get(key, "lagi") == TA
    healed.close()

    other = ReadingCache(path, {"pillow": "9", "torch": "2"})  # Pillow lain bisa merender berbeda: tidak dipakai
    assert (other.loaded, other.foreign) == (0, 5) and other.get(key, "ckpt") is None
    other.close()
    fresh = ReadingCache(path, env, fresh=True)
    assert fresh.loaded == 0 and fresh.get(key, "ckpt") is None
    fresh.close()
    assert path.read_text(encoding="utf-8") == ""


# --- baris yang dipakai ------------------------------------------------------------------------------------


def small_tokenizer() -> Tokenizer:
    """Tabel reorder hanya mengenal aksara + taling, jadi pasangan bertaling tidak round-trip (seperti 0,3% val)."""
    return Tokenizer([" ", KA, NA, TA, SA, PANGKON, TALING, WULU], {"L A9BA": [1, 0]})


def test_lines_are_kept_or_dropped_for_every_condition_at_once(monkeypatch):
    tok = small_tokenizer()
    good, other = KA + NA + " " + TA + TALING, SA + WULU + " " + KA + NA
    # Berspasi round-trip; tanpa spasi "ꦤ꧀ꦠꦺ" menjadi satu suku kata yang polanya tidak ada di tabel reorder.
    merged = KA + NA + PANGKON + " " + TA + TALING
    assert tok.to_logical(tok.to_visual(merged)) == merged
    assert tok.to_logical(tok.to_visual(merged.replace(" ", ""))) != merged.replace(" ", "")

    kept, dropped = select_lines([good, merged, other], [7, 3, 5], NOTO, tok, (0.0, 0.3))
    assert kept == [{"i": 0, "split_index": 7, "text": {NO_SPACE: good.replace(" ", ""), SPACED: good}},
                    {"i": 2, "split_index": 5, "text": {NO_SPACE: other.replace(" ", ""), SPACED: other}}]
    assert dropped == [{"i": 1, "split_index": 3, "reason": "tidak round-trip tokenizer (tanpa-spasi)", "text": merged}]
    outside = select_lines(["ꦧ" + KA], [0], NOTO, tok, (0.0,))[1]  # aksara di luar charset
    assert outside[0]["reason"] == "tidak round-trip tokenizer (tanpa-spasi)"

    real_render = eval_spacing.render_line

    def narrow_at_wide_tracking(text, font, size, tracking=0.0):
        if text == other and tracking == 0.3:
            return Image.new("L", (8, 96), 255)  # 2 frame untuk 5 target: melanggar T >= 1,5L
        if text == good.replace(" ", "") and tracking == 0.3:
            raise EmptyRender("uji")
        return real_render(text, font, size, tracking=tracking)

    monkeypatch.setattr(eval_spacing, "render_line", narrow_at_wide_tracking)
    kept, dropped = select_lines([good, other, KA + NA], [0, 1, 2], NOTO, tok, (0.0, 0.3))
    # Gagal di SATU kondisi = keluar dari semua kondisi, supaya setiap jarak memakai baris yang sama.
    assert [line["i"] for line in kept] == [2]
    assert [entry["reason"] for entry in dropped] == ["gagal render (tanpa-spasi, 0.3 em): uji",
                                                      "T < 1,5L (berspasi, 0.3 em)"]

    # Font tanpa glyph untuk sebuah aksara merender kotak .notdef tanpa galat: labelnya tidak cocok dengan citra.
    # Spasi tidak diperiksa (GumregahNew membentuk celah kata lewat shaping, tanpa glyph spasi).
    covered = eval_spacing.font_codepoints(str(NOTO))
    monkeypatch.setattr(eval_spacing, "font_codepoints", lambda path: covered - {ord(SA), ord(TA), ord(" ")})
    kept, dropped = select_lines([good, other, KA + " " + NA], [0, 1, 2], NOTO, tok, (0.0,))
    assert [line["i"] for line in kept] == [2]
    assert [entry["reason"] for entry in dropped] == ["glyph tidak ada di font (U+A9A0)",
                                                      "glyph tidak ada di font (U+A9B1)"]


# --- skrip utuh --------------------------------------------------------------------------------------------


class Interrupted(Exception):
    """Pengganti proses yang dimatikan di tengah jalan."""


@needs_tokenizer
def test_main_with_fake_readers_resumes_from_cache_and_writes_dose_response(tmp_path, monkeypatch, capsys):
    tok = Tokenizer.load(TOKENIZER_JSON)
    real_render = eval_spacing.render_line

    def render_with_label(text, font, size, tracking=0.0):
        image = real_render(text, font, size, tracking=tracking)
        image.info.update(label=text, tracking=tracking)  # pembaca palsu membaca label yang ditempel pada citra
        return image

    readers = {
        "rapat": lambda text, tracking: text,  # dilatih dengan jarak: membaca persis di setiap jarak
        "renggang": lambda text, tracking: spread(text) if tracking >= 0.3 else text,  # gejala G3
        "buta": lambda text, tracking: text.replace(" ", ""),  # tidak pernah menulis spasi
        "baru": lambda text, tracking: text,
    }
    calls: list[str] = []
    budget = [40]  # bacaan yang boleh dihitung sebelum "proses mati"; None = tanpa batas

    def fake_predict(image, model, tokenizer, **_):
        if "label" not in image.info:
            return ""  # citra pemanasan
        if budget[0] is not None:
            if budget[0] == 0:
                raise Interrupted
            budget[0] -= 1
        calls.append(model)
        return readers[model](image.info["label"], image.info["tracking"])

    monkeypatch.setattr(eval_spacing, "render_line", render_with_label)
    monkeypatch.setattr(eval_spacing, "load_checkpoint", lambda path: (Path(path).stem, tok))
    monkeypatch.setattr(eval_spacing, "predict", fake_predict)
    monkeypatch.setattr(eval_spacing, "read_split", lambda name: list(LINES))
    monkeypatch.setattr(eval_spacing, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    paths = {name: tmp_path / f"{name}.pt" for name in readers}
    for name, path in paths.items():
        tracked = name == "rapat"
        torch.save({"step": 7, "name": name, "args": {"track_prob": 0.5 if tracked else None,
                                                      "track_max": 0.3 if tracked else None}}, path)
    out = tmp_path / "out"
    names = ["rapat", "renggang", "buta"]
    argv = ["--lines", "6", "--tracking", "0.3", "0", "--fonts", str(NOTO), "--out", str(out), "--bootstrap", "200",
            "--checkpoints", f"rapat={paths['rapat']}", "belum_ada", f"renggang={paths['renggang']}",
            f"buta={paths['buta']}"]

    # Jalan pertama terputus: laporan akhir belum ada, tetapi kondisi yang selesai dan setiap bacaan sudah di disk.
    with pytest.raises(Interrupted):
        eval_spacing.main(argv)
    assert not (out / "spacing_synthetic.json").exists()
    partial = [json.loads(line) for line in (out / "spacing_synthetic_partial.jsonl").read_text("utf-8").splitlines()]
    assert [(p["form"], p["tracking"]) for p in partial] == [(NO_SPACE, 0.0), (NO_SPACE, 0.3)]
    assert partial[1]["results"]["renggang"]["false_per_boundary"] == 1.0
    cached = (out / "spacing_synthetic_cache.jsonl").read_text("utf-8").splitlines()
    assert len(cached) == 13 and calls == names * 13 + ["rapat"]  # citra ke-14 terputus sebelum tersimpan

    # Jalan ulang melanjutkan: hanya bacaan yang belum ada yang dihitung.
    calls.clear()
    budget[0] = None
    eval_spacing.main(argv)
    assert len(calls) == 6 * 2 * 2 * 3 - 39
    assert not (out / "spacing_synthetic_partial.jsonl").exists()
    report = json.loads((out / "spacing_synthetic.json").read_text(encoding="utf-8"))
    assert report["cache"]["computed"] == 33 and report["cache"]["reused"] == 39
    assert list(report["checkpoints"]) == names  # urutan --checkpoints = arah selisih
    assert [entry["name"] for entry in report["skipped_checkpoints"]] == ["belum_ada"]
    assert report["checkpoints"]["rapat"]["track_prob"] == 0.5 and report["checkpoints"]["rapat"]["step"] == 7
    settings = report["settings"]
    assert (settings["lines"], settings["tracking"], settings["size"]) == (6, [0.0, 0.3], 64)

    block = report["fonts"][NOTO.name]
    assert block["lines"] == 6 and block["dropped"] == [] and block["note"].startswith("font training")
    assert block["reference"][NO_SPACE]["spaces"] == 0 < block["reference"][SPACED]["spaces"]
    assert block["reference"][NO_SPACE]["chars"] + block["reference"][SPACED]["spaces"] \
        == block["reference"][SPACED]["chars"]
    results = block["results"]
    for form in (NO_SPACE, SPACED):
        assert [row["tracking"] for row in results["rapat"][form]] == [0.0, 0.3]  # baris tabel = jarak, menaik
        assert all(row["false_per_boundary"] == 0 and row["cer"] == 0 for row in results["rapat"][form])
        near, far = results["renggang"][form]
        assert (near["false_per_boundary"], far["false_per_boundary"]) == (0.0, 1.0)  # 0 -> 100 per 100 batas
        assert far["false_spaces"] == far["boundaries"] == block["reference"][form]["boundaries"]
        assert far["cer"] > 0 and far["cer_space_insensitive"] == 0  # bentuk aksara tetap terbaca
        assert all(row["false_per_boundary"] == 0 for row in results["buta"][form])
    assert all(row["cer_hyp_no_space"] == 0 for row in results["renggang"][NO_SPACE])
    assert all(row["space_recall"] is None for name in names for row in results[name][NO_SPACE])
    assert all(row["space_recall"] == 1.0 for row in results["rapat"][SPACED] + results["renggang"][SPACED])
    for row in results["buta"][SPACED]:
        assert row["space_recall"] == 0.0 and row["cer_space_insensitive"] == 0
        assert row["cer"] == row["cer_hyp_no_space"] == pytest.approx(row["ref_spaces"] / row["reference_chars"])

    comparisons = block["comparisons"]
    assert [(c["a"], c["b"], c["primary"]) for c in comparisons] == [
        ("rapat", "renggang", True), ("rapat", "buta", False), ("renggang", "buta", False)]
    first = {(row["form"], row["tracking"]): row["metrics"] for row in comparisons[0]["rows"]}
    assert first[NO_SPACE, 0.3]["false_per_boundary"]["diff"] == -1.0
    assert first[NO_SPACE, 0.3]["false_per_boundary"]["bootstrap"]["ci95"] == [-1.0, -1.0]
    assert first[NO_SPACE, 0.0]["false_per_boundary"]["diff"] == 0
    assert first[SPACED, 0.3]["space_recall"]["diff"] == 0 and "space_recall" not in first[NO_SPACE, 0.3]
    second = {(row["form"], row["tracking"]): row["metrics"] for row in comparisons[1]["rows"]}
    assert second[SPACED, 0.0]["space_recall"]["diff"] == 1.0

    # Kontrol positif hanya untuk checkpoint tanpa jarak saat training.
    assert [(entry["checkpoint"], entry["low"]["false_per_boundary"], entry["high"]["false_per_boundary"])
            for entry in report["sanity"]] == [("renggang", 0.0, 1.0), ("buta", 0.0, 0.0)]
    assert report["timing"]["estimate_full"] is None  # bentuk tanpa-spasi seluruhnya dari cache: waktunya tidak terukur

    records = [json.loads(line) for line in (out / "spacing_synthetic_lines.jsonl").read_text("utf-8").splitlines()]
    assert len(records) == 6 * 2 * 2 and {record["split_index"] for record in records} == set(range(6))
    assert all(set(record["hyp"]) == set(names) and record["hyp"]["rapat"] == record["reference"]
               for record in records)
    assert all(" " not in record["reference"] for record in records if record["form"] == NO_SPACE)
    markdown = (out / "spacing_synthetic.md").read_text(encoding="utf-8")
    assert "# Evaluasi sintetis dosis-respons jarak antar suku kata" in markdown
    assert "| jarak (em) | `rapat` | `renggang` | `buta` |\n|---:|---:|---:|---:|\n| 0 | 0.0 | 0.0 | 0.0 |\n" \
           "| 0.3 | 0.0 | 100.0 | 0.0 |" in markdown
    assert "| spasi palsu per 100 batas | tanpa-spasi | 0.3 | 0.0 | 100.0 | -100.00 pt | [-100.00, -100.00] pt |" \
        in markdown
    assert "DILEWATI, berkas tidak ada: `belum_ada`" in markdown and "`rapat` (A) - `renggang` (B), pasangan pertama" \
        in markdown
    printed = capsys.readouterr().out
    assert "dilewati: checkpoint 'belum_ada' tidak ada" in printed
    assert "bacaan (citra x checkpoint): 72, 33 belum ada di cache" in printed
    assert "perkiraan jalan penuh: tidak ada" in printed
    assert "renggang, NotoSansJavanese-Regular.ttf, tanpa-spasi: 0 em -> 0.0 spasi palsu per 100 batas" in printed

    # Checkpoint yang menyusul dan jarak tambahan: hanya bacaan baru yang dihitung, sisanya dari cache.
    calls.clear()
    eval_spacing.main([*argv[:3], "0", "0.3", "0.45", *argv[5:], f"baru={paths['baru']}"])
    assert calls.count("baru") == 6 * 2 * 3 and len(calls) == 6 * 2 * 3 + 6 * 2 * 3
    report = json.loads((out / "spacing_synthetic.json").read_text(encoding="utf-8"))
    assert report["cache"] == {"file": (out / "spacing_synthetic_cache.jsonl").resolve().as_posix(), "version": 1,
                               "computed": 72, "reused": 72}
    assert report["skipped_checkpoints"] == [{"name": "belum_ada", "path": (
        tmp_path / "checkpoints" / "belum_ada" / "last_snapshot.pt").resolve().as_posix()}]
    far = report["fonts"][NOTO.name]["results"]["renggang"][NO_SPACE][-1]
    assert (far["tracking"], far["false_per_boundary"]) == (0.45, 1.0)
    assert "| 0.45* | 0.0 | 100.0 | 0.0 | 0.0 |" in (out / "spacing_synthetic.md").read_text(encoding="utf-8")
    estimate = report["timing"]["estimate_full"]  # setiap kondisi punya bacaan baru, jadi waktunya terukur
    assert (estimate["lines"], estimate["checkpoints"]) == (200, 5)
    assert estimate["trackings"] == [0, 0.1, 0.2, 0.3, 0.45, 0.6]
    assert "perkiraan jalan penuh dengan cache kosong (200 baris x 2 bentuk x 6 jarak" in capsys.readouterr().out

    # Semua dari cache: tidak ada satu pun bacaan baru, angka sama.
    calls.clear()
    budget[0] = 0
    eval_spacing.main([*argv[:3], "0", "0.3", "0.45", *argv[5:], f"baru={paths['baru']}"])
    again = json.loads((out / "spacing_synthetic.json").read_text(encoding="utf-8"))
    assert calls == [] and again["cache"]["computed"] == 0 and again["fonts"][NOTO.name]["results"] \
        == report["fonts"][NOTO.name]["results"]
    assert "perkiraan jalan penuh: tidak ada" in capsys.readouterr().out

    # Baris yang ditolak tidak dibaca dan tidak dihitung, dan jumlah yang dibandingkan dengan ">= 300 baris" aturan
    # analisis adalah baris yang DIPAKAI, bukan --lines (sampel seed 0 yang sebenarnya: --lines 300 memakai 299).
    monkeypatch.setattr(eval_spacing, "read_split", lambda name: [*LINES, LINES[0] + "a"])  # "a" di luar charset
    eval_spacing.main(["--lines", "7", *argv[2:]])
    printed = capsys.readouterr().out
    shorter = json.loads((out / "spacing_synthetic.json").read_text(encoding="utf-8"))
    block = shorter["fonts"][NOTO.name]
    assert calls == [] and shorter["settings"]["lines"] == 7 and block["lines"] == 6
    assert [entry["reason"] for entry in block["dropped"]] == ["tidak round-trip tokenizer (tanpa-spasi)"]
    assert block["results"] == results  # enam baris yang sama dengan jalan pertama: angka tidak berubah
    records = [json.loads(line) for line in (out / "spacing_synthetic_lines.jsonl").read_text("utf-8").splitlines()]
    assert len(records) == 6 * 2 * 2 + 1 and [record["reason"] for record in records if "reason" in record] \
        == ["tidak round-trip tokenizer (tanpa-spasi)"]
    assert "7 baris val (seed 0) x 2 bentuk" in printed and "6 baris dipakai (ditolak: tidak round-trip tokenizer 1)" \
        in printed
    assert "run ini memakai 6 baris (dari 7 yang diambil, sesudah penolakan) x 2 bentuk = 12 citra" in printed
    assert "sesudah penolakan run ini memakai 6 baris x 2 bentuk = 12 citra per jarak dan font, KURANG dari yang " \
           "direncanakan." in (out / "spacing_synthetic.md").read_text(encoding="utf-8")


@needs_tokenizer
def test_main_stops_when_no_checkpoint_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_spacing, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    with pytest.raises(SystemExit, match="galat: tidak ada satu pun checkpoint"):
        eval_spacing.main(["--fonts", str(NOTO), "--out", str(tmp_path / "out")])
    with pytest.raises(SystemExit, match="galat: font tidak ada"):
        eval_spacing.main(["--fonts", str(tmp_path / "tidak_ada.ttf"), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
