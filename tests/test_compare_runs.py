"""Test scripts/compare_runs.py: perbandingan berpasangan dua pipeline, dengan data buatan."""

import json
import random
import sys
from functools import lru_cache
from pathlib import Path

import pytest
from rapidfuzz.distance import Levenshtein

from src.align import line_cer
from src.decode import cer

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))  # scripts/ bukan paket

import compare_runs  # noqa: E402
from compare_runs import (  # noqa: E402
    CompareError,
    alignment,
    bootstrap_diffs,
    compare,
    glyph_class,
    load_rare,
    matched_mask,
    paired_lines,
    sign_test,
)

KA, NA, TA, SA, DA, YA = "ꦏ", "ꦤ", "ꦠ", "ꦱ", "ꦢ", "ꦪ"
KA_MURDA, SA_MURDA, ADEG, LETTER_A = "ꦑ", "ꦯ", "꧋", "ꦄ"
RARE = [KA_MURDA, SA_MURDA, ADEG, LETTER_A]
WINDU, ZERO, LETTER_E, PA_MURDA, EIGHT, NGA_LELET, TWO, SIX = "꧆", "꧐", "ꦌ", "ꦦ", "꧘", "ꦊ", "꧒", "꧖"
LAYAR, WULU, TALING = "ꦂ", "ꦶ", "ꦺ"  # sandhangan (kategori Unicode Mn/Mc)


def rows_of(*items) -> list[dict]:
    """(referensi, keluaran A, keluaran B[, halaman]) -> baris seperti keluaran paired_lines."""
    return [{"id": f"L{i}", "source_id": item[3] if len(item) > 3 else "P1", "reference": item[0],
             "a": item[1], "b": item[2]} for i, item in enumerate(items)]


def contract(items) -> tuple[list[dict], list[dict]]:
    """Baris lines.jsonl dan predictions.jsonl (pipeline p_a, p_b) dari tuple rows_of."""
    lines, preds = [], []
    for row in rows_of(*items):
        lines.append({"dataset": "uji", "id": row["id"], "reference": row["reference"], "source_id": row["source_id"]})
        for side in "ab":
            preds.append({"line": row["id"], "pipeline": f"p_{side}", "text": row[side],
                          "cer": line_cer(row["reference"], row[side])})
    return lines, preds


# --- CER ---------------------------------------------------------------------------------------------------


def test_cer_is_micro_and_equals_official_function():
    rows = rows_of((KA + NA + TA + SA, KA + NA + TA + TA, KA + NA + TA + SA), (KA + NA, KA + NA, ""))
    report = compare(rows, RARE, resamples=0)
    # Mikro: 1 edit / 6 karakter. Rata-rata CER per baris akan memberi (1/4 + 0) / 2 = 12,5%.
    assert report["cer"]["a"] == pytest.approx(1 / 6)
    assert report["cer"]["b"] == pytest.approx(2 / 6)
    assert report["cer"]["diff"] == pytest.approx(-1 / 6)
    refs = [row["reference"] for row in rows]
    assert report["cer"]["a"] == cer(refs, [row["a"] for row in rows])
    assert report["cer"]["b"] == cer(refs, [row["b"] for row in rows])
    assert report["exact_line_accuracy"] == {"a": 0.5, "b": 0.5}
    assert report["lines"] == 2 and report["reference_chars"] == 6


def test_no_space_variant_strips_spaces_from_hypothesis_only():
    rows = rows_of((KA + NA, KA + " " + NA, KA + NA), (KA + " " + NA, KA + " " + NA, KA + NA))
    report = compare(rows, RARE, resamples=0)
    assert report["cer"]["a"] == pytest.approx(1 / 5)  # satu spasi tambahan
    assert report["cer"]["b"] == pytest.approx(1 / 5)  # satu spasi label tidak dibaca
    # Spasi keluaran dibuang, label apa adanya: spasi di label baris kedua kini tidak terbaca oleh A juga.
    assert report["cer_hyp_no_space"]["a"] == pytest.approx(1 / 5)
    assert report["cer_hyp_no_space"]["b"] == pytest.approx(1 / 5)
    only_first = compare(rows[:1], RARE, resamples=0)
    assert only_first["cer"]["a"] == pytest.approx(1 / 2) and only_first["cer_hyp_no_space"]["a"] == 0


def test_paired_counts_and_sign_test():
    ref = KA + NA + TA
    rows = rows_of((ref, ref, KA), (ref, ref, ""), (ref, KA, KA), (ref, "", ref, "P2"))
    report = compare(rows, RARE, resamples=0)
    assert report["paired_lines"] == {"units": 4, "a_better": 2, "b_better": 1, "tie": 1, "sign_test_p": 1.0}
    # Halaman P1: A 2 edit, B 7 edit. Halaman P2: A 3 edit, B 0.
    assert report["paired_clusters"]["a_better"] == 1 and report["paired_clusters"]["b_better"] == 1
    assert report["clusters"] == 2 and report["cluster_sizes"] == {"min": 1, "median": 2.0, "max": 3}

    assert sign_test(0, 0) == 1.0 and sign_test(3, 3) == 1.0
    assert sign_test(5, 0) == sign_test(0, 5) == 2 / 2**5
    assert sign_test(1, 9) == pytest.approx(22 / 1024)
    assert 0 < sign_test(413, 157) < 1e-20


# --- recall dan presisi dari penjajaran --------------------------------------------------------------------


def test_matched_mask_marks_only_equal_positions():
    assert matched_mask("abc", "abc") == [True, True, True]
    assert matched_mask("abc", "axc") == [True, False, True]  # substitusi
    assert matched_mask("abc", "ac") == [True, False, True]  # hapus
    assert matched_mask("ac", "abc") == [True, True]  # sisipan tidak punya posisi di referensi
    assert matched_mask("abc", "") == [False, False, False] and matched_mask("", "abc") == []
    # Seri biaya (2 edit): aksara hilang + spasi tambahan. Opcodes rapidfuzz menjajarkannya sebagai dua substitusi,
    # jadi ꦪ yang terbaca benar ikut dihitung salah; penjajaran dengan kecocokan terbanyak tidak.
    assert matched_mask(DA + YA, YA + " ") == [False, True]
    assert matched_mask(KA + DA + YA + NA, KA + YA + " " + NA) == [True, False, True, True]


def best_counts(ref: str, hyp: str) -> tuple[int, int, int]:
    """(biaya, kecocokan, substitusi sekelas) penjajaran terbaik menurut urutan leksikografis (biaya minimum, kecocokan
    terbanyak, substitusi sekelas terbanyak), rekursi dari depan (pembanding)."""

    @lru_cache(maxsize=None)
    def best(i: int, j: int) -> tuple[int, int, int]:  # (biaya, -kecocokan, -sekelas) untuk ref[i:] vs hyp[j:]
        if i == len(ref) or j == len(hyp):
            return len(ref) - i + len(hyp) - j, 0, 0
        cost, neg, same = best(i + 1, j + 1)
        if ref[i] == hyp[j]:
            diag = (cost, neg - 1, same)
        else:
            diag = (cost + 1, neg, same - (glyph_class(ref[i]) == glyph_class(hyp[j])))
        cost, neg, same = best(i + 1, j)
        delete = (cost + 1, neg, same)
        cost, neg, same = best(i, j + 1)
        return min(diag, delete, (cost + 1, neg, same))

    cost, neg, same = best(0, 0)
    return cost, -neg, -same


def test_alignment_is_minimum_cost_with_most_matches_then_same_class_substitutions(monkeypatch):
    assert [glyph_class(ch) for ch in (" ", LAYAR, TALING, "꧀", KA, TWO, ADEG)] == [0, 1, 1, 1, 2, 2, 2]
    rng = random.Random(0)
    alphabet = [KA, NA, TA, " ", KA_MURDA, TWO, LAYAR, WULU]
    for _ in range(1000):
        ref = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 7)))
        hyp = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 7)))
        ops = alignment(ref, hyp)
        # Penjajaran utuh: setiap karakter dipakai sekali, berurutan; equal hanya untuk karakter yang sama.
        assert [i for tag, i, _ in ops if tag != "insert"] == list(range(len(ref)))
        assert [j for tag, _, j in ops if tag != "delete"] == list(range(len(hyp)))
        assert all((ref[i] == hyp[j]) == (tag == "equal") for tag, i, j in ops if tag in ("equal", "replace"))
        cost, matches, same_class = best_counts(ref, hyp)
        assert sum(tag != "equal" for tag, _, _ in ops) == cost == Levenshtein.distance(ref, hyp)
        assert sum(matched_mask(ref, hyp)) == matches
        assert matches >= sum(i2 - i1 for tag, i1, i2, _, _ in Levenshtein.opcodes(ref, hyp) if tag == "equal")
        assert sum(tag == "replace" and glyph_class(ref[i]) == glyph_class(hyp[j]) for tag, i, j in ops) == same_class

    # Penjajaran yang tidak berbiaya minimum menghentikan perbandingan alih-alih memberi recall yang salah.
    monkeypatch.setattr(compare_runs, "alignment", lambda ref, hyp: [("delete", i, None) for i in range(len(ref))]
                        + [("insert", None, j) for j in range(len(hyp))])
    with pytest.raises(CompareError, match="tidak berbiaya minimum"):
        compare(rows_of((KA + NA, KA + NA, KA + NA)), RARE, resamples=0)


def test_substitution_and_deletion_lower_recall_insertion_lowers_precision():
    with_murda, plain = KA + KA_MURDA + NA, KA + NA
    rows = rows_of(
        (with_murda, KA + KA + NA, with_murda),  # substitusi: tidak terbaca
        (with_murda, KA + NA, with_murda),  # hapus: tidak terbaca
        (with_murda, with_murda, with_murda),  # benar
        (plain, with_murda, plain),  # sisipan: tidak ada di referensi
    )
    report = compare(rows, RARE, resamples=0)
    group = report["rare"]["groups"]["langka"]
    assert group["ref"] == 3 and group["hits_a"] == 1 and group["hyp_a"] == 2
    assert group["recall"]["a"] == pytest.approx(1 / 3)  # sisipan tidak mengubah recall
    assert group["precision"]["a"] == pytest.approx(1 / 2)  # sisipan menurunkan presisi
    assert group["recall"]["b"] == 1.0 and group["precision"]["b"] == 1.0
    assert group["recall"]["diff"] == pytest.approx(-2 / 3)
    assert report["rare"]["groups"]["murda"]["recall"]["a"] == pytest.approx(1 / 3)

    (entry,) = report["rare"]["per_codepoint"]
    assert entry["codepoint"] == "U+A991" and entry["name"] == "JAVANESE LETTER KA MURDA" and entry["ref"] == 3
    assert entry["recall_a"] == pytest.approx(1 / 3) and entry["recall_b"] == 1.0
    assert entry["hyp_a"] == 2 and entry["precision_a"] == pytest.approx(1 / 2)


def test_codepoints_that_never_occur_are_not_reported_as_zero_percent():
    ref = KA + KA_MURDA + NA
    rows = rows_of((ref, KA + SA_MURDA + NA, KA + NA), (KA + NA, KA + NA, KA + NA))
    rare = compare(rows, RARE, resamples=0)["rare"]
    assert rare["codepoints"] == 4 and rare["in_reference"] == 1
    assert [entry["codepoint"] for entry in rare["per_codepoint"]] == ["U+A991"]
    assert rare["absent"] == ["U+A984", "U+A9CB"]  # tidak ada di referensi maupun keluaran: tanpa angka
    # Dibaca salah oleh keduanya: recall 0% yang sungguhan. Tidak pernah dikeluarkan: presisi tidak ada.
    assert rare["per_codepoint"][0]["recall_a"] == 0.0 and rare["per_codepoint"][0]["precision_a"] is None
    (hallucinated,) = rare["only_in_hypotheses"]
    assert hallucinated["codepoint"] == "U+A9AF" and hallucinated["recall_a"] is None
    assert hallucinated["precision_a"] == 0.0 and hallucinated["precision_b"] is None
    assert rare["groups"]["langka"]["precision"] == {
        "a": 0.0, "b": None, "diff": None, "bootstrap_lines": None, "bootstrap_clusters": None}

    no_rare = compare(rows_of((KA + NA, KA + NA, KA)), RARE, resamples=200)["rare"]
    assert no_rare["per_codepoint"] == [] and no_rare["in_reference"] == 0
    assert no_rare["groups"]["langka"]["recall"] == {
        "a": None, "b": None, "diff": None, "bootstrap_lines": None, "bootstrap_clusters": None}


def test_adeg_adeg_at_line_start_is_counted_per_line():
    rows = rows_of(
        (ADEG + KA + NA, ADEG + KA + NA, KA + NA),  # A membaca pembuka, B tidak
        (ADEG + KA, KA, KA),  # keduanya tidak
        (KA + ADEG + NA, KA + ADEG + NA, KA + ADEG + NA),  # di tengah baris: bukan pembuka
        (KA + NA, ADEG + KA + NA, KA + NA),  # A menulis pembuka yang tidak ada
    )
    report = compare(rows, RARE, resamples=0)
    opener = dict(report["line_initial_adeg_adeg"])
    recall, false_rate = opener.pop("recall"), opener.pop("false_start_rate")
    assert opener == {
        "codepoint": "U+A9CB", "lines": 2, "read_a": 1, "read_b": 0, "both": 0, "only_a": 1, "only_b": 0,
        "neither": 1, "sign_test_p": 1.0, "lines_without": 2, "false_start_a": 1, "false_start_b": 0}
    # Recall pembuka & pembuka palsu sebagai rasio per baris, dengan bootstrap seperti metrik lain.
    assert (recall["a"], recall["b"], recall["diff"]) == (0.5, 0.0, 0.5)
    assert (false_rate["a"], false_rate["b"], false_rate["diff"]) == (0.5, 0.0, 0.5)
    (entry,) = report["rare"]["per_codepoint"]
    assert entry["ref"] == 3 and entry["recall_a"] == pytest.approx(2 / 3) and entry["recall_b"] == pytest.approx(1 / 3)
    assert entry["precision_a"] == pytest.approx(2 / 3) and entry["precision_b"] == 1.0


def test_cer_on_lines_without_rare_characters():
    rows = rows_of(
        (KA + NA + TA, KA + NA, KA + NA + TA, "P1"),  # tanpa aksara langka: A 1 edit, B 0
        (KA + KA_MURDA, KA + KA, KA + KA_MURDA, "P1"),  # memuat aksara langka: tidak ikut
        (NA + TA, NA + TA, NA, "P2"),  # A 0, B 1
        (SA + KA + NA + TA, SA, SA + KA + NA + TA, "P3"),  # A 3, B 0
    )
    plain = compare(rows, RARE, resamples=300)["cer_without_rare"]
    assert plain["lines"] == 3 and plain["reference_chars"] == 9
    assert plain["a"] == pytest.approx(4 / 9) and plain["b"] == pytest.approx(1 / 9)
    assert plain["diff"] == pytest.approx(3 / 9)
    low, high = plain["bootstrap_clusters"]["ci95"]
    assert low <= 3 / 9 <= high

    # Beda hanya di baris beraksara langka: CER keseluruhan berbeda, CER tanpa aksara langka sama persis di setiap
    # resample (baris beraksara langka berbobot 0, bukan sekadar jarang terambil).
    rows = rows_of((KA + NA, KA + NA, KA + NA, "P1"), (KA + KA_MURDA, KA + KA, KA + KA_MURDA, "P1"), (TA, "", "", "P2"))
    report = compare(rows, RARE, resamples=300)
    assert report["cer"]["a"] == pytest.approx(2 / 5) and report["cer"]["b"] == pytest.approx(1 / 5)
    plain = report["cer_without_rare"]
    assert plain["lines"] == 2 and plain["reference_chars"] == 3
    assert plain["a"] == plain["b"] == pytest.approx(1 / 3) and plain["diff"] == 0
    assert plain["bootstrap_lines"]["ci95"] == plain["bootstrap_clusters"]["ci95"] == [0.0, 0.0]


def test_homoglyph_confusions_are_counted_separately():
    rows = rows_of(
        (KA + PA_MURDA + NA, KA + EIGHT + NA, KA + PA_MURDA + NA),  # A: pa murda -> angka delapan
        (KA + " " + ZERO + " " + NA, KA + " " + ZERO + " " + NA, KA + " " + WINDU + " " + NA),  # B: nol -> windu
        (KA + NA, KA + NA + LETTER_E, KA + NA),  # A: E padahal referensi tanpa E
        (NGA_LELET + KA, TWO + KA, NGA_LELET + KA + NGA_LELET),  # B: nga lelet lebih, referensinya memuat nga lelet
    )
    homoglyphs = {e["letter_char"]: e for e in compare(rows, RARE, resamples=0)["homoglyphs"]}
    assert list(homoglyphs) == [WINDU, LETTER_E, PA_MURDA, NGA_LELET]
    windu, e, pa_murda, nga_lelet = homoglyphs.values()
    assert windu["digit"] == "U+A9D0" and windu["letter_name"] == "JAVANESE PADA WINDU"
    assert (windu["ref_letter"], windu["ref_digit"]) == (0, 1)
    assert (windu["digit_as_letter_a"], windu["digit_as_letter_b"]) == (0, 1)
    assert (windu["false_letter_a"], windu["false_letter_b"], windu["false_letter_lines_b"]) == (0, 1, 1)
    assert (e["false_letter_a"], e["false_letter_lines_a"], e["false_letter_b"]) == (1, 1, 0)
    assert (pa_murda["ref_letter"], pa_murda["letter_as_digit_a"], pa_murda["letter_as_digit_b"]) == (1, 1, 0)
    assert (nga_lelet["letter_as_digit_a"], nga_lelet["letter_as_digit_b"]) == (1, 0)
    # Kelebihan nga lelet di baris yang memang memuatnya bukan "X palsu" (itu masuk presisi), dan tidak ada
    # substitusi aksara -> angka yang tercatat di arah sebaliknya.
    assert nga_lelet["false_letter_b"] == 0 and nga_lelet["digit_as_letter_b"] == 0


def test_homoglyph_next_to_extra_space_or_dropped_mark_is_still_counted():
    # Seri biaya dan kecocokan: angka bisa dipasangkan dengan aksaranya atau dengan spasi/sandhangan di sebelahnya.
    # Pola nyata (crnn_fonts): label tanpa spasi, keluaran menulis spasi sesudah glyph, atau sandhangan tidak terbaca.
    assert alignment(KA + NGA_LELET, KA + TWO + " ") == [("equal", 0, 0), ("replace", 1, 1), ("insert", None, 2)]
    assert alignment(PA_MURDA + LAYAR, EIGHT) == [("replace", 0, 0), ("delete", 1, None)]
    ne, ti = NA + TALING, TA + WULU
    rows = rows_of(
        (KA + NGA_LELET + NA, KA + TWO + " " + NA, KA + " " + TWO + NA),  # spasi sesudah / sebelum angka
        (ne + PA_MURDA + LAYAR + ti, ne + EIGHT + ti, ne + EIGHT + " " + ti),  # layar hilang / terbaca spasi
        (NA + LETTER_E + KA, NA + " " + SIX + " " + KA, NA + LETTER_E + KA),  # spasi di kedua sisi
    )
    report = compare(rows, RARE, resamples=0)
    homoglyphs = {e["letter_char"]: e for e in report["homoglyphs"]}
    assert (homoglyphs[NGA_LELET]["letter_as_digit_a"], homoglyphs[NGA_LELET]["letter_as_digit_b"]) == (1, 1)
    assert (homoglyphs[PA_MURDA]["letter_as_digit_a"], homoglyphs[PA_MURDA]["letter_as_digit_b"]) == (1, 1)
    assert (homoglyphs[LETTER_E]["letter_as_digit_a"], homoglyphs[LETTER_E]["letter_as_digit_b"]) == (1, 0)
    # CER tidak berubah oleh pemutus seri (compare memeriksa jumlah edit terhadap src.evaluate.summarize).
    assert report["cer"]["a"] == cer([r["reference"] for r in rows], [r["a"] for r in rows])


# --- bootstrap ---------------------------------------------------------------------------------------------


def test_bootstrap_is_deterministic_for_a_seed():
    stats = {"x": ([1, 0, 2, 3, 0, 1], [4] * 6, [0, 1, 1, 2, 1, 0], [4] * 6)}
    first = bootstrap_diffs(stats, 500, seed=7)
    assert first == bootstrap_diffs(stats, 500, seed=7)
    assert first != bootstrap_diffs(stats, 500, seed=8)
    low, high = first["x"]["ci95"]
    assert low < 7 / 24 - 5 / 24 < high and first["x"]["valid"] == 500
    assert bootstrap_diffs(stats, 0, seed=7) == {"x": None}

    rows = rows_of((KA + NA, KA, KA + NA, "P1"), (KA + NA + TA, KA + NA + TA, NA, "P2"), (KA, KA, "", "P3"))
    assert compare(rows, RARE, resamples=300, seed=1) == compare(rows, RARE, resamples=300, seed=1)


def test_identical_pipelines_have_a_zero_interval():
    rows = rows_of((KA + NA, KA, KA, "P1"), (KA + NA + TA, KA + NA + TA, KA + NA + TA, "P2"))
    report = compare(rows, RARE, resamples=200)
    assert report["cer"]["diff"] == 0
    assert report["cer"]["bootstrap_lines"]["ci95"] == [0.0, 0.0]
    assert report["cer"]["bootstrap_clusters"]["ci95"] == [0.0, 0.0]


def test_cluster_bootstrap_resamples_whole_pages():
    ref = KA + NA
    rows = rows_of(*[(ref, ref, "", "P1")] * 20, *[(ref, "", ref, "P2")] * 20)
    report = compare(rows, RARE, resamples=2000)
    assert report["clusters"] == 2 and report["cer"]["diff"] == 0
    low, high = report["cer"]["bootstrap_lines"]["ci95"]
    assert -0.6 < low < 0 < high < 0.6  # baris dianggap bebas: selang sempit
    # Dua halaman berlawanan: resample halaman bisa berisi satu halaman saja, jadi selangnya seluruh rentang.
    assert report["cer"]["bootstrap_clusters"]["ci95"] == [-1.0, 1.0]


# --- kontrak data ------------------------------------------------------------------------------------------


def test_line_missing_in_one_pipeline_is_a_clear_error():
    lines, preds = contract([(KA + NA, KA, KA + NA), (KA, KA, KA), (NA, NA, "")])
    incomplete = [p for p in preds if not (p["pipeline"] == "p_b" and p["line"] == "L1")]
    with pytest.raises(CompareError, match=r"'p_b' tidak punya 1 dari 3 baris \(mis\. L1\).*--common-only"):
        paired_lines(lines, incomplete, "p_a", "p_b")
    assert [row["id"] for row in paired_lines(lines, incomplete, "p_a", "p_b", common_only=True)] == ["L0", "L2"]
    with pytest.raises(CompareError, match=r"'p_x' tidak punya prediksi.*tersedia: p_a, p_b"):
        paired_lines(lines, preds, "p_a", "p_x")
    with pytest.raises(CompareError, match="prediksi ganda"):
        paired_lines(lines, preds + preds[:1], "p_a", "p_b")
    with pytest.raises(CompareError, match="tidak ada di lines.jsonl"):
        paired_lines(lines[:2], preds, "p_a", "p_b")


def test_paired_lines_normalizes_to_nfc_and_checks_stored_cer():
    decomposed, composed = "e" + chr(0x301), chr(0xE9)  # e + aksen gabung; bentuk NFC-nya satu codepoint
    lines = [{"id": "L0", "reference": decomposed + KA}]  # tanpa source_id: baris jadi klaster sendiri
    preds = [{"line": "L0", "pipeline": "p_a", "text": composed + KA, "cer": 0.0},
             {"line": "L0", "pipeline": "p_b", "text": decomposed, "cer": 0.5}]
    assert paired_lines(lines, preds, "p_a", "p_b") == [
        {"id": "L0", "source_id": "L0", "reference": composed + KA, "a": composed + KA, "b": composed}]
    preds[1]["cer"] = 0.25
    with pytest.raises(CompareError, match="CER tersimpan"):
        paired_lines(lines, preds, "p_a", "p_b")


def test_rare_set_is_cached_and_recomputed_when_its_inputs_change(tmp_path, monkeypatch):
    calls, params = [], {"train_lines": 10}

    def fake_training():
        calls.append(1)
        return [{"codepoint": "U+A991", "char": KA_MURDA, "name": "JAVANESE LETTER KA MURDA", "train_lines_with": 3}]

    monkeypatch.setattr(compare_runs, "rare_from_training", fake_training)
    monkeypatch.setattr(compare_runs, "rare_params", lambda: dict(params))
    cache = tmp_path / "compare" / "rare_codepoints.json"
    first = load_rare(cache, expected=1)
    assert first["count"] == 1 and first["codepoints"][0]["char"] == KA_MURDA
    assert load_rare(cache, expected=1) == first and len(calls) == 1  # dari cache
    params["train_lines"] = 20
    assert load_rare(cache, expected=1)["params"] == {"train_lines": 20} and len(calls) == 2
    load_rare(cache, refresh=True, expected=1)
    assert len(calls) == 3
    with pytest.raises(CompareError, match="1 codepoint.*mencatat 44"):
        load_rare(cache)


# --- skrip utuh --------------------------------------------------------------------------------------------


def test_main_writes_report_and_stops_on_manifest_mismatch(tmp_path, monkeypatch, capsys):
    items = [(ADEG + KA + KA_MURDA, ADEG + KA + KA_MURDA, KA + " " + KA, "P1"),
             (KA + NA + TA, KA + NA, KA + NA + TA, "P1"),
             (NA + SA_MURDA, NA + SA, NA + SA_MURDA, "P2"),
             (TA + SA, TA + SA, TA, "P3")]
    lines, preds = contract(items)
    refs = [line["reference"] for line in lines]
    metrics = []
    for key in ("p_a", "p_b"):
        hyps = [p["text"] for p in preds if p["pipeline"] == key]
        metrics.append({"scope": "uji_4", "pipeline": key, "lines": 4, "cer": cer(refs, hyps),
                        "cer_no_space": cer(refs, [h.replace(" ", "") for h in hyps])})
    manifest = {"schema": 1, "generated": "2026-01-01T00:00:00", "metrics": metrics,
                "pipelines": [{"key": "p_a", "label": "Pipeline A", "config": "uji a"},
                              {"key": "p_b", "label": "Pipeline B", "config": "uji b"}]}
    results, out = tmp_path / "results", tmp_path / "compare"
    results.mkdir()

    def dump(name, data):
        text = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else "".join(
            json.dumps(item, ensure_ascii=False) + "\n" for item in data)
        (results / name).write_text(text, encoding="utf-8")

    dump("lines.jsonl", lines)
    dump("predictions.jsonl", preds)
    dump("manifest.json", manifest)
    (tmp_path / "eval").mkdir()
    for metric, shift in zip(metrics, (0.0, 0.05)):  # laporan src.evaluate: A sama, B berbeda
        block = {"lines": 4, "cer": metric["cer"] + shift, "cer_hyp_no_space": metric["cer_no_space"]}
        (tmp_path / "eval" / f"{metric['pipeline']}_G3_full.json").write_text(
            json.dumps({"results": {"semua": block}}), encoding="utf-8")
    rare = {"params": {"train_lines": 100_000, "max_line_fraction": 0.001},
            "codepoints": [{"codepoint": f"U+{ord(ch):04X}", "char": ch} for ch in RARE]}
    monkeypatch.setattr(compare_runs, "load_rare", lambda cache, refresh=False: rare)
    argv = ["p_a", "p_b", "--results", str(results), "--out", str(out), "--bootstrap", "200"]

    compare_runs.main(argv)
    report = json.loads((out / "p_a_vs_p_b.json").read_text(encoding="utf-8"))
    assert report["a"] == {"key": "p_a", "label": "Pipeline A", "config": "uji a"}
    assert report["lines"] == 4 and report["clusters"] == 3 and report["subset"] == "semua baris"
    assert report["cer"]["a"] == metrics[0]["cer"] and report["cer"]["b"] == metrics[1]["cer"]
    assert report["cer_hyp_no_space"]["b"] == metrics[1]["cer_no_space"] < metrics[1]["cer"]
    assert report["checks"]["a"]["manifest"] == report["checks"]["b"]["manifest"] == {"scope": "uji_4", "match": True}
    assert report["checks"]["a"]["official"]["match"] is True and report["checks"]["b"]["official"]["match"] is False
    assert report["line_initial_adeg_adeg"]["lines"] == 1 and report["line_initial_adeg_adeg"]["only_a"] == 1
    assert len(report["cer"]["bootstrap_clusters"]["ci95"]) == 2
    # Baris 1 dan 3 tanpa aksara langka (5 karakter): A 1 edit, B 1 edit.
    assert report["cer_without_rare"]["lines"] == 2 and report["cer_without_rare"]["a"] == pytest.approx(1 / 5)
    assert [e["letter"] for e in report["homoglyphs"]] == ["U+A9C6", "U+A98C", "U+A9A6", "U+A98A"]
    markdown = (out / "p_a_vs_p_b.md").read_text(encoding="utf-8")
    assert "# Perbandingan pipeline: `p_a` (A) vs `p_b` (B)" in markdown
    assert "| U+A9AF ꦯ | LETTER SA MURDA | 1 | 0.00% | 100.00% | 0 | - | 1 | 100.00% |" in markdown
    assert "| CER, hanya 2 baris yang referensinya tanpa aksara langka (5 karakter; " in markdown
    windu_row = "| U+A9C6 ꧆ PADA WINDU ~ U+A9D0 ꧐ DIGIT ZERO | 0 / 0 | 0 / 0 | 0 / 0 | 0 (0 baris) | 0 (0 baris) |"
    assert windu_row in markdown
    assert "B 0 dari 3 baris." in markdown  # awalan adeg-adeg palsu dari baris yang referensinya tanpa adeg-adeg
    printed = capsys.readouterr().out
    assert "A = p_a" in printed and "adeg-adeg di awal baris: 1 baris; terbaca A 1, B 0" in printed
    assert "CER baris tanpa aksara langka (n=2)" in printed and "homoglif U+A9A6 ꦦ LETTER PA MURDA" in printed
    assert "PERINGATAN: 'p_b'" in printed and "PERINGATAN: 'p_a'" not in printed  # laporan resmi beda: tidak fatal

    with pytest.raises(SystemExit):  # A dan B harus berbeda
        compare_runs.main(["p_a", "p_a", "--results", str(results), "--out", str(out)])
    dump("predictions.jsonl", [p for p in preds if not (p["pipeline"] == "p_b" and p["line"] == "L3")])
    with pytest.raises(SystemExit, match="galat: .*'p_b' tidak punya 1 dari 4 baris"):
        compare_runs.main(argv)
    dump("predictions.jsonl", preds)
    manifest["metrics"][0]["cer"] += 0.01
    dump("manifest.json", manifest)
    with pytest.raises(SystemExit, match="galat: 'p_a': cer .* berbeda dari manifest.json"):
        compare_runs.main(argv)
