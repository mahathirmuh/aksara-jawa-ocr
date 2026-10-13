"""Test scripts/eval_spelling.py: hitungan bentuk ejaan baku per baris dan ringkasannya, tanpa model."""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))  # scripts/ bukan paket

import eval_spelling as es  # noqa: E402


def test_form_positions_find_the_five_standard_forms():
    text = "ꦥꦚ꧀ꦕꦶ ꦚꦮ ꦉꦒ ꦊꦩꦃ ꦠꦽꦱ꧀ꦤ ꧇꧗꧇ ꦥꦚ꧀ꦗꦶ ꦥꦚ꧀ꦠ"
    found = es.form_positions(text)
    assert sorted(found.values()) == ["cerek", "keret", "lelet", "nya", "nya", "pangkat", "pangkat"]
    # nya yang berdiri sendiri, atau di depan pangkon + aksara lain, bukan bentuk yang dihitung.
    assert [text[i] for i, form in found.items() if form == "nya"] == ["ꦚ", "ꦚ"] and text.count("ꦚ") == 4
    assert es.form_positions("ꦏꦸꦭ ꦫꦼꦒ") == {}


def test_line_counts_separate_recall_from_false_output():
    ref = "ꦉꦒ ꦊꦩꦃ ꦏꦸꦭ"
    # Terbaca: pa cerek benar, nga lelet menjadi la + pepet, dan satu keret palsu di akhir.
    row = es.line_counts(ref, "ꦉꦒ ꦭꦼꦩꦃ ꦏꦸꦭꦽ")
    assert (dict(row["ref"]), dict(row["hit"])) == ({"cerek": 1, "lelet": 1}, {"cerek": 1, "lelet": 0})
    assert (dict(row["out"]), dict(row["right"])) == ({"cerek": 1, "keret": 1}, {"cerek": 1, "keret": 0})
    assert (row["edits"], row["chars"]) == (3, len(ref))
    perfect = es.line_counts(ref, ref)
    assert perfect["hit"] == perfect["ref"] == perfect["out"] == perfect["right"] and perfect["edits"] == 0


def test_summary_and_page_bootstrap(tmp_path):
    rows_a = [es.line_counts("ꦉꦒ", "ꦉꦒ"), es.line_counts("ꦊꦩꦃ", "ꦭꦼꦩꦃ"), es.line_counts("ꦏꦸꦭ", "ꦏꦸꦭ")]
    rows_b = [es.line_counts("ꦉꦒ", "ꦫꦼꦒ"), es.line_counts("ꦊꦩꦃ", "ꦭꦼꦩꦃ"), es.line_counts("ꦏꦸꦭ", "ꦏꦸꦤ")]
    a, b = es.summarize(rows_a), es.summarize(rows_b)
    assert (a["in_reference"], a["read"], a["recall"], a["emitted"], a["precision"]) == (2, 1, 0.5, 1, 1.0)
    assert (b["read"], b["recall"], b["emitted"], b["precision"]) == (0, 0.0, 0, None)
    assert a["forms"]["cerek"] == {"in_reference": 1, "read": 1, "emitted": 1, "emitted_right": 1}
    assert (a["lines_without_forms"], a["cer_without_forms"], b["cer_without_forms"]) == (1, 0.0, 1 / 3)
    assert a["cer"] == 2 / 8 and b["cer"] == 5 / 8
    boot = es.page_bootstrap(["h1", "h1", "h2"], rows_a, rows_b, resamples=200, seed=1)
    assert boot["pages"] == 2 and boot["resamples"] == 200
    low, high = boot["recall"]
    assert 0.0 <= low <= high <= 1.0 and boot["cer"][1] <= 0  # A tidak pernah lebih buruk di sampel mana pun
    assert es.page_bootstrap(["h1", "h1", "h2"], rows_a, rows_b, resamples=200, seed=1) == boot  # deterministik

    # Dari berkas hasil: laporan tidak memuat teks label, dan pipeline tanpa prediksi lengkap ditolak.
    results = tmp_path / "results"
    results.mkdir()
    refs = ["ꦉꦒ", "ꦊꦩꦃ", "ꦏꦸꦭ"]
    (results / "lines.jsonl").write_text("".join(json.dumps({"id": f"{k}.png", "reference": ref, "source_id": f"h{k // 2}"}) + "\n"
                                                 for k, ref in enumerate(refs)), encoding="utf-8")
    preds = [("x", "ꦉꦒ"), ("x", "ꦭꦼꦩꦃ"), ("x", "ꦏꦸꦭ"), ("y", "ꦫꦼꦒ"), ("y", "ꦭꦼꦩꦃ"), ("y", "ꦏꦸꦤ")]
    (results / "predictions.jsonl").write_text("".join(
        json.dumps({"line": f"{k % 3}.png", "pipeline": key, "text": text}) + "\n" for k, (key, text) in enumerate(preds)), encoding="utf-8")
    report = es.real("x", "y", results, tmp_path / "out")
    assert (report["a"]["recall"], report["b"]["recall"], report["diff"]["recall"], report["lines"]) == (0.5, 0.0, 0.5, 3)
    written = (tmp_path / "out/spelling_x_vs_y.json").read_text(encoding="utf-8")
    assert json.loads(written)["a"]["key"] == "x" and not any(ref in written for ref in refs)
    import pytest

    with pytest.raises(SystemExit, match="pipeline z tidak punya prediksi untuk semua 3 baris"):
        es.real("x", "z", results, tmp_path / "out")
