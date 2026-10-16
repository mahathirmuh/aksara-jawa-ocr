"""Test scripts/eval_slant.py: penaksir kemiringan, kelompok, dan laporan per kelompok, tanpa model."""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))  # scripts/ bukan paket

import eval_slant as es  # noqa: E402
from src.dataset import TRAIN_FONTS, oblique  # noqa: E402
from src.render import render_line  # noqa: E402

TEXT = "ꦲꦶꦁ ꦱꦶꦱꦶꦃ ꦭꦺꦴꦂ ꦮꦺꦠꦤ꧀ ꦏꦶꦢꦸꦭ꧀ ꦲꦤ ꦏꦭꦶ ꦧꦱ ꦗꦮ"


@pytest.mark.parametrize("font", TRAIN_FONTS)
def test_slant_estimate_recovers_a_known_shear(font):
    img = render_line(TEXT, font, 56).convert("L")
    for shear in (0.0, 0.2, 0.4, 0.6):
        assert es.slant(es.ink(oblique(img, shear) if shear else img)) == pytest.approx(shear, abs=0.051)
    assert es.slant(es.ink(Image.new("L", (50, 20), 255))) == 0.0  # tanpa tinta


def test_groups_follow_the_thresholds():
    assert [es.group(v) for v in (-0.3, 0.0, 0.05, 0.1, 0.25, 0.3, 0.9)] == [
        "tegak", "tegak", "tegak", "agak miring", "agak miring", "miring", "miring"]


def test_real_report_per_group_without_label_text(tmp_path, monkeypatch):
    # Tiga baris: tegak (citra Noto), miring (dimiringkan 0,5), di dua halaman. A lebih baik di baris miring.
    root = tmp_path
    (root / "img").mkdir()
    upright = render_line(TEXT, TRAIN_FONTS[0], 56).convert("L")
    italic = oblique(upright, 0.5)
    for name, img in (("a.png", upright), ("b.png", italic), ("c.png", italic)):
        img.save(root / "img" / name)
    monkeypatch.setattr(es, "ROOT", root)
    refs = {"a.png": "ꦏꦸꦭ", "b.png": "ꦧꦝꦺ", "c.png": "ꦠꦶꦤ"}
    results = root / "results"
    results.mkdir()
    (results / "lines.jsonl").write_text("".join(
        json.dumps({"id": k, "image": f"img/{k}", "reference": v, "source_id": "h1" if k != "c.png" else "h2"}) + "\n"
        for k, v in refs.items()), encoding="utf-8")
    preds = {"x": {"a.png": "ꦏꦸꦭ", "b.png": "ꦧꦝꦺ", "c.png": "ꦠꦶ"}, "y": {"a.png": "ꦏꦸ", "b.png": "ꦧ", "c.png": "ꦠ"}}
    (results / "predictions.jsonl").write_text("".join(
        json.dumps({"line": line, "pipeline": key, "text": text}) + "\n" for key, rows in preds.items() for line, text in rows.items()),
        encoding="utf-8")
    report = es.real("x", "y", results, tmp_path / "out", resamples=200)
    g = report["groups"]
    assert (g["tegak"]["lines"], g["miring"]["lines"], g["agak miring"]["lines"]) == (1, 2, 0)
    assert g["tegak"]["cer_a"] == 0 and g["tegak"]["cer_b"] == pytest.approx(1 / 3)
    assert g["miring"]["cer_a"] == pytest.approx(1 / 6) and g["miring"]["cer_b"] == pytest.approx(4 / 6)
    assert g["semua"]["chars"] == 9 and g["agak miring"]["cer_a"] is None
    low, high = report["ci95_pages"]["miring"]
    assert low <= high < 0 and report["ci95_pages"]["agak miring"] is None and report["ci95_pages"]["pages"] == 2
    written = (tmp_path / "out" / "slant_x_vs_y.json").read_text(encoding="utf-8")
    assert not any(ref in written for ref in refs.values())
    with pytest.raises(SystemExit, match="pipeline z tidak punya prediksi"):
        es.real("x", "z", results, tmp_path / "out")
