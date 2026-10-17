"""Test scripts/eval_neighbor.py: tanda sisipan, pecahan tepi, potong rapat, laporan baris cetak, citra sintetis."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))  # scripts/ bukan paket

import eval_neighbor as en  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_inserted_counts_only_inserted_marks():
    # wulu dan cecak (atas) serta suku (bawah) disisipkan; ka yang disisipkan bukan tanda.
    assert en.inserted("ꦏꦭ", "ꦏꦶꦭꦸꦁ") == {"tanda atas": 2, "tanda bawah": 1}
    assert en.inserted("ꦏꦶ", "ꦏꦼ") == {"tanda atas": 0, "tanda bawah": 0}  # tertukar, bukan disisipkan
    assert en.inserted("ꦏ", "ꦏꦏ") == {"tanda atas": 0, "tanda bawah": 0}
    assert en.inserted("ꦏꦁ", "ꦏ") == {"tanda atas": 0, "tanda bawah": 0}  # hilang, bukan disisipkan
    assert en.TOP_MARKS.isdisjoint(en.BOTTOM_MARKS) and len(en.TOP_MARKS) == 6 and len(en.BOTTOM_MARKS) == 5


def test_edge_fragments_counts_short_components_touching_an_edge():
    mask = np.zeros((100, 200), dtype=bool)
    mask[30:70, 10:180] = True  # badan baris: tidak menyentuh tepi
    assert en.edge_fragments(mask) == 0
    mask[0:10, 5:8] = True  # dua pecahan di tepi atas
    mask[0:5, 100:110] = True
    mask[92:100, 50:60] = True  # satu di tepi bawah
    mask[85:88, 150:160] = True  # kecil tetapi tidak menyentuh tepi
    mask[60:100, 195:198] = True  # menyentuh tepi bawah tetapi tinggi (40% > 22%)
    assert en.edge_fragments(mask) == 3
    mask[0:21, 120:123] = True  # 21% dari 100: masih pecahan
    mask[0:23, 140:143] = True  # 23%: bukan
    assert en.edge_fragments(mask) == 4
    # Dua potong yang hanya bersentuhan sudut = satu komponen (8-tetangga), tingginya 30%: bukan pecahan.
    mask[0:10, 160:163] = True
    mask[10:30, 163:166] = True
    assert en.edge_fragments(mask) == 4


def test_tight_crops_to_the_ink_box():
    img = Image.new("L", (50, 30), 255)
    ImageDraw.Draw(img).rectangle((10, 5, 19, 14), fill=0)
    assert en.tight(img).size == (10, 10)
    blank = Image.new("L", (50, 30), 255)
    assert en.tight(blank).size == (50, 30)


def _line_image(fragments: int) -> Image.Image:
    img = Image.new("L", (200, 60), 255)
    draw = ImageDraw.Draw(img)
    draw.rectangle((10, 15, 189, 44), fill=0)
    for x in (20, 60)[:fragments]:  # pecahan pendek di tepi atas
        draw.rectangle((x, 0, x + 6, 4), fill=0)
    return img


def test_real_report_on_printed_lines_without_label_text(tmp_path, monkeypatch):
    root = tmp_path
    (root / "img").mkdir()
    for name, frag in (("a.png", 2), ("b.png", 0), ("c.png", 2), ("d.png", 1)):
        _line_image(frag).save(root / "img" / name)
    monkeypatch.setattr(en, "ROOT", root)
    refs = {"a.png": "ꦏꦭ", "b.png": "ꦠꦤ", "c.png": "ꦧꦝ", "d.png": "ꦩꦤ"}
    pages = {"a.png": "nusaaksara_hal_0", "b.png": "nusaaksara_hal_2", "c.png": "nusaaksara_hal_1",  # hal 1 = tulisan tangan
             "d.png": "nusaaksara_hal_3"}
    results = root / "results"
    results.mkdir()
    (results / "lines.jsonl").write_text("".join(
        json.dumps({"id": k, "image": f"img/{k}", "reference": v, "source_id": pages[k]}) + "\n"
        for k, v in refs.items()), encoding="utf-8")
    preds = {"x": {"a.png": "ꦏꦭ", "b.png": "ꦠꦤ", "c.png": "ꦧ", "d.png": "ꦩꦤ"},
             "y": {"a.png": "ꦏꦁꦭ", "b.png": "ꦠꦤꦸꦸ", "c.png": "", "d.png": "ꦩꦤ"}}
    (results / "predictions.jsonl").write_text("".join(
        json.dumps({"line": line, "pipeline": key, "text": text}) + "\n" for key, rows in preds.items() for line, text in rows.items()),
        encoding="utf-8")
    report = en.real("x", "y", results, tmp_path / "out", resamples=200)
    g = report["groups"]
    assert report["lines"] == 3 and report["kind"] == "cetak"  # baris tulisan tangan tidak dihitung
    assert (g["cetak"]["chars"], g["cetak"]["cer_a"], g["cetak"]["cer_b"]) == (6, 0, pytest.approx(3 / 6))
    # Berpecahan = paling sedikit dua komponen pendek di tepi (d.png hanya satu).
    assert (g["cetak berpecahan"]["lines"], g["cetak berpecahan"]["cer_b"]) == (1, pytest.approx(1 / 2))
    assert (g["cetak tanpa pecahan"]["lines"], g["cetak tanpa pecahan"]["pages"], g["cetak tanpa pecahan"]["cer_b"]) == (
        2, 2, pytest.approx(2 / 4))
    marks = report["inserted_marks"]
    assert (marks["tanda atas"]["total_a"], marks["tanda atas"]["total_b"], marks["tanda atas"]["b"]) == (0, 1, pytest.approx(1 / 3))
    assert (marks["tanda bawah"]["total_a"], marks["tanda bawah"]["total_b"], marks["tanda bawah"]["b"]) == (0, 2, pytest.approx(2 / 3))
    ci = report["ci95_pages"]
    assert ci["pages"] == 3 and ci["cetak berpecahan"] == [pytest.approx(-0.5)] * 2
    assert ci["cetak tanpa pecahan"] == [pytest.approx(-1.0), pytest.approx(0.0)]
    low, high = ci["cetak"]
    assert -1.0 <= low < high <= 0
    low, high = ci["sisipan tanda bawah per baris"]
    assert -2 <= low < high <= 0
    written = (tmp_path / "out" / "neighbor_x_vs_y.json").read_text(encoding="utf-8")
    assert not any(ref in written for ref in refs.values())
    with pytest.raises(SystemExit, match="pipeline z tidak punya prediksi"):
        en.real("x", "z", results, tmp_path / "out")


@pytest.mark.skipif(not (Path(en.HELD_OUT_FONT).exists() and (ROOT / "data/splits/val.txt").exists()),
                    reason="butuh javatext dan split val")
def test_synthetic_images_are_deterministic_and_add_fragments():
    first, second = en.synthetic_images(4), en.synthetic_images(4)
    assert len(first) == 4
    taller = 0
    for (text, variants), (text2, variants2) in zip(first, second):
        assert text == text2 and set(variants) == set(en.CONDITIONS)
        for name in en.CONDITIONS:
            assert variants[name].tobytes() == variants2[name].tobytes()
        assert variants["bersih rapat"].size == en.tight(variants["bersih"]).size
        assert variants["pecahan rapat"].height >= variants["bersih rapat"].height
        taller += variants["pecahan rapat"].height > variants["bersih rapat"].height
    assert taller >= 1
