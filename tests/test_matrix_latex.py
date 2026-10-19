"""scripts/matrix_latex.py: tabel LaTeX makalah dari matrix.json. Semua test memakai matriks buatan, bukan data nyata."""

import copy
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))  # scripts/ bukan paket

import matrix_latex as ml  # noqa: E402

SUBSETS = {"cetak": {"lines": 10, "pages": 3}, "tulisan_tangan": {"lines": 5, "pages": 2},
           "semua": {"lines": 15, "pages": 5}}


def cell(cer, ned, name, delta=None, full=False):
    c = {"cer": cer, "ned": ned, "lines": SUBSETS[name]["lines"], "pages": SUBSETS[name]["pages"]}
    if delta is not None:
        c |= {"delta": delta, "ci95": [delta - 0.01, delta + 0.01], "ned_delta": delta,
              "ned_ci95": [delta - 0.01, delta + 0.01]}
    if full:
        c |= {"cer_no_space": cer, "exact": 0.1, "collapsed": 0.0, "non_javanese": 0.0}
    return c


def method(key, group, cer, ned, delta=None, all_lines=True, **extra):
    return {"key": key, "label": "x", "group": group, "offline": group != "vlm", "trained": group == "kita",
            "note": "", "params": 4_590_909 if group == "kita" else None, "seconds_per_line": 0.25,
            "cetak": cell(cer, ned, "cetak", delta, True),
            "tulisan_tangan": cell(cer, ned, "tulisan_tangan", delta) if all_lines else None,
            "semua": cell(cer, ned, "semua", delta) if all_lines else None, "partial": None, **extra}


def pair(group, a, b, status, d=-0.01):
    sub = {"cer_a": 0.1, "cer_b": 0.11, "delta": d, "ci95_pages": [d - 0.005, d + 0.005], "lines": 10, "pages": 3}
    return {"group": group, "label": "x", "a": a, "b": b, "status": status,
            "cetak": dict(sub), "tulisan_tangan": dict(sub), "semua": dict(sub)}


def stage(name, delta, ci95):
    return {"stage": name, "transformer": "t", "crnn": "c", "val_transformer": 0.004, "val_crnn": 0.002,
            "cetak": {"cer_t": 0.14, "cer_c": 0.12, "ned_t": 0.13, "ned_c": 0.12, "delta": delta, "ci95": ci95},
            "semua": {"cer_t": 0.22, "cer_c": 0.2}}


def fake() -> dict:
    return {
        "schema": 1, "generated": "2026-01-01 00:00:00", "reference": "crnn_fase9_ctrl", "prompt": "p",
        "prompt_sha": "abc123def456", "rules": "r", "subsets": copy.deepcopy(SUBSETS),
        "methods": [
            method("crnn_fase9_ctrl_beam", "kita", 0.11, 0.10, -0.015),
            method("crnn_fase9_ctrl", "kita", 0.123, 0.12),
            method("vlm_claude_opus55", "vlm", 0.84, 0.77, 0.65, all_lines=False, seconds_per_line=None),
            method("gpt_gpt-4o", "vlm", 1.12, 0.86, 0.74, output_tokens_per_line=87.1, usd_per_1000_lines=2.17,
                   model_served=["gpt-4o-2024-08-06"]),
            method("tesseract_jav", "umum", 1.09, 0.997, 0.87),
            {"key": "vlm_zeroshot", "label": "x", "group": "vlm", "offline": False, "trained": False, "note": "",
             "params": None, "seconds_per_line": None, "cetak": None, "tulisan_tangan": None, "semua": None,
             "partial": {"lines": 7, "cer": 0.9, "reference_cer": 0.2, "ned": 0.78, "reference_ned": 0.19}},
        ],
        "architecture": [{"key": "arch_crnn_base", "params": 4_590_909, "g1": 0.008, "g2": 0.54},
                         {"key": "arch_transformer", "params": 4_571_709, "g1": 0.06, "g2": 0.72},
                         {"key": "arch_crnn_full", "params": 4_590_909, "g1": 0.0027, "g2": 0.0074},
                         {"key": "archT_fase9_ctrl", "params": 4_571_709, "g1": 0.004, "g2": 0.0126}],
        "architecture_stages": [stage("base", -0.02, [-0.05, 0.01]), stage("fase9_ctrl", 0.0177, [0.011, 0.0235])],
        "ablation": [pair("data", "crnn_fonts", "crnn_core", "dipakai", -0.2),
                     pair("data", "crnn_fase9_slant", "crnn_fase9_ctrl", "tidak dipakai", 0.0075),
                     pair("baca", "crnn_fase9_ctrl_beam", "crnn_fase9_ctrl", "tersedia (Demo)", -0.0156),
                     pair("baca", "abl_beam_nolm", "crnn_fase9_ctrl", "pembanding", -0.0018),
                     pair("rujukan", "crnn_fase11_ctrl", "crnn_fase10_ctrl", "rujukan", -0.0004)],
        "augment": [{"k": 0, "op": "(kontrol, tanpa augmentasi)", "clean": 0.003, "heavy": 0.65, "g3": 0.75,
                     "g3_no_space": 0.7},
                    {"k": 1, "op": "rotate", "clean": 0.003, "heavy": 0.5, "g3": 0.80, "g3_no_space": 0.7},
                    {"k": 2, "op": "tight", "clean": 0.003, "heavy": 0.4, "g3": 0.60, "g3_no_space": 0.5}],
    }


FULL = ["crnn_fase9_ctrl_beam", "crnn_fase9_ctrl", "vlm_claude_opus55", "gpt_gpt-4o", "tesseract_jav"]


def rows(text: str) -> list[str]:
    return re.findall(r"% row: (\S+)$", text, re.M)


def pair_rows(text: str) -> list[str]:
    """Baris ablasi ditandai '% row: a - b'; dikembalikan kunci perlakuan a."""
    return [a for a, _ in re.findall(r"% row: (\S+) - (\S+)$", text, re.M)]


def test_tex_escapes_specials_and_rejects_non_ascii():
    assert ml.tex("a_b & c % d # e $") == r"a\_b \& c \% d \# e \$"
    assert ml.tex("{x}^~\\") == r"\{x\}\textasciicircum{}\textasciitilde{}\textbackslash{}"
    for bad in ("café", "ꦏ"):
        with pytest.raises(ValueError, match="non-ASCII"):
            ml.tex(bad)


def test_number_formats():
    assert ml.pct(0.1221715) == "12.22"
    assert ml.pct(1.1196) == "111.96"  # CER boleh di atas 100%
    assert ml.signed(-0.015323) == "$-1.53$"
    assert ml.signed(0.0111) == "$+1.11$"
    assert ml.signed(0.00004) == ml.signed(-0.00004) == "$0.00$"
    assert ml.ci([-0.0196, 0.0111]) == "[$-1.96$, $+1.11$]"
    assert ml.delta_ci(-0.0153, [-0.0196, -0.0111], "x") == "$-1.53$ points (95\\% CI [$-1.96$, $-1.11$])"
    assert ml.secs(0.2176) == "0.22" and ml.secs(117.47) == "117"
    with pytest.raises(ValueError):
        ml.ci([0.02, 0.01])
    with pytest.raises(ValueError):
        ml.pct(float("nan"))


def test_every_method_once_and_reference_marked():
    files = ml.build(fake(), readings=7)
    tab = files["tab_methods.tex"]
    assert rows(tab) == FULL  # urutan JSON, tiap kunci sekali
    assert rows(files["tab_practical.tex"]) == FULL
    for line in tab.splitlines():
        if line.endswith("% row: crnn_fase9_ctrl"):
            assert r"\textbf{CRNN, greedy (official)}" in line and r"\textit{reference}" in line
            assert line.count(r"\textbf{") >= 8
        elif "% row:" in line:
            assert "reference" not in line and r"\textbf" not in line and "official" not in line
    # Selisih dan SK dari JSON, NED pembanding dari JSON.
    gpt = next(line for line in tab.splitlines() if line.endswith("% row: gpt_gpt-4o"))
    assert "86.00 & $+74.00$ & {\\scriptsize[$+73.00$, $+75.00$]} & 112.00" in gpt
    claude = next(line for line in tab.splitlines() if line.endswith("% row: vlm_claude_opus55"))
    assert claude.count("& -- &") == 1 and claude.endswith("& -- & -- & 10.00 \\\\ % row: vlm_claude_opus55")
    fig = files["fig_methods.tex"]
    for v in ("(10.00,0) [10.00]", "(12.00,1) [12.00]", "(77.00,2) [77.00]", "(99.70,4) [99.70]"):
        assert v in fig
    assert "xmin=0, xmax=100" in fig and "NED on 10 printed lines" in fig


def test_partial_method_goes_to_note_not_row():
    files = ml.build(fake(), readings=7)
    tab = files["tab_methods.tex"]
    assert "vlm_zeroshot" not in rows(tab)
    notes = re.findall(r"% note: (\S+)$", tab, re.M)
    assert notes == ["vlm_zeroshot"]
    note = next(line for line in tab.splitlines() if line.endswith("% note: vlm_zeroshot"))
    assert "on 7 of the 15 lines" in note and "NED 78.00\\% (CER 90.00\\%)" in note and "19.00\\% (20.00\\%)" in note
    assert "vlm_zeroshot" not in files["tab_practical.tex"] and "blind test" not in files["fig_methods.tex"]


def test_unknown_keys_raise():
    m = fake()
    m["ablation"].append(pair("data", "crnn_new", "crnn_old", "dipakai"))
    with pytest.raises(ValueError, match="crnn_new"):
        ml.build(m)
    m = fake()
    m["ablation"][0]["status"] = "entah"
    with pytest.raises(ValueError, match="status"):
        ml.build(m)
    m = fake()
    m["methods"].append(method("gpt_gpt-9", "vlm", 0.9, 0.8, 0.7))
    with pytest.raises(ValueError, match="gpt_gpt-9"):
        ml.build(m)
    m = fake()
    m["architecture_stages"].append(stage("fase99", 0.01, [0.0, 0.02]))
    with pytest.raises(ValueError, match="fase99"):
        ml.build(m)
    m = fake()
    m["augment"].append({"k": 3, "op": "sharpen", "clean": 0.0, "heavy": 0.1, "g3": 0.5, "g3_no_space": 0.4})
    with pytest.raises(ValueError, match="sharpen"):
        ml.build(m)


@pytest.mark.parametrize("where", ["method", "negative", "partial", "stage", "subset"])
def test_ned_outside_0_100_raises(where):
    m = fake()
    if where == "method":
        m["methods"][2]["cetak"]["ned"] = 1.2
    elif where == "negative":
        m["methods"][0]["cetak"]["ned"] = -0.1
    elif where == "partial":
        m["methods"][5]["partial"]["reference_ned"] = 1.5
    elif where == "stage":
        m["architecture_stages"][0]["cetak"]["ned_t"] = 1.01
    else:
        m["methods"][4]["semua"]["ned"] = 2.0
    with pytest.raises(ValueError, match="NED"):
        ml.build(m)


def test_line_counts_must_match_subsets():
    m = fake()
    m["methods"][3]["cetak"]["lines"] = 9
    with pytest.raises(ValueError, match="gpt_gpt-4o"):
        ml.build(m)


def test_ablation_tables():
    files = ml.build(fake(), readings=7)
    dec = files["tab_decoding.tex"]
    assert pair_rows(dec) == ["abl_beam_nolm", "crnn_fase9_ctrl_beam"]  # urutan DECODE_ORDER: beam saja dulu
    assert "Beam + standard LM" in dec and "crnn_fase11_ctrl" not in dec
    data = files["tab_data_ablation.tex"]
    assert pair_rows(data) == ["crnn_fonts", "crnn_fase9_slant", "crnn_fase11_ctrl"]  # rujukan di blok terakhir
    fonts = next(line for line in data.splitlines() if "% row: crnn_fonts" in line)
    assert fonts.startswith("8 extra fonts & $-20.00$ & {\\scriptsize[$-20.50$, $-19.50$]}") and "& yes \\\\" in fonts
    assert "& no \\\\" in next(line for line in data.splitlines() if "% row: crnn_fase9_slant" in line)
    arch = files["tab_architecture.tex"]
    assert "Base & 0.40 & 0.20 & 13.00 & 12.00 & 14.00 & 12.00 & $-2.00$" in arch
    assert "4.57M parameters" in arch and "end of the chain T 0.40 / 1.26, B 0.27 / 0.74" in arch
    aug = files["tab_augment.tex"]
    assert "1 & rotation & 0.30 & 50.00 & 80.00 & $+5.00$" in aug and "tight crop$^\\ast$" in aug


def test_output_is_ascii_and_section_inputs_everything():
    files = ml.build(fake(), readings=7)
    for name, text in files.items():
        text.encode("ascii")
    section = files["matrix_section.tex"]
    assert section.count(r"\providecommand{\matrixdir}{matrix/}") == 1
    for name in ("tab_methods", "fig_methods", "tab_practical", "tab_decoding", "tab_data_ablation",
                 "tab_architecture", "tab_augment"):
        assert section.count(f"\\input{{\\matrixdir {name}}}") == 1
    assert r"\cite{sun2019lsvt}" in section and r"\bibitem{sun2019lsvt}" in files["matrix_bib.tex"]
    assert r"\label{eq:mx-ned}" in section and "at least 7 times" in section
    assert "\\newcommand{\\matrixdir}{}" in files["matrix_preview.tex"]
    assert "\\input{matrix_section}" in files["matrix_preview.tex"]


def test_main_writes_files_and_reads_card(tmp_path):
    (tmp_path / "m.json").write_text(json.dumps(fake()), encoding="utf-8")
    ml.main(["--matrix", str(tmp_path / "m.json"), "--out", str(tmp_path / "o"),
             "--datasets", str(tmp_path / "tidak_ada.json")])
    assert sorted(p.name for p in (tmp_path / "o").iterdir()) == sorted(ml.build(fake()))
    assert "evaluated repeatedly" in (tmp_path / "o" / "tab_methods.tex").read_text(encoding="ascii")
    card = {"usage": {"real": {"reports": 52, "others": [{}] * 6}}}
    (tmp_path / "card.json").write_text(json.dumps(card), encoding="utf-8")
    assert ml.load_readings(tmp_path / "card.json") == 58
