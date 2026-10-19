"""Tabel LaTeX untuk makalah (docs/paper) dari matriks metode out/results/matrix.json (scripts/method_matrix.py export,
skema 1).

  python scripts/matrix_latex.py [--matrix out/results/matrix.json] [--out docs/paper/matrix]

Menulis tab_methods.tex, tab_practical.tex, tab_decoding.tex, tab_data_ablation.tex, tab_architecture.tex,
tab_augment.tex, fig_methods.tex, matrix_bib.tex, matrix_section.tex, dan matrix_preview.tex (pratinjau mandiri).
Semua angka hasil dibaca dari matrix.json; satu-satunya angka lain = jumlah pembacaan data uji nyata selama pengembangan,
dari kartu data out/results/datasets.json (laporan G3 + pembacaan beam; batas bawah) bila berkasnya ada. Label Inggris
dipetakan per kunci metode, pasangan ablasi (a, b), tahap, dan operasi augmentasi; kunci yang tidak dikenal menghentikan
skrip supaya baris baru tidak diam-diam mendapat label yang salah. NED di luar 0-100% juga menghentikan skrip.
Keluaran hanya ASCII (aman untuk pdflatex) dan tidak memuat teks label NusaAksara (matrix.json hanya berisi agregat).

Di docs/paper/main.tex: \\input{matrix/matrix_section} sebelum \\section{Discussion and Limitations}, dan
\\input{matrix/matrix_bib} sebelum \\end{thebibliography}.
"""

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GROUPS = {"kita": "ours", "vlm": "VLM, zero-shot", "umum": "general OCR"}

# Kunci metode -> label tabel, label pendek (gambar dan tabel praktis), cara waktu baca diukur di
# scripts/method_matrix.py ("cpu" = subperintah latency lewat jalur layanan Demo, "process" = satu proses Tesseract per
# baris dengan beberapa pekerja paralel, "api" = waktu satu permintaan API, "none" = tidak diukur), kalimat protokol
# untuk pembanding, dan keterangan untuk catatan bila metode hanya punya hasil parsial.
METHODS = {
    "crnn_fase9_ctrl_beam": {"label": r"CRNN, beam + char.\ LM", "short": "CRNN, beam + LM", "timing": "cpu",
                             "prose": "beam search with the character language model",
                             "how": "with beam search and the language model"},
    "crnn_fase9_ctrl": {"label": "CRNN, greedy", "short": "CRNN, greedy", "timing": "cpu",
                        "prose": "greedy decoding", "how": "with greedy decoding"},
    "vlm_claude_opus55": {"label": "Claude Opus 5.5", "short": "Claude Opus 5.5", "timing": "none",
                          "protocol": "Claude Opus 5.5 read randomly renamed copies of the crops through parallel Claude "
                                      "Code subagents, some of which opened several images at once instead of one at a "
                                      "time."},
    "gpt_gpt-5.6-sol": {"label": "GPT-5.6 Sol", "short": "GPT-5.6 Sol", "timing": "api",
                        "protocol": "GPT-5.6 Sol was queried through the OpenAI Responses API with its default "
                                    "settings, under which it reasons before answering."},
    "gpt_gpt-4o": {"label": "GPT-4o", "short": "GPT-4o", "timing": "api",
                   "protocol": r"GPT-4o, which the NusaAksara authors also evaluated~\cite{nusaaksara}, was queried "
                               r"through the OpenAI Responses API with its default settings."},
    "tesseract_jav": {"label": r"Tesseract 5.4 (\texttt{jav})", "short": "Tesseract 5.4", "timing": "process",
                      "protocol": r"Tesseract~5.4 was run with the \texttt{jav} model and single-line page "
                                  r"segmentation. Its model collection has no model for Javanese script, and "
                                  r"\texttt{jav} is Javanese in Latin script, so this row shows what an off-the-shelf "
                                  r"engine returns, not a tuned baseline."},
    "vlm_zeroshot": {"label": "Claude, earlier blind test", "short": "Claude, blind test", "timing": "none",
                     "detail": "an earlier blind test of a Claude model with another prompt (model version not "
                               "recorded)"},
}
TIMING = {"cpu": "mean CPU time per line through the demo service, measured while the integrated GPU was training "
                 "another model",
          "process": "mean wall-clock time of one process per line, start-up included, with several processes in "
                     "parallel",
          "api": "mean wall-clock time of one API request, including the remote model's reasoning and the network",
          "none": "not timed"}

# Pasangan ablasi (perlakuan a, kontrol b) -> label tabel, nama dalam kalimat.
PAIRS = {
    ("crnn_fonts", "crnn_core"): ("8 extra fonts", "the eight extra fonts"),
    ("crnn_fase7_track", "crnn_fase7_ctrl"): ("Syllable tracking", "syllable tracking"),
    ("crnn_fase8_std", "crnn_fase8_ctrl"): ("Standard spelling", "standard spelling"),
    ("crnn_fase6_rare", "crnn_fase6_ctrl"): ("Rare v1", "rare-character insertion v1"),
    ("crnn_fase7_track_rare", "crnn_fase7_track"): ("Rare v2", "rare-character insertion v2"),
    ("crnn_fase10_rare", "crnn_fase10_ctrl"): ("Rare v2, rerun", "the rerun of v2 on the official model"),
    ("crnn_fase9_slant", "crnn_fase9_ctrl"): ("Italic rendering", "italic (slanted) rendering"),
    ("crnn_fase11_neighbor", "crnn_fase11_ctrl"): ("Neighbour lines", "fragments of neighbouring lines"),
    ("abl_beam_nolm", "crnn_fase9_ctrl"): ("Beam, no LM", "with beam search alone"),
    ("abl_beam_lm_lama", "crnn_fase9_ctrl"): ("Beam + corpus LM",
                                              "with the language model in the spelling of the corpus"),
    ("crnn_fase9_ctrl_beam", "crnn_fase9_ctrl"): ("Beam + standard LM",
                                                  "with the language model in standard spelling"),
    ("crnn_fase11_ctrl", "crnn_fase10_ctrl"): ("Same recipe again", "training the same recipe twice"),
    ("crnn_fase10_ctrl", "crnn_fase9_ctrl"): ("1,500 more steps", "1,500 more steps without treatment"),
}
# Urutan baris tabel cara membaca: komponen bertambah (beam saja, + LM ejaan korpus, + LM ejaan baku).
DECODE_ORDER = [("abl_beam_nolm", "crnn_fase9_ctrl"), ("abl_beam_lm_lama", "crnn_fase9_ctrl"),
                ("crnn_fase9_ctrl_beam", "crnn_fase9_ctrl")]
STATUS = {"dipakai": "yes", "tidak dipakai": "no", "rujukan": "--", "tersedia (Demo)": "demo", "pembanding": "no"}

# Tahap rantai model resmi (architecture_stages) -> label tabel, penjelasan untuk catatan.
STAGES = {
    "base": ("Base", "two core fonts, no degradations, trained from scratch"),
    "fase5_fonts": ("+ fonts", "scan-imitating degradations, then eight extra fonts"),
    "fase6_ctrl": ("+ steps", "more training with the same recipe"),
    "fase7_track": ("+ tracking", "syllable tracking"),
    "fase8_std": ("+ spelling", "standard spelling on half of the lines"),
    "fase9_ctrl": ("+ font fixes", "a font that the training environment rendered without shaping is dropped and one "
                                   "glyph conflict is avoided (official model)"),
}
# Baris architecture -> (resep, lengan): T = encoder Transformer, B = BiLSTM.
ARCH = {"arch_transformer": ("base", "T"), "arch_crnn_base": ("base", "B"),
        "archT_fase9_ctrl": ("full", "T"), "arch_crnn_full": ("full", "B")}
ARCH_GATE_LINES = 2000  # out/eval/<kunci>_G1_2k.json dan _G2_2k.json (scripts/method_matrix.py export)

# Operasi ablasi augmentasi kumulatif (src/augment.py ABLATION_STEPS) -> label, tiruan pindaian atau bukan.
AUG_OPS = {
    "(kontrol, tanpa augmentasi)": ("none (control)", False),
    "rotate": ("rotation", False), "noise": ("noise", False), "blur": ("blur", False),
    "elastic": ("elastic distortion", False), "contrast": ("contrast", False), "jpeg": ("JPEG compression", False),
    "texture": ("paper texture", False), "tight": ("tight crop", True), "stroke": ("stroke width", True),
    "binarize": ("binarisation", True), "speckle": ("speckle", True),
}
AUG_STEPS = 600  # langkah per run ablasi augmentasi (scripts/ablation.py --steps 600; CLAUDE.md, gerbang Fase 5)

NED_KEYS = {"ned", "reference_ned", "ned_t", "ned_c"}
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
         10: "ten", 11: "eleven", 12: "twelve"}

PREAMBLE = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{array}
\usepackage{url}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
"""
BIB_NUSAAKSARA = (r"\bibitem{nusaaksara} M. F. Adilazuarda \emph{et al.}, ``NusaAksara: A multimodal and multilingual "
                  r"benchmark for preserving Indonesian indigenous scripts,'' in \emph{Proc. 63rd Annu. Meeting Assoc. "
                  r"Comput. Linguistics (ACL)}, 2025.")
# Judul, penulis pertama, dan arXiv diperiksa 2026-10-19; nomor halaman prosiding belum, jadi tidak ditulis (isi dari
# IEEE Xplore sebelum paper dikirim).
BIB_SUN = (r"\bibitem{sun2019lsvt} Y. Sun \emph{et al.}, ``ICDAR 2019 competition on large-scale street view text with "
           r"partial labeling -- RRC-LSVT,'' in \emph{Proc. Int. Conf. Document Anal. Recognit. (ICDAR)}, 2019, "
           r"arXiv:1909.07741.")

SPECIAL = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
           "^": r"\textasciicircum{}", "_": r"\_", "%": r"\%", "~": r"\textasciitilde{}"}


# --- format dan pemeriksaan ---------------------------------------------------------------------------------------

def ascii_only(text: str, what: str = "keluaran") -> str:
    bad = sorted({f"U+{ord(c):04X}" for c in text if ord(c) > 127})
    if bad:
        raise ValueError(f"{what}: karakter non-ASCII {', '.join(bad)}")
    return text


def tex(text) -> str:
    """Teks biasa (dari JSON) -> LaTeX: karakter khusus di-escape, non-ASCII ditolak."""
    return ascii_only("".join(SPECIAL.get(c, c) for c in str(text)), repr(text))


def num(x, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{what}: bukan angka: {x!r}")
    return float(x)


def frac(x, what: str, upper: float | None = None) -> float:
    x = num(x, what)
    if x < 0:
        raise ValueError(f"{what}: pecahan negatif: {x!r}")
    if upper is not None and x > upper:
        raise ValueError(f"{what}: harus 0-{100 * upper:.0f}%, didapat {100 * x:.2f}%")
    return x


def pct(x, what: str = "nilai", upper: float | None = None) -> str:
    """Pecahan -> persen dua desimal, tanpa tanda %."""
    return f"{100 * frac(x, what, upper):.2f}"


def signed(x, what: str = "selisih") -> str:
    """Selisih pecahan -> poin persen bertanda dalam mode matematika, mis. $-1.53$."""
    s = f"{100 * num(x, what):+.2f}"
    if s in ("+0.00", "-0.00"):
        s = "0.00"
    return f"${s}$"


def ci(pair, what: str = "SK 95%") -> str:
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError(f"{what}: SK harus [bawah, atas], didapat {pair!r}")
    lo, hi = (num(v, what) for v in pair)
    if lo > hi:
        raise ValueError(f"{what}: batas bawah lebih besar dari batas atas: {pair!r}")
    return f"[{signed(lo, what)}, {signed(hi, what)}]"


def delta_ci(delta, pair, what: str) -> str:
    """Selisih bertanda + SK untuk kalimat: '$-1.53$ points (95\\% CI [$-1.96$, $-1.11$])'."""
    return f"{signed(delta, what)} points (95\\% CI {ci(pair, what)})"


def points(x, what: str = "selisih") -> str:
    """Besar selisih tanpa tanda, untuk kalimat yang sudah menyebut arahnya ('lowers ... by 8.10 points')."""
    return f"{100 * abs(num(x, what)):.2f}"


def size_ci(delta, pair, what: str) -> str:
    """Besar selisih tanpa tanda + SK bertanda, untuk kalimat yang menyebut arahnya ('raises ... by 0.75 points (...)')."""
    return f"{points(delta, what)} points (95\\% CI {ci(pair, what)})"


def secs(v) -> str:
    """Detik per baris dengan presisi yang masuk akal: 0.22, 12.3, 117."""
    v = frac(v, "seconds_per_line")
    return f"{v:.2f}" if v < 10 else (f"{v:.1f}" if v < 100 else f"{v:.0f}")


def capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def thousands(n: float) -> str:
    return f"{round(n):,}"


def words(n: int) -> str:
    return WORDS.get(n, str(n))


def join(items) -> str:
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def check_ned(obj, path: str = "matrix") -> None:
    """Setiap NED di mana pun di matriks harus 0-100% (NED berbatas; CER tidak)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in NED_KEYS and v is not None:
                frac(v, f"{path}.{k} (NED)", upper=1.0)
            check_ned(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_ned(v, f"{path}[{i}]")


def method_info(key: str) -> dict:
    if key not in METHODS:
        raise ValueError(f"kunci metode tidak dikenal: {key}; tambahkan label Inggris di METHODS "
                         f"(scripts/matrix_latex.py)")
    return METHODS[key]


def pair_info(pair: dict) -> tuple[str, str]:
    key = (pair["a"], pair["b"])
    if key not in PAIRS:
        raise ValueError(f"pasangan ablasi tidak dikenal: {key[0]} - {key[1]}; tambahkan label Inggris di PAIRS "
                         f"(scripts/matrix_latex.py)")
    if pair["status"] not in STATUS:
        raise ValueError(f"status ablasi tidak dikenal: {pair['status']!r} ({key[0]} - {key[1]})")
    return PAIRS[key]


def stage_info(stage: str) -> tuple[str, str]:
    if stage not in STAGES:
        raise ValueError(f"tahap arsitektur tidak dikenal: {stage}; tambahkan label Inggris di STAGES")
    return STAGES[stage]


def aug_info(op: str) -> tuple[str, bool]:
    if op not in AUG_OPS:
        raise ValueError(f"operasi augmentasi tidak dikenal: {op!r}; tambahkan label Inggris di AUG_OPS")
    return AUG_OPS[op]


def split_methods(m: dict) -> tuple[list[dict], list[dict]]:
    """Metode dengan hasil baris cetak (baris tabel) dan metode yang hanya punya hasil parsial (catatan)."""
    sub = m["subsets"]
    full, partial = [], []
    for x in m["methods"]:
        info = method_info(x["key"])
        if x.get("group") not in GROUPS:
            raise ValueError(f"metode {x['key']}: kelompok tidak dikenal {x.get('group')!r}")
        if x.get("cetak"):
            for name in ("cetak", "tulisan_tangan", "semua"):
                cell = x.get(name)
                if cell and (cell.get("lines"), cell.get("pages")) != (sub[name]["lines"], sub[name]["pages"]):
                    raise ValueError(f"metode {x['key']}: {name} {cell.get('lines')} baris / {cell.get('pages')} "
                                     f"halaman, subset {sub[name]['lines']} / {sub[name]['pages']}")
            full.append(x)
        elif x.get("partial"):
            if not info.get("detail"):
                raise ValueError(f"metode {x['key']} hanya punya hasil parsial: tambahkan 'detail' di METHODS")
            partial.append(x)
        else:
            raise ValueError(f"metode {x['key']}: tidak ada hasil baris cetak maupun parsial")
    if m["reference"] not in {x["key"] for x in full}:
        raise ValueError(f"pembanding {m['reference']} tidak punya hasil baris cetak")
    return full, partial


def check_matrix(m: dict) -> None:
    if m.get("schema") != 1:
        raise ValueError(f"skema matrix.json {m.get('schema')!r}, yang dikenal 1")
    keys = [x["key"] for x in m["methods"]]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise ValueError(f"kunci metode ganda: {', '.join(dup)}")
    check_ned(m)
    split_methods(m)
    for p in m.get("ablation") or []:
        pair_info(p)
    for s in m.get("architecture_stages") or []:
        stage_info(s["stage"])
    for a in m.get("architecture") or []:
        if a["key"] not in ARCH:
            raise ValueError(f"baris arsitektur tidak dikenal: {a['key']}; tambahkan di ARCH")
    for r in m.get("augment") or []:
        aug_info(r["op"])


def header(m: dict) -> str:
    return (f"% Generated by scripts/matrix_latex.py from out/results/matrix.json (generated {tex(m['generated'])}).\n"
            f"% Do not edit by hand: re-run the script after scripts/method_matrix.py export.\n")


def note_block(width: str, sentences: list[str]) -> list[str]:
    return [r"\par\smallskip", rf"\begin{{minipage}}{{{width}}}\footnotesize", *sentences, r"\end{minipage}"]


def bold_cells(cells: list[str]) -> list[str]:
    return [c if c.startswith(r"\multicolumn") else rf"\textbf{{{c}}}" for c in cells]


def readings_sentence(readings: int | None) -> str:
    how = f"at least {readings} times" if readings else "repeatedly"
    return (f"Each comparator read the test lines once, whereas the real lines were evaluated {how} during the "
            f"development of our model, so our figures are slightly optimistic.")


# --- tabel --------------------------------------------------------------------------------------------------------

def methods_table(m: dict, readings: int | None) -> str:
    full, partial = split_methods(m)
    ref, sub = m["reference"], m["subsets"]
    c, h, a = sub["cetak"], sub["tulisan_tangan"], sub["semua"]
    out = [r"\begin{table*}[t]",
           r"\caption{Comparison with other methods on the NusaAksara line crops (values in \%, lower is better)}",
           r"\label{tab:mx-methods}",
           r"\centering",
           r"\footnotesize",
           r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{@{}llrr@{\ }lrr@{\ }lrrr@{}}",
           r"\toprule",
           rf" & & \multicolumn{{6}}{{c}}{{Printed lines ({c['lines']})}} & Handwr. & All & Exact \\",
           r"\cmidrule(lr){3-8}\cmidrule(lr){9-9}\cmidrule(lr){10-10}\cmidrule(l){11-11}",
           r"Method & Type & NED & \multicolumn{2}{c}{$\Delta$NED [95\% CI]} & CER & "
           rf"\multicolumn{{2}}{{c}}{{$\Delta$CER [95\% CI]}} & ({h['lines']}) & ({a['lines']}) & ({c['lines']}) \\",
           r"\midrule"]
    prev = None
    for x in full:
        if prev is not None and x["group"] != prev:
            out.append(r"\midrule")
        prev = x["group"]
        key, cell = x["key"], x["cetak"]
        is_ref = key == ref
        label = method_info(key)["label"] + (" (official)" if is_ref else "")
        if is_ref:
            dn = dc = r"\multicolumn{2}{c}{\textbf{\textit{reference}}}"
        else:
            dn = signed(cell["ned_delta"], key) + r" & {\scriptsize" + ci(cell["ned_ci95"], key) + "}"
            dc = signed(cell["delta"], key) + r" & {\scriptsize" + ci(cell["ci95"], key) + "}"
        hw = pct(x["tulisan_tangan"]["cer"], key) if x.get("tulisan_tangan") else "--"
        al = pct(x["semua"]["cer"], key) if x.get("semua") else "--"
        cells = [label, GROUPS[x["group"]], pct(cell["ned"], key, 1.0), dn, pct(cell["cer"], key), dc, hw, al,
                 pct(cell["exact"], key, 1.0)]
        out.append(" & ".join(bold_cells(cells) if is_ref else cells) + rf" \\ % row: {key}")
    out += [r"\bottomrule", r"\end{tabular}"]
    notes = [r"NED $=\sum_i D(r_i,h_i)/\sum_i\max(|r_i|,|h_i|)$ is the corpus-level normalised edit distance between "
             r"reference $r_i$ and output $h_i$ of line $i$, bounded 0--100\%; CER divides the same edits by "
             r"$\sum_i|r_i|$ and has no upper bound, because insertions are not capped.",
             r"$\Delta$: method minus the official model (bold) on the same lines, in percentage points, with a 95\% "
             r"bootstrap CI over pages.",
             rf"Printed: {c['lines']} lines on {c['pages']} pages; Handwr.: CER on {h['lines']} handwritten lines on "
             rf"{h['pages']} pages; All: CER on all {a['lines']} lines ({a['pages']} pages); Exact: share of printed "
             r"lines read without error; --: lines not read. The print/handwriting split is a page-level visual "
             r"classification that has not been verified by a human."]
    if any(x["group"] == "vlm" for x in full):
        notes.append(rf"The VLMs received one fixed English prompt, written before any VLM read a test line "
                     rf"(SHA-256 prefix \texttt{{{tex(m['prompt_sha'])}}}).")
    notes.append(readings_sentence(readings))
    for x in partial:
        p = x["partial"]
        notes.append(rf"Not shown as a row: {method_info(x['key'])['detail']}, on {p['lines']} of the {a['lines']} "
                     rf"lines, gave NED {pct(p['ned'], x['key'], 1.0)}\% (CER {pct(p['cer'], x['key'])}\%), against "
                     rf"{pct(p['reference_ned'], x['key'], 1.0)}\% ({pct(p['reference_cer'], x['key'])}\%) for the "
                     rf"official model on the same lines. % note: {x['key']}")
    out += note_block(r"\textwidth", notes) + [r"\end{table*}"]
    return header(m) + "\n".join(out) + "\n"


def practical_table(m: dict) -> str:
    full, _ = split_methods(m)
    ref = m["reference"]

    def yesno(v):
        return "--" if v is None else ("yes" if v else "no")

    def opt(v, fmt):
        return "--" if v is None else fmt(v)

    out = [r"\begin{table}[t]",
           r"\caption{Practical properties of the compared methods}",
           r"\label{tab:mx-practical}",
           r"\centering",
           r"\footnotesize",
           r"\setlength{\tabcolsep}{2.5pt}",
           r"\begin{tabular}{@{}lccrrrr@{}}",
           r"\toprule",
           r"Method & Off. & Jav. & Params & s/line & Tok./line & USD/1k \\",
           r"\midrule"]
    prev = None
    timing: dict[str, list[str]] = {}
    served = []
    for x in full:
        if prev is not None and x["group"] != prev:
            out.append(r"\midrule")
        prev = x["group"]
        info = method_info(x["key"])
        is_ref = x["key"] == ref
        cells = [info["short"], yesno(x.get("offline")), yesno(x.get("trained")),
                 opt(x.get("params"), lambda v: f"{num(v, 'params') / 1e6:.2f}M"),
                 opt(x.get("seconds_per_line"), secs),
                 opt(x.get("output_tokens_per_line"), lambda v: thousands(frac(v, "output_tokens_per_line"))),
                 opt(x.get("usd_per_1000_lines"), lambda v: f"{frac(v, 'usd_per_1000_lines'):.2f}")]
        out.append(" & ".join(bold_cells(cells) if is_ref else cells) + rf" \\ % row: {x['key']}")
        how = info["timing"] if x.get("seconds_per_line") is not None else "none"
        who = "ours" if x["group"] == "kita" else info["short"]
        if who not in timing.setdefault(how, []):
            timing[how].append(who)
        if x.get("model_served"):
            served.append(f"{', '.join(tex(s) for s in x['model_served'])} ({info['short']})")
    out += [r"\bottomrule", r"\end{tabular}"]
    notes = [r"Bold: the official model. Off.: runs without a network connection. Jav.: trained on Javanese "
             r"script. Params: --, not disclosed or not recorded.",
             "s/line: " + "; ".join(f"{join(who)}, {TIMING[how]}" for how, who in timing.items()) + "."]
    if any(x.get("output_tokens_per_line") is not None for x in full):
        notes.append("Tok./line: output tokens per line as reported by the API, reasoning tokens included.")
    if any(x.get("usd_per_1000_lines") is not None for x in full):
        notes.append("USD/1k: cost of 1,000 lines from the token counts at the public list price, given only where a "
                     "price is recorded.")
    if served:
        notes.append("Model versions reported by the API: " + "; ".join(served) + ".")
    out += note_block(r"\columnwidth", notes) + [r"\end{table}"]
    return header(m) + "\n".join(out) + "\n"


def pairs_of(m: dict, group: str) -> list[dict]:
    rows = [p for p in m.get("ablation") or [] if p["group"] == group]
    for p in rows:
        pair_info(p)
    return rows


def decoding_table(m: dict) -> str:
    rows = pairs_of(m, "baca")
    order = {k: i for i, k in enumerate(DECODE_ORDER)}
    rows.sort(key=lambda p: order.get((p["a"], p["b"]), len(order)))
    sub = m["subsets"]
    out = [r"\begin{table}[t]",
           r"\caption{Decoding ablation on the official model: CER difference to greedy decoding (percentage points)}",
           r"\label{tab:mx-decoding}",
           r"\centering",
           r"\footnotesize",
           r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{@{}lccc@{}}",
           r"\toprule",
           r" & Printed & Handwr. & All \\",
           rf"Decoding & ({sub['cetak']['lines']}) & ({sub['tulisan_tangan']['lines']}) & ({sub['semua']['lines']}) \\",
           r"\midrule"]
    for p in rows:
        label, _ = pair_info(p)
        tag = f"{p['a']} - {p['b']}"
        cells = [p[k] for k in ("cetak", "tulisan_tangan", "semua")]
        out.append(" & ".join([label] + [signed(c["delta"], tag) for c in cells]) + rf" \\ % row: {tag}")
        out.append(" & ".join([""] + [rf"{{\scriptsize {ci(c['ci95_pages'], tag)}}}" for c in cells]) + r" \\[1pt]")
    out += [r"\bottomrule", r"\end{tabular}"]
    notes = [r"Beam: beam search of width 16. Corpus / standard LM: character 5-gram language model built from the "
             r"training text in the spelling of the corpus / with standard spelling on half of the lines; the LM "
             r"weights were tuned on synthetic development lines. Below each difference: 95\% bootstrap CI over "
             r"pages."]
    base = {(p["b"], k): p[k]["cer_b"] for p in rows for k in ("cetak", "tulisan_tangan", "semua")}
    bases = {p["b"] for p in rows}
    if len(bases) == 1:
        b = bases.pop()
        notes.append(rf"Greedy CER: printed {pct(base[(b, 'cetak')])}, handwritten "
                     rf"{pct(base[(b, 'tulisan_tangan')])}, all {pct(base[(b, 'semua')])}.")
    demo = [pair_info(p)[0] for p in rows if p["status"] == "tersedia (Demo)"]
    if demo:
        notes.append(f"{join(demo)}: available in the demo; the official figures use greedy decoding.")
    notes.append(r"The CIs come from a separate bootstrap run from Table~\ref{tab:mx-methods} and can differ from it in "
                 r"the last digit.")
    out += note_block(r"\columnwidth", notes) + [r"\end{table}"]
    return header(m) + "\n".join(out) + "\n"


def data_table(m: dict) -> str:
    sub = m["subsets"]
    out = [r"\begin{table}[t]",
           r"\caption{Training-data ablation: CER difference, treatment minus control (percentage points)}",
           r"\label{tab:mx-data}",
           r"\centering",
           r"\footnotesize",
           r"\setlength{\tabcolsep}{2.5pt}",
           r"\begin{tabular}{@{}lr@{\ }lrrc@{}}",
           r"\toprule",
           rf" & \multicolumn{{2}}{{c}}{{Printed ({sub['cetak']['lines']})}} & Handwr. & All & \\",
           r"\cmidrule(lr){2-3}",
           rf"Treatment & $\Delta$ & 95\% CI & ({sub['tulisan_tangan']['lines']}) & ({sub['semua']['lines']}) & Used \\",
           r"\midrule"]

    def row(p):
        label, _ = pair_info(p)
        tag = f"{p['a']} - {p['b']}"
        c = p["cetak"]
        cells = [label, signed(c["delta"], tag), rf"{{\scriptsize{ci(c['ci95_pages'], tag)}}}",
                 signed(p["tulisan_tangan"]["delta"], tag), signed(p["semua"]["delta"], tag), STATUS[p["status"]]]
        return " & ".join(cells) + rf" \\ % row: {tag}"

    out += [row(p) for p in pairs_of(m, "data")]
    refs = pairs_of(m, "rujukan")
    if refs:
        out += [r"\midrule", r"\multicolumn{6}{@{}l}{\textit{Reference pairs, no treatment}} \\"]
        out += [row(p) for p in refs]
    out += [r"\bottomrule", r"\end{tabular}"]
    notes = [r"Each treatment is compared with a control run that has the same starting checkpoint, number of steps "
             r"and learning-rate schedule; one run per arm. The 95\% CI is a bootstrap over pages (printed lines). "
             r"Rare: insertion of rare characters into the training text. Used: part of the recipe of the official "
             r"model."]
    if refs:
        notes.append(r"Reference pairs: the recipe of the control arm trained twice (only the random rendering "
                     r"differs), and 1,500 more steps without treatment; they show the size of run-to-run "
                     r"differences.")
    out += note_block(r"\columnwidth", notes) + [r"\end{table}"]
    return header(m) + "\n".join(out) + "\n"


def arch_rows(m: dict) -> dict:
    """architecture -> {(resep, lengan): baris}; parameter tiap lengan harus sama di semua resep."""
    found = {}
    for a in m.get("architecture") or []:
        if a["key"] not in ARCH:
            raise ValueError(f"baris arsitektur tidak dikenal: {a['key']}; tambahkan di ARCH")
        found[ARCH[a["key"]]] = a
    for arm in ("T", "B"):
        params = {r.get("params") for (recipe, a), r in found.items() if a == arm}
        if len(params) > 1:
            raise ValueError(f"jumlah parameter lengan {arm} berbeda antar resep: {sorted(params)}")
    return found


def params_of(rows: dict, arm: str) -> str | None:
    for (recipe, a), r in rows.items():
        if a == arm and r.get("params"):
            return f"{num(r['params'], 'params') / 1e6:.2f}M"
    return None


def architecture_table(m: dict) -> str:
    stages = m.get("architecture_stages") or []
    rows = arch_rows(m)
    out = [r"\begin{table}[t]",
           r"\caption{Transformer encoder (T) versus BiLSTM (B) on the same CNN, by training stage (\%)}",
           r"\label{tab:mx-arch}",
           r"\centering",
           r"\footnotesize",
           r"\setlength{\tabcolsep}{2.5pt}",
           r"\begin{tabular}{@{}lrrrrrrc@{}}",
           r"\toprule",
           r" & \multicolumn{2}{c}{Val CER} & \multicolumn{2}{c}{Print NED} & \multicolumn{2}{c}{Print CER} "
           r"& $\Delta$CER \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-8}",
           r"Stage & T & B & T & B & T & B & T$-$B \\",
           r"\midrule"]
    for s in stages:
        name, _ = stage_info(s["stage"])
        c = s["cetak"]
        tag = s["stage"]
        vals = [pct(s["val_transformer"], tag) if s.get("val_transformer") is not None else "--",
                pct(s["val_crnn"], tag) if s.get("val_crnn") is not None else "--",
                pct(c["ned_t"], tag, 1.0), pct(c["ned_c"], tag, 1.0), pct(c["cer_t"], tag), pct(c["cer_c"], tag)]
        out.append(" & ".join([name] + vals + [signed(c["delta"], tag)]) + rf" \\ % stage: {tag}")
        out.append(" & " * 7 + rf"{{\scriptsize {ci(c['ci95'], tag)}}} \\[1pt]")
    out += [r"\bottomrule", r"\end{tabular}"]
    pt, pb = params_of(rows, "T"), params_of(rows, "B")
    notes = [(f"T: Transformer encoder ({pt} parameters); B: BiLSTM, the official architecture ({pb}). " if pt and pb
              else "T: Transformer encoder; B: BiLSTM, the official architecture. ") +
             r"One run per architecture and stage; the Transformer chain repeats the recipe of each BiLSTM stage with "
             r"the current rendering code. Val: last clean synthetic validation CER of the training run. "
             rf"Print: the {m['subsets']['cetak']['lines']} printed real lines. $\Delta$: T minus B in percentage "
             r"points, 95\% bootstrap CI over pages below; positive means the BiLSTM is better.",
             "Stages, each continuing the previous one: " +
             "; ".join(f"{stage_info(s['stage'])[0]}: {stage_info(s['stage'])[1]}" for s in stages) + "."]
    gates = []
    for recipe, what in (("base", "base stage"), ("full", "end of the chain")):
        t, b = rows.get((recipe, "T")), rows.get((recipe, "B"))
        if t and b and None not in (t.get("g1"), t.get("g2"), b.get("g1"), b.get("g2")):
            gates.append(f"{what} T {pct(t['g1'])} / {pct(t['g2'])}, B {pct(b['g1'])} / {pct(b['g2'])}")
    if gates:
        notes.append(f"Synthetic gates G1 (clean) / G2 (heavy degradation) on {ARCH_GATE_LINES:,} test lines in the "
                     f"held-out font: " + "; ".join(gates) + ".")
    out += note_block(r"\columnwidth", notes) + [r"\end{table}"]
    return header(m) + "\n".join(out) + "\n"


def augment_rows(m: dict) -> list[dict]:
    rows = sorted(m.get("augment") or [], key=lambda r: r["k"])
    if [r["k"] for r in rows] != list(range(len(rows))):
        raise ValueError(f"langkah ablasi augmentasi tidak berurutan dari 0: {[r['k'] for r in rows]}")
    for r in rows:
        aug_info(r["op"])
    return rows


def augment_table(m: dict) -> str:
    rows = augment_rows(m)
    out = [r"\begin{table}[t]",
           r"\caption{Cumulative augmentation ablation (CER in \%)}",
           r"\label{tab:mx-augment}",
           r"\centering",
           r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{@{}rlrrrr@{}}",
           r"\toprule",
           rf" & & \multicolumn{{2}}{{c}}{{Synthetic val}} & \multicolumn{{2}}{{c}}{{G3, {m['subsets']['semua']['lines']} "
           r"real lines} \\",
           r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
           r"Step & Operation added & Clean & Heavy & CER & $\Delta$ \\",
           r"\midrule"]
    prev = None
    for r in rows:
        label, scan = aug_info(r["op"])
        tag = f"augment {r['k']}"
        d = "--" if prev is None else signed(num(r["g3"], tag) - num(prev, tag), tag)
        out.append(" & ".join([str(r["k"]), label + (r"$^\ast$" if scan else ""), pct(r["clean"], tag),
                               pct(r["heavy"], tag), pct(r["g3"], tag), d]) + rf" \\ % row: {tag}")
        prev = r["g3"]
    out += [r"\bottomrule", r"\end{tabular}"]
    notes = [rf"Each run starts from the base model and adds one operation to those of the previous run; "
             rf"{AUG_STEPS} training steps per run and one seed, so interactions and run-to-run variance are mixed "
             rf"into each step. $^\ast$Scan-imitating operation. Synthetic val: validation lines without (clean) and "
             rf"with heavy degradation. G3: all {m['subsets']['semua']['lines']} real lines, print and handwriting; "
             r"$\Delta$: change from the previous step in percentage points."]
    out += note_block(r"\columnwidth", notes) + [r"\end{table}"]
    return header(m) + "\n".join(out) + "\n"


def methods_figure(m: dict) -> str:
    full, _ = split_methods(m)
    ref = m["reference"]
    labels, ours, others = [], [], []
    for i, x in enumerate(full):
        info = method_info(x["key"])
        labels.append("{" + info["short"] + (" (official)" if x["key"] == ref else "") + "}")
        v = pct(x["cetak"]["ned"], x["key"], 1.0)
        (ours if x["group"] == "kita" else others).append(f"({v},{i}) [{v}]")
    out = [r"\begin{figure}[t]",
           r"\centering",
           r"\begin{tikzpicture}",
           r"\begin{axis}[",
           r"  xbar, bar shift=0pt, scale only axis, width=0.5\columnwidth,",
           rf"  height={0.5 * len(full) + 0.4:.1f}cm, bar width=6pt,",
           r"  xmin=0, xmax=100, xtick={0,20,40,60,80,100}, clip=false,",
           rf"  xlabel={{NED on {m['subsets']['cetak']['lines']} printed lines (\%)}},",
           rf"  ytick={{0,...,{len(full) - 1}}}, yticklabels={{{','.join(labels)}}},",
           r"  y dir=reverse, enlarge y limits={abs=0.5},",
           r"  axis x line*=bottom, axis y line*=left, xmajorgrids, major grid style={black!12},",
           r"  point meta=explicit symbolic, nodes near coords, nodes near coords align={horizontal},",
           r"  every node near coord/.append style={font=\scriptsize, anchor=west},",
           r"  tick label style={font=\footnotesize}, label style={font=\footnotesize}]"]
    if ours:
        out.append(r"\addplot[fill=black!75, draw=none] coordinates {" + " ".join(ours) + "};")
    if others:
        out.append(r"\addplot[fill=black!25, draw=none] coordinates {" + " ".join(others) + "};")
    out += [r"\end{axis}",
            r"\end{tikzpicture}",
            r"\caption{Corpus-level NED on the printed lines (Table~\ref{tab:mx-methods}). Dark: our model; light: "
            r"zero-shot VLMs and general OCR.}",
            r"\label{fig:mx-methods}",
            r"\end{figure}"]
    return header(m) + "\n".join(out) + "\n"


def bib_file(m: dict) -> str:
    return (header(m) + "% In main.tex: \\input{matrix/matrix_bib} just before \\end{thebibliography}.\n" + BIB_SUN + "\n")


# --- teks bagian --------------------------------------------------------------------------------------------------

def compare_text(m: dict, readings: int | None) -> list[str]:
    full, _ = split_methods(m)
    ref = m["reference"]
    sub = m["subsets"]
    c, h, a = sub["cetak"], sub["tulisan_tangan"], sub["semua"]
    by_key = {x["key"]: x for x in full}
    r = by_key[ref]["cetak"]
    comps = [x for x in full if x["group"] != "kita"]
    ours = [x for x in full if x["group"] == "kita" and x["key"] != ref]

    paras = [r"\subsection{Comparison with Other Methods}\label{sec:mx-compare}",
             r"We compared our recogniser with zero-shot vision-language models (VLMs) and an off-the-shelf OCR engine "
             r"on the same NusaAksara line crops. Every output was scored by one rule: it was normalised to NFC, only "
             r"trailing line breaks and tabs were removed (spaces count), and the Levenshtein distance $D$ to the "
             r"reference was computed in logical order without any other cleaning, so Latin letters or explanations "
             r"in an output count as errors. CER has no upper bound, because insertions are not capped and an empty "
             r"output already scores 100\%. The main scale is therefore the corpus-level normalised edit distance"
             "\n" r"\begin{equation}" "\n"
             r"\mathrm{NED}=\frac{\sum_i D(r_i,h_i)}{\sum_i \max(|r_i|,|h_i|)},\label{eq:mx-ned}" "\n"
             r"\end{equation}" "\n"
             r"where $r_i$ and $h_i$ are the reference and the output of line $i$. NED is pooled over lines in the "
             r"same way as corpus-level CER and lies between 0 and 100\%; the line-averaged variant of this "
             r"normalisation is used in the ICDAR 2019 robust reading competitions~\cite{sun2019lsvt}.",
             rf"A page-level visual inspection showed that {h['lines']} of the {a['lines']} lines, on {h['pages']} of "
             rf"the {a['pages']} pages, reproduce handwriting rather than print. This classification has not yet been "
             rf"verified by a human. Handwriting is outside the scope of this work, so the {c['lines']} printed lines "
             rf"({c['pages']} pages) are the main test subset, and the handwritten lines and the full set are reported "
             rf"beside them."]

    missing = [x["key"] for x in comps if not method_info(x["key"]).get("protocol")]
    if missing:
        raise ValueError(f"pembanding tanpa kalimat protokol di METHODS: {', '.join(missing)}")
    vlms = [x for x in comps if x["group"] == "vlm"]
    proto = [method_info(x["key"])["protocol"] for x in vlms]
    if vlms:
        proto.append(rf"All VLMs received the same English prompt, fixed before any VLM read a test line (SHA-256 "
                     rf"prefix \texttt{{{tex(m['prompt_sha'])}}}).")
    proto += [method_info(x["key"])["protocol"] for x in comps if x["group"] != "vlm"]
    print_only = [method_info(x["key"])["short"] for x in comps if not x.get("semua")]
    if print_only:
        proto.append(f"{join(print_only)} read only the printed lines.")
    proto.append(readings_sentence(readings))
    paras.append(" ".join(proto))

    res = [rf"Table~\ref{{tab:mx-methods}} and Fig.~\ref{{fig:mx-methods}} give the results. On the printed lines the "
           rf"official model reaches an NED of {pct(r['ned'], ref, 1.0)}\% (CER {pct(r['cer'], ref)}\%)."]
    for x in ours:
        cell = x["cetak"]
        verb = "lowers" if cell["ned_delta"] < 0 else "raises"
        res.append(rf"{capital(method_info(x['key'])['prose'])} {verb} it to {pct(cell['ned'], x['key'], 1.0)}\% "
                   rf"(CER {pct(cell['cer'], x['key'])}\%), a difference of "
                   rf"{delta_ci(cell['ned_delta'], cell['ned_ci95'], x['key'])}.")
    if comps:
        best = min(comps, key=lambda x: x["cetak"]["ned"])
        cell = best["cetak"]
        side = "above" if cell["ned_delta"] > 0 else "below"
        res.append(rf"The best comparator, {method_info(best['key'])['short']}, reaches "
                   rf"{pct(cell['ned'], best['key'], 1.0)}\%, {points(cell['ned_delta'])} points {side} the official "
                   rf"model (95\% CI of the difference {ci(cell['ned_ci95'], best['key'])}).")
        top = max(x["cetak"]["exact"] for x in comps)
        if top == 0:
            res.append(rf"No comparator reads a single printed line exactly, against {pct(r['exact'], ref, 1.0)}\% "
                       rf"of the lines for the official model.")
        else:
            res.append(rf"The comparators read at most {pct(top, 'exact', 1.0)}\% of the printed lines exactly, "
                       rf"against {pct(r['exact'], ref, 1.0)}\% for the official model.")
        over = [x for x in comps if x["cetak"]["cer"] > 1]
        if over:
            names = join(method_info(x["key"])["short"] for x in over)
            values = join(pct(x["cetak"]["cer"], x["key"]) + "\\%" for x in over)
            res.append(rf"The CER of {names} exceeds 100\% ({values}), whereas NED is bounded by construction.")
        foreign = [x for x in comps if x["cetak"].get("non_javanese")]
        if foreign:
            names = join(method_info(x["key"])["short"] for x in foreign)
            values = join(pct(x["cetak"]["non_javanese"], x["key"], 1.0) + "\\%" for x in foreign)
            res.append(f"Outputs with characters outside the Javanese block occur on {values} of the printed lines "
                       f"for {names}" + (", respectively." if len(foreign) > 1 else "."))
    paras.append(" ".join(res))
    paras += [r"\input{\matrixdir tab_methods}", r"\input{\matrixdir fig_methods}"]

    timed = [x for x in full if x.get("seconds_per_line") is not None]
    prac = [r"Table~\ref{tab:mx-practical} lists practical properties."]
    ours_t = [x for x in timed if x["group"] == "kita"]
    if ours_t:
        ours_t.sort(key=lambda x: x["key"] != ref)
        prac.append("On a CPU, our model reads a line in " +
                    join(f"{frac(x['seconds_per_line'], 's'):.2f}~s {method_info(x['key'])['how']}"
                         for x in ours_t) + ".")
    api = [x for x in timed if method_info(x["key"])["timing"] == "api"]
    if api:
        prac.append("Through the API, a request takes on average " +
                    join(f"{secs(x['seconds_per_line'])}~s for {method_info(x['key'])['short']}" for x in api) + ".")
    toks = [x for x in full if x.get("output_tokens_per_line") is not None]
    if toks:
        prac.append("The API models produce on average " +
                    join(f"{thousands(frac(x['output_tokens_per_line'], 'tokens'))}"
                         f"{' output tokens per line' if i == 0 else ''} for {method_info(x['key'])['short']}"
                         for i, x in enumerate(toks)) + ".")
    paras.append(" ".join(prac))
    paras.append(r"\input{\matrixdir tab_practical}")
    return paras


def ablation_text(m: dict) -> list[str]:
    paras = [r"\subsection{Architecture and Ablation}\label{sec:mx-ablation}"]

    def cell_text(p, k="cetak"):
        return delta_ci(p[k]["delta"], p[k]["ci95_pages"], f"{p['a']} - {p['b']}")

    dec = pairs_of(m, "baca")
    if dec:
        order = {k: i for i, k in enumerate(DECODE_ORDER)}
        dec.sort(key=lambda p: order.get((p["a"], p["b"]), len(order)))
        by = {(p["a"], p["b"]): p for p in dec}
        changes = join(f"by {cell_text(p)} {pair_info(p)[1]}" for p in dec)
        txt = [r"\textbf{Decoding.} Table~\ref{tab:mx-decoding} decodes the official model in other ways. On the "
               rf"printed lines, the CER changes {changes}."]
        nolm, lms = by.get(DECODE_ORDER[0]), [by[k] for k in DECODE_ORDER[1:] if k in by]
        if nolm and lms and all(p["cetak"]["delta"] < 0 for p in lms) and \
                abs(nolm["cetak"]["delta"]) < 0.5 * min(abs(p["cetak"]["delta"]) for p in lms):
            txt.append("Most of the gain therefore comes from the language model, not from the beam search itself.")
        paras.append(" ".join(txt))
        paras.append(r"\input{\matrixdir tab_decoding}")

    data = pairs_of(m, "data")
    if data:
        txt = [r"\textbf{Training data.} Table~\ref{tab:mx-data} repeats the ablation of the training data on the "
               r"split into printed and handwritten lines."]

        def named(ps):
            return join(f"{pair_info(p)[1]} ({signed(p['cetak']['delta'])})" for p in ps)

        better = [p for p in data if p["cetak"]["ci95_pages"][1] < 0]
        worse = [p for p in data if p["cetak"]["ci95_pages"][0] > 0]
        unclear = [p for p in data if p not in better and p not in worse]
        parts = ([f"below zero for {named(better)}"] if better else []) + \
                ([f"above zero for {named(worse)}"] if worse else []) + \
                ([f"includes zero for {named(unclear)}"] if unclear else [])
        if len(parts) > 1:
            parts[-1] = "and " + parts[-1]
        txt.append(r"On the printed lines, the 95\% CI of the difference lies " + "; ".join(parts) + ".")
        for p in data:
            hw = p["tulisan_tangan"]
            if p["cetak"]["ci95_pages"][0] > 0 and hw["ci95_pages"][1] < 0:
                txt.append(f"{capital(pair_info(p)[1])} lowers the CER of the handwritten lines by "
                           f"{points(hw['delta'])} points but raises that of the printed lines by "
                           f"{size_ci(p['cetak']['delta'], p['cetak']['ci95_pages'], p['a'])}" +
                           (", so it is not used." if p["status"] == "tidak dipakai" else "."))
        by = {(p["a"], p["b"]): p for p in data}
        v2 = by.get(("crnn_fase7_track_rare", "crnn_fase7_track"))
        rep = by.get(("crnn_fase10_rare", "crnn_fase10_ctrl"))
        if v2 and rep:
            txt.append(f"Rare-character insertion v2 changed the printed-line CER by {cell_text(v2)} in its first run "
                       f"and by {cell_text(rep)} when it was repeated on the official model.")
        noise = next((p for p in pairs_of(m, "rujukan") if (p["a"], p["b"]) == ("crnn_fase11_ctrl", "crnn_fase10_ctrl")),
                     None)
        if noise:
            txt.append(f"Training the same recipe twice moved the CER by {cell_text(noise)} on the printed lines and by "
                       f"{cell_text(noise, 'tulisan_tangan')} on the handwritten lines, one draw of run-to-run noise.")
        paras.append(" ".join(txt))
        paras.append(r"\input{\matrixdir tab_data_ablation}")

    stages = m.get("architecture_stages") or []
    if stages:
        rows = arch_rows(m)
        pt, pb = params_of(rows, "T"), params_of(rows, "B")
        txt = [r"\textbf{Architecture.} To test the recurrent layer, we replaced the two BiLSTM layers by a four-layer "
               r"pre-norm Transformer encoder (four heads, feed-forward width 768, sinusoidal positions, padding mask) "
               r"on the same CNN and projection" + (f", with {pt} parameters against {pb}" if pt and pb else "") +
               r", and trained it through the same chain of stages as the official model, one run per architecture "
               r"and stage (Table~\ref{tab:mx-arch})."]
        first, rest = stages[0], stages[1:]
        name = stage_info(first["stage"])[0].lower()
        diff = delta_ci(first["cetak"]["delta"], first["cetak"]["ci95"], first["stage"])
        lo, hi = first["cetak"]["ci95"]
        if lo <= 0 <= hi:
            txt.append(f"After the {name} stage the two cannot be distinguished on the printed lines: the difference "
                       f"is {diff}.")
        else:
            txt.append(f"After the {name} stage the difference on the printed lines is {diff}.")
        if rest:
            last = rest[-1]
            end = delta_ci(last["cetak"]["delta"], last["cetak"]["ci95"], last["stage"])
            if all(s["cetak"]["ci95"][0] > 0 for s in rest):
                ds = [s["cetak"]["delta"] for s in rest]
                txt.append(f"At every later stage the BiLSTM is better, by {points(min(ds))} to {points(max(ds))} "
                           f"points, and by {end} at the end of the chain.")
            else:
                txt.append(f"At the end of the chain the difference is {end}.")
        t, b = rows.get(("full", "T")), rows.get(("full", "B"))
        if t and b and None not in (t.get("g1"), t.get("g2"), b.get("g1"), b.get("g2")):
            txt.append(rf"On the synthetic gates the final models reach G1 {pct(t['g1'])}\% and G2 {pct(t['g2'])}\% "
                       rf"(Transformer) against {pct(b['g1'])}\% and {pct(b['g2'])}\% (BiLSTM). With this training "
                       r"budget the BiLSTM is the better choice; we did not test larger budgets, from which a "
                       r"Transformer may profit more.")
        paras.append(" ".join(txt))
        paras.append(r"\input{\matrixdir tab_architecture}")

    rows = augment_rows(m)
    if rows:
        scan = [aug_info(r["op"])[1] for r in rows]
        txt = [rf"\textbf{{Degradations.}} Table~\ref{{tab:mx-augment}} gives the full cumulative augmentation "
               rf"ablation ({AUG_STEPS} steps per run, one seed)."]
        first_scan = scan.index(True) if True in scan else 0
        if first_scan > 1 and all(scan[first_scan:]) and not any(scan[:first_scan]):
            g = [num(r["g3"], "g3") for r in rows]
            txt.append(rf"Together, the {words(first_scan - 1)} generic operations changed G3 by "
                       rf"{signed(g[first_scan - 1] - g[0])} points and the {words(len(rows) - first_scan)} "
                       rf"scan-imitating ones by {signed(g[-1] - g[first_scan - 1])}.")
        paras.append(" ".join(txt))
        paras.append(r"\input{\matrixdir tab_augment}")
    return paras


def section_file(m: dict, readings: int | None) -> str:
    paras = compare_text(m, readings) + ablation_text(m)
    return (header(m) + "% In main.tex: \\input{matrix/matrix_section} before \\section{Discussion and Limitations}.\n"
            + "\\providecommand{\\matrixdir}{matrix/}\n\n" + "\n\n".join(paras) + "\n")


def preview_file(m: dict) -> str:
    return (header(m) + PREAMBLE + "\\newcommand{\\matrixdir}{}\n\\begin{document}\n"
            "\\title{Comparison Matrix (Preview)}\n\\maketitle\n\\section{Results}\n\\input{matrix_section}\n\n"
            "\\begin{thebibliography}{00}\n" + BIB_NUSAAKSARA + "\n\\input{matrix_bib}\n\\end{thebibliography}\n"
            "\\end{document}\n")


def build(m: dict, readings: int | None = None) -> dict[str, str]:
    check_matrix(m)
    files = {"tab_methods.tex": methods_table(m, readings), "tab_practical.tex": practical_table(m),
             "tab_decoding.tex": decoding_table(m), "tab_data_ablation.tex": data_table(m),
             "tab_architecture.tex": architecture_table(m), "tab_augment.tex": augment_table(m),
             "fig_methods.tex": methods_figure(m), "matrix_bib.tex": bib_file(m),
             "matrix_section.tex": section_file(m, readings), "matrix_preview.tex": preview_file(m)}
    for name, text in files.items():
        ascii_only(text, name)
    return files


def load_readings(path: Path) -> int | None:
    """Jumlah pembacaan data uji nyata selama pengembangan menurut kartu data (scripts/export_datasets.py): laporan G3
    di out/eval + pembacaan beam di out/beam. Batas bawah; None bila kartu tidak ada."""
    if not path.exists():
        return None
    real = json.loads(path.read_text(encoding="utf-8")).get("usage", {}).get("real", {})
    reports, others = real.get("reports"), real.get("others")
    if not isinstance(reports, int) or not isinstance(others, list):
        raise ValueError(f"{path}: usage.real.reports / usage.real.others tidak terbaca")
    return reports + len(others)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matrix", default=str(ROOT / "out/results/matrix.json"))
    parser.add_argument("--out", default=str(ROOT / "docs/paper/matrix"))
    parser.add_argument("--datasets", default=str(ROOT / "out/results/datasets.json"),
                        help="kartu data untuk jumlah pembacaan data uji nyata (dilewati bila tidak ada)")
    args = parser.parse_args(argv)
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    files = build(matrix, load_readings(Path(args.datasets)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_bytes(text.encode("ascii"))
    print(f"ditulis ke {out}: {', '.join(files)}")


if __name__ == "__main__":
    main()
