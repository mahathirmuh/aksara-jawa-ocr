"""Titik akhir fase 8 (CLAUDE.md, "Aturan analisis fase 8"): bentuk ejaan baku yang nyaris tidak ada di korpus latih.

Lima bentuk: nya di depan ca/ja, pa cerek, nga lelet, keret, pada pangkat (src/spelling.py).

  real       dua pipeline di out/results (lines.jsonl + predictions.jsonl): recall dan presisi kelima bentuk, CER baris
             yang tidak memuat satu pun bentuk itu, selisih A - B dengan SK 95% bootstrap per halaman.
               .venv/Scripts/python scripts/eval_spelling.py real crnn_fase8_std crnn_fase8_ctrl
  synthetic  baris val yang kena aturan ejaan, dirender bersih dengan javatext dalam ejaan korpus DAN ejaan baku
             (citra yang sama untuk semua checkpoint): CER tiap ejaan dan recall bentuk baku. Tanpa NusaAksara.
               .venv/Scripts/python scripts/eval_spelling.py synthetic fase8_std fase8_ctrl fase7_track --lines 300

Hasil: out/compare/spelling_<A>_vs_<B>.json dan out/compare/spelling_synthetic.json. Teks label tidak ditulis.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.spelling import CA, JA, KERET, NGA_LELET, NYA, PA_CEREK, PADA_PANGKAT, PANGKON, RULES, standardize  # noqa: E402

FORMS = ("nya", "cerek", "lelet", "keret", "pangkat")
SINGLE = {PA_CEREK: "cerek", NGA_LELET: "lelet", KERET: "keret", PADA_PANGKAT: "pangkat"}
HELD_OUT_FONT = "C:/Windows/Fonts/javatext.ttf"


def form_positions(text: str) -> dict[int, str]:
    """Posisi karakter yang merupakan salah satu dari lima bentuk ejaan baku -> nama bentuknya."""
    found = {}
    for i, ch in enumerate(text):
        if ch in SINGLE:
            found[i] = SINGLE[ch]
        elif ch == NYA and text[i + 1: i + 2] == PANGKON and text[i + 2: i + 3] in (CA, JA):
            found[i] = "nya"
    return found


def line_counts(reference: str, hypothesis: str) -> dict:
    """Per baris: jumlah tiap bentuk di label, yang terbaca benar, yang dikeluarkan, dan keluaran yang benar; jarak
    sunting dan panjang label."""
    ref_ok, hyp_ok = [False] * len(reference), [False] * len(hypothesis)
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(reference, hypothesis):
        if tag == "equal":
            for k in range(i2 - i1):
                ref_ok[i1 + k] = hyp_ok[j1 + k] = True
    out = {"edits": Levenshtein.distance(reference, hypothesis), "chars": len(reference),
           "ref": Counter(), "hit": Counter(), "out": Counter(), "right": Counter()}
    for i, form in form_positions(reference).items():
        out["ref"][form] += 1
        out["hit"][form] += ref_ok[i]
    for j, form in form_positions(hypothesis).items():
        out["out"][form] += 1
        out["right"][form] += hyp_ok[j]
    return out


def summarize(rows: list[dict]) -> dict:
    """Angka satu pipeline dari daftar `line_counts`."""
    total = lambda key: sum((row[key] for row in rows), Counter())  # noqa: E731
    ref, hit, out, right = total("ref"), total("hit"), total("out"), total("right")
    ratio = lambda a, b: a / b if b else None  # noqa: E731
    plain = [row for row in rows if not row["ref"]]
    return {
        "recall": ratio(sum(hit.values()), sum(ref.values())), "precision": ratio(sum(right.values()), sum(out.values())),
        "in_reference": sum(ref.values()), "read": sum(hit.values()), "emitted": sum(out.values()), "emitted_right": sum(right.values()),
        "forms": {form: {"in_reference": ref[form], "read": hit[form], "emitted": out[form], "emitted_right": right[form]}
                  for form in FORMS},
        "cer": ratio(sum(row["edits"] for row in rows), sum(row["chars"] for row in rows)),
        "cer_without_forms": ratio(sum(row["edits"] for row in plain), sum(row["chars"] for row in plain)),
        "lines_without_forms": len(plain),
    }


def page_bootstrap(pages: list[str], a: list[dict], b: list[dict], resamples: int = 10_000, seed: int = 0) -> dict:
    """SK 95% (persentil) selisih A - B untuk recall bentuk baku, CER, dan CER baris tanpa bentuk baku; yang diambil
    ulang adalah HALAMAN (baris satu halaman saling mirip)."""
    by_page = defaultdict(list)
    for k, page in enumerate(pages):
        by_page[page].append(k)
    names = sorted(by_page)
    rng = random.Random(seed)
    diffs = defaultdict(list)
    for _ in range(resamples):
        idx = [k for name in rng.choices(names, k=len(names)) for k in by_page[name]]
        sa, sb = summarize([a[k] for k in idx]), summarize([b[k] for k in idx])
        for key in ("recall", "cer", "cer_without_forms"):
            if sa[key] is not None and sb[key] is not None:
                diffs[key].append(sa[key] - sb[key])

    def interval(values: list[float]) -> list[float] | None:
        if not values:
            return None
        values = sorted(values)
        return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]

    return {key: interval(values) for key, values in diffs.items()} | {"resamples": resamples, "seed": seed, "pages": len(names)}


def real(a: str, b: str, results: Path, out_dir: Path) -> dict:
    lines = [json.loads(x) for x in (results / "lines.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    texts: dict[str, dict[str, str]] = defaultdict(dict)
    for x in (results / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        if x.strip():
            row = json.loads(x)
            if row["pipeline"] in (a, b):
                texts[row["pipeline"]][row["line"]] = row["text"]
    missing = [key for key in (a, b) if len(texts[key]) != len(lines)]
    if missing:
        raise SystemExit(f"pipeline {', '.join(missing)} tidak punya prediksi untuk semua {len(lines)} baris di {results}")
    rows = {key: [line_counts(line["reference"], texts[key][line["id"]]) for line in lines] for key in (a, b)}
    sa, sb = summarize(rows[a]), summarize(rows[b])
    report = {
        "a": {"key": a, **sa}, "b": {"key": b, **sb}, "lines": len(lines),
        "diff": {key: (sa[key] - sb[key] if sa[key] is not None and sb[key] is not None else None)
                 for key in ("recall", "precision", "cer", "cer_without_forms")},
        "ci95_pages": page_bootstrap([line["source_id"] for line in lines], rows[a], rows[b]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"spelling_{a}_vs_{b}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    pct = lambda v: "-" if v is None else f"{v:.2%}"  # noqa: E731
    for side in (report["a"], report["b"]):
        print(f"{side['key']}: recall bentuk baku {side['read']}/{side['in_reference']} ({pct(side['recall'])}), keluaran "
              f"{side['emitted']} benar {side['emitted_right']} (presisi {pct(side['precision'])}), CER {pct(side['cer'])}, "
              f"CER {side['lines_without_forms']} baris tanpa bentuk baku {pct(side['cer_without_forms'])}")
        print("   ", {form: f"{v['read']}/{v['in_reference']}" for form, v in side["forms"].items()})
    ci = report["ci95_pages"]
    for key in ("recall", "cer", "cer_without_forms"):
        low, high = ci[key] or (None, None)
        print(f"selisih {key}: {report['diff'][key]:+.4f} SK halaman [{low:+.4f}; {high:+.4f}]")
    print(f"ditulis: {path}")
    return report


def synthetic(runs: list[str], lines: int, out_dir: Path, seed: int = 0) -> dict:
    from src.beam import greedy_ids, line_log_probs
    from src.infer import load_checkpoint
    from src.render import EmptyRender, RenderClipped, render_line
    from src.tokenizer import nfc

    pool = [nfc(x) for x in (ROOT / "data/splits/val.txt").read_text(encoding="utf-8").splitlines()]
    hit = [x for x in pool if standardize(x) != x]
    random.Random(seed).shuffle(hit)
    images = []
    for text in hit:
        if len(images) == lines:
            break
        try:
            images.append((text, render_line(text, HELD_OUT_FONT, 64), standardize(text), render_line(standardize(text), HELD_OUT_FONT, 64)))
        except (RenderClipped, EmptyRender):
            continue
    report = {"settings": {"split": "val", "lines": len(images), "font": Path(HELD_OUT_FONT).name, "size": 64, "seed": seed,
                           "rules": [name for name, *_ in RULES]}, "checkpoints": {}, "results": {}}
    for run in runs:
        path = ROOT / "out/checkpoints" / run / "last_snapshot.pt"
        model, tok = load_checkpoint(str(path), "cpu")
        read = lambda img: tok.to_logical(tok.decode(greedy_ids(line_log_probs(img, model, "cpu", 0.0))))  # noqa: E731
        corpus = [line_counts(text, read(img)) for text, img, _, _ in images]
        standard = [line_counts(text, read(img)) for _, _, text, img in images]
        report["results"][run] = {"corpus_spelling": summarize(corpus), "standard_spelling": summarize(standard)}
        c, s = report["results"][run]["corpus_spelling"], report["results"][run]["standard_spelling"]
        print(f"{run}: ejaan korpus CER {c['cer']:.2%} (bentuk baku palsu {c['emitted']}); ejaan baku CER {s['cer']:.2%}, "
              f"recall bentuk baku {s['read']}/{s['in_reference']} ({s['recall']:.1%})", flush=True)
        print("   ", {form: f"{v['read']}/{v['in_reference']}" for form, v in s["forms"].items()}, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "spelling_synthetic.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ditulis: {path}")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("real")
    r.add_argument("a")
    r.add_argument("b")
    r.add_argument("--results", default=str(ROOT / "out/results"))
    s = sub.add_parser("synthetic")
    s.add_argument("runs", nargs="+")
    s.add_argument("--lines", type=int, default=300)
    for p in (r, s):
        p.add_argument("--out", default=str(ROOT / "out/compare"))
    args = parser.parse_args(argv)
    if args.mode == "real":
        real(args.a, args.b, Path(args.results), Path(args.out))
    else:
        synthetic(args.runs, args.lines, Path(args.out))


if __name__ == "__main__":
    main()
