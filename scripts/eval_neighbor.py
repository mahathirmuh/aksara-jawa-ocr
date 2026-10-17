"""Titik akhir fase 11 (CLAUDE.md, "Aturan analisis fase 11"): pecahan baris tetangga di tepi atas dan bawah potongan.

Potongan baris cetak nyata sering memuat sisa tinta baris di atas dan di bawahnya (lembar kontak 2026-10-17); render
sintetis tidak pernah memuatnya. Fase 11 melatih dengan pecahan itu (`src/neighbors.py`, `src.train --neighbor-prob`).

  real       dua pipeline di out/results, baris CETAK saja (`eval_slant.kind`): tanda atas dan tanda bawah yang
             DISISIPKAN keluaran (tidak ada di label; jalur sisip Levenshtein), per baris, dan CER baris yang tepi atas
             atau bawahnya memuat pecahan kecil (>= 2 komponen tinta yang menyentuh tepi itu dan lebih pendek dari 22%
             tinggi potongan; ukuran kasar yang ikut menghitung tanda baris itu sendiri) lawan baris lain; selisih A - B
             dan SK 95% bootstrap per halaman.
               .venv/Scripts/python scripts/eval_neighbor.py real crnn_fase11_neighbor crnn_fase11_ctrl
  synthetic  baris val dirender bersih dengan javatext, lalu tiga kondisi pada citra yang sama untuk semua checkpoint:
             bersih, bersih dipotong rapat, dan pecahan baris tetangga (`with_neighbors`, dua baris val lain, javatext,
             tanpa spasi; rentang pita dan jarak = rentang latih) yang dipotong rapat seperti potongan nyata. Tanpa
             NusaAksara.
               .venv/Scripts/python scripts/eval_neighbor.py synthetic fase11_neighbor fase11_ctrl fase9_ctrl --lines 300

Hasil: out/compare/neighbor_<A>_vs_<B>.json dan out/compare/neighbor_synthetic.json. Teks label tidak ditulis.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_slant import HELD_OUT_FONT, checkpoint_identity, ink, kind  # noqa: E402

TOP_MARKS = frozenset("ꦁꦂꦃꦶꦷꦼ")  # cecak, layar, wignyan, wulu, wulu melik, pepet
BOTTOM_MARKS = frozenset("ꦸꦹꦽꦾꦿ")  # suku, suku mendut, keret, pengkal, cakra
FRAGMENT_HEIGHT = 0.22  # komponen tepi yang lebih pendek dari ini (relatif tinggi potongan) dihitung pecahan
CONDITIONS = ("bersih", "bersih rapat", "pecahan rapat")


def inserted(ref: str, hyp: str) -> dict:
    """Jumlah tanda atas dan tanda bawah di keluaran yang tidak dipasangkan dengan karakter label (operasi sisip)."""
    counts = {"tanda atas": 0, "tanda bawah": 0}
    for op in Levenshtein.editops(ref, hyp):
        if op.tag == "insert":
            ch = hyp[op.dest_pos]
            if ch in TOP_MARKS:
                counts["tanda atas"] += 1
            elif ch in BOTTOM_MARKS:
                counts["tanda bawah"] += 1
    return counts


def edge_fragments(mask: np.ndarray) -> int:
    """Banyaknya komponen tinta (8-tetangga) yang menyentuh tepi atas atau bawah dan lebih pendek dari
    FRAGMENT_HEIGHT x tinggi citra."""
    from scipy import ndimage

    h = mask.shape[0]
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    found = 0
    for index, rows in enumerate(ndimage.find_objects(labels), start=1):
        if rows is None:
            continue
        top, bottom = rows[0].start, rows[0].stop
        if (top == 0 or bottom == h) and bottom - top < FRAGMENT_HEIGHT * h:
            found += 1
    return found


def tight(img: Image.Image) -> Image.Image:
    """Potong rapat ke kotak tinta (seperti potongan nyata yang tintanya menyentuh tepi)."""
    box = Image.eval(img.convert("L"), lambda v: 255 if v < 128 else 0).getbbox()
    return img.crop(box) if box else img


def page_bootstrap(rows: list[dict], keys: dict, resamples: int = 10_000, seed: int = 0) -> dict:
    """SK 95% (persentil) per kunci: `keys[k]` = (pilih baris, pembilang A, pembilang B, penyebut); yang diambil ulang
    adalah HALAMAN. Pembilang dan penyebut dijumlahkan di baris terpilih, selisih = (A - B) / penyebut."""
    pages = defaultdict(list)
    for row in rows:
        pages[row["page"]].append(row)
    names = sorted(pages)
    rng = random.Random(seed)
    diffs = defaultdict(list)
    for _ in range(resamples):
        sample = [row for name in rng.choices(names, k=len(names)) for row in pages[name]]
        for key, (pick, num_a, num_b, den) in keys.items():
            sel = [r for r in sample if pick(r)]
            total = sum(den(r) for r in sel)
            if total:
                diffs[key].append((sum(num_a(r) for r in sel) - sum(num_b(r) for r in sel)) / total)

    def interval(values):
        values = sorted(values)
        return [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]] if values else None

    return {key: interval(diffs[key]) for key in keys} | {"resamples": resamples, "seed": seed, "pages": len(names)}


def real(a: str, b: str, results: Path, out_dir: Path, resamples: int = 10_000) -> dict:
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
    rows = []
    for line in lines:
        if kind(line["source_id"]) != "cetak":
            continue
        ref = line["reference"]
        ins_a, ins_b = inserted(ref, texts[a][line["id"]]), inserted(ref, texts[b][line["id"]])
        rows.append({"page": line["source_id"], "chars": len(ref),
                     "fragments": edge_fragments(ink(Image.open(ROOT / line["image"]))) >= 2,
                     "a": Levenshtein.distance(ref, texts[a][line["id"]]),
                     "b": Levenshtein.distance(ref, texts[b][line["id"]]),
                     "top_a": ins_a["tanda atas"], "top_b": ins_b["tanda atas"],
                     "bottom_a": ins_a["tanda bawah"], "bottom_b": ins_b["tanda bawah"]})

    def cer(sel):
        chars = sum(r["chars"] for r in sel)
        return {"lines": len(sel), "pages": len({r["page"] for r in sel}), "chars": chars,
                "cer_a": sum(r["a"] for r in sel) / chars if chars else None,
                "cer_b": sum(r["b"] for r in sel) / chars if chars else None}

    groups = {"cetak": cer(rows), "cetak berpecahan": cer([r for r in rows if r["fragments"]]),
              "cetak tanpa pecahan": cer([r for r in rows if not r["fragments"]])}
    per_line = {mark: {"a": sum(r[f"{key}_a"] for r in rows) / len(rows), "b": sum(r[f"{key}_b"] for r in rows) / len(rows),
                       "total_a": sum(r[f"{key}_a"] for r in rows), "total_b": sum(r[f"{key}_b"] for r in rows)}
                for mark, key in (("tanda atas", "top"), ("tanda bawah", "bottom"))} if rows else {}
    keys = {
        "cetak": (lambda r: True, lambda r: r["a"], lambda r: r["b"], lambda r: r["chars"]),
        "cetak berpecahan": (lambda r: r["fragments"], lambda r: r["a"], lambda r: r["b"], lambda r: r["chars"]),
        "cetak tanpa pecahan": (lambda r: not r["fragments"], lambda r: r["a"], lambda r: r["b"], lambda r: r["chars"]),
        "sisipan tanda atas per baris": (lambda r: True, lambda r: r["top_a"], lambda r: r["top_b"], lambda r: 1),
        "sisipan tanda bawah per baris": (lambda r: True, lambda r: r["bottom_a"], lambda r: r["bottom_b"], lambda r: 1),
    }
    report = {"a": a, "b": b, "lines": len(rows), "kind": "cetak", "groups": groups, "inserted_marks": per_line,
              "fragment_rule": {"min_components": 2, "max_height": FRAGMENT_HEIGHT},
              "ci95_pages": page_bootstrap(rows, keys, resamples)}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"neighbor_{a}_vs_{b}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for key, value in groups.items():
        if value["cer_a"] is None:
            print(f"{key:22s}   0 baris")
            continue
        low, high = report["ci95_pages"][key] or (float("nan"), float("nan"))
        print(f"{key:22s} {value['lines']:3d} baris {value['pages']:2d} halaman: {a} {value['cer_a']:.2%} lawan {b} "
              f"{value['cer_b']:.2%}, selisih {value['cer_a'] - value['cer_b']:+.4f} SK halaman [{low:+.4f}; {high:+.4f}]")
    for mark, value in per_line.items():
        low, high = report["ci95_pages"][f"sisipan {mark} per baris"] or (float("nan"), float("nan"))
        print(f"sisipan {mark}: {value['total_a']} lawan {value['total_b']} ({value['a']:.3f} lawan {value['b']:.3f} per "
              f"baris), selisih SK halaman [{low:+.3f}; {high:+.3f}]")
    print(f"ditulis: {path}")
    return report


def synthetic_images(lines: int, seed: int = 0) -> list[tuple[str, dict]]:
    """(teks, {kondisi: citra}) untuk `lines` baris val acak yang bisa dirender dengan javatext; pecahan dari dua baris
    val lain (tanpa spasi), diundi dengan seed tetap supaya citra sama untuk semua checkpoint."""
    from src.neighbors import with_neighbors
    from src.render import EmptyRender, RenderClipped, render_line
    from src.tokenizer import nfc

    pool = [nfc(x) for x in (ROOT / "data/splits/val.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    random.Random(seed).shuffle(pool)
    rng = random.Random(seed + 1)
    out = []
    for text in pool:
        if len(out) == lines:
            break
        try:
            img = render_line(text, HELD_OUT_FONT, 64).convert("L")
            above, below = (render_line(pool[rng.randrange(len(pool))].replace(" ", ""), HELD_OUT_FONT, 64)
                            for _ in range(2))
        except (RenderClipped, EmptyRender):
            continue
        out.append((text, {"bersih": img, "bersih rapat": tight(img),
                           "pecahan rapat": tight(with_neighbors(img, above, below, rng))}))
    return out


def synthetic(runs: list[str], lines: int, out_dir: Path, seed: int = 0) -> dict:
    from src.beam import greedy_ids, line_log_probs
    from src.infer import load_checkpoint

    images = synthetic_images(lines, seed)
    report = {"settings": {"split": "val", "lines": len(images), "font": Path(HELD_OUT_FONT).name, "size": 64, "seed": seed,
                           "conditions": list(CONDITIONS)}, "checkpoints": {}, "results": {}}
    for run in runs:
        path = ROOT / "out/checkpoints" / run / "last_snapshot.pt"
        model, tok = load_checkpoint(str(path), "cpu")
        report["checkpoints"][run] = checkpoint_identity(path)
        report["results"][run] = {}
        for condition in CONDITIONS:
            edits = chars = 0
            marks = {"tanda atas": 0, "tanda bawah": 0}
            for text, variants in images:
                hyp = tok.to_logical(tok.decode(greedy_ids(line_log_probs(variants[condition], model, "cpu", 0.0))))
                edits += Levenshtein.distance(text, hyp)
                chars += len(text)
                for mark, count in inserted(text, hyp).items():
                    marks[mark] += count
            report["results"][run][condition] = {"cer": edits / chars, "chars": chars, "inserted_marks": marks}
        print(f"{run}: " + ", ".join(f"{c}: {report['results'][run][c]['cer']:.2%}" for c in CONDITIONS), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "neighbor_synthetic.json"
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
