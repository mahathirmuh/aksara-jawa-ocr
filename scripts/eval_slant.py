"""Titik akhir fase 9 (CLAUDE.md, "Aturan analisis fase 9" dan adendumnya): baris miring, dan baris cetak lawan tulisan
tangan.

Analisis galat fase8_std (2026-10-16): 270 dari 745 baris nyata (20 halaman, 32% karakter) miring, CER 28,3% lawan 15,7%
baris tegak, dan menyumbang 45% suntingan; render bersih yang dimiringkan 0,5 dibaca 15,4% (tegak 0,16%). Kemudian
ternyata baris miring itu semuanya reproduksi TULISAN TANGAN, bukan cetak miring: HANDWRITTEN_PAGES.

  real       dua pipeline di out/results: CER per kelompok kemiringan baris (tegak < 0,1, agak miring 0,1-0,3, miring
             >= 0,3; ditaksir dari citra dengan `slant`) dan per jenis tulisan (cetak, tulisan tangan; menurut halaman),
             selisih A - B dan SK 95% bootstrap per halaman.
               .venv/Scripts/python scripts/eval_slant.py real crnn_fase9_slant crnn_fase9_ctrl
  synthetic  baris val dirender bersih dengan javatext lalu dimiringkan 0 / 0,2 / 0,35 / 0,5 / 0,7 (0,7 di luar rentang
             latih 0-0,6), citra yang sama untuk semua checkpoint: CER per kemiringan. Tanpa NusaAksara.
               .venv/Scripts/python scripts/eval_slant.py synthetic fase9_slant fase9_ctrl fase8_std --lines 300

Hasil: out/compare/slant_<A>_vs_<B>.json dan out/compare/slant_synthetic.json. Teks label tidak ditulis.
"""

import argparse
import hashlib
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

# Geser yang dicoba penaksir; render yang dimiringkan 0,2 / 0,4 / 0,6 ditaksir persis (Noto, Tuladha).
SHEARS = np.round(np.arange(-0.3, 0.91, 0.05), 2)
GROUPS = (("tegak", -9.0, 0.1), ("agak miring", 0.1, 0.3), ("miring", 0.3, 9.0))
SYNTH_SHEARS = (0.0, 0.2, 0.35, 0.5, 0.7)
HELD_OUT_FONT = "C:/Windows/Fonts/javatext.ttf"
# Halaman NusaAksara (source_id "nusaaksara_hal_<N>") yang berisi reproduksi naskah tulisan tangan atau contoh aksara yang
# digambar tangan (87-89), bukan huruf cetak: klasifikasi visual 2026-10-16 dari satu sampai empat baris per halaman, belum
# diperiksa manusia (CLAUDE.md, adendum fase 9). Tulisan tangan di luar scope repo ini, tetapi G3 menghitungnya, jadi
# laporan memisahkan keduanya. Halaman lain = cetak.
HANDWRITTEN_PAGES = frozenset({1, 6, 8, 14, 29, 32, 43, 47, 52, 60, 62, 65, 71, 81, 86, 87, 88, 89, 90, 92})
KINDS = ("cetak", "tulisan tangan")


def ink(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("L"), dtype=np.float32)
    return np.abs(a - np.median(a)) > 60


def slant(mask: np.ndarray) -> float:
    """Kemiringan gaya Vinciarelli-Luettin: untuk tiap geser x' = x + s*(y - h/2), jumlah kuadrat panjang kolom yang
    tintanya bersambung tanpa celah; batang tegak yang lurus memberi kolom bersambung panjang. Positif = condong ke kanan
    seperti italic. Potongan bertekstur raster kadang tertaksir tegak (titik raster memutus kolom)."""
    h, _ = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return 0.0
    best, best_s = -1.0, 0.0
    for s in SHEARS:
        nx = np.round(xs + s * (ys - h / 2)).astype(int)
        nx -= nx.min()
        n = int(nx.max()) + 1
        count = np.bincount(nx, minlength=n)
        top = np.full(n, h, dtype=int)
        bot = np.full(n, -1, dtype=int)
        np.minimum.at(top, nx, ys)
        np.maximum.at(bot, nx, ys)
        whole = (count > 0) & (bot - top + 1 == count)
        score = float((count[whole].astype(np.float64) ** 2).sum())
        if score > best:
            best, best_s = score, float(s)
    return best_s


def group(value: float) -> str:
    return next(name for name, lo, hi in GROUPS if lo <= value < hi)


def kind(source_id: str) -> str:
    """Jenis tulisan baris nyata menurut halamannya: "tulisan tangan" untuk HANDWRITTEN_PAGES, selain itu "cetak"."""
    tail = str(source_id).rsplit("_", 1)[-1]
    return KINDS[1] if tail.isdigit() and int(tail) in HANDWRITTEN_PAGES else KINDS[0]


def page_bootstrap(rows: list[dict], resamples: int = 10_000, seed: int = 0) -> dict:
    """SK 95% (persentil) selisih CER A - B per kelompok kemiringan, per jenis tulisan, dan keseluruhan; yang diambil
    ulang adalah HALAMAN."""
    pages = defaultdict(list)
    for row in rows:
        pages[row["page"]].append(row)
    names = sorted(pages)
    rng = random.Random(seed)
    diffs = defaultdict(list)
    keys = [name for name, _, _ in GROUPS] + list(KINDS) + ["semua"]
    for _ in range(resamples):
        sample = [row for name in rng.choices(names, k=len(names)) for row in pages[name]]
        for key in keys:
            sel = [r for r in sample if key == "semua" or r["group"] == key or r.get("kind") == key]
            chars = sum(r["chars"] for r in sel)
            if chars:
                diffs[key].append((sum(r["a"] for r in sel) - sum(r["b"] for r in sel)) / chars)

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
        s = slant(ink(Image.open(ROOT / line["image"])))
        ref = line["reference"]
        rows.append({"page": line["source_id"], "slant": s, "group": group(s), "kind": kind(line["source_id"]),
                     "chars": len(ref), "a": Levenshtein.distance(ref, texts[a][line["id"]]),
                     "b": Levenshtein.distance(ref, texts[b][line["id"]])})

    def stats(sel: list[dict]) -> dict:
        chars = sum(r["chars"] for r in sel)
        return {"lines": len(sel), "pages": len({r["page"] for r in sel}), "chars": chars,
                "cer_a": sum(r["a"] for r in sel) / chars if chars else None,
                "cer_b": sum(r["b"] for r in sel) / chars if chars else None}

    summary = {key: stats([r for r in rows if key == "semua" or r["group"] == key])
               for key in [name for name, _, _ in GROUPS] + ["semua"]}
    kinds = {key: stats([r for r in rows if r["kind"] == key]) for key in KINDS}
    report = {"a": a, "b": b, "lines": len(lines), "groups": summary, "kinds": kinds,
              "thresholds": {name: [lo, hi] for name, lo, hi in GROUPS},
              "handwritten_pages": sorted(HANDWRITTEN_PAGES), "ci95_pages": page_bootstrap(rows, resamples)}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"slant_{a}_vs_{b}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for key, value in {**summary, **kinds}.items():
        if value["cer_a"] is None:
            print(f"{key:12s}   0 baris")
            continue
        low, high = report["ci95_pages"][key] or (float("nan"), float("nan"))
        print(f"{key:12s} {value['lines']:3d} baris {value['pages']:2d} halaman: {a} {value['cer_a']:.2%} lawan {b} "
              f"{value['cer_b']:.2%}, selisih {value['cer_a'] - value['cer_b']:+.4f} SK halaman [{low:+.4f}; {high:+.4f}]")
    print(f"ditulis: {path}")
    return report


def checkpoint_identity(path: Path) -> dict:
    import torch

    with open(path, "rb") as handle:
        sha = hashlib.file_digest(handle, "sha1").hexdigest()[:12]
    return {"step": torch.load(path, map_location="cpu", weights_only=False)["step"], "sha1": sha}


def synthetic(runs: list[str], lines: int, out_dir: Path, seed: int = 0) -> dict:
    from src.beam import greedy_ids, line_log_probs
    from src.dataset import oblique
    from src.infer import load_checkpoint
    from src.render import EmptyRender, RenderClipped, render_line
    from src.tokenizer import nfc

    pool = [nfc(x) for x in (ROOT / "data/splits/val.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    random.Random(seed).shuffle(pool)
    images = []
    for text in pool:
        if len(images) == lines:
            break
        try:
            images.append((text, render_line(text, HELD_OUT_FONT, 64).convert("L")))
        except (RenderClipped, EmptyRender):
            continue
    report = {"settings": {"split": "val", "lines": len(images), "font": Path(HELD_OUT_FONT).name, "size": 64, "seed": seed,
                           "shears": list(SYNTH_SHEARS)}, "checkpoints": {}, "results": {}}
    for run in runs:
        path = ROOT / "out/checkpoints" / run / "last_snapshot.pt"
        model, tok = load_checkpoint(str(path), "cpu")
        report["checkpoints"][run] = checkpoint_identity(path)
        report["results"][run] = {}
        for shear in SYNTH_SHEARS:
            edits = chars = 0
            for text, img in images:
                hyp = tok.to_logical(tok.decode(greedy_ids(line_log_probs(oblique(img, shear) if shear else img, model, "cpu", 0.0))))
                edits += Levenshtein.distance(text, hyp)
                chars += len(text)
            report["results"][run][str(shear)] = {"cer": edits / chars, "chars": chars}
        print(f"{run}: " + ", ".join(f"{s}: {report['results'][run][str(s)]['cer']:.2%}" for s in SYNTH_SHEARS), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "slant_synthetic.json"
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
