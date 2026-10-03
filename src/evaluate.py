"""Evaluasi CER sebuah checkpoint terhadap target PLAN.md §1.2.

  G1  python -m src.evaluate CKPT --lines 10000 --fonts C:/Windows/Fonts/javatext.ttf
  G2  python -m src.evaluate CKPT --lines 10000 --fonts C:/Windows/Fonts/javatext.ttf --augment heavy
  G3  python -m src.evaluate CKPT --real data/real/labels.tsv

Hasil ke out/eval/<nama>.json, termasuk kesalahan terburuk.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein
from torch.utils.data import DataLoader

from src.dataset import TRAIN_FONTS, LengthBucketSampler, SyntheticLines, collate
from src.decode import cer, greedy_decode
from src.infer import load_checkpoint, predict
from src.tokenizer import nfc

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits"
EVAL_DIR = ROOT / "out" / "eval"
TARGETS = {"G1": 0.02, "G2": 0.05, "G3": 0.08}


@torch.no_grad()
def evaluate_synthetic(model, tokenizer, lines, font, augment, batch_size, workers) -> tuple[list, int]:
    dataset = SyntheticLines(lines, tokenizer, [font], augment=augment, deterministic=True, seed=12345)
    loader = DataLoader(
        dataset,
        batch_sampler=LengthBucketSampler(lines, batch_size, shuffle=False),
        collate_fn=collate,
        num_workers=workers,
    )
    pairs = []
    for batch in loader:
        if batch is None:
            continue
        X, Y, in_lens, tgt_lens = batch
        # in_lens wajib: tanpa itu LSTM membaca padding batch dan hasilnya berbeda
        # dari inference satu baris.
        hyps = greedy_decode(model(X, in_lens), in_lens, tokenizer)
        refs = [tokenizer.to_logical(tokenizer.decode(y.tolist())) for y in torch.split(Y, tgt_lens.tolist())]
        pairs += zip(refs, hyps)
    return pairs, len(lines) - len(pairs)


def evaluate_real(model, tokenizer, labels_path: Path, pad_ratio: float = 0.0, limit: int = 0) -> dict[str, list]:
    groups: dict[str, list] = defaultdict(list)
    with labels_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if 0 < limit < len(rows):
        # Sampel acak tetap: baris awal labels.tsv hanya berasal dari beberapa halaman pertama.
        rows = random.Random(0).sample(rows, limit)
    for row in rows:
        image = Image.open(labels_path.parent / row["image_path"])
        pair = (nfc(row["text"]), predict(image, model, tokenizer, pad_ratio=pad_ratio))
        groups["semua"].append(pair)
        groups[f"kondisi:{row.get('condition', '?')}"].append(pair)
    return groups


def summarize(pairs) -> dict:
    worst = sorted(pairs, key=lambda p: Levenshtein.distance(*p) / max(1, len(p[0])), reverse=True)
    return {
        "lines": len(pairs),
        "cer": cer([r for r, _ in pairs], [h for _, h in pairs]),
        # Bukan angka gerbang: label NusaAksara umumnya ditulis tanpa spasi, sedangkan model
        # menulis spasi di celah antarkata. Dicatat terpisah supaya porsinya terlihat.
        "cer_hyp_no_space": cer([r for r, _ in pairs], [h.replace(" ", "") for _, h in pairs]),
        "exact_line_accuracy": sum(r == h for r, h in pairs) / max(1, len(pairs)),
        "worst": [{"ref": r, "hyp": h} for r, h in worst[:20] if r != h],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--split", default="test")
    parser.add_argument("--lines", type=int, default=10_000)
    parser.add_argument("--fonts", nargs="+", default=[str(p) for p in TRAIN_FONTS])
    parser.add_argument("--augment", default="none")
    parser.add_argument("--real", default="", help="labels.tsv citra nyata (G3)")
    parser.add_argument("--pad-ratio", type=float, default=0.0, help="margin putih x tinggi citra nyata")
    parser.add_argument("--name", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)

    model, tokenizer = load_checkpoint(args.checkpoint)
    report: dict = {"checkpoint": args.checkpoint}

    if args.real:
        goal = "G3"
        groups = evaluate_real(model, tokenizer, Path(args.real), args.pad_ratio, args.lines)
        report["results"] = {name: summarize(pairs) for name, pairs in groups.items()}
        report["pad_ratio"] = args.pad_ratio
        overall = report["results"]["semua"]["cer"]
    else:
        goal = "G2" if args.augment != "none" else "G1"
        augment = None
        if args.augment != "none":
            from src.augment import build_augment

            augment = build_augment(args.augment)
        lines = (SPLITS_DIR / f"{args.split}.txt").read_text(encoding="utf-8").splitlines()[: args.lines]
        all_pairs, results = [], {}
        for font in args.fonts:
            pairs, rejected = evaluate_synthetic(model, tokenizer, lines, font, augment, args.batch_size, args.workers)
            results[Path(font).name] = {**summarize(pairs), "rejected_by_min_frames": rejected}
            all_pairs += pairs
            print(f"{Path(font).name:40} CER {results[Path(font).name]['cer']:.4%} ({len(pairs)} baris, ditolak {rejected})")
        results["semua"] = summarize(all_pairs)
        report.update({"split": args.split, "augment": args.augment, "results": results})
        overall = results["semua"]["cer"]

    report.update({"goal": goal, "target": TARGETS[goal], "cer": overall, "passed": overall < TARGETS[goal]})
    name = args.name or f"{Path(args.checkpoint).parent.name}_{goal}"
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / f"{name}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    verdict = "LULUS" if report["passed"] else "BELUM"
    print(f"{goal}: CER {overall:.4%} (target < {TARGETS[goal]:.0%}) -> {verdict}; detail {out_path}")
    if args.real:
        no_space = report["results"]["semua"]["cer_hyp_no_space"]
        print(f"    info, bukan gerbang: CER bila spasi dibuang dari keluaran = {no_space:.4%}")


if __name__ == "__main__":
    main()
