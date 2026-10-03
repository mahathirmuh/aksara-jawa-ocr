"""Beam search + LM karakter vs greedy pada keluaran CRNN yang sama (kondisi "kamus", ablasi koreksi).

Protokol (CLAUDE.md): alpha/beta disetel HANYA di dev set = split val yang dirender dengan font training
dan augmentasi `fase5`, lalu dibaca CRNN. NusaAksara (test) dievaluasi SEKALI dengan setelan terpilih.
Render bersih split test dengan font held-out ikut diukur untuk memastikan G1 tidak memburuk.

  .venv/Scripts/python scripts/beam_eval.py out/checkpoints/fase5_fonts/last_snapshot.pt \\
      --lm data/charlm/o5_n300000_ds0.5.pkl --name fonts_o5

Hasil: out/beam/<name>.json.
"""

import argparse
import csv
import itertools
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment import build_augment  # noqa: E402
from src.beam import greedy_ids, line_log_probs, prefix_beam_search  # noqa: E402
from src.charlm import CharLM  # noqa: E402
from src.dataset import TRAIN_FONTS, SyntheticLines  # noqa: E402
from src.decode import cer  # noqa: E402
from src.infer import load_checkpoint  # noqa: E402
from src.render import EmptyRender, RenderClipped  # noqa: E402
from src.tokenizer import nfc  # noqa: E402

OUT_DIR = ROOT / "out" / "beam"
HELD_OUT_FONT = "C:/Windows/Fonts/javatext.ttf"


def synthetic_items(model, tokenizer, split, n, fonts, augment, drop_space_prob, seed) -> list:
    lines = (ROOT / "data" / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    lines = random.Random(seed).sample(lines, min(n, len(lines)))
    dataset = SyntheticLines(lines, tokenizer, fonts, augment=build_augment(augment), deterministic=True,
                             seed=seed, drop_space_prob=drop_space_prob)
    items = []
    for i in range(len(lines)):
        try:
            img, text = dataset.sample(i)
        except (RenderClipped, EmptyRender):
            continue
        items.append((nfc(text), line_log_probs(img, model)))
    return items


def real_items(model, labels: Path, pad_ratio: float) -> list:
    with labels.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [(nfc(r["text"]), line_log_probs(Image.open(labels.parent / r["image_path"]), model, pad_ratio=pad_ratio))
            for r in rows]


def decode(items, tokenizer, lm=None, alpha=0.0, beta=0.0, width=1, greedy=False) -> list[str]:
    out = []
    for _, lp in items:
        ids = greedy_ids(lp) if greedy else prefix_beam_search(lp, tokenizer.charset, lm, alpha, beta, width)
        out.append(tokenizer.to_logical(tokenizer.decode(ids)))
    return out


def compare(items, greedy_hyps, beam_hyps) -> dict:
    refs = [r for r, _ in items]
    better = worse = 0
    for r, g, b in zip(refs, greedy_hyps, beam_hyps):
        dg, db = Levenshtein.distance(r, g), Levenshtein.distance(r, b)
        better += db < dg
        worse += db > dg
    return {
        "lines": len(refs),
        "greedy_cer": cer(refs, greedy_hyps),
        "beam_cer": cer(refs, beam_hyps),
        "greedy_cer_hyp_no_space": cer(refs, [h.replace(" ", "") for h in greedy_hyps]),
        "beam_cer_hyp_no_space": cer(refs, [h.replace(" ", "") for h in beam_hyps]),
        "lines_better": better,
        "lines_worse": worse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--lm", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dev-lines", type=int, default=300)
    parser.add_argument("--dev-augment", default="fase5")
    parser.add_argument("--dev-drop-space-prob", type=float, default=0.5)
    parser.add_argument("--extra-fonts", default="fonts/extra", help="ikut dipakai merender dev (seperti training)")
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1,1.5")
    parser.add_argument("--betas", default="0,0.5,1,2")
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--clean-lines", type=int, default=300)
    parser.add_argument("--real", default="data/real/nusaaksara/labels.tsv")
    parser.add_argument("--pad-ratio", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    sys.stdout.reconfigure(encoding="utf-8")
    model, tokenizer = load_checkpoint(args.checkpoint)
    lm = CharLM.load(Path(args.lm))
    fonts = list(TRAIN_FONTS) + sorted(p for p in (ROOT / args.extra_fonts).glob("*") if p.suffix.lower() in {".ttf", ".otf"})
    report = {"checkpoint": args.checkpoint, "lm": args.lm, "lm_meta": lm.meta, "lm_order": lm.order,
              "beam_width": args.beam_width}
    started = time.time()

    dev = synthetic_items(model, tokenizer, "val", args.dev_lines, fonts, args.dev_augment,
                          args.dev_drop_space_prob, seed=4242)
    dev_greedy = decode(dev, tokenizer, greedy=True)
    grid = []
    for alpha, beta in itertools.product(map(float, args.alphas.split(",")), map(float, args.betas.split(","))):
        hyps = decode(dev, tokenizer, lm, alpha, beta, args.beam_width)
        grid.append({"alpha": alpha, "beta": beta, "cer": cer([r for r, _ in dev], hyps)})
        print(f"dev alpha {alpha:<4} beta {beta:<3} CER {grid[-1]['cer']:.4%}", flush=True)
    chosen = min(grid, key=lambda g: (g["cer"], g["alpha"], g["beta"]))
    dev_beam = decode(dev, tokenizer, lm, chosen["alpha"], chosen["beta"], args.beam_width)
    report["dev"] = {"augment": args.dev_augment, "drop_space_prob": args.dev_drop_space_prob, "grid": grid,
                     "chosen": chosen, **compare(dev, dev_greedy, dev_beam)}
    print(f"dev: greedy {report['dev']['greedy_cer']:.4%} -> beam {report['dev']['beam_cer']:.4%} "
          f"(alpha {chosen['alpha']}, beta {chosen['beta']})", flush=True)

    alpha, beta = chosen["alpha"], chosen["beta"]
    clean = synthetic_items(model, tokenizer, "test", args.clean_lines, [HELD_OUT_FONT], "none", 0.0, seed=12345)
    report["clean_heldout_font"] = compare(clean, decode(clean, tokenizer, greedy=True),
                                           decode(clean, tokenizer, lm, alpha, beta, args.beam_width))
    real = real_items(model, ROOT / args.real, args.pad_ratio)
    report["real"] = {"labels": args.real, "pad_ratio": args.pad_ratio,
                      **compare(real, decode(real, tokenizer, greedy=True),
                                decode(real, tokenizer, lm, alpha, beta, args.beam_width))}
    report["hours"] = (time.time() - started) / 3600
    for key in ("clean_heldout_font", "real"):
        r = report[key]
        print(f"{key}: greedy {r['greedy_cer']:.4%} -> beam {r['beam_cer']:.4%} "
              f"(tanpa spasi {r['greedy_cer_hyp_no_space']:.4%} -> {r['beam_cer_hyp_no_space']:.4%}); "
              f"baris membaik {r['lines_better']}, memburuk {r['lines_worse']} dari {r['lines']}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.name}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    print(f"hasil: {out}")


if __name__ == "__main__":
    main()
