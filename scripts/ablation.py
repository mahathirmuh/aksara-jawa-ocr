"""Fase 5 — ablation augmentasi: tambah satu jenis per iterasi, ukur dampaknya.

  .venv-xpu/Scripts/python scripts/ablation.py --init out/checkpoints/base/best_4b_step1500.pt --steps 600

Langkah k melatih lanjut dari checkpoint yang SAMA dengan augmentasi kumulatif
ABLATION_STEPS[:k] dan jumlah langkah yang sama; k = 0 adalah kontrol tanpa
augmentasi, supaya efek "latihan lebih lama" tidak terbaca sebagai efek
augmentasi. Font tambahan dan --drop-space-prob sama untuk semua k. Setiap run
dievaluasi pada val sintetis bersih, val ter-augmentasi berat, dan (opsional) citra
nyata --real. Hasil: out/ablation.md dan out/ablation.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.augment import ABLATION_STEPS  # noqa: E402

OUT_JSON = ROOT / "out" / "ablation.json"
OUT_MD = ROOT / "out" / "ablation.md"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def evaluate(checkpoint: Path, name: str, options: list[str]) -> dict:
    run([sys.executable, "-m", "src.evaluate", str(checkpoint), *options, "--name", name])
    return json.loads((ROOT / "out" / "eval" / f"{name}.json").read_text(encoding="utf-8"))


def pct(value) -> str:
    return "-" if value is None else f"{value:.2%}"


def delta(row: dict, prev: dict | None, key: str) -> str:
    if prev is None or row.get(key) is None or prev.get(key) is None:
        return ""
    return f"{(row[key] - prev[key]) * 100:+.2f} pt"


def write_markdown(rows: list[dict], args) -> None:
    lines = [
        "# Ablation augmentasi (Fase 5)",
        "",
        f"Semua run mulai dari `{args.init}`, {args.steps} langkah, batch {args.batch_size}, "
        f"lr {args.lr}, {args.train_lines:,} baris sintetis, font training inti + `{args.extra_fonts or '-'}`, "
        f"drop-space {args.drop_space_prob}. Evaluasi checkpoint terakhir (bukan terbaik, supaya tidak memihak "
        f"val bersih): {args.eval_lines} baris val font inti, bersih dan augmentasi berat.",
    ]
    if args.real:
        lines += [
            "",
            f"G3 = seluruh baris `{args.real}`. **Peringatan:** itu test set. Memilih augmentasi dari kolom ini "
            "berarti memilih pada data test, jadi angka G3 akhir menjadi bias optimistis. Pakai sebagai arah, "
            "dan laporkan batasan ini.",
        ]
    lines += [
        "",
        "| k | augmentasi ditambahkan | CER bersih | Δ bersih | CER berat | Δ berat | G3 | Δ G3 | G3 tanpa spasi |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    prev = None
    for row in rows:
        lines.append(
            f"| {row['k']} | {row['added']} | {pct(row['clean'])} | {delta(row, prev, 'clean')} | "
            f"{pct(row['heavy'])} | {delta(row, prev, 'heavy')} | {pct(row.get('G3'))} | {delta(row, prev, 'G3')} | "
            f"{pct(row.get('G3_no_space'))} |"
        )
        prev = row
    lines += ["", "Δ negatif = CER turun = augmentasi membantu."]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--init", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-lines", type=int, default=50_000)
    parser.add_argument("--eval-lines", type=int, default=500)
    parser.add_argument("--extra-fonts", default="", help="direktori font tambahan untuk data training (semua k)")
    parser.add_argument("--drop-space-prob", type=float, default=0.0)
    parser.add_argument("--real", default="", help="labels.tsv citra nyata; kosong = tanpa kolom G3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--first-k", type=int, default=0, help="lanjutkan ablation yang terhenti")
    args = parser.parse_args()

    rows = json.loads(OUT_JSON.read_text(encoding="utf-8")) if args.first_k and OUT_JSON.exists() else []
    rows = [row for row in rows if row["k"] < args.first_k]
    for k in range(args.first_k, len(ABLATION_STEPS) + 1):
        augment = "+".join(ABLATION_STEPS[:k]) or "none"
        run_name = f"ablation_{k}"
        train = [sys.executable, "-m", "src.train", "--run", run_name, "--init", args.init, "--augment", augment,
                 "--steps", str(args.steps), "--batch-size", str(args.batch_size), "--lr", str(args.lr),
                 "--train-lines", str(args.train_lines), "--val-lines", "200", "--eval-every", str(args.steps),
                 "--log-every", "100", "--drop-space-prob", str(args.drop_space_prob),
                 "--workers", str(args.workers), "--device", args.device]
        if args.extra_fonts:
            train += ["--extra-fonts", args.extra_fonts]
        run(train)
        checkpoint = ROOT / "out" / "checkpoints" / run_name / "last.pt"
        val = ["--split", "val", "--lines", str(args.eval_lines)]
        row = {
            "k": k,
            "added": ABLATION_STEPS[k - 1] if k else "(kontrol, tanpa augmentasi)",
            "augment": augment,
            "clean": evaluate(checkpoint, f"{run_name}_clean", [*val, "--augment", "none"])["cer"],
            "heavy": evaluate(checkpoint, f"{run_name}_heavy", [*val, "--augment", "heavy"])["cer"],
        }
        if args.real:
            report = evaluate(checkpoint, f"{run_name}_G3", ["--real", args.real, "--lines", "0"])
            row["G3"] = report["cer"]
            row["G3_no_space"] = report["results"]["semua"]["cer_hyp_no_space"]
        rows.append(row)
        OUT_JSON.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
        write_markdown(rows, args)
    print(f"tabel ablation: {OUT_MD}")


if __name__ == "__main__":
    main()
