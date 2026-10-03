"""Terjemahan batch untuk `php artisan aksara:translate` (tahap 3 alur).

  python tools/translate_batch.py INPUT.jsonl OUTPUT.jsonl SUMMARY.json

INPUT: satu baris JSON per teks {dataset, external_id, source, input, reference}. `reference` = terjemahan
manusia NusaAksara (boleh kosong). OUTPUT: baris yang sama ditambah `output` dan `chrf` (sentence chrF).
SUMMARY: chrF & BLEU korpus per `source` (sacrebleu), model, jumlah baris.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import sacrebleu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nllb import MODEL, translate  # noqa: E402


def main(argv: list[str]) -> None:
    src, dst, summary_path = map(Path, argv[1:4])
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    started = time.time()
    outputs = translate([r["input"] for r in rows])
    by_source = defaultdict(lambda: ([], []))
    with dst.open("w", encoding="utf-8", newline="\n") as f:
        for row, hyp in zip(rows, outputs):
            ref = (row.get("reference") or "").strip()
            row["output"] = hyp
            row["chrf"] = round(sacrebleu.sentence_chrf(hyp, [ref]).score, 2) if ref and hyp else None
            if ref:
                by_source[row["source"]][0].append(hyp)
                by_source[row["source"]][1].append(ref)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "model": MODEL,
        "seconds": round(time.time() - started, 1),
        "runs": [
            {"source": source, "lines": len(hyps),
             "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
             "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2)}
            for source, (hyps, refs) in by_source.items()
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for run in summary["runs"]:
        print(f"{run['source']}: chrF {run['chrf']} BLEU {run['bleu']} ({run['lines']} baris)", flush=True)
    print(f"{len(rows)} teks dalam {summary['seconds']} dtk", flush=True)


if __name__ == "__main__":
    main(sys.argv)
