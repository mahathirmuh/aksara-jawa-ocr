"""Fase 6 — impor hasil verifikasi rekan menjadi labels.tsv.

  python scripts/import_review.py hasil_verifikasi.tsv

Masukan: TSV yang diekspor halaman verifikasi, kolom `id  status  text  note`,
status `benar` | `koreksi` | `buang`. Baris tanpa status dianggap belum diverifikasi
dan tidak ikut.

Keluaran di data/real/commons/:
  labels.tsv       — hanya baris benar/koreksi, tervalidasi (NFC, charset, tinggi)
  ATTRIBUTION.md   — atribusi CC BY-SA untuk setiap foto yang dipakai
"""

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.real import RealLine, validate, write_labels  # noqa: E402
from src.tokenizer import TOKENIZER_PATH, Tokenizer  # noqa: E402

COMMONS = ROOT / "data" / "real" / "commons"
DRAFTS = COMMONS / "drafts.jsonl"
LABELS = COMMONS / "labels.tsv"
ATTRIBUTION = COMMONS / "ATTRIBUTION.md"
ACCEPTED = {"benar", "koreksi"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("review", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    drafts = {d["id"]: d for d in map(json.loads, DRAFTS.read_text(encoding="utf-8").splitlines())}
    # Halaman verifikasi mengekspor CSV; TSV lama tetap diterima.
    delimiter = "," if args.review.suffix.lower() == ".csv" else "\t"
    with args.review.open(encoding="utf-8-sig", newline="") as f:
        reviews = {row["id"]: row for row in csv.DictReader(f, delimiter=delimiter)}

    status_counts = Counter()
    kept: list[RealLine] = []
    used_sources = {}
    for line_id, draft in drafts.items():
        review = reviews.get(line_id)
        status = (review or {}).get("status", "").strip()
        status_counts[status or "belum diverifikasi"] += 1
        if status not in ACCEPTED:
            continue
        text = unicodedata.normalize("NFC", review["text"].strip())
        if not text:
            status_counts["benar/koreksi tapi teks kosong"] += 1
            continue
        # Satu foto = satu sumber: baris dari foto yang sama tidak boleh terpisah antar split.
        source = Path(draft["source_file"]).stem
        kept.append(RealLine(COMMONS / draft["image"], text, source, "papan_nama"))
        used_sources[source] = draft

    unknown_ids = sorted(set(reviews) - set(drafts))
    tokenizer = Tokenizer.load(TOKENIZER_PATH)
    problems = validate(kept, tokenizer)

    print("status:", dict(status_counts))
    if unknown_ids:
        print(f"PERINGATAN: {len(unknown_ids)} id di TSV tidak ada di drafts.jsonl, contoh {unknown_ids[:3]}")
    for problem in problems:
        print("  MASALAH", problem)
    if problems:
        print("labels.tsv TIDAK ditulis; perbaiki baris di atas (atau tandai 'buang') lalu impor ulang.")
        sys.exit(1)

    write_labels(kept, LABELS)
    lines = ["# Atribusi foto (Wikimedia Commons)", "",
             "Potongan baris di `images/` diturunkan dari foto berikut; lisensi mengikuti foto aslinya.", ""]
    for source, d in sorted(used_sources.items()):
        lines.append(f"- `{source}`: [{d['title']}]({d['page']}) oleh {d['artist'] or 'tidak disebut'}, "
                     f"{d['license']} ({d['license_url']})")
    ATTRIBUTION.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(kept)} baris dari {len(used_sources)} foto -> {LABELS}")
    print(f"atribusi -> {ATTRIBUTION}")


if __name__ == "__main__":
    main()
