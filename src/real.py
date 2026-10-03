"""Fase 6 — citra baris nyata: validasi anotasi, split per sumber, dataset fine-tune.

Format data/real/labels.tsv (PLAN.md §4.6 C), path citra relatif terhadap file tsv:
  image_path<TAB>text<TAB>source_id<TAB>condition

  python -m src.real check data/real/labels.tsv
  python -m src.real split data/real/labels.tsv     # -> labels_{train,val,test}.tsv

Split per source_id, BUKAN per baris: baris dari papan nama yang sama tidak
boleh ada di train dan test sekaligus, atau CER test terlihat bagus secara palsu.
"""

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.dataset import Augment, is_trainable, to_tensor
from src.infer import add_margin, normalize_polarity
from src.tokenizer import TOKENIZER_PATH, Tokenizer, is_well_formed, nfc

FIELDS = ["image_path", "text", "source_id", "condition"]
MIN_LINE_HEIGHT = 40
MIN_TEST_LINES = 200


@dataclass(frozen=True)
class RealLine:
    image_path: Path
    text: str
    source_id: str
    condition: str


def read_labels(path: str | Path) -> list[RealLine]:
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: kolom hilang {sorted(missing)}")
        return [
            RealLine(path.parent / row["image_path"], row["text"], row["source_id"], row["condition"])
            for row in reader
        ]


def write_labels(rows: list[RealLine], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(FIELDS)
        for row in rows:
            image = row.image_path.relative_to(path.parent).as_posix()
            writer.writerow([image, row.text, row.source_id, row.condition])


def validate(rows: list[RealLine], tokenizer: Tokenizer) -> list[str]:
    """Daftar masalah anotasi; kosong berarti siap dipakai."""
    problems = []
    usage = Counter(row.image_path for row in rows)
    for number, row in enumerate(rows, start=2):  # baris 1 = header
        where = f"baris {number} ({row.image_path.name})"
        if row.text != nfc(row.text):
            problems.append(f"{where}: teks belum ternormalisasi NFC")
        unknown = sorted({c for c in nfc(row.text) if c not in tokenizer.char_to_id})
        if unknown:
            problems.append(f"{where}: karakter di luar charset: {', '.join(f'U+{ord(c):04X}' for c in unknown)}")
        if not is_well_formed(row.text):
            problems.append(f"{where}: ada tanda yang tidak menempel pada aksara")
        if not row.source_id.strip():
            problems.append(f"{where}: source_id kosong")
        if usage[row.image_path] > 1:
            problems.append(f"{where}: citra yang sama dianotasi lebih dari sekali")
        if not row.image_path.exists():
            problems.append(f"{where}: citra tidak ditemukan")
            continue
        with Image.open(row.image_path) as img:
            if img.height < MIN_LINE_HEIGHT:
                problems.append(f"{where}: tinggi {img.height}px < {MIN_LINE_HEIGHT}px, sandhangan akan hilang")
    return problems


def split_by_source(
    rows: list[RealLine], min_test_lines: int = MIN_TEST_LINES, val_fraction: float = 0.1, seed: int = 0
) -> dict[str, list[RealLine]]:
    """Seluruh baris satu source_id selalu jatuh ke split yang sama."""
    by_source: dict[str, list[RealLine]] = defaultdict(list)
    for row in rows:
        by_source[row.source_id].append(row)
    sources = sorted(by_source)
    random.Random(seed).shuffle(sources)

    splits: dict[str, list[RealLine]] = {"train": [], "val": [], "test": []}
    remaining = []
    for source in sources:
        if len(splits["test"]) < min_test_lines:
            splits["test"] += by_source[source]
        else:
            remaining.append(source)
    val_target = round(val_fraction * sum(len(by_source[s]) for s in remaining))
    for source in remaining:
        splits["val" if len(splits["val"]) < val_target else "train"] += by_source[source]
    return splits


class RealLines(Dataset):
    """Citra baris nyata -> (citra (1, H, W), target urutan visual). Tidak ada render."""

    def __init__(
        self,
        rows: list[RealLine],
        tokenizer: Tokenizer,
        augment: Augment | None = None,
        deterministic: bool = False,
        seed: int = 0,
        pad_ratio: float = 0.0,
    ):
        self.rows = list(rows)
        self.lines = [nfc(row.text) for row in self.rows]
        self.tokenizer = tokenizer
        self.augment = augment
        self.deterministic = deterministic
        self.seed = seed
        self.pad_ratio = pad_ratio

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        rng = random.Random(f"{self.seed}:{idx}") if self.deterministic else random.Random()
        with Image.open(self.rows[idx].image_path) as img:
            img = add_margin(normalize_polarity(img), getattr(self, "pad_ratio", 0.0))
        if self.augment:
            img = self.augment(img, rng)
        x = to_tensor(img)
        y = torch.tensor(self.tokenizer.encode(self.tokenizer.to_visual(self.lines[idx])), dtype=torch.long)
        return (x, y) if is_trainable(len(y), x.shape[-1]) else None


class ConcatLines(Dataset):
    """Gabungan dataset baris, masing-masing dengan faktor ulang; `lines` ikut digabung."""

    def __init__(self, parts: list[tuple[Dataset, int]]):
        self.index = [(dataset, i) for dataset, repeat in parts for _ in range(repeat) for i in range(len(dataset))]
        self.lines = [dataset.lines[i] for dataset, i in self.index]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, k: int):
        dataset, i = self.index[k]
        return dataset[i]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "split"])
    parser.add_argument("labels")
    parser.add_argument("--min-test-lines", type=int, default=MIN_TEST_LINES)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    labels = Path(args.labels)
    rows = read_labels(labels)
    problems = validate(rows, Tokenizer.load(TOKENIZER_PATH))
    sources = Counter(row.source_id for row in rows)
    print(f"{len(rows)} baris dari {len(sources)} sumber; kondisi: {dict(Counter(r.condition for r in rows))}")
    for problem in problems:
        print(f"  MASALAH {problem}")
    if problems:
        sys.exit(1)
    if args.command == "check":
        print("anotasi valid")
        return

    splits = split_by_source(rows, args.min_test_lines, args.val_fraction, args.seed)
    for name, part in splits.items():
        write_labels(part, labels.with_name(f"labels_{name}.tsv"))
        print(f"{name:5}: {len(part):4} baris, {len({r.source_id for r in part})} sumber")
    if len(splits["test"]) < args.min_test_lines:
        print(f"PERINGATAN: test hanya {len(splits['test'])} baris; G3 menuntut >= {args.min_test_lines}")


if __name__ == "__main__":
    main()
