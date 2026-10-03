"""Fase 6 — potong baris aksara dari foto papan nama Wikimedia Commons + draf label.

  python scripts/commons_lines.py grid c012.jpg [c013.jpg ...]   # pratinjau berkisi koordinat
  python scripts/commons_lines.py crop boxes.json                # potong + rapatkan + draf label

boxes.json berisi daftar kotak dalam piksel citra data/real/commons/raw/:
  [{"file": "c012.jpg", "box": [x0, y0, x1, y1], "latin": "PEMERINTAH KOTA SURAKARTA", "note": ""}, ...]

Draf label = transliterasi teks Latin papan (pipeline korpus), spasi dihapus seperti
penulisan papan nama umumnya. Baris tanpa padanan Latin diberi draf kosong.
DRAF WAJIB DIVERIFIKASI pembaca aksara Jawa sebelum dipakai untuk training/test.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus import clean_words, transliterate_words, valid_aksara  # noqa: E402

COMMONS = ROOT / "data" / "real" / "commons"
RAW_DIR = COMMONS / "raw"
IMAGE_DIR = COMMONS / "images"
DRAFTS = COMMONS / "drafts.jsonl"
GRID_DIR = ROOT / "out" / "commons_grid"
MARGIN = 6


def grid(files: list[str]) -> None:
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=18)
    for name in files:
        img = Image.open(RAW_DIR / name).convert("RGB")
        draw = ImageDraw.Draw(img)
        for x in range(0, img.width, 100):
            draw.line([(x, 0), (x, img.height)], fill=(255, 0, 255) if x % 500 == 0 else (0, 200, 255), width=1)
            draw.text((x + 2, 2), str(x), fill=(255, 0, 255), font=font)
        for y in range(0, img.height, 100):
            draw.line([(0, y), (img.width, y)], fill=(255, 0, 255) if y % 500 == 0 else (0, 200, 255), width=1)
            draw.text((2, y + 2), str(y), fill=(255, 0, 255), font=font)
        out = GRID_DIR / name
        img.save(out, quality=90)
        print(f"{name}: {img.width}x{img.height} -> {out}")


def tighten(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Batas tinta di dalam potongan: piksel yang jauh dari warna latar (median)."""
    background = np.median(gray)
    ink = np.abs(gray.astype(np.int16) - background) > 60
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    if len(rows) == 0:
        return 0, 0, gray.shape[1], gray.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def crop(boxes_path: Path) -> None:
    sources = {
        json.loads(line)["file"]: json.loads(line)
        for line in (COMMONS / "sources.jsonl").read_text(encoding="utf-8").splitlines()
    }
    boxes = json.loads(boxes_path.read_text(encoding="utf-8"))

    # Huruf kecil dulu: clean_words menganggap kata berhuruf kapital semua sebagai
    # singkatan dan membuangnya, padahal papan nama umumnya ditulis kapital.
    for b in boxes:
        b["latin"] = b.get("latin", "").lower()
    words = {w for b in boxes for w in clean_words(b["latin"]) if w}
    table = transliterate_words(words) if words else {}

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    counters: dict[str, int] = {}
    drafts = []
    for b in boxes:
        src = sources[b["file"]]
        img = Image.open(RAW_DIR / b["file"]).convert("RGB")
        x0, y0, x1, y1 = b["box"]
        region = img.crop((max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)))
        gx0, gy0, gx1, gy1 = tighten(np.asarray(region.convert("L")))
        line = region.crop((max(0, gx0 - MARGIN), max(0, gy0 - MARGIN),
                            min(region.width, gx1 + MARGIN), min(region.height, gy1 + MARGIN)))
        k = counters.get(b["file"], 0)
        counters[b["file"]] = k + 1
        line_id = f"{Path(b['file']).stem}_l{k}"
        line.save(IMAGE_DIR / f"{line_id}.png")

        latin_words = [w for w in clean_words(b.get("latin", "")) if w]
        parts = [table[w][0] for w in latin_words if w in table and valid_aksara(table[w][0])]
        draft = "".join(parts) if latin_words and len(parts) == len(latin_words) else ""
        note = b.get("note", "")
        if b.get("latin") and not draft:
            note = (note + "; " if note else "") + "transliterasi otomatis gagal (singkatan/nama asing)"
        drafts.append({
            "id": line_id, "image": f"images/{line_id}.png", "width": line.width, "height": line.height,
            "latin": b.get("latin", ""), "draft": draft, "note": note,
            "source_file": b["file"], "title": src["title"], "page": src["page"],
            "artist": src["artist"], "license": src["license"], "license_url": src["license_url"],
        })
        flag = "" if line.height >= 40 else "  <-- tinggi < 40px"
        print(f"{line_id}: {line.width}x{line.height} draf={'ada' if draft else 'KOSONG'}{flag}")

    with DRAFTS.open("w", encoding="utf-8", newline="\n") as f:
        for d in drafts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(drafts)} baris -> {DRAFTS}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("grid").add_argument("files", nargs="+")
    sub.add_parser("crop").add_argument("boxes", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.command == "grid":
        grid(args.files)
    else:
        crop(args.boxes)


if __name__ == "__main__":
    main()
