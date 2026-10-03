"""Fase 0 — render kasus uji shaping ke out/shaping/ untuk diperiksa manusia.

Pemeriksaan otomatis di sini hanya menangkap kegagalan KASAR: font tanpa tabel
OpenType, glyph tidak ada, atau pasangan tidak terbentuk. Ia tidak membuktikan
shaping benar. Fase 0 baru lulus setelah manusia membuka PNG-nya.

Jalankan dari root repo:  python scripts/verify_shaping.py
"""

import argparse
import struct
import sys
import unicodedata
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.render import (  # noqa: E402
    DEFAULT_PAD,
    DEFAULT_SIZE,
    assert_raqm,
    load_font,
    render_with_font,
)

FONTS_DIR = ROOT / "fonts"
OUT_DIR = ROOT / "out" / "shaping"
MIN_FONTS = 2
# Pasangan yang terbentuk kira-kira selebar aksara dasarnya; tanpa shaping,
# pangkon dan ta berjajar di sampingnya sehingga lebarnya lebih dari 2x.
MAX_PASANGAN_WIDTH_RATIO = 1.4
# Seberapa jauh (dalam em) pasangan harus turun di bawah dasar aksara ka.
MIN_PASANGAN_DROP_EM = 0.25
# Tanda atas yang menempel benar tidak membuat suku kata jauh lebih lebar dari aksaranya.
MAX_MARK_WIDTH_RATIO = 1.3
WULU = "ꦶ"

KA = "ꦏ"
# Codepoint tak-teralokasi di blok Jawa: font mana pun menggambarnya sebagai .notdef.
NOTDEF_PROBE = "꧎"
# ka + pangkon + ta: pasangan yang pasti menumpuk di bawah.
STACK_PROBE = "A98F A9C0 A9A0"
GRAPHITE_TABLES = {"Silf", "Glat", "Gloc"}

# (nama, codepoint, yang harus terlihat) — tabel PLAN.md §7 Fase 0, dengan satu koreksi.
#
# PLAN.md menulis pasangan sebagai A9A5 berlabel "ta". A9A5 adalah PA, dan
# pasangan ha/sa/pa ditulis di SAMPING, bukan di bawah — terukur sama persis di
# Noto Sans Javanese dan Tuladha Jejeg OT. Dengan A9A5 kasus itu tidak bisa
# menguji penumpukan, jadi dipakai A9A0 (TA) sesuai maksud labelnya. Kasus
# A9A5 aslinya dipertahankan sebagai pasangan_samping.
CASES = [
    ("nglegena", "A98F A9A5 A9A9", "ka pa ma: tiga aksara sejajar"),
    ("taling_prebase", "A98F A9BA", "taling di KIRI ka"),
    ("taling_tarung", "A98F A9BA A9B4", "taling kiri + tarung kanan, mengapit ka"),
    ("pasangan", STACK_PROBE, "ta menumpuk di BAWAH ka, pangkon hilang"),
    ("pasangan_susun3", "A98F A9C0 A9A0 A9C0 A9A9", "tiga tingkat ka/ta/ma, tak tumpang tindih"),
    ("sandhangan_atas", "A98F A9B6", "wulu di atas ka"),
    ("sandhangan_bawah", "A98F A9B8", "suku di bawah ka"),
    ("layar", "A98F A982", "layar di atas kanan"),
    ("cakra", "A98F A9BF", "cakra di bawah kanan"),
    ("angka", "A9D1 A9D2 A9D3", "tiga digit: 1 2 3"),
    ("pada", "A9C8 A9C9", "pada lingsa & pada lungsi"),
    # Di luar 11 kasus PLAN.md.
    # Kriteria lulus menuntut pangkon terlihat di akhir kata, tapi tabelnya tidak punya kasusnya.
    ("pangkon_akhir", "A98F A9C0", "pangkon TERLIHAT di akhir kata"),
    ("pasangan_samping", "A98F A9C0 A9A5", "pa di SAMPING ka, tidak menumpuk"),
]


def to_text(codepoints: str) -> str:
    return "".join(chr(int(h, 16)) for h in codepoints.split())


def font_tables(path: Path) -> set[str]:
    """Tag tabel sfnt di file font."""
    data = path.read_bytes()
    (num_tables,) = struct.unpack(">H", data[4:6])
    return {data[12 + 16 * i : 16 + 16 * i].decode("latin-1") for i in range(num_tables)}


def try_render(text: str, font: ImageFont.FreeTypeFont, pad: int = DEFAULT_PAD) -> Image.Image | None:
    try:
        return render_with_font(text, font, pad)
    except ValueError:  # render kosong
        return None


def same_image(a: Image.Image | None, b: Image.Image | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.size == b.size and ImageChops.difference(a, b).getbbox() is None


def missing_codepoints(font: ImageFont.FreeTypeFont) -> list[str]:
    """Codepoint yang dirender identik dengan .notdef — tidak ada di font."""
    notdef_alone = try_render(NOTDEF_PROBE, font)
    notdef_after_ka = try_render(KA + NOTDEF_PROBE, font)
    missing = []
    for cp in sorted({h for _, cps, _ in CASES for h in cps.split()}):
        ch = chr(int(cp, 16))
        # Tanda (Mn/Mc) sendirian diberi dotted circle oleh HarfBuzz;
        # uji di atas ka supaya perbandingannya setara.
        if unicodedata.category(ch).startswith("M"):
            got, ref = try_render(KA + ch, font), notdef_after_ka
        else:
            got, ref = try_render(ch, font), notdef_alone
        if same_image(got, ref):
            missing.append(f"U+{cp} {unicodedata.name(ch)}")
    return missing


def report(label: str, ok: bool, detail: str = "") -> bool:
    status = "OK" if ok else "MASALAH"
    suffix = f": {detail}" if detail and not ok else ""
    print(f"  [{status}] {label}{suffix}")
    return ok


def check_font(path: Path, raqm: ImageFont.FreeTypeFont) -> int:
    """Jalankan pemeriksaan kasar; kembalikan jumlah masalah."""
    results = []

    # RAQM hanya membaca shaping OpenType. Font Graphite-only tetap dirender
    # tanpa error, tapi pasangan tidak terbentuk dan taling tidak pindah.
    tables = font_tables(path)
    absent = [t for t in ("GSUB", "GPOS") if t not in tables]
    detail = "tidak ada " + ", ".join(absent)
    if tables & GRAPHITE_TABLES:
        detail += " (font Graphite, tidak dibaca RAQM)"
    results.append(report("tabel OpenType GSUB & GPOS ada", not absent, detail))

    missing = missing_codepoints(raqm)
    results.append(report("glyph tersedia untuk semua codepoint uji", not missing, ", ".join(missing)))

    # Tinggi saja tidak cukup untuk membuktikan pasangan terbentuk: glyph
    # pangkon punya ekor ke bawah, jadi render tanpa shaping pun ikut "turun".
    # Pasangan yang benar turun DAN tetap kira-kira selebar ka.
    ka_left, _, ka_right, ka_bottom = raqm.getbbox(KA, anchor="ls")
    left, _, right, bottom = raqm.getbbox(to_text(STACK_PROBE), anchor="ls")
    drop_em = (bottom - ka_bottom) / raqm.size
    width_ratio = (right - left) / (ka_right - ka_left)
    stacked = drop_em >= MIN_PASANGAN_DROP_EM and width_ratio <= MAX_PASANGAN_WIDTH_RATIO
    results.append(
        report(
            f"pasangan ta menumpuk (turun {drop_em:.2f} em, lebar {width_ratio:.2f}x ka;"
            f" butuh >= {MIN_PASANGAN_DROP_EM} em dan <= {MAX_PASANGAN_WIDTH_RATIO}x)",
            stacked,
        )
    )

    # Tanda atas harus menempel di atas aksara dasar. Font dengan mark attachment GPOS
    # rusak (Istaka, Nawatura) lolos semua cek di atas tetapi menggeser wulu ke kiri.
    _, ka_top, _, _ = raqm.getbbox(KA, anchor="ls")
    wulu_left, wulu_top, wulu_right, _ = raqm.getbbox(KA + WULU, anchor="ls")
    mark_width_ratio = (wulu_right - wulu_left) / (ka_right - ka_left)
    results.append(
        report(
            f"wulu menempel di atas ka (lebar {mark_width_ratio:.2f}x ka, butuh <= {MAX_MARK_WIDTH_RATIO}x;"
            f" puncak {ka_top - wulu_top} px di atas ka)",
            mark_width_ratio <= MAX_MARK_WIDTH_RATIO and wulu_top < ka_top,
        )
    )

    return results.count(False)


def contact_sheet(columns: list[tuple[str, list[Image.Image | None]]]) -> Image.Image:
    label_font = ImageFont.load_default(size=16)
    gap, label_w, header_h, min_row = 12, 440, 40, 64

    col_ws = [
        int(max([im.width for im in imgs if im is not None] + [label_font.getlength(header), 160])) + 2 * gap
        for header, imgs in columns
    ]
    row_hs = []
    for r in range(len(CASES)):
        heights = [imgs[r].height for _, imgs in columns if imgs[r] is not None]
        row_hs.append(max(heights + [min_row]) + 2 * gap)

    sheet = Image.new("L", (label_w + sum(col_ws), header_h + sum(row_hs)), 255)
    draw = ImageDraw.Draw(sheet)

    x = label_w
    for (header, _), w in zip(columns, col_ws):
        draw.line([(x, 0), (x, sheet.height)], fill=200)
        draw.text((x + gap, gap), header, font=label_font, fill=0)
        x += w

    y = header_h
    for r, (name, cps, expect) in enumerate(CASES):
        draw.line([(0, y), (sheet.width, y)], fill=200)
        draw.text((gap, y + gap), f"{r:02d} {name}\n{cps}\n{expect}", font=label_font, fill=0, spacing=4)
        x = label_w
        for (_, imgs), w in zip(columns, col_ws):
            if imgs[r] is None:
                draw.text((x + gap, y + gap), "(render kosong)", font=label_font, fill=0)
            else:
                sheet.paste(imgs[r], (x + gap, y + gap))
            x += w
        y += row_hs[r]
    return sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fonts-dir", type=Path, default=FONTS_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--per-sheet", type=int, default=4, help="jumlah font per lembar kontak")
    # argv None (mis. dipanggil dari notebook) = tanpa argumen, bukan sys.argv milik kernel.
    args = parser.parse_args([] if argv is None else argv)
    assert_raqm()

    font_paths = []
    if args.fonts_dir.is_dir():
        font_paths = sorted(p for p in args.fonts_dir.iterdir() if p.suffix.lower() in {".ttf", ".otf"})
    if not font_paths:
        print(f"Tidak ada font di {args.fonts_dir}. Lihat PLAN.md §4.5.")
        return 2
    if len(font_paths) < MIN_FONTS:
        print(f"PERINGATAN: hanya {len(font_paths)} font; gerbang Fase 0 butuh minimal {MIN_FONTS}.")

    args.out.mkdir(parents=True, exist_ok=True)
    problems = 0
    columns = []

    for path in font_paths:
        raqm = load_font(path, DEFAULT_SIZE)
        print(f"\n== {path.name} ==")

        font_dir = args.out / path.stem
        font_dir.mkdir(exist_ok=True)
        for old in font_dir.glob("*.png"):
            old.unlink()
        imgs = []
        for i, (name, cps, _) in enumerate(CASES):
            img = try_render(to_text(cps), raqm)
            if img is None:
                problems += not report(f"render {name}", False, "kosong")
            else:
                img.save(font_dir / f"{i:02d}_{name}.png")
            imgs.append(img)
        columns.append((f"{path.stem} (RAQM)", imgs))

        problems += check_font(path, raqm)

    # Satu lembar per kelompok font; kolom terakhir tiap lembar adalah pembanding
    # seperti apa render yang SALAH (tanpa shaping) untuk font pertama kelompok itu.
    sheet_paths = []
    for start in range(0, len(font_paths), args.per_sheet):
        group = font_paths[start : start + args.per_sheet]
        basic = ImageFont.truetype(str(group[0]), DEFAULT_SIZE, layout_engine=ImageFont.Layout.BASIC)
        comparison = ("TANPA SHAPING (pembanding salah)", [try_render(to_text(cps), basic) for _, cps, _ in CASES])
        single = len(font_paths) <= args.per_sheet
        sheet_path = args.out / ("contact_sheet.png" if single else f"contact_sheet_{start // args.per_sheet + 1}.png")
        contact_sheet(columns[start : start + args.per_sheet] + [comparison]).save(sheet_path)
        sheet_paths.append(sheet_path)

    print(f"\nPNG per font : {args.out}/<font>/")
    print("Ringkasan    : " + ", ".join(str(p) for p in sheet_paths))
    print(f"Masalah otomatis: {problems}")
    print("\nFase 0 BELUM lulus. Pemeriksaan di atas hanya menangkap kegagalan kasar.")
    print("Buka PNG-nya dan periksa sendiri terhadap kolom 'yang harus terlihat'.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
