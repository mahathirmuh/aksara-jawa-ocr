"""Pecahan baris tetangga di tepi atas dan bawah citra baris (augmentasi data sintetis).

Potongan baris nyata hasil segmentasi halaman sering memuat sisa tinta baris di atas dan di bawahnya: ujung bawah
pasangan dan sandhangan bawah baris sebelumnya di tepi atas, tanda atas baris berikutnya di tepi bawah (CLAUDE.md,
"pecahan baris tetangga", 2026-10-17). Render sintetis tidak pernah memuatnya, jadi model belajar membaca semua tinta
sebagai teks. `with_neighbors` menempelkan pita bawah sebuah baris lain di atas citra dan pita atas baris lain di
bawahnya; labelnya tetap teks baris utama. Augmentasi `tight` sesudahnya memotong rapat sampai pecahan menyentuh tepi.
"""

import random

import numpy as np
from PIL import Image

# Bagian tinggi tinta baris tetangga yang ikut terbawa (pita bawah untuk baris di atas, pita atas untuk baris di bawah),
# dan jarak kosong antara pita itu dan tinta baris utama, keduanya relatif terhadap tinggi tinta baris utama.
BAND_RANGE = (0.05, 0.25)  # pecahan nyata umumnya jarang: tanda atas atau ujung pasangan, bukan badan aksara
GAP_RANGE = (0.02, 0.15)
INK = 128  # piksel < INK = tinta (citra render: hitam di atas putih)


def ink_rows(img: Image.Image) -> tuple[int, int] | None:
    """Baris piksel pertama dan sesudah-terakhir yang memuat tinta, atau None bila citra kosong."""
    rows = np.where((np.asarray(img.convert("L")) < INK).any(axis=1))[0]
    return (int(rows[0]), int(rows[-1]) + 1) if len(rows) else None


def band(img: Image.Image, fraction: float, bottom: bool) -> Image.Image | None:
    """Pita tinta selebar citra: `fraction` tinggi tinta terbawah (bottom) atau teratas, tanpa baris kosong."""
    rows = ink_rows(img)
    if rows is None:
        return None
    top, end = rows
    take = max(1, int(round((end - top) * fraction)))
    strip = img.crop((0, end - take, img.width, end) if bottom else (0, top, img.width, top + take))
    return strip if ink_rows(strip) is not None else None


def with_neighbors(img: Image.Image, above: Image.Image | None, below: Image.Image | None,
                   rng: random.Random) -> Image.Image:
    """Citra baris `img` dengan pita bawah `above` di atasnya dan pita atas `below` di bawahnya (yang None dilewati).

    Pita digeser mendatar secara acak dan dipotong selebar `img`, dengan jarak kosong ke tinta baris utama, jadi pecahan
    tidak pernah menimpa teks yang dilabeli. Citra yang tidak bertinta dikembalikan apa adanya."""
    img = img.convert("L")
    rows = ink_rows(img)
    if rows is None:
        return img
    ink_height = rows[1] - rows[0]
    parts = []
    for neighbor, bottom in ((above, True), (below, False)):
        strip = band(neighbor.convert("L"), rng.uniform(*BAND_RANGE), bottom) if neighbor is not None else None
        if strip is None:
            parts.append(None)
            continue
        canvas = Image.new("L", (img.width, strip.height), 255)
        canvas.paste(strip, (rng.randint(-max(0, strip.width - img.width), max(0, img.width - strip.width) // 4), 0))
        parts.append((canvas, max(1, int(round(ink_height * rng.uniform(*GAP_RANGE))))))
    if parts == [None, None]:
        return img
    # Tinta baris utama mulai di rows[0] dan berakhir di rows[1]; jarak kosong diukur dari tinta, bukan dari tepi citra
    # render. Kanvas hanya diperluas sejauh pita dan jaraknya tidak muat di tepi kosong citra render.
    top_part, bottom_part = parts
    head = max(0, top_part[0].height + top_part[1] - rows[0]) if top_part else 0
    tail = max(0, rows[1] + bottom_part[1] + bottom_part[0].height - img.height) if bottom_part else 0
    out = Image.new("L", (img.width, head + img.height + tail), 255)
    out.paste(img, (0, head))
    if top_part:
        out.paste(top_part[0], (0, head + rows[0] - top_part[1] - top_part[0].height), mask=_ink_mask(top_part[0]))
    if bottom_part:
        out.paste(bottom_part[0], (0, head + rows[1] + bottom_part[1]), mask=_ink_mask(bottom_part[0]))
    return out


def _ink_mask(strip: Image.Image) -> Image.Image:
    """Topeng tempel: hanya piksel tinta yang ditempel, jadi latar putih pita tidak menutupi citra utama."""
    return Image.fromarray(((np.asarray(strip) < INK) * 255).astype(np.uint8))
