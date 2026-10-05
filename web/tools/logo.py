"""Buat semua berkas lambang Aksara OCR Lab dari satu gambar sumber: resources/brand/logo.png.

Gambar sumber dipilih user (2026-10-05): ikon aplikasi 1254 px berupa ubin putih bersudut bulat berisi lambang
(keris di atas atap joglo, diapit bulir padi, di atas ombak), sebaris tulisan bergaya aksara Jawa, dan garis hias.

Keluaran, semuanya di public/:
  img/logo/logo.png       ubin utuh, sudut transparan, 384 px       halaman masuk
  img/logo/mark.png       lambang saja di ubin putih, 144 px        sidebar (tulisan tidak terbaca di bawah ~100 px)
  favicon.ico             lambang saja, 16 / 32 / 48 px             tintanya ditebalkan sedikit: garis tipis lambang
  favicon-192.png         lambang saja, 192 px                      memudar jadi biru pucat bila hanya diperkecil
  apple-touch-icon.png    ubin utuh, persegi tanpa transparansi, 180 px (iOS memotong sudutnya sendiri)

Ukuran di bawah (TILE, EMBLEM) diukur dari gambar sumber ini; kalau gambarnya diganti, ukur ulang. Skrip berhenti
bila kotak lambang tidak lagi cocok dengan tinta yang ditemukan. Pakai (dari akar repo):
  .venv/Scripts/python web/tools/logo.py web
"""
import os
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFilter

# Satuan piksel gambar sumber.
SOURCE_SIZE = 1254
TILE = (92, 92, 1161, 1161)         # ubin putih, 5 px di dalam tepinya supaya bayangan luar tidak ikut
RADIUS = .222                       # sudut ubin: 240 dari 1079 px
EMBLEM = (237, 122, 1016, 906)      # kotak pembatas lambang; tulisan mulai di y = 918
INK_SATURATION = 70                 # maks - min kanal di atas ini = tinta lambang (latar hiasnya <= 40)
# Latar ubin punya hiasan ombak sangat pucat. Untuk lambang saja, piksel yang nyaris putih dijadikan putih:
# kegelapan (255 - kanal terkecil) <= CLEAN[0] dibuang, >= CLEAN[1] dipertahankan utuh (tinta paling muda ~120).
CLEAN = (40, 96)


def channel_extremes(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    r, g, b = image.convert('RGB').split()
    return ImageChops.lighter(ImageChops.lighter(r, g), b), ImageChops.darker(ImageChops.darker(r, g), b)


def ink_box(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Kotak pembatas tinta jenuh di dalam `box`, dalam koordinat gambar."""
    highest, lowest = channel_extremes(image.crop(box))
    found = ImageChops.subtract(highest, lowest).point(lambda v: 255 if v > INK_SATURATION else 0).getbbox()
    assert found, 'tidak ada tinta di kotak lambang'
    return (box[0] + found[0], box[1] + found[1], box[0] + found[2] - 1, box[1] + found[3] - 1)


def rounded(square: Image.Image, size: int, grow: float = 0.0) -> Image.Image:
    """Perkecil ke `size` px dengan sudut bulat transparan. `grow` = penebalan tinta tiap sisi, dalam piksel hasil."""
    if grow:
        # Ditebalkan pada ukuran antara (beberapa kali ukuran hasil): filter minimum melebarkan warna tinta ke putih.
        factor = max(1, min(8, square.width // size))
        square = square.resize((size * factor, size * factor), Image.LANCZOS).filter(ImageFilter.MinFilter(2 * round(grow * factor) + 1))
    side = square.width
    mask = Image.new('L', (side, side), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, side - 1, side - 1], radius=round(RADIUS * side), fill=255)
    square = square.convert('RGB')
    square.putalpha(mask)
    return square.resize((size, size), Image.LANCZOS)


def tile(source: Image.Image) -> Image.Image:
    """Ubin utuh (lambang + tulisan + garis hias) tanpa bayangan luarnya."""
    return source.crop((TILE[0], TILE[1], TILE[2] + 1, TILE[3] + 1))


def emblem_tile(source: Image.Image, fill: float) -> Image.Image:
    """Lambang saja, dipusatkan di ubin putih; `fill` = bagian sisi ubin yang ditempati lambang."""
    margin = 6
    box = (EMBLEM[0] - margin, EMBLEM[1] - margin, EMBLEM[2] + 1 + margin, EMBLEM[3] + 1 + margin)
    crop = source.crop(box)
    _, lowest = channel_extremes(crop)
    low, high = CLEAN
    alpha = ImageChops.invert(lowest).point(lambda v: max(0, min(255, round((v - low) * 255 / (high - low)))))
    side = round(max(EMBLEM[2] - EMBLEM[0], EMBLEM[3] - EMBLEM[1]) / fill)
    canvas = Image.new('RGB', (side, side), '#ffffff')
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2), alpha)
    return canvas


def main(web_dir: str) -> None:
    source = Image.open(os.path.join(web_dir, 'resources', 'brand', 'logo.png')).convert('RGB')
    assert source.size == (SOURCE_SIZE, SOURCE_SIZE), f'gambar sumber {source.size}, ukuran di skrip untuk {SOURCE_SIZE} px'
    found = ink_box(source, (TILE[0], TILE[1], TILE[2], EMBLEM[3] + 6))
    assert all(abs(a - b) <= 4 for a, b in zip(found, EMBLEM)), f'kotak lambang {EMBLEM} tidak cocok dengan tinta {found}'

    public = os.path.join(web_dir, 'public')
    os.makedirs(os.path.join(public, 'img', 'logo'), exist_ok=True)
    full = tile(source)
    rounded(full, 384).save(os.path.join(public, 'img', 'logo', 'logo.png'), optimize=True)
    rounded(emblem_tile(source, .82), 144, grow=.5).save(os.path.join(public, 'img', 'logo', 'mark.png'), optimize=True)
    # (ukuran, bagian ubin yang ditempati lambang, penebalan): makin kecil makin besar dan makin tebal.
    frames = [rounded(emblem_tile(source, fill), size, grow) for size, fill, grow in ((48, .86, .125), (32, .88, .125), (16, .9, .25))]
    frames[0].save(os.path.join(public, 'favicon.ico'), sizes=[frame.size for frame in frames], append_images=frames[1:])
    rounded(emblem_tile(source, .86), 192).save(os.path.join(public, 'favicon-192.png'), optimize=True)
    full.resize((180, 180), Image.LANCZOS).save(os.path.join(public, 'apple-touch-icon.png'), optimize=True)
    print('lambang ditulis ke', public)


if __name__ == '__main__':
    main(sys.argv[1])
