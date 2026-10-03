"""Fase 5 — augmentasi degradasi untuk citra baris hasil render.

Untuk teks cetak, render font sudah menghasilkan bentuk glyph yang benar; gap
ke dunia nyata tinggal kanal degradasi (kertas, tinta, optik, sensor,
kompresi). Modul ini memodelkan kanal itu dengan PIL + numpy.

Setiap operasi diterapkan dalam urutan fisik (geometri -> tinta/kertas ->
optik -> sensor -> kompresi), apa pun urutan penyebutannya. Semua keacakan
berasal dari `rng` yang diberikan dataset, jadi mode deterministik tetap
reproducible. JPEG dikodekan di memori, bukan ke disk.

  python -m src.augment      # contoh visual ke out/augment_examples.png
"""

import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PATH = ROOT / "out" / "augment_examples.png"

# Urutan penambahan untuk ablation, mengikuti PLAN.md §7 Fase 5. Variasi
# ukuran font dan font berbeda adalah baseline (sudah ada di dataset).
ABLATION_STEPS = ["rotate", "noise", "blur", "elastic", "contrast", "jpeg", "texture",
                  # tambahan setelah membandingkan dengan pindaian nyata (NusaAksara)
                  "tight", "stroke", "binarize", "speckle"]


def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32)


def _img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _np_rng(rng: random.Random) -> np.random.Generator:
    return np.random.default_rng(rng.getrandbits(32))


def _scale(img: Image.Image) -> float:
    """Kekuatan dinyatakan relatif terhadap baris setinggi 96 px."""
    return img.height / 96


def _smooth_field(np_rng: np.random.Generator, h: int, w: int, cell: int) -> np.ndarray:
    coarse = np_rng.uniform(-1, 1, (h // cell + 2, w // cell + 2)).astype(np.float32)
    return np.asarray(Image.fromarray(coarse).resize((w, h), Image.Resampling.BICUBIC))


def _bilinear(arr: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, w - 2)
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, h - 2)
    fx = np.clip(xs - x0, 0, 1)
    fy = np.clip(ys - y0, 0, 1)
    top = arr[y0, x0] * (1 - fx) + arr[y0, x0 + 1] * fx
    bottom = arr[y0 + 1, x0] * (1 - fx) + arr[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def margin(img: Image.Image, rng: random.Random, heavy: bool) -> Image.Image:
    """Potongan baris nyata jarang pas ke tinta."""
    h = img.height
    top, bottom = rng.randint(0, h // 4), rng.randint(0, h // 4)
    left, right = rng.randint(0, h // 2), rng.randint(0, h // 2)
    out = Image.new("L", (img.width + left + right, h + top + bottom), 255)
    out.paste(img, (left, top))
    return out


def tight(img: Image.Image, rng: random.Random, heavy: bool) -> Image.Image:
    """Potongan rapat seperti pindaian buku: tinta menyentuh tepi atas dan bawah.

    94% potongan NusaAksara begitu, sedangkan render sintetis selalu diberi margin.
    Ujung tumpukan pasangan/sandhangan kadang ikut terpangkas sedikit; batasnya kecil
    supaya label tetap sesuai citra. Kebalikan `margin`, jadi tidak masuk preset.
    """
    arr = _arr(img)
    ink = arr < 128
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if len(rows) == 0:
        return img
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    trim = int((bottom - top) * rng.uniform(0, 0.04 if heavy else 0.02))
    top += rng.randint(0, trim)
    bottom -= rng.randint(0, trim)
    side = max(1, img.height // 40)
    left = max(0, left - rng.randint(0, side))
    right = min(img.width, right + rng.randint(0, side))
    return img.crop((left, top, right, max(top + 1, bottom)))


def rotate(img: Image.Image, rng: random.Random, heavy: bool) -> Image.Image:
    """Kemiringan baseline ±1.5°."""
    angle = rng.uniform(-1.5, 1.5)
    return img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=255)


def elastic(img: Image.Image, rng: random.Random, heavy: bool) -> Image.Image:
    """Distorsi halus lokal: kertas bergelombang, cetakan tidak rata."""
    arr = _arr(img)
    h, w = arr.shape
    np_rng = _np_rng(rng)
    alpha = rng.uniform(0.5, 2.5 if heavy else 1.5) * _scale(img)
    cell = max(8, h // 3)
    dx = _smooth_field(np_rng, h, w, cell) * alpha
    dy = _smooth_field(np_rng, h, w, cell) * alpha
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return _img(_bilinear(arr, ys + dy, xs + dx))


def stroke(img: Image.Image, rng: random.Random, heavy: bool) -> Image.Image:
    """Tebal-tipis goresan: tinta meluber di cetakan lama, atau tipis/pudar.

    Menipiskan hanya dengan kernel 3: kernel lebih besar menghapus goresan tipis
    (label tidak lagi sesuai citra).
    """
    if rng.random() < 0.6:
        return img.filter(ImageFilter.MinFilter(rng.choice((3, 5) if heavy else (3,))))
    return img.filter(ImageFilter.MaxFilter(3))


def contrast(img: Image.Image, rng: random.Random, heavy: bool) -> Image.Image:
    """Tinta tidak selalu hitam, latar tidak selalu putih."""
    ink = rng.uniform(0, 100 if heavy else 60)
    paper = rng.uniform(160 if heavy else 190, 255)
    return _img(ink + (paper - ink) * (_arr(img) / 255.0))


def texture(img: Image.Image, rng: random.Random, heavy: bool) -> Image.Image:
    """Bayangan tidak rata dan butiran kertas."""
    arr = _arr(img)
    h, w = arr.shape
    np_rng = _np_rng(rng)
    strength = rng.uniform(0.05, 0.3 if heavy else 0.15)
    low = (_smooth_field(np_rng, h, w, max(4, h // 2)) + 1) / 2
    grain = np_rng.normal(0, 1, (h, w)).astype(np.float32)
    return _img(arr * (1.0 - strength * low) + grain * 40 * strength)


def blur(img: Image.Image, rng: random.Random, heavy: bool) -> Image.Image:
    """Fokus meleset (gaussian) atau tangan bergerak (motion horizontal)."""
    s = _scale(img)
    if rng.random() < 0.5:
        return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6 if heavy else 1.0) * s))
    k = max(2, round(rng.uniform(1.5, 5 if heavy else 3) * s))
    arr = _arr(img)
    shifted = sum(np.roll(arr, shift, axis=1) for shift in range(-(k // 2), k - k // 2))
    return _img(shifted / k)


def noise(img: Image.Image, rng: random.Random, heavy: bool) -> Image.Image:
    """Noise sensor."""
    sigma = rng.uniform(2, 20 if heavy else 10)
    return _img(_arr(img) + _np_rng(rng).normal(0, sigma, (img.height, img.width)))


def binarize(img: Image.Image, rng: random.Random, heavy: bool) -> Image.Image:
    """Pindaian hitam-putih: ambang global, tepi goresan jadi bergerigi.

    Ambang ditaruh relatif terhadap level tinta dan latar citra saat itu, sehingga
    tetap bekerja setelah contrast/texture/blur.
    """
    arr = _arr(img)
    ink, paper = float(np.percentile(arr, 1)), float(np.percentile(arr, 50))
    if paper - ink < 16:
        return img
    threshold = ink + (paper - ink) * rng.uniform(0.35, 0.75 if heavy else 0.6)
    return _img(np.where(arr < threshold, 0.0, 255.0))


def speckle(img: Image.Image, rng: random.Random, heavy: bool) -> Image.Image:
    """Bintik debu/raster pindaian: titik hitam di latar, titik putih di tinta."""
    arr = _arr(img)
    h, w = arr.shape
    np_rng = _np_rng(rng)
    dots = np_rng.random((h, w)) < rng.uniform(0.002, 0.02 if heavy else 0.008)
    if rng.random() < 0.5:  # bintik 2x2, lebih mirip raster cetak
        dots = dots | np.roll(dots, 1, axis=0) | np.roll(dots, 1, axis=1)
    dark = arr < 128
    out = arr.copy()
    out[dots & ~dark] = 0.0
    out[dots & dark] = 255.0
    return _img(out)


def jpeg(img: Image.Image, rng: random.Random, heavy: bool) -> Image.Image:
    """Artefak kompresi (foto ponsel, dokumen hasil unduhan)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=rng.randint(15 if heavy else 40, 85))
    buf.seek(0)
    return Image.open(buf).convert("L")


OPS = {
    "margin": margin,
    "tight": tight,
    "rotate": rotate,
    "elastic": elastic,
    "stroke": stroke,
    "contrast": contrast,
    "texture": texture,
    "blur": blur,
    "noise": noise,
    "binarize": binarize,
    "speckle": speckle,
    "jpeg": jpeg,
}
PHYSICAL_ORDER = list(OPS)
# Definisi G2 ("augmentasi berat"): JANGAN diubah, supaya angka G2 antar-run sebanding.
# Operasi yang ditambahkan belakangan tidak mengonsumsi angka acak preset ini.
PRESET_OPS = ["margin", "rotate", "elastic", "contrast", "texture", "blur", "noise", "jpeg"]
# Preset training Fase 5: semua operasi, termasuk yang meniru pindaian nyata
# (tight, stroke, binarize, speckle). margin & tight sama-sama p=0.5; tight berlaku
# setelah margin, jadi ~50% sampel rapat, ~25% bermargin lebar.
SCAN_OPS = PHYSICAL_ORDER


class Augment:
    """Komposisi operasi; tiap operasi aktif dengan peluang `p`. Picklable untuk worker."""

    def __init__(self, names: list[str], p: float, heavy: bool):
        unknown = set(names) - set(OPS)
        if unknown:
            raise ValueError(f"Augmentasi tidak dikenal: {sorted(unknown)}; pilihan: {PHYSICAL_ORDER}")
        self.names = [n for n in PHYSICAL_ORDER if n in names]
        self.p = p
        self.heavy = heavy

    def __call__(self, img: Image.Image, rng: random.Random) -> Image.Image:
        img = img.convert("L")
        for name in self.names:
            if rng.random() < self.p:
                img = OPS[name](img, rng, self.heavy)
        return img

    def __repr__(self) -> str:
        return f"Augment({'+'.join(self.names)}, p={self.p}, heavy={self.heavy})"


def build_augment(spec: str) -> Augment | None:
    """'none' | 'heavy' (G2, p=1) | 'train' (op G2, p=0.5) | 'fase5' (semua op, p=0.5) | 'rotate+noise+...' (p=0.5)."""
    if spec in ("", "none"):
        return None
    if spec == "fase5":
        return Augment(SCAN_OPS, p=0.5, heavy=True)
    if spec == "heavy":
        return Augment(PRESET_OPS, p=1.0, heavy=True)
    if spec == "train":
        return Augment(PRESET_OPS, p=0.5, heavy=True)
    return Augment(spec.split("+"), p=0.5, heavy=True)


def main() -> None:
    from src.dataset import TRAIN_FONTS
    from src.render import render_line

    line = "ꦧꦱ ꦗꦮ ꦲꦶꦏꦸ ꦧꦱ ꦏꦁ ꦢꦶꦲꦼꦁꦒꦺꦴ ꦢꦼꦤꦶꦁ ꦮꦺꦴꦁ ꦗꦮ꧉"
    label_font = ImageFont.load_default(size=16)
    rows = [("asli", None)] + [(n, Augment([n], 1.0, True)) for n in PHYSICAL_ORDER]
    rows += [("heavy #1", build_augment("heavy")), ("heavy #2", build_augment("heavy")), ("heavy #3", build_augment("heavy"))]
    tiles = []
    for i, (label, aug) in enumerate(rows):
        img = render_line(line, TRAIN_FONTS[i % len(TRAIN_FONTS)], 64)
        if aug is not None:
            img = aug(img, random.Random(i))
        tiles.append((label, img))
    width = 140 + max(t.width for _, t in tiles) + 20
    height = sum(t.height + 20 for _, t in tiles)
    sheet = Image.new("L", (width, height), 200)
    y = 0
    for label, tile in tiles:
        ImageDraw.Draw(sheet).text((10, y + 10), label, font=label_font, fill=0)
        sheet.paste(tile, (140, y + 10))
        y += tile.height + 20
    EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(EXAMPLES_PATH)
    print(f"contoh augmentasi: {EXAMPLES_PATH}")


if __name__ == "__main__":
    main()
