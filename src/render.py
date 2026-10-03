"""Render satu baris teks aksara Jawa ke citra grayscale.

Shaping OpenType — pasangan menumpuk, taling pindah ke kiri — hanya terjadi
lewat layout engine RAQM. Tanpa RAQM, Pillow tetap merender tanpa error,
glyph hanya berjajar apa adanya. Karena kegagalannya senyap, engine
diperiksa eksplisit, bukan diasumsikan.
"""

import unicodedata
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, features

from src.tokenizer import logical_syllables

DEFAULT_SIZE = 64
DEFAULT_PAD = 8


class RenderClipped(RuntimeError):
    """Glyph tetap menyentuh tepi kanvas walau kanvas sudah diperbesar."""


class EmptyRender(ValueError):
    """Render tidak menghasilkan tinta sama sekali."""


def assert_raqm() -> None:
    # raise, bukan `assert`: `python -O` membuang pernyataan assert.
    if not features.check("raqm"):
        raise RuntimeError(
            "RAQM tidak aktif di Pillow ini — shaping aksara Jawa akan salah "
            "secara senyap. Lihat PLAN.md §4.4."
        )


# 16 font x 17 ukuran render: cache kecil membuat worker memuat ulang file font terus.
@lru_cache(maxsize=512)
def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    assert_raqm()
    font = ImageFont.truetype(font_path, size, layout_engine=ImageFont.Layout.RAQM)
    # Pillow jatuh ke BASIC diam-diam kalau RAQM tidak bisa dipakai.
    if font.layout_engine != ImageFont.Layout.RAQM:
        raise RuntimeError(f"{font_path}: layout engine bukan RAQM")
    return font


def load_font(font_path: str | Path, size: int = DEFAULT_SIZE) -> ImageFont.FreeTypeFont:
    return _load_font(str(font_path), size)


def syllable_origins(text: str, font: ImageFont.FreeTypeFont) -> list[tuple[str, float]]:
    """Suku kata ortografis beserta posisi awalnya (piksel) dalam tata letak baris utuh.

    Posisi diambil dari panjang awalan baris, bukan dari jumlah advance tiap suku kata, supaya kerning
    kontekstual antar suku kata tetap ikut (Noto & Tuladha merenggangkan suku + pasangan, GumregahNew
    merapatkan lebih dari separuh pasangan suku kata). Menggambar tiap suku kata di posisi ini menghasilkan
    citra yang sama dengan menggambar barisnya sekaligus.
    """
    origins, prefix = [], ""
    for syllable in logical_syllables(text):
        prefix += syllable
        origins.append((syllable, font.getlength(prefix) - font.getlength(syllable)))
    return origins


def render_with_font(
    text: str, font: ImageFont.FreeTypeFont, pad: int = DEFAULT_PAD, tracking: float = 0.0
) -> Image.Image:
    """Render `text` dengan objek font apa adanya, TANPA memeriksa layout engine.

    Pakai `render_line` untuk semua keperluan normal. Fungsi ini terbuka hanya
    supaya verify_shaping bisa membuat pembanding tanpa shaping (BASIC).

    `tracking` (em) = jarak tambahan sesudah tiap suku kata, termasuk spasi: cetakan Jawa tanpa spasi
    diratakan dengan merenggangkan aksara, dan model yang hanya melihat jarak bawaan font membaca
    renggang itu sebagai spasi kata. 0 = satu panggilan gambar seperti semula.
    """
    text = unicodedata.normalize("NFC", text)
    size = font.size
    if tracking > 0:
        step = tracking * size
        parts = [(syllable, x + i * step) for i, (syllable, x) in enumerate(syllable_origins(text, font))]
        extra = step * max(0, len(parts) - 1)
    else:
        parts, extra = [(text, 0)], 0
    # Tumpukan pasangan turun jauh di bawah baseline dan sandhangan naik di atasnya.
    # Kata serapan di korpus bisa menghasilkan rantai 5-6 pasangan, jadi kanvas
    # diperbesar bertahap sampai tinta tidak menyentuh tepi, lalu dipotong ke tinta.
    for margin_em in (3, 6, 12):
        margin = margin_em * size
        width = int(font.getlength(text) + extra) + 2 * margin
        height = 2 * margin
        canvas = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for part, x in parts:
            draw.text((margin + x, margin), part, font=font, fill=255, anchor="ls")

        bbox = canvas.getbbox()
        if bbox is None:
            raise EmptyRender(f"Render kosong untuk {text!r} — font tidak punya glyph-nya?")
        left, top, right, bottom = bbox
        if left > 0 and top > 0 and right < width and bottom < height:
            break
    else:
        raise RenderClipped(f"Glyph terpotong tepi kanvas saat merender {text!r}")

    ink = canvas.crop(bbox)
    out = Image.new("L", (ink.width + 2 * pad, ink.height + 2 * pad), 0)
    out.paste(ink, (pad, pad))
    return ImageOps.invert(out)


def render_line(
    text: str,
    font_path: str | Path,
    size: int = DEFAULT_SIZE,
    pad: int = DEFAULT_PAD,
    tracking: float = 0.0,
) -> Image.Image:
    """Render satu baris teks jadi citra grayscale mode "L", hitam di atas putih.

    Citra dipotong ketat ke batas tinta lalu diberi `pad` piksel di tiap sisi.
    Tinggi belum dinormalisasi ke H=96 — itu tugas dataset (Fase 3).
    """
    return render_with_font(text, load_font(font_path, size), pad, tracking)
