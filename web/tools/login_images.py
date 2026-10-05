"""Siapkan gambar halaman masuk dari Wikimedia Commons (lisensi bebas): tiga foto panel kiri dan satu latar.

Sumber dan lisensi tercatat di SOURCES di bawah dan disalin ke public/img/login/SUMBER.md. Berkas asli diunduh ke
folder sementara, dipotong ke rasio panel (10:11), lalu diperkecil; latar = foto naskah yang diburamkan dan
digelapkan. Pakai (dari akar repo): .venv/Scripts/python web/tools/login_images.py web/public/img/login
"""
import io
import os
import sys
import tempfile
import urllib.request

from PIL import Image, ImageEnhance, ImageFilter

COMMONS = 'https://upload.wikimedia.org/wikipedia/commons/'
PAGE = 'https://commons.wikimedia.org/wiki/File:'

# kunci: (berkas hasil, jalur berkas asli, halaman Commons, pusat potongan x/y 0..1, judul, pembuat, lisensi)
SOURCES = [
    ('naskah-sonobudoyo.jpg', '7/79/Naskah_aksara_Jawa_koleksi_Museum_Sonobudoyo.jpg',
     'Naskah_aksara_Jawa_koleksi_Museum_Sonobudoyo.jpg', (0.66, 0.5),
     'Naskah beraksara Jawa, koleksi Museum Sonobudoyo, Yogyakarta', 'Candramawa99', 'CC0 1.0'),
    ('serat-damar-wulan.jpg', '9/99/Serat_Damar_Wulan_%28page_25_crop%29.jpg',
     'Serat_Damar_Wulan_(page_25_crop).jpg', (0.40, 0.5),
     'Serat Damar Wulan, akhir abad ke-18 (British Library, MSS Jav 89)', 'British Library', 'domain publik'),
    ('kraton-yogyakarta.jpg', '9/92/Jogja_-_Kraton_Yogyakarta_-_Donopratono_gate_%282025%29_-_img_01.jpg',
     'Jogja_-_Kraton_Yogyakarta_-_Donopratono_gate_(2025)_-_img_01.jpg', (0.5, 0.5),
     'Gerbang Donopratono, Kraton Yogyakarta', 'Chainwit.', 'CC BY 4.0'),
]
RATIO = 10 / 11          # lebar : tinggi panel kiri kartu masuk
SLIDE = (1200, 1320)     # ~1,9x ukuran tampil (642 x 704 px) supaya tetap tajam di layar rapat
BACKDROP_WIDTH = 1600


def fetch(path: str, cache: str) -> Image.Image:
    local = os.path.join(cache, os.path.basename(path).replace('%', '_'))
    if not os.path.exists(local):
        request = urllib.request.Request(COMMONS + path, headers={'User-Agent': 'AksaraOCRLab/1.0 (riset lokal)'})
        with urllib.request.urlopen(request, timeout=300) as response, io.open(local, 'wb') as out:
            out.write(response.read())
    return Image.open(local).convert('RGB')


def crop_to_ratio(image: Image.Image, centre: tuple[float, float]) -> Image.Image:
    width, height = image.size
    crop_h = height
    crop_w = crop_h * RATIO
    if crop_w > width:
        crop_w, crop_h = width, width / RATIO
    left = min(max(centre[0] * width - crop_w / 2, 0), width - crop_w)
    top = min(max(centre[1] * height - crop_h / 2, 0), height - crop_h)
    return image.crop((round(left), round(top), round(left + crop_w), round(top + crop_h)))


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(tempfile.gettempdir(), 'aksara-login-src')
    os.makedirs(cache, exist_ok=True)
    notes = ['# Sumber gambar halaman masuk', '',
             'Dibuat `web/tools/login_images.py` dari Wikimedia Commons; dipotong dan diperkecil.', '',
             '| Berkas | Isi | Pembuat | Lisensi | Sumber |', '|---|---|---|---|---|']
    for name, path, page, centre, title, author, licence in SOURCES:
        image = fetch(path, cache)
        slide = crop_to_ratio(image, centre).resize(SLIDE, Image.LANCZOS)
        slide.save(os.path.join(out_dir, name), 'JPEG', quality=80, optimize=True, progressive=True)
        notes.append(f'| `{name}` | {title} | {author} | {licence} | <{PAGE}{page}> |')
        print(name, image.size, '->', slide.size, os.path.getsize(os.path.join(out_dir, name)) // 1024, 'KB')
        if name.startswith('naskah'):
            ratio = BACKDROP_WIDTH / image.width
            backdrop = image.resize((BACKDROP_WIDTH, round(image.height * ratio)), Image.LANCZOS)
            backdrop = ImageEnhance.Brightness(backdrop.filter(ImageFilter.GaussianBlur(10))).enhance(0.5)
            backdrop.save(os.path.join(out_dir, 'latar.jpg'), 'JPEG', quality=70, optimize=True, progressive=True)
            notes.append(f'| `latar.jpg` | Foto naskah yang sama, diburamkan dan digelapkan | {author} | {licence} | <{PAGE}{page}> |')
            print('latar.jpg', backdrop.size, os.path.getsize(os.path.join(out_dir, 'latar.jpg')) // 1024, 'KB')
    io.open(os.path.join(out_dir, 'SUMBER.md'), 'w', encoding='utf-8', newline='\n').write('\n'.join(notes) + '\n')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join('web', 'public', 'img', 'login'))
