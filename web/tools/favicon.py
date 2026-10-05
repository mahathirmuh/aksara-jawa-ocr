"""Buat ikon situs (favicon.svg + favicon.ico) dari aksara ha (U+A9B2) Noto Sans Javanese di atas kotak biru.
Garis luar glyph diambil lewat uharfbuzz, jadi SVG tidak bergantung pada font yang terpasang di browser.
Pakai (dari akar repo): .venv/Scripts/python web/tools/favicon.py fonts/NotoSansJavanese-Regular.ttf web/public"""
import io
import sys

import uharfbuzz as hb
from PIL import Image, ImageDraw, ImageFont

font_path, out_dir = sys.argv[1], sys.argv[2]
BLUE = '#3b82f6'
SIZE = 64          # kanvas SVG
RADIUS = 14
BOX = 38           # sisi kotak tempat glyph dipaskan

blob = hb.Blob.from_file_path(font_path)
face = hb.Face(blob)
font = hb.Font(face)
gid = font.get_nominal_glyph(0xA9B2)
assert gid, 'glyph ha tidak ada di font'

cmds = []
funcs = hb.DrawFuncs()
funcs.set_move_to_func(lambda x, y, _: cmds.append(('M', x, y)))
funcs.set_line_to_func(lambda x, y, _: cmds.append(('L', x, y)))
funcs.set_quadratic_to_func(lambda cx, cy, x, y, _: cmds.append(('Q', cx, cy, x, y)))
funcs.set_cubic_to_func(lambda c1x, c1y, c2x, c2y, x, y, _: cmds.append(('C', c1x, c1y, c2x, c2y, x, y)))
funcs.set_close_path_func(lambda _: cmds.append(('Z',)))
font.draw_glyph(gid, funcs)

xs = [v for c in cmds for v in c[1::2]]
ys = [v for c in cmds for v in c[2::2]]
x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
scale = BOX / max(x1 - x0, y1 - y0)
dx = (SIZE - (x1 - x0) * scale) / 2 - x0 * scale
dy = (SIZE + (y1 - y0) * scale) / 2 + y0 * scale     # sumbu y font ke atas, SVG ke bawah


def pt(x, y):
    return f'{x * scale + dx:.2f} {dy - y * scale:.2f}'


d = ' '.join(c[0] + ' '.join(pt(c[i], c[i + 1]) for i in range(1, len(c), 2)) for c in cmds)
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}">'
       f'<rect width="{SIZE}" height="{SIZE}" rx="{RADIUS}" fill="{BLUE}"/>'
       f'<path fill="#fff" d="{d}"/></svg>\n')
io.open(out_dir + '/favicon.svg', 'w', encoding='utf-8', newline='\n').write(svg)

# ICO untuk browser yang meminta /favicon.ico: gambar raster dari font yang sama.
big = 256
img = Image.new('RGBA', (big, big), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)
draw.rounded_rectangle([0, 0, big - 1, big - 1], radius=round(RADIUS * big / SIZE), fill=BLUE)
pil_font = ImageFont.truetype(font_path, 10)
l, t, r, b = pil_font.getbbox(chr(0xA9B2))
px = int(10 * (BOX * big / SIZE) / max(r - l, b - t))
pil_font = ImageFont.truetype(font_path, px)
l, t, r, b = pil_font.getbbox(chr(0xA9B2))
draw.text(((big - (r - l)) / 2 - l, (big - (b - t)) / 2 - t), chr(0xA9B2), font=pil_font, fill='#ffffff')
img.save(out_dir + '/favicon.ico', sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
img.resize((180, 180), Image.LANCZOS).save(out_dir + '/apple-touch-icon.png')
print('glyph', gid, 'perintah', len(cmds), 'svg', len(svg), 'byte')
