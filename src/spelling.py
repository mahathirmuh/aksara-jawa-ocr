"""Ejaan baku untuk baris korpus (hanya data training dan model bahasa).

Korpus ditransliterasi dengan konvensi `scripts/translit/translit.js`, sedangkan cetakan memakai ejaan baku.
Terukur 2026-10-13: di 300.000 baris latih nya + pangkon + ca/ja 0 kali (na + pangkon + ja 14.741), pa cerek 33,
nga lelet 26, keret 30, dan 40.822 bilangan tanpa pada pangkat; di 745 label nyata pa cerek, nga lelet, dan keret
terbaca 0%, dan model bahasa mengganti nya di depan ca/ja yang sudah benar.

`standardize` menulis ulang teks aksara SEBELUM dirender, jadi citra dan label tetap sama persis. Aturannya sama
dengan ejaan baku `web/app/Support/AksaraWriter.php` (yang dicocokkan dengan ejaan kamus), diterapkan pada aksara:

  nya     na + pangkon + ca/ja      -> nya + pangkon + ca/ja      (di dalam kata; spasi di antaranya = dua kata)
  cerek   ra + pepet                -> pa cerek                   (bukan sesudah pangkon: pasangan tetap ra berpepet)
  lelet   la + pepet                -> nga lelet                  (sama)
  keret   cakra + pepet             -> keret
  pangkat deret angka               -> diapit pada pangkat

Tiap aturan hanya dipakai bila font perender punya glyph hasilnya (`covered`), dan hasil yang tidak well-formed atau
tidak lolos round-trip tokenizer dibuang (teks asal dipertahankan).
"""

import re
from typing import Callable, Collection

NA, NYA, CA, JA, RA, LA = "ꦤ", "ꦚ", "ꦕ", "ꦗ", "ꦫ", "ꦭ"
PANGKON, PEPET, CAKRA, KERET = "꧀", "ꦼ", "ꦿ", "ꦽ"
PA_CEREK, NGA_LELET, PADA_PANGKAT = "ꦉ", "ꦊ", "꧇"
DIGITS = "꧐-꧙"

# (nama, pola, pengganti, codepoint yang harus ada di font)
RULES: tuple[tuple[str, re.Pattern, str, str], ...] = (
    ("nya", re.compile(f"{NA}(?={PANGKON}[{CA}{JA}])"), NYA, NYA),
    ("cerek", re.compile(f"(?<!{PANGKON}){RA}{PEPET}"), PA_CEREK, PA_CEREK),
    ("lelet", re.compile(f"(?<!{PANGKON}){LA}{PEPET}"), NGA_LELET, NGA_LELET),
    ("keret", re.compile(f"{CAKRA}{PEPET}"), KERET, KERET),
    # Deret angka (boleh berpemisah titik, koma, titik dua di antara angka: 2,5 atau 10:30) yang belum berpangkat.
    ("pangkat", re.compile(f"(?<![{DIGITS}{PADA_PANGKAT}])([{DIGITS}]+(?:[.,:][{DIGITS}]+)*)(?![{DIGITS}{PADA_PANGKAT}])"),
     f"{PADA_PANGKAT}\\1{PADA_PANGKAT}", PADA_PANGKAT),
)
RULE_NAMES = tuple(rule[0] for rule in RULES)
# Contoh sebelum dan sesudah tiap aturan penulisan ulang aksara, untuk memeriksa apakah sebuah font merender keduanya
# KEMBAR (font yang meligaturkan cakra + pepet menjadi keret). Terukur 2026-10-13 di lingkungan training: hanya
# GumregahNew, aturan keret; di `.venv` BasaJan juga kembar untuk cerek, lelet, dan keret.
PROBES = {"nya": ("ꦥꦤ꧀ꦕ", "ꦥꦚ꧀ꦕ"), "cerek": ("ꦫꦼ", "ꦉ"), "lelet": ("ꦭꦼ", "ꦊ"), "keret": ("ꦏꦿꦼ", "ꦏꦽ")}


def standardize(text: str, covered: frozenset[int] | None = None, valid: Callable[[str], bool] | None = None,
                only: Collection[str] | None = None) -> str:
    """Teks dengan ejaan baku. `covered` = codepoint di cmap font perender (None = semua ada); `valid(teks)` = teks
    boleh dipakai (well-formed dan lolos round-trip); `only` = nama aturan yang dipakai (None = semua). Aturan yang
    hasilnya tidak sah dilewati satu per satu."""
    for name, pattern, replacement, needs in RULES:
        if (only is not None and name not in only) or (covered is not None and ord(needs) not in covered):
            continue
        new = pattern.sub(replacement, text)
        if new != text and (valid is None or valid(new)):
            text = new
    return text
