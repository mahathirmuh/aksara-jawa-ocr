"""Bangun berkas kamus kata untuk halaman Kamus: bahasa Jawa (jv) dan bahasa Indonesia (id).

Semua isinya data pihak ketiga yang diunduh skrip ini; tidak ada yang dihitung proyek. Keluaran (bawaan
web/database/dictionaries/), dibaca `php artisan aksara:dictionary`:
  sources.json    sumber tiap kamus: nama, lisensi, tautan, tanggal data diambil
  jv.jsonl.gz     satu entri per baris: word, glosses[], gloss_lang, source, lalu pos / aksara / register / note /
                  examples[] / url bila ada. Dua sumber: Wiktionary bahasa Inggris (arti berbahasa Inggris, ejaan
                  aksara, ragam) dan bagian bahasa Jawa Wikikamus (arti berbahasa Indonesia, contoh kalimat)
  id.jsonl.gz

Unduhan mentah disimpan di web/storage/app/dictionaries/ (tidak ikut git) dan dipakai ulang; --refresh mengunduh lagi.
Hanya pustaka bawaan Python. Pakai (dari akar repo):
  .venv/Scripts/python web/tools/dictionaries.py
  cd web && php artisan aksara:dictionary
"""
import argparse
import bz2
import datetime
import gzip
import json
import os
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from collections import Counter

USER_AGENT = 'aksara-ocr-lab/1.0 (riset non-komersial; pembangun kamus halaman Kamus)'
JAVA = re.compile('[ꦀ-꧟]')
JAVA_RUN = re.compile('[ꦀ-꧟‌]+(?: [ꦀ-꧟‌]+)*')

# Kelas kata wiktextract -> istilah Indonesia yang dipakai halaman Kamus.
POS = {
    'noun': 'nomina', 'verb': 'verba', 'adj': 'adjektiva', 'adv': 'adverbia', 'name': 'nama diri', 'num': 'numeralia',
    'conj': 'konjungsi', 'pron': 'pronomina', 'det': 'pewatas', 'intj': 'interjeksi', 'particle': 'partikel',
    'prep': 'preposisi', 'prefix': 'awalan', 'suffix': 'akhiran', 'circumfix': 'konfiks', 'root': 'akar kata',
    'phrase': 'frasa', 'character': 'aksara', 'article': 'artikula', 'punct': 'tanda baca', 'proverb': 'peribahasa',
    'classifier': 'kata penggolong', 'prep_phrase': 'frasa preposisi', 'symbol': 'simbol', 'infix': 'sisipan',
    'postp': 'posposisi', 'abbrev': 'singkatan', 'affix': 'imbuhan', 'contraction': 'kontraksi', 'interfix': 'interfiks',
}


def fetch(url: str, path: str, refresh: bool) -> str:
    """Unduh `url` ke `path` kecuali sudah ada. Mengembalikan tanggal berkas (YYYY-MM-DD) sebagai tanggal data diambil."""
    if refresh or not os.path.isfile(path) or os.path.getsize(path) == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print('mengunduh', url)
        request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        with urllib.request.urlopen(request, timeout=600) as response, open(path + '.part', 'wb') as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        os.replace(path + '.part', path)
    return datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat()


def nfc(text: str) -> str:
    return unicodedata.normalize('NFC', ' '.join(text.split()))


def plain(text: str) -> str:
    """Kunci pembanding: huruf kecil tanpa diakritik Latin (sama maksudnya dengan DictionaryEntry::normalize di PHP)."""
    text = ''.join(ch for ch in unicodedata.normalize('NFD', text) if unicodedata.category(ch) != 'Mn' or JAVA.match(ch))
    return ' '.join(text.lower().split())


# Induk arti bersarang yang lebih panjang dari ini ditulis sekali sebagai arti sendiri (median induk 9 karakter,
# persentil 95 = 32; yang lebih panjang berupa kalimat penuh).
CHAIN_LIMIT = 40


def sense_glosses(sense: dict) -> list[str]:
    """Arti satu sense wiktextract, dengan label pemakaiannya ("(krama) water").

    Sense bersarang disimpan sebagai rantai [arti induk, ..., arti sendiri], mis. ["house:", "abode"]. Induk yang
    pendek dirangkai dengan anaknya ("house: abode") supaya arti anak tidak hilang dan terbaca dalam konteksnya.
    Induk yang panjang ditulis sekali sebagai arti sendiri dan anaknya menyusul (pemanggil membuang ulangannya);
    induk yang hanya label ("(by extension)") dirangkai dengan spasi.
    """
    chain = [nfc(part).rstrip(' :,;') for part in (sense.get('raw_glosses') or sense.get('glosses') or [])]
    chain = list(dict.fromkeys(part for part in chain if part))
    # Sisa templat yang gagal diurai ekstraksi sumber ("{{place|en|...", ":Template:initialism of") bukan arti.
    if any('{{' in part or ':Template:' in part for part in chain):
        return []
    if len(chain) < 2:
        return chain
    out, lead, inherited, labels = [], '', '', []
    for parent in chain[:-1]:
        short = parent.rstrip('.')
        label = re.match(r'\([^()]*\)\s*', short)
        body = short[label.end():] if label else short
        labels += [label.group().strip()] if label else []
        if not body:                            # induk hanya label
            lead += short + ' '
        elif len(body) <= CHAIN_LIMIT:          # panjang diukur tanpa labelnya
            lead += short + ': '
        else:
            out.append(lead + parent)
            lead, inherited = '', label.group().strip() + ' ' if label else ''
    own = chain[-1]
    if re.fullmatch(r'\([^()]*\)', own):       # anak hanya label: "nucleus (chemistry, nuclear physics)"
        return out + [(lead.rstrip(': ') + ' ' + own).strip()]
    if own.startswith('('):
        # Anak yang berlabel sendiri tidak mewarisi label induk, dan label yang sama dengan induknya ditulis sekali.
        inherited = ''
        repeated = next((label for label in labels if own.startswith(label + ' ')), None)
        if repeated and lead.strip() == repeated:
            lead = ''
        elif repeated and lead:
            own = own[len(repeated):].lstrip()
    return out + [inherited + lead + own]


# ---------- bahasa Jawa dari Wiktionary bahasa Inggris (ekstraksi kaikki.org / wiktextract) ----------

KAIKKI_JV = 'https://kaikki.org/dictionary/Javanese/kaikki.org-dictionary-Javanese.jsonl'
# Ragam yang bisa menjadi ragam sebuah entri.
REGISTER_ORDER = ['ngoko', 'krama-ngoko', 'madya', 'krama', 'krama inggil', 'krama andhap', 'kawi']
# Urutan di catatan padanan: ditambah "alus", yang di sumber hanya muncul sebagai padanan (ngoko alus, krama alus).
NOTE_ORDER = ['ngoko', 'ngoko alus', 'krama-ngoko', 'madya', 'krama', 'krama alus', 'krama inggil', 'krama andhap', 'kawi']
# Label ragam di argumen templat kepala: {{head|jv|noun|ngoko|omah|krama|griya}} (jv-noun|n=|k=|ki=|kn=|kw= berujung ke situ).
# "Lugu" adalah bentuk biasa ragam itu.
HEAD_REGISTER = {'ngoko': 'ngoko', 'ngoko lugu': 'ngoko', 'ngoko alus': 'ngoko alus', 'krama ngoko': 'krama-ngoko',
                 'krama-ngoko': 'krama-ngoko', 'ngoko-krama': 'krama-ngoko', 'madya': 'madya', 'krama madya': 'madya', 'krama': 'krama',
                 'krama lugu': 'krama', 'krama alus': 'krama alus', 'krama inggil': 'krama inggil', 'krama andhap': 'krama andhap',
                 'kawi': 'kawi'}
# Judul kolom tabel ragam yang tidak terurai wiktextract ("Javanese register set ꦏꦿꦩ : émah ꦔꦺꦴꦏꦺꦴ : omah omah").
SET_REGISTER = {'ꦔꦺꦴꦏꦺꦴ': 'ngoko', 'ꦏꦿꦩ': 'krama', 'ꦏꦿꦩꦲꦶꦁꦒꦶꦭ꧀': 'krama inggil', 'ꦏꦿꦩꦔꦺꦴꦏꦺꦴ': 'krama-ngoko',
                'ꦏꦿꦩꦲꦤ꧀ꦝꦥ꧀': 'krama andhap', 'ꦏꦮꦶ': 'kawi'}
# Label ragam yang ditulis penyunting dalam kata-kata: "(krama inggil) to walk", "krama inggil dalem", "Krama of gedhang."
WORD_REGISTER = {'ngoko': 'ngoko', 'krama': 'krama', 'krama inggil': 'krama inggil', 'krama andhap': 'krama andhap',
                 'krama-ngoko': 'krama-ngoko', 'krama ngoko': 'krama-ngoko', 'madya': 'madya', 'krama madya': 'madya',
                 'kawi': 'kawi'}
REGISTER_WORDS = '|'.join(sorted(map(re.escape, WORD_REGISTER), key=len, reverse=True))
LABELLED = re.compile('^(' + REGISTER_WORDS + ') (.+)$', re.I)
REGISTER_OF = re.compile('^(' + REGISTER_WORDS + ') of ', re.I)
# "dated spelling of endhas", "(Surinamese) alternative form of bangsa", "alternative spelling of ꦧꦶꦩ (Bima)".
VARIANT_OF = re.compile(r'^(?:\([^()]*\) )?(?:\w+ )?(?:spelling|form) of (?:[ꦀ-꧟\u200c]+ \()?([^\s(),.;:“”"]+)')
SPELLING_OF = re.compile(r'^(?:Carakan|Javanese script) spelling of (.+?)\.?$')
SET_COLUMN = re.compile(r'([ꦀ-꧟]+) : (.*?)(?= [ꦀ-꧟]+ : |$)')


# Bunyi aksara untuk pembanding kasar (bukan transliterator: hanya untuk menolak ejaan yang jelas milik kata lain).
AKSARA_CONSONANT = {
    'ꦏ': 'k', 'ꦐ': 'k', 'ꦑ': 'k', 'ꦒ': 'g', 'ꦓ': 'g', 'ꦔ': 'ng', 'ꦕ': 'c', 'ꦖ': 'c', 'ꦗ': 'j', 'ꦘ': 'ny', 'ꦙ': 'j', 'ꦚ': 'ny',
    'ꦛ': 'th', 'ꦜ': 'th', 'ꦝ': 'dh', 'ꦞ': 'dh', 'ꦟ': 'n', 'ꦠ': 't', 'ꦡ': 't', 'ꦢ': 'd', 'ꦣ': 'dh', 'ꦤ': 'n', 'ꦥ': 'p', 'ꦦ': 'p',
    'ꦧ': 'b', 'ꦨ': 'b', 'ꦩ': 'm', 'ꦪ': 'y', 'ꦫ': 'r', 'ꦬ': 'r', 'ꦭ': 'l', 'ꦮ': 'w', 'ꦯ': 's', 'ꦰ': 's', 'ꦱ': 's', 'ꦲ': 'h',
}
AKSARA_VOWEL = {'ꦴ': 'a', 'ꦵ': 'o', 'ꦶ': 'i', 'ꦷ': 'i', 'ꦸ': 'u', 'ꦹ': 'u', 'ꦺ': 'e', 'ꦻ': 'ai', 'ꦼ': 'e'}
AKSARA_OTHER = {'ꦄ': 'a', 'ꦅ': 'i', 'ꦆ': 'i', 'ꦇ': 'i', 'ꦈ': 'u', 'ꦌ': 'e', 'ꦍ': 'ai', 'ꦎ': 'o', 'ꦉ': 're', 'ꦊ': 'le',
                'ꦀ': 'm', 'ꦁ': 'ng', 'ꦂ': 'r', 'ꦃ': 'h', 'ꦿ': 'r', 'ꦾ': 'y', 'ꦽ': 're'}


def reads_as(aksara: str, word: str) -> bool:
    """Apakah ejaan aksara ini bisa dibaca sebagai kata Latin itu, dengan toleransi konvensi ejaan.

    Yang diabaikan: tanda diakritik, h (ꦲ sering tak berbunyi), dh/th lawan d/t, huruf rangkap, luncuran y/w, ny di depan
    c/j, f/v, dan a/o di akhir kata (Herjuna = Herjuno). Yang tidak: vokal dan konsonan lain.
    """
    sounds, index = [], 0
    while index < len(aksara):
        char = aksara[index]
        index += 1
        if char in AKSARA_CONSONANT:
            consonant, vowel = AKSARA_CONSONANT[char], 'a'
            if aksara[index:index + 1] == '꦳':                # cecak telu: aksara rekan (ꦥ꦳ = f/v)
                consonant, index = {'p': 'f', 'w': 'f', 'j': 'z'}.get(consonant, consonant), index + 1
            while aksara[index:index + 1] in ('ꦿ', 'ꦾ', 'ꦽ'):    # cakra, pengkal, keret
                consonant, vowel = consonant + AKSARA_OTHER[aksara[index]][0], 'e' if aksara[index] == 'ꦽ' else vowel
                index += 1
            signs = ''
            while aksara[index:index + 1] in AKSARA_VOWEL or aksara[index:index + 1] == '꧀':
                signs, index = signs + aksara[index], index + 1
            if '꧀' in signs:
                vowel = ''
            elif signs:
                vowel = 'o' if 'ꦺ' in signs and 'ꦴ' in signs else ''.join(AKSARA_VOWEL[sign] for sign in signs)
            sounds.append(consonant + vowel)
        else:
            sounds.append(AKSARA_OTHER.get(char, ''))

    def frame(latin: str) -> str:
        latin = re.sub('[^a-z]', '', plain(latin)).replace('dh', 'd').replace('th', 't').replace('v', 'f').replace('h', '')
        latin = re.sub('ny(?=[cj])', 'n', latin).replace('ngng', 'ng')
        latin = re.sub('(?<=[ie])y(?=[aeiou])|(?<=[uo])w(?=[aeiou])', '', latin)
        return re.sub('o$', 'a', re.sub(r'(.)\1+', r'\1', latin))

    return frame(''.join(sounds)) == frame(word)


def tidy_latin(word: str | None) -> str | None:
    """Rapikan romanisasi otomatis di kepala entri: nya di depan ca/ja ditulis "n" (ꦗꦚ꧀ꦕꦸꦏ꧀ = jancuk, bukan janycuk)."""
    return re.sub('ny(?=[cj])', 'n', word) if word else word


def mend(text: str) -> str:
    """Rapatkan tanda aksara yang terlepas dari aksaranya di ekstraksi sumber: "ꦢꦶꦤ ꧀ꦠꦼꦤ꧀" -> "ꦢꦶꦤ꧀ꦠꦼꦤ꧀".

    Sandhangan dan pangkon tidak bisa mengawali kata, jadi spasi di depannya pasti artefak (tautan wiki yang
    berakhir di tengah kata).
    """
    return re.sub(r'(?<=[ꦀ-꧟])\s+(?=[ꦀ-ꦃ꦳-꧀])', '', text)


def head_latin(entry: dict, headword: str) -> str | None:
    """Romanisasi kata kepala menurut kepala entrinya sendiri: "ꦲꦗꦶ (aji)" -> aji, "ꦲꦸꦭꦩ꧀ (ulam, oelam)" -> ulam.

    Satu halaman beraksara bisa memuat beberapa kata (ꦲꦗꦶ = haji dan aji), jadi romanisasi dibaca per entri.
    """
    for head in entry.get('head_templates', []):
        own = re.match(re.escape(headword) + r' \(([^()]+)\)', nfc(head.get('expansion', '')))
        first = own.group(1).split(',')[0].strip() if own else ''
        if first and not JAVA.search(first) and not LABELLED.match(first) and first.lower() not in WORD_REGISTER:
            return first
    return None


def head_registers(entry: dict, headword: str) -> list[str]:
    """Ragam yang disebut kepala entri untuk kata itu sendiri: "ꦲꦒꦼꦁ (ageng) (krama)", "kang (krama, ngoko)".

    Tanda kurung pertama sesudah judul beraksara adalah romanisasinya, bukan label: "ꦏꦿꦩ (krama)" = kata "krama".
    """
    found = []
    for head in entry.get('head_templates', []):
        text = nfc(head.get('expansion', ''))
        if JAVA.search(headword):
            text = re.sub('^' + re.escape(headword) + r' \([^()]*\)', '', text)
        tail = re.search(r'\(([^()]+)\)\s*$', text)
        names = [name.strip().lower() for name in tail.group(1).split(',')] if tail else []
        if names and all(name in WORD_REGISTER for name in names):
            found += [WORD_REGISTER[name] for name in names]
    return list(dict.fromkeys(found))


def spelling_targets(sense: dict) -> list[str]:
    """Kata Latin yang ditunjuk sense "Carakan spelling of ...": "timah (“tin”)" -> [timah], "alis or halis" -> dua kata."""
    pointer = SPELLING_OF.match(nfc((sense.get('glosses') or [''])[0]))
    if not pointer:
        return []
    linked = [nfc(target['word']) for target in sense.get('alt_of', []) if not JAVA.search(target['word'])]
    named = [part.strip() for part in re.split(r' or |[,;]', re.sub(r'\s*\(.*$', '', pointer.group(1))) if part.strip()]
    return list(dict.fromkeys(named + [word for word in linked if ' ' not in word]))


def register_words(text: str) -> list[str]:
    """Kata-kata satu ragam seperti ditulis sumber, dipisah koma, titik koma, atau pada lingsa (koma aksara Jawa).

    Bentuk beraksara yang diikuti bentuk Latin adalah SATU kata dengan romanisasinya (dan ejaan lamanya):
    "ꦕꦶꦭꦶꦏ꧀, cilik, tyilik, tjilik" -> [cilik]; "ꦲꦺꦱ꧀ꦠꦿꦶ (èstri)" -> [èstri]; "ꦧꦺꦴꦕꦃ, bocah; ꦲꦤꦏ꧀, anak" -> [bocah, anak].
    Tanpa romanisasi bentuk beraksaranya dipertahankan: "ꦩꦂꦒ꧈ ꦩꦼꦂꦒ" -> dua kata beraksara.
    """
    words: list[str] = []
    state = 'kata'      # 'aksara' = baru saja bentuk beraksara; 'romanisasi' = romanisasinya sudah diambil
    for item in (part.strip() for part in re.split('[,;꧈]', mend(nfc(text)))):
        paired = re.fullmatch(r'[ꦀ-꧟\u200c -]+\(([^()]+)\)', item)
        if paired:
            words.append(paired.group(1).strip())
            state = 'romanisasi'
        elif JAVA.search(item):
            words.append(item)
            state = 'aksara'
        elif item and state == 'aksara':
            words[-1], state = item, 'romanisasi'
        elif item and state == 'kata':
            words.append(item)
    return words


def register_set(entry: dict, headword: str) -> dict[str, list[str]]:
    """Padanan antar-ragam di kepala entri: {'ngoko': ['mangan'], 'krama': ['nedha'], ...}, ejaan seperti di sumber."""
    found: dict[str, list[str]] = {}
    # 1. Argumen templat kepala: pasangan (label ragam, kata) di posisi ganjil-genap mulai argumen ke-3. Dibaca dari
    #    argumennya, bukan dari teks hasilnya, supaya batas tiap ragam tidak perlu ditebak.
    for head in entry.get('head_templates', []):
        args = head.get('args') or {}
        for key, label in args.items():
            register = HEAD_REGISTER.get(label.strip().lower()) if key.isdigit() and int(key) >= 3 and int(key) % 2 else None
            if register and args.get(str(int(key) + 1), '').strip():
                found.setdefault(register, []).extend(register_words(args[str(int(key) + 1)]))
    # 2. Tabel ragam ("Javanese register set ꦏꦿꦩ : émah ꦔꦺꦴꦏꦺꦴ : omah omah") sampai sebagai bentuk 'canonical', kadang
    #    terpotong di koma menjadi beberapa bentuk.
    table = ', '.join(nfc(form.get('form', '')) for form in entry.get('forms', []) if 'canonical' in form.get('tags', []))
    for part in table.split('Javanese register set')[1:]:
        columns = SET_COLUMN.findall(mend(part.strip()))
        for index, (label, value) in enumerate(columns):
            tokens = value.split()
            # Sesudah kolom terakhir wiktextract menempelkan kata kepalanya sendiri.
            if index == len(columns) - 1 and len(tokens) > 1 and tokens[-1] == headword:
                tokens = tokens[:-1]
            if label in SET_REGISTER:
                found.setdefault(SET_REGISTER[label], []).extend(register_words(' '.join(tokens)))
    return {register: list(dict.fromkeys(words)) for register, words in found.items() if words}


def labelled_register(glosses: list[str]) -> str | None:
    """Ragam dari label di arti, hanya bila SEMUA arti menyebut satu ragam yang sama."""
    seen = set()
    for gloss in glosses:
        labels = re.match(r'^\(([^)]*)\)', gloss)
        mine = {WORD_REGISTER[label.strip().lower()] for label in (labels.group(1).split(',') if labels else [])
                if label.strip().lower() in WORD_REGISTER}
        of = REGISTER_OF.match(gloss)
        if of:
            mine.add(WORD_REGISTER[of.group(1).lower()])
        if mine == {'ngoko', 'krama'}:      # "(ngoko, krama) ..." = dipakai di kedua ragam
            mine = {'krama-ngoko'}
        if len(mine) != 1:
            return None
        seen |= mine
    return seen.pop() if len(seen) == 1 else None


def enwiktionary_javanese(cache: str, refresh: bool):
    path = os.path.join(cache, 'enwiktionary-javanese.jsonl')
    retrieved = fetch(KAIKKI_JV, path, refresh)
    with open(path, encoding='utf-8') as handle:
        raw = [json.loads(line) for line in handle if line.strip()]

    # Halaman penunjuk "romanization of <aksara>": kata Latin -> judul beraksara. Dikenali dari tag sense-nya, karena
    # sebagian halaman penunjuk berkelas kata biasa (pendhidhikan). Satu judul beraksara bisa punya beberapa penunjuk,
    # termasuk ejaan tidak baku ("gedang" untuk gedhang).
    pointers: dict[str, list[str]] = {}
    for entry in raw:
        for sense in entry.get('senses', []):
            if 'romanization' in sense.get('tags', []) and not JAVA.search(entry['word']):
                for target in sense.get('alt_of', []):
                    if JAVA.search(target['word']):
                        pointers.setdefault(nfc(target['word']), []).append(nfc(entry['word']))
    # Romanisasi per JUDUL beraksara, untuk entri yang kepalanya tidak menyebut romanisasi dan untuk anggota tabel ragam.
    # Bentuk "romanization" di forms tidak dipakai sendirian: pada entri bertabel ragam, yang pertama sering
    # romanisasi kata LAIN di tabel itu (geni -> "latu"); penunjuk yang juga tercatat di forms didahulukan.
    latin_of: dict[str, str] = {}
    for entry in raw:
        headword = nfc(entry['word'])
        if not JAVA.search(headword) or headword in latin_of:
            continue
        own = tidy_latin(head_latin(entry, headword))
        romanized = [nfc(form['form']) for form in entry.get('forms', [])
                     if 'romanization' in form.get('tags', []) and not LABELLED.match(nfc(form['form']))]
        candidates = pointers.get(headword, [])
        agreed = [candidate for candidate in candidates if candidate in romanized]
        # Di antara beberapa romanisasi (tabel ragam: "latu, geni"), yang milik judul ini adalah yang terbaca sebagai judulnya.
        reading = list(dict.fromkeys(candidate for candidate in romanized if reads_as(headword, candidate)))
        chosen = own or (agreed or (reading if len(reading) == 1 else []) or (romanized if len(romanized) == 1 else candidates) or [None])[0]
        if chosen:
            latin_of[headword] = chosen
    for script, candidates in pointers.items():
        latin_of.setdefault(script, candidates[0])
    # Ejaan aksara lema Latin dari halaman beraksara yang isinya "Carakan spelling of <latin>". Dikunci kata persisnya
    # (huruf besar dan tanda ikut): "enèm" bukan "enem", "Arya" bukan "arya".
    aksara_of: dict[str, str] = {}
    for entry in raw:
        if JAVA.search(entry['word']):
            for sense in entry.get('senses', []):
                for target in spelling_targets(sense):
                    aksara_of.setdefault(target, nfc(entry['word']))

    merged: dict[tuple, dict] = {}
    skipped = {'tanpa ejaan Latin': 0, 'tanpa arti': 0, 'kembar digabung': 0}
    for entry in raw:
        headword = nfc(entry['word'])
        scripted = bool(JAVA.search(headword))
        if scripted:
            word, aksara = tidy_latin(head_latin(entry, headword)) or latin_of.get(headword), headword
        else:
            word, aksara = headword, None
            # Argumen j=/c= templat kepala (jv-noun|j=ꦥꦶꦱꦁ) = ejaan aksara kata ini menurut penyuntingnya.
            for head in entry.get('head_templates', []):
                args = head.get('args') or {}
                stated = JAVA_RUN.fullmatch(nfc(args.get('j') or args.get('c') or '')) if head.get('name', '').startswith('jv-') else None
                aksara = aksara or (stated.group() if stated and reads_as(stated.group(), word) else None)
            for form in entry.get('forms', []):
                tags, spelled = form.get('tags', []), JAVA_RUN.fullmatch(nfc(form.get('form', '')).removeprefix('spelling '))
                # "Carakan"/"Javanese" = ejaan kata ini menurut kepala entrinya sendiri.
                if spelled and not aksara and ('Carakan' in tags or 'Javanese' in tags) and reads_as(spelled.group(), word):
                    aksara = spelled.group()
            aksara = aksara or aksara_of.get(headword)
            for form in entry.get('forms', []):
                # Bentuk beraksara lain hanya dipakai bila memang kata ini: romanisasi halamannya sama, atau terbaca sama.
                spelled = JAVA_RUN.fullmatch(nfc(form.get('form', '')))
                if spelled and not aksara and 'canonical' not in form.get('tags', []) and (
                        plain(latin_of.get(spelled.group(), '')) == plain(word) or reads_as(spelled.group(), word)):
                    aksara = spelled.group()
        glosses, stated = [], set()
        for sense in entry.get('senses', []):
            tags, text = set(sense.get('tags', [])), nfc((sense.get('glosses') or [''])[0])
            # "nonstandard spelling of apa, romanization of ꦲꦥ" bertag romanization tetapi bukan penunjuk murni: itu
            # entri varian ejaan (opo, sego, gulo), dan aksaranya disebut di teksnya.
            variant = VARIANT_OF.match(text)
            if spelling_targets(sense) or 'no-gloss' in tags or ('romanization' in tags and not variant):
                continue
            glosses += sense_glosses(sense)
            stated |= set(re.findall('romanization of ([ꦀ-꧟\u200c]+)', text)) if variant else set()
        glosses = list(dict.fromkeys(glosses))
        if not scripted and not aksara and len(stated) == 1:
            aksara = stated.pop()
        if not glosses:
            skipped['tanpa arti'] += 1      # halaman penunjuk (romanisasi, ejaan Carakan) atau sense tanpa arti
            continue
        # Tanpa romanisasi, atau judulnya bukan huruf Latin (ejaan Pegon, simbol): tidak bisa dicari sebagai kata.
        if not word or not re.search('[a-z]', plain(word)):
            skipped['tanpa ejaan Latin'] += 1
            continue

        # Padanan beraksara ditulis Latin bila romanisasinya diketahui: dari halaman kata itu, atau dari bentuk Latin
        # di entri ini yang terbaca sama (daftar romanisasi tabel ragam, kolom roman).
        latin_forms = [nfc(text) for form in entry.get('forms', []) for text in (form.get('form', ''), form.get('roman', ''))
                       if text and not JAVA.search(text) and not LABELLED.match(nfc(text))]

        def romanize(value: str) -> str:
            if not JAVA.search(value):
                return value
            return latin_of.get(value) or next((latin for latin in latin_forms if reads_as(value, latin)), value)

        registers = {name: list(dict.fromkeys(romanize(value) for value in values))
                     for name, values in register_set(entry, headword).items()}
        # Kata yang sama di ngoko dan krama = krama-ngoko (sumber kadang menulisnya di kedua ragam).
        shared = [value for value in registers.get('ngoko', []) if value in registers.get('krama', [])]
        if shared:
            registers['krama-ngoko'] = list(dict.fromkeys(registers.get('krama-ngoko', []) + shared))
            registers.update({name: [value for value in registers[name] if value not in shared] for name in ('ngoko', 'krama')})
        # Ragam kata ini: yang disebut kepala entrinya, kalau tidak ada dari kolom ragam yang memuat kata ini.
        own = head_registers(entry, headword) or [name for name in REGISTER_ORDER if any(plain(value) == plain(word) for value in registers.get(name, []))]
        if 'krama-ngoko' in own or {'ngoko', 'krama'} <= set(own):
            register = 'krama-ngoko'
        elif own:
            register = next(name for name in REGISTER_ORDER if name in own)
        else:
            register = labelled_register(glosses)
        # Catatan = padanan di ragam LAIN; kata ini sendiri (ejaan Latin maupun aksaranya) sudah ditandai lewat `register`.
        # Satu-satunya bentuk beraksara di kolom ragam kata ini sendiri adalah ejaan kata itu, bukan padanan.
        mine = {register, 'ngoko', 'krama'} if register == 'krama-ngoko' else {register}
        others = {name: [value for value in values if plain(value) != plain(word) and value != aksara
                         and not (name in mine and len(values) == 1 and JAVA.search(value))]
                  for name, values in registers.items()}
        note = ' · '.join(f'{name} {", ".join(others[name])}' for name in NOTE_ORDER if others.get(name))

        item = {'word': word, 'glosses': glosses, 'gloss_lang': 'en', 'source': 'enwiktionary',
                'url': 'https://en.wiktionary.org/wiki/' + urllib.parse.quote(headword.replace(' ', '_')) + '#Javanese'}
        if entry.get('pos') in POS:
            item['pos'] = POS[entry['pos']]
        if aksara:
            item['aksara'] = aksara
        if register:
            item['register'] = register
        if note:
            item['note'] = note
        # Kata yang sama sering punya dua halaman (judul Latin dan judul beraksara) dengan arti yang sama: satukan;
        # isi halaman berjudul Latin didahulukan, halaman beraksara melengkapi (ejaan aksara, ragam, catatan).
        key = (plain(word), item.get('pos'), tuple(glosses))
        if key in merged:
            skipped['kembar digabung'] += 1
            kept = merged[key]['word']
            merged[key] = {**item, **merged[key]} if scripted else {**merged[key], **item}
            # Dua halaman menulis kata yang sama; ejaan yang menandai é/è (kéwan) lebih berguna daripada yang polos (kewan).
            merged[key]['word'] = max(kept, word, key=lambda spelled: (len(unicodedata.normalize('NFD', spelled)), spelled == kept))
        else:
            merged[key] = item

    # Varian ejaan yang hanya beda diakritik dari kata yang ditunjuknya ("dheweke" = "dhèwèké", "êndhas" = "endhas")
    # ditulis sama dalam aksara. Hanya bila kata yang ditunjuk punya SATU ejaan aksara; kata lain yang kebetulan mirip
    # ("enèm" bukan "enem") tidak menunjuk ke situ, jadi tidak kena.
    spellings: dict[str, set[str]] = {}
    for item in merged.values():
        if item.get('aksara'):
            spellings.setdefault(item['word'], set()).add(item['aksara'])
    for item in merged.values():
        # Semua arti harus menunjuk SATU kata ("jene" menunjuk jené dan jéné, dua ejaan aksara: tidak diwarisi).
        targets = {match.group(1) for match in map(VARIANT_OF.match, item['glosses']) if match}
        target = targets.pop() if 'aksara' not in item and len(targets) == 1 and all(map(VARIANT_OF.match, item['glosses'])) else None
        known = spellings.get(target, set()) if target and plain(target) == plain(item['word']) else set()
        if len(known) == 1:
            item['aksara'] = next(iter(known))

    # Lema yang sama (kata persis dan kelas kata sama) kadang punya dua entri dengan arti berbeda, satu dari halaman
    # berjudul Latin dan satu dari halaman beraksara ("latu"): ejaan aksaranya sama, asal hanya ada satu.
    by_lemma: dict[tuple, set[str]] = {}
    for item in merged.values():
        if item.get('aksara'):
            by_lemma.setdefault((item['word'], item.get('pos')), set()).add(item['aksara'])
    for item in merged.values():
        known = by_lemma.get((item['word'], item.get('pos')), set())
        if 'aksara' not in item and len(known) == 1:
            item['aksara'] = next(iter(known))

    # Judul halaman penunjuk romanisasi yang bukan kata kepala entri sasarannya ("candi" untuk ꦕꦤ꧀ꦝꦶ = candhi) tetap
    # bisa dicari: entri kecil yang menunjuk ke kata itu.
    reachable = {plain(item['word']) for item in merged.values()}
    words_of: dict[str, str] = {}
    for item in merged.values():
        if item.get('aksara'):
            words_of.setdefault(item['aksara'], item['word'])
    for script, titles in pointers.items():
        for title in dict.fromkeys(titles):
            if script in words_of and plain(title) not in reachable and re.search('[a-z]', plain(title)):
                reachable.add(plain(title))
                merged[(plain(title), None, script)] = {
                    'word': title, 'glosses': [f'romanization of {script} ({words_of[script]})'], 'gloss_lang': 'en',
                    'source': 'enwiktionary', 'aksara': script,
                    'url': 'https://en.wiktionary.org/wiki/' + urllib.parse.quote(title.replace(' ', '_')) + '#Javanese'}

    meta = {'key': 'enwiktionary', 'name': 'Wiktionary bahasa Inggris (entri bahasa Jawa, ekstraksi kaikki.org)',
            'license': 'CC BY-SA 4.0', 'url': 'https://kaikki.org/dictionary/Javanese/', 'retrieved': retrieved}
    return meta, list(merged.values()), skipped


# ---------- bahasa Indonesia dari Wiktionary bahasa Inggris (ekstraksi kaikki.org): arti berbahasa Inggris ----------

KAIKKI_ID = 'https://kaikki.org/dictionary/Indonesian/kaikki.org-dictionary-Indonesian.jsonl'


def enwiktionary_indonesian(cache: str, refresh: bool):
    path = os.path.join(cache, 'enwiktionary-indonesian.jsonl')
    retrieved = fetch(KAIKKI_ID, path, refresh)
    merged: dict[tuple, dict] = {}
    skipped = {'tanpa arti': 0, 'kembar digabung': 0, 'judul bukan huruf Latin': 0}
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            word = nfc(entry['word'])
            if not re.search('[a-z]', plain(word)) or len(word) > 120:
                skipped['judul bukan huruf Latin'] += 1
                continue
            glosses = []
            for sense in entry.get('senses', []):
                if 'no-gloss' not in sense.get('tags', []):
                    glosses += sense_glosses(sense)
            glosses = list(dict.fromkeys(glosses))
            if not glosses:
                skipped['tanpa arti'] += 1
                continue
            item = {'word': word, 'glosses': glosses, 'gloss_lang': 'en', 'source': 'enwiktionary',
                    'url': 'https://en.wiktionary.org/wiki/' + urllib.parse.quote(word.replace(' ', '_')) + '#Indonesian'}
            if entry.get('pos') in POS:
                item['pos'] = POS[entry['pos']]
            key = (word, item.get('pos'), tuple(glosses))
            if key in merged:
                skipped['kembar digabung'] += 1
            merged[key] = item
    meta = {'key': 'enwiktionary', 'name': 'Wiktionary bahasa Inggris (entri bahasa Indonesia, ekstraksi kaikki.org)',
            'license': 'CC BY-SA 4.0', 'url': 'https://kaikki.org/dictionary/Indonesian/', 'retrieved': retrieved}
    return meta, list(merged.values()), skipped


# ---------- bahasa Jawa dari Wikikamus (Wiktionary bahasa Indonesia): arti berbahasa Indonesia ----------

# Dump bertanggal supaya hasilnya bisa diulang. Wikimedia menghapus dump lama sesudah beberapa bulan: bila --refresh
# gagal 404, ganti tanggalnya (lihat https://dumps.wikimedia.org/idwiktionary/) dan periksa lagi keluarannya.
IDWIKT_DUMP = '20261001'
IDWIKT_URL = f'https://dumps.wikimedia.org/idwiktionary/{IDWIKT_DUMP}/idwiktionary-{IDWIKT_DUMP}-pages-articles.xml.bz2'

# Subjudul kelas kata: templat {{-n-|jv}} atau judul biasa ===Nomina===. Nama lain = pengalihan di situsnya.
WIKT_POS = {
    'n': 'nomina', 'nomina': 'nomina', 'nom': 'nomina', 'kata benda': 'nomina', 'k_b': 'nomina',
    'v': 'verba', 'verba': 'verba', 'verb': 'verba', 'kata kerja': 'verba', 'k_k': 'verba',
    'a': 'adjektiva', 'adj': 'adjektiva', 'adjektiva': 'adjektiva', 'kata sifat': 'adjektiva', 'k_s': 'adjektiva',
    'adv': 'adverbia', 'adverbia': 'adverbia', 'adverb': 'adverbia',
    'pron': 'pronomina', 'pronomina': 'pronomina',
    'num': 'numeralia', 'numeralia': 'numeralia',
    'p': 'partikel', 'part': 'partikel', 'partikel': 'partikel', 'ptcl': 'partikel',
    'prep': 'preposisi', 'preposisi': 'preposisi',
    'conj': 'konjungsi', 'konjungsi': 'konjungsi', 'kon': 'konjungsi',
    'intj': 'interjeksi', 'interjeksi': 'interjeksi',
    'pn': 'nama diri', 'nama diri': 'nama diri',
    'akhiran': 'akhiran', 'suf': 'akhiran', 'awalan': 'awalan', 'frasa': 'frasa', 'art': 'artikula', 'artikula': 'artikula',
    'akronim': 'singkatan', 'singkatan': 'singkatan', 'definisi': None,
}
# Subjudul yang tidak menutup blok definisi (penyunting kadang menaruh definisi berikutnya di bawahnya).
WIKT_NEUTRAL = {'lafal', 'pelafalan', 'pengucapan'}
# Templat label di baris definisi: tingkat tutur, dialek, bidang.
WIKT_LABELS = {
    'ngoko': 'ngoko', 'krama': 'krama', 'krama inggil': 'krama inggil', 'krama-ngoko': 'krama-ngoko', 'krama andhap': 'krama andhap',
    'kedu': 'Kedu', 'surabaya': 'Surabaya', 'banyumas': 'Banyumas', 'banyumasan': 'Banyumas', 'banten': 'Banten',
    'semarang': 'Semarang', 'semarangan': 'Semarang', 'malang': 'Malang', 'mataraman': 'Mataraman',
    'enggon-enggonan': 'dialek', 'cak': 'cakapan', 'ki': 'kiasan', 'informal': 'informal', 'hormat': 'hormat', 'lemes': 'halus',
    'ling': 'linguistik', 'geo': 'geografi', 'antr': 'antropologi', 'komp': 'komputer',
}
# Judul bagian bahasa: baku `=={{bahasa|jv}}==`, ditambah bentuk lama yang masih tersisa (sebagian menyebut dialeknya).
WIKT_HEADING = re.compile(r'^==\s*([^=].*?)\s*==\s*$', re.M)
WIKT_JAVANESE = re.compile(r'\{\{bahasa\s*\|\s*jv\}\}|\{\{bhs\|jv\}\}|\[\[bahasa jawa\]\]|\{\{(jawa(?: [a-z ]+)?|semarang)\}\}(?: (kedu))?', re.I)
WIKT_DIALECT = {'jawa kedu': 'Kedu', 'kedu': 'Kedu', 'semarang': 'Semarang', 'jawa krama inggil': 'krama inggil', 'jawa ngoko': 'ngoko',
                'jawa ngapak': 'Banyumas'}
# Isian contoh yang lupa diganti penyuntingnya.
WIKT_PLACEHOLDER = {'def', 'definisi', 'definisi/terjemah', 'terjemah', 'contoh', 'arti'}
WIKT_SUBHEAD = re.compile(r'^\s*(?:\{\{-([^|{}]+?)-(?:\|[^{}]*)?\}\}|={3,}\s*(?:\{\{)?([^={}|]+?)(?:\}\})?\s*={3,})')
WIKT_TEMPLATE = re.compile(r'\{\{([^{}]*)\}\}')


def wikt_text(text: str, labels: list | None = None) -> str:
    """Satu baris wikitext -> teks polos. Templat label dikumpulkan ke `labels`; templat yang tidak dikenal dibuang."""
    def template(match: re.Match) -> str:
        parts = [part.strip() for part in match.group(1).split('|')]
        name = parts[0].lower()
        named = dict(part.split('=', 1) for part in parts[1:] if '=' in part)
        args = [part for part in parts[1:] if '=' not in part]
        if name in ('label', 'lb') and len(args) > 1:
            if labels is not None:
                labels.extend(WIKT_LABELS.get(arg.lower(), arg) for arg in args[1:] if arg)
            return ''
        if name in WIKT_LABELS:
            if labels is not None:
                labels.append(WIKT_LABELS[name])
            return ''
        if name in ('l', 'sebut', 'lang', 'jav') and args:
            word = args[-1] if name in ('lang', 'jav') or len(args) < 2 else args[1]
            return word + (f" ({named['t']})" if named.get('t') else '')
        if name.endswith(' dari') and len(args) > 1:        # "ragam dari", "ejaan dari", "sinonim dari"
            return {'ragam dari': 'ragam bentuk dari', 'ejaan dari': 'ragam ejaan dari'}.get(name, name) + ' ' + args[1]
        if name in ('q', 'qual', 'glos', 'gloss') and args:
            return '(' + args[0] + ')'
        if name in ('non-gloss', 'bukan glos') and args:
            return args[0]
        if name == 'nonbaku' and args:
            return 'bentuk tidak baku dari ' + args[0]
        return ''

    text = re.sub(r'<ref[^>]*/>|<ref[^>]*>.*?</ref>|<sup>.*?</sup>|<!--.*?-->', '', text, flags=re.S)
    for _ in range(4):      # pranala dulu (bisa berada di dalam templat), lalu templat dari yang terdalam
        text, links = re.subn(r'\[\[(?:[^\[\]|]*\|)?([^\[\]|]*)\]\]', r'\1', text)
        text, templates = WIKT_TEMPLATE.subn(template, text)
        if not templates and not links:
            break
    text = re.sub(r'\[https?://\S+ ?([^\]]*)\]', r'\1', text)
    text = re.sub(r"'{2,}|<[^>]+>|\(\s*\)", '', text)
    return nfc(text.replace('&nbsp;', ' ')).strip(' ;,:=')


def wikt_gloss(line: str, title: str) -> tuple[str, list[str]]:
    """Baris definisi -> (arti berbahasa Indonesia dengan labelnya, contoh yang menempel sesudah <br>)."""
    body, _, tail = re.sub(r'<br\s*/?>', '\n', line).partition('\n')
    labels: list[str] = []
    gloss = wikt_text(body, labels)
    typed = re.match(r'^\(([^()]{1,30})\)\s*(.+)$', gloss)       # label yang diketik biasa: "(ngoko) ompong"
    if typed:
        labels.append(typed.group(1))
        gloss = typed.group(2)
    # Banyak definisi mengulang lemanya dulu ("mung; hanya, cuma", "butuh (perlu)"): yang diambil artinya.
    echo = re.match(r'^(.+?)\s*(?:[;:]\s*(.+)|\((.+)\))$', gloss)
    if echo and plain(echo.group(1)) == plain(title):
        gloss = (echo.group(2) or echo.group(3)).strip(' ;,:')
    # Bukan arti berbahasa Indonesia: arti yang hanya mengulang lemanya, isian kosong, dan ejaan Jawa bertanda ê/é/è
    # (halaman berjudul kata Indonesia yang "artinya" kata Jawanya: pijat -> pijêt); penunjuk "ragam ... dari" tetap.
    pointer = re.match(r'(ragam|bentuk|sinonim) [a-z ]*dari ', gloss)
    if not gloss or plain(gloss) == plain(title) or gloss.lower() in WIKT_PLACEHOLDER or (re.search('[êéè]', gloss, re.I) and not pointer):
        return '', []
    labels = list(dict.fromkeys(labels))
    return (f"({', '.join(labels)}) " if labels else '') + gloss, [text for text in [wikt_text(tail)] if text]


def wikt_examples(lines: list[str], title: str) -> tuple[list[str], list[str]]:
    """Baris `#:` di bawah satu definisi -> (contoh "kalimat Jawa — terjemahan", sinonim)."""
    examples, synonyms, pending, slanted = [], [], None, False
    for raw in lines:
        level = len(raw) - len(raw.lstrip(':'))
        body = raw.lstrip(':').strip()
        quoted = re.match(r'\{\{(?:contoh|ux)\|jv\|(.*)\}\}$', body, re.S)
        related = re.match(r'\{\{(syn|sinonim|sin)\|jv\|(.*)\}\}$', body)
        if quoted:
            parts = [wikt_text(part) for part in re.split(r'\|(?![^\[\]]*\]\])', quoted.group(1)) if not re.match(r'\w+=', part)]
            if parts and parts[0]:
                examples.append(' — '.join(part for part in parts[:2] if part))
            continue
        if related:
            parts = related.group(2).split('|')
            label = next((part.split('=', 1)[1] for part in parts if part.startswith('q=')), '')
            synonyms += [word + (f' ({label})' if label else '') for word in (wikt_text(part) for part in parts if '=' not in part) if word]
            continue
        text = wikt_text(body)
        if not text or body.startswith('{{') or text.lower() in WIKT_PLACEHOLDER:
            continue
        # Kalimat Jawa memuat lemanya dan biasanya dicetak miring; baris sesudahnya yang lebih menjorok, tanpa lema itu,
        # atau tegak sesudah kalimat miring adalah terjemahannya.
        italic = body.startswith("''")
        starts = level == 1 and plain(title) in plain(text) and not (slanted and not italic)
        if pending is not None and not starts:
            examples.append(pending + ' — ' + text)
            pending = None
        else:
            if pending is not None:
                examples.append(pending)
            pending, slanted = text, italic
    if pending is not None:
        examples.append(pending)
    # Tepat dua baris tak berpasangan = kalimat dan terjemahannya yang sama-sama memuat lemanya (aku maca buku / aku membaca buku).
    if len(examples) == 2 and not any(' — ' in example for example in examples):
        examples = [' — '.join(examples)]
    return examples, synonyms


def wikt_entries(title: str, text: str, skipped: Counter) -> list[dict]:
    """Entri kamus dari bagian bahasa Jawa satu halaman Wikikamus: satu entri per blok kelas kata yang berdefinisi."""
    entries = []
    headings = list(WIKT_HEADING.finditer(text))
    for index, heading in enumerate(headings):
        javanese = WIKT_JAVANESE.fullmatch(heading.group(1))
        if not javanese:
            continue
        dialect = WIKT_DIALECT.get((javanese.group(2) or javanese.group(1) or '').lower())
        section = text[heading.end():headings[index + 1].start() if index + 1 < len(headings) else len(text)]
        # Entri hasil impor KBBI (teks berhak cipta Badan Bahasa) tidak diambil.
        if re.search(r'\{\{rfv\|jv\|[^{}]*KBBI|\{\{R:KBBI', section):
            skipped['bertanda impor KBBI'] += 1
            continue
        blocks: list[dict] = []
        current, collecting = None, True
        for line in section.split('\n'):
            sub = WIKT_SUBHEAD.match(line)
            if sub:
                name = (sub.group(1) or sub.group(2)).strip().lower()
                if name in WIKT_POS:
                    current, collecting = {'pos': WIKT_POS[name], 'defs': [], 'box': {}}, True
                    blocks.append(current)
                elif name not in WIKT_NEUTRAL:
                    collecting = False
                continue
            box = re.search(r'\{\{jvword\|([^{}]*)\}\}', line)
            if box and current is not None:
                current['box'] = dict(part.split('=', 1) for part in box.group(1).split('|') if '=' in part)
            elif collecting and re.match(r'#(?![:*#])', line):
                if current is None:
                    current = {'pos': None, 'defs': [], 'box': {}}
                    blocks.append(current)
                current['defs'].append([line[1:], []])
            elif collecting and line.startswith('#:') and current and current['defs']:
                current['defs'][-1][1].append(line[1:])
        for block in blocks:
            glosses, examples, synonyms = [], [], []
            for line, below in block['defs']:
                gloss, inline = wikt_gloss(line, title)
                if not gloss:
                    skipped['definisi kosong atau hanya mengulang lema'] += 1
                    continue
                found, related = wikt_examples(below, title)
                if dialect and not gloss.startswith('('):
                    gloss = f'({dialect}) {gloss}'
                glosses.append(gloss)
                examples += inline + found
                synonyms += related
            glosses = list(dict.fromkeys(glosses))
            if not glosses:
                continue
            item = {'word': nfc(title), 'glosses': glosses, 'gloss_lang': 'id', 'source': 'idwiktionary',
                    'url': 'https://id.wiktionary.org/wiki/' + urllib.parse.quote(title.replace(' ', '_')) + '#Bahasa_Jawa'}
            if block['pos']:
                item['pos'] = block['pos']
            if examples:
                item['examples'] = [example for example in dict.fromkeys(examples) if len(example) <= 300][:3]
            # Kotak padanan tingkat tutur hanya berguna bila bentuknya berbeda dari lema.
            levels = {'n': 'ngoko', 'm': 'madya', 'k': 'krama', 'ki': 'krama inggil', 'ka': 'krama andhap'}
            notes = [' · '.join(f'{levels[key]} {nfc(value)}' for key, value in block['box'].items()
                                if key in levels and value.strip() and plain(value) != plain(title))]
            notes.append('sinonim: ' + ', '.join(dict.fromkeys(synonyms)) if synonyms else '')
            if any(notes):
                item['note'] = '; '.join(note for note in notes if note)
            entries.append(item)
    return entries


def wikt_pages(path: str):
    """(judul, wikitext) tiap halaman ruang nama utama yang bukan pengalihan, dibaca mengalir dari dump .xml.bz2."""
    with bz2.open(path, 'rb') as handle:
        title = namespace = text = None
        redirect = False
        for _, element in ElementTree.iterparse(handle):
            tag = element.tag.rsplit('}', 1)[-1]
            if tag == 'title':
                title = element.text
            elif tag == 'ns':
                namespace = element.text
            elif tag == 'redirect':
                redirect = True
            elif tag == 'text':
                text = element.text or ''
            elif tag == 'page':
                if namespace == '0' and not redirect and title and text:
                    yield title, text
                title = namespace = text = None
                redirect = False
                element.clear()


def idwiktionary_javanese(cache: str, refresh: bool):
    path = os.path.join(cache, f'idwiktionary-{IDWIKT_DUMP}-pages-articles.xml.bz2')
    retrieved = fetch(IDWIKT_URL, path, refresh)
    entries: list[dict] = []
    skipped: Counter = Counter()
    for title, text in wikt_pages(path):
        if not WIKT_JAVANESE.search(text):
            continue
        # Halaman berjudul aksara Jawa atau simbol tidak bisa dicari sebagai kata Latin.
        if not re.search('[a-z]', plain(title)):
            skipped['judul bukan huruf Latin'] += 1
            continue
        entries += wikt_entries(title, text, skipped)
    meta = {'key': 'idwiktionary', 'name': f'Wikikamus bahasa Indonesia (bagian bahasa Jawa, dump {IDWIKT_DUMP[:4]}-{IDWIKT_DUMP[4:6]}-{IDWIKT_DUMP[6:]})',
            'license': 'CC BY-SA 4.0', 'url': 'https://id.wiktionary.org/', 'retrieved': retrieved}
    return meta, entries, dict(skipped)


# ---------- keluaran ----------

BUILDERS = {'jv': [enwiktionary_javanese, idwiktionary_javanese], 'id': [enwiktionary_indonesian]}


def write_jsonl(path: str, entries: list[dict]) -> None:
    # mtime=0 dan tanpa nama berkas di kepala gzip: isi yang sama menghasilkan berkas yang sama (git tidak melihat beda semu).
    with open(path, 'wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as out:
        for entry in entries:
            out.write((json.dumps(entry, ensure_ascii=False) + '\n').encode('utf-8'))


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', default=os.path.join(here, '..', 'database', 'dictionaries'))
    parser.add_argument('--cache', default=os.path.join(here, '..', 'storage', 'app', 'dictionaries'))
    parser.add_argument('--only', choices=sorted(BUILDERS))
    parser.add_argument('--refresh', action='store_true', help='unduh ulang sumber walau sudah ada di --cache')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sources_path = os.path.join(args.out, 'sources.json')
    sources = json.load(open(sources_path, encoding='utf-8')) if os.path.isfile(sources_path) else {}

    for dictionary, builders in BUILDERS.items():
        if args.only and dictionary != args.only or not builders:
            continue
        merged, metas = [], []
        for build in builders:
            meta, entries, skipped = build(args.cache, args.refresh)
            # Urutan tetap supaya berkas hasilnya sama dari satu jalan ke jalan berikutnya.
            entries.sort(key=lambda e: (plain(e['word']), e['word'], e.get('pos', ''), e['glosses'][0]))
            merged += entries
            metas.append(meta)
            print(f"{dictionary} · {meta['key']}: {len(entries):,} entri; dilewati: {skipped}")
        assert merged, f'kamus {dictionary} kosong'
        write_jsonl(os.path.join(args.out, f'{dictionary}.jsonl.gz'), merged)
        sources[dictionary] = metas
    with open(sources_path, 'w', encoding='utf-8', newline='\n') as out:
        json.dump(sources, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print('ditulis ke', os.path.abspath(args.out))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
