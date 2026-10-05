"""Bangun berkas kamus kata untuk halaman Kamus: bahasa Jawa (jv) dan bahasa Indonesia (id).

Semua isinya data pihak ketiga yang diunduh skrip ini; tidak ada yang dihitung proyek. Keluaran (bawaan
web/database/dictionaries/), dibaca `php artisan aksara:dictionary`:
  sources.json    sumber tiap kamus: nama, lisensi, tautan, tanggal data diambil
  jv.jsonl.gz     satu entri per baris: word, glosses[], gloss_lang, source, lalu pos / aksara / register / note / url bila ada
  id.jsonl.gz

Unduhan mentah disimpan di web/storage/app/dictionaries/ (tidak ikut git) dan dipakai ulang; --refresh mengunduh lagi.
Hanya pustaka bawaan Python. Pakai (dari akar repo):
  .venv/Scripts/python web/tools/dictionaries.py
  cd web && php artisan aksara:dictionary
"""
import argparse
import datetime
import gzip
import json
import os
import re
import sys
import unicodedata
import urllib.parse
import urllib.request

USER_AGENT = 'aksara-ocr-lab/1.0 (riset non-komersial; pembangun kamus halaman Kamus)'
JAVA = re.compile('[ꦀ-꧟]')
JAVA_RUN = re.compile('[ꦀ-꧟‌]+(?: [ꦀ-꧟‌]+)*')

# Kelas kata wiktextract -> istilah Indonesia yang dipakai halaman Kamus.
POS = {
    'noun': 'nomina', 'verb': 'verba', 'adj': 'adjektiva', 'adv': 'adverbia', 'name': 'nama diri', 'num': 'numeralia',
    'conj': 'konjungsi', 'pron': 'pronomina', 'det': 'pewatas', 'intj': 'interjeksi', 'particle': 'partikel',
    'prep': 'preposisi', 'prefix': 'awalan', 'suffix': 'akhiran', 'circumfix': 'konfiks', 'root': 'akar kata',
    'phrase': 'frasa', 'character': 'aksara', 'article': 'artikula', 'punct': 'tanda baca',
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


def sense_gloss(sense: dict) -> str:
    """Arti satu sense wiktextract, dengan label pemakaiannya ("(krama) water"). Titik dua penutup = sense yang punya anak."""
    return nfc((sense.get('raw_glosses') or sense.get('glosses') or [''])[0]).rstrip(':').rstrip()


# ---------- bahasa Jawa dari Wiktionary bahasa Inggris (ekstraksi kaikki.org / wiktextract) ----------

KAIKKI_JV = 'https://kaikki.org/dictionary/Javanese/kaikki.org-dictionary-Javanese.jsonl'
REGISTER_ORDER = ['ngoko', 'krama-ngoko', 'madya', 'krama', 'krama inggil', 'krama andhap', 'kawi']
# Tag bentuk di kepala entri (templat jv-noun|n=|k=|ki=) -> ragam.
FORM_REGISTER = {'informal': 'ngoko', 'krama': 'krama', 'honorific': 'krama inggil', 'krama-ngoko': 'krama-ngoko', 'humble': 'krama andhap'}
# Judul kolom tabel ragam yang tidak terurai wiktextract ("Javanese register set ꦏꦿꦩ : émah ꦔꦺꦴꦏꦺꦴ : omah omah").
SET_REGISTER = {'ꦔꦺꦴꦏꦺꦴ': 'ngoko', 'ꦏꦿꦩ': 'krama', 'ꦏꦿꦩꦲꦶꦁꦒꦶꦭ꧀': 'krama inggil', 'ꦏꦿꦩꦔꦺꦴꦏꦺꦴ': 'krama-ngoko',
                'ꦏꦿꦩꦲꦤ꧀ꦝꦥ꧀': 'krama andhap', 'ꦏꦮꦶ': 'kawi'}
# Label ragam yang ditulis penyunting dalam kata-kata: "(krama inggil) to walk", "krama inggil dalem", "Krama of gedhang."
WORD_REGISTER = {'ngoko': 'ngoko', 'krama': 'krama', 'krama inggil': 'krama inggil', 'krama andhap': 'krama andhap',
                 'krama-ngoko': 'krama-ngoko', 'krama ngoko': 'krama-ngoko', 'madya': 'madya', 'krama madya': 'madya'}
REGISTER_WORDS = '|'.join(sorted(map(re.escape, WORD_REGISTER), key=len, reverse=True))
LABELLED = re.compile('^(' + REGISTER_WORDS + ') (.+)$', re.I)
REGISTER_OF = re.compile('^(' + REGISTER_WORDS + ') of ', re.I)
SPELLING_OF = re.compile(r'^(?:Carakan|Javanese script) spelling of (.+?)\.?$')
SET_COLUMN = re.compile(r'([ꦀ-꧟]+) : (.*?)(?= [ꦀ-꧟]+ : |$)')


def register_set(entry: dict, headword: str) -> dict[str, list[str]]:
    """Padanan antar-ragam di kepala entri: {'ngoko': ['mangan'], 'krama': ['nedha'], ...}, ejaan seperti di sumber."""
    found: dict[str, list[str]] = {}
    for form in entry.get('forms', []):
        tags, text = form.get('tags', []), nfc(form.get('form', ''))
        for tag in tags:
            if tag in FORM_REGISTER and text:
                found.setdefault(FORM_REGISTER[tag], []).append(text)
        # "(madya griya, ngoko omah, krama inggil dalem)" di kepala entri sampai ke sini sebagai bentuk "romanization".
        labelled = LABELLED.match(text) if 'romanization' in tags else None
        if labelled:
            found.setdefault(WORD_REGISTER[labelled.group(1).lower()], []).append(labelled.group(2))
        if 'canonical' in tags and 'Javanese register set' in text:
            for table in text.split('Javanese register set')[1:]:
                columns = SET_COLUMN.findall(table.strip())
                for index, (label, value) in enumerate(columns):
                    tokens = value.split()
                    # Sesudah kolom terakhir wiktextract menempelkan kata kepalanya sendiri.
                    if index == len(columns) - 1 and len(tokens) > 1 and tokens[-1] == headword:
                        tokens = tokens[:-1]
                    if label in SET_REGISTER and tokens:
                        found.setdefault(SET_REGISTER[label], []).append(' '.join(tokens))
    return found


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
        if len(mine) != 1:
            return None
        seen |= mine
    return seen.pop() if len(seen) == 1 else None


def enwiktionary_javanese(cache: str, refresh: bool):
    path = os.path.join(cache, 'enwiktionary-javanese.jsonl')
    retrieved = fetch(KAIKKI_JV, path, refresh)
    with open(path, encoding='utf-8') as handle:
        raw = [json.loads(line) for line in handle if line.strip()]

    # Ejaan Latin tiap lema beraksara. Urutan kepercayaan: kepala entri "<aksara> (latin)"; lalu halaman penunjuk
    # "romanization of <aksara>" (bisa lebih dari satu, termasuk ejaan tidak baku seperti "gedang" untuk gedhang, jadi
    # yang juga tercatat sebagai bentuk romanisasi di entrinya didahulukan). Bentuk "romanization" di forms tidak
    # dipakai sendirian: pada entri bertabel ragam, yang pertama sering romanisasi kata LAIN di tabel (geni -> "latu").
    pointers: dict[str, list[str]] = {}
    for entry in raw:
        if entry.get('pos') == 'romanization':
            for sense in entry.get('senses', []):
                for target in sense.get('alt_of', []):
                    if JAVA.search(target['word']):
                        pointers.setdefault(nfc(target['word']), []).append(nfc(entry['word']))
    latin_of: dict[str, str] = {}
    for entry in raw:
        headword = nfc(entry['word'])
        if not JAVA.search(headword) or headword in latin_of:
            continue
        for head in entry.get('head_templates', []):
            own = re.match(re.escape(headword) + r' \(([^(),]+)\)', nfc(head.get('expansion', '')))
            if own and not LABELLED.match(own.group(1)) and not JAVA.search(own.group(1)):
                latin_of[headword] = own.group(1)
                break
        else:
            romanized = [nfc(form['form']) for form in entry.get('forms', [])
                         if 'romanization' in form.get('tags', []) and not LABELLED.match(nfc(form['form']))]
            candidates = pointers.get(headword, [])
            agreed = [candidate for candidate in candidates if candidate in romanized]
            if agreed or len(romanized) == 1 or candidates:
                latin_of[headword] = (agreed or (romanized if len(romanized) == 1 else candidates))[0]
    for script, candidates in pointers.items():      # kata yang hanya muncul sebagai anggota tabel ragam
        latin_of.setdefault(script, candidates[0])
    # Ejaan aksara lema Latin: entri beraksara yang isinya hanya "Carakan spelling of <latin>".
    aksara_of: dict[str, str] = {}
    for entry in raw:
        if JAVA.search(entry['word']):
            for sense in entry.get('senses', []):
                spelling = SPELLING_OF.match(nfc((sense.get('glosses') or [''])[0]))
                if spelling:
                    aksara_of.setdefault(plain(spelling.group(1)), nfc(entry['word']))

    merged: dict[tuple, dict] = {}
    skipped = {'tanpa ejaan Latin': 0, 'tanpa arti': 0, 'kembar digabung': 0}
    for entry in raw:
        if entry.get('pos') == 'romanization':
            continue
        headword = nfc(entry['word'])
        scripted = bool(JAVA.search(headword))
        if scripted:
            word, aksara = latin_of.get(headword), headword
        else:
            word, aksara = headword, aksara_of.get(plain(headword))
            for form in entry.get('forms', []):
                tags, spelled = form.get('tags', []), JAVA_RUN.fullmatch(nfc(form.get('form', '')).removeprefix('spelling '))
                # "Carakan"/"Javanese" = ejaan kata ini; bentuk beraksara lain hanya dipakai bila romanisasinya memang kata ini.
                if spelled and not aksara and ('Carakan' in tags or 'Javanese' in tags
                                               or plain(latin_of.get(spelled.group(), '')) == plain(word)):
                    aksara = spelled.group()
        glosses = []
        for sense in entry.get('senses', []):
            text = sense_gloss(sense)
            pointer = SPELLING_OF.match(nfc((sense.get('glosses') or [''])[0]))
            if text and not pointer and not {'romanization', 'no-gloss'} & set(sense.get('tags', [])):
                glosses.append(text)
        glosses = list(dict.fromkeys(glosses))
        if not glosses:
            skipped['tanpa arti'] += 1
            continue
        # Tanpa romanisasi, atau judulnya bukan huruf Latin (ejaan Pegon, simbol): tidak bisa dicari sebagai kata.
        if not word or not re.search('[a-z]', plain(word)):
            skipped['tanpa ejaan Latin'] += 1
            continue

        registers = {name: list(dict.fromkeys(latin_of.get(value, value) for value in values))
                     for name, values in register_set(entry, headword).items()}
        own = [name for name in REGISTER_ORDER if any(plain(value) == plain(word) for value in registers.get(name, []))]
        for head in entry.get('head_templates', []):       # kepala entri kadang menyebut ragamnya sendiri: "(krama)"
            explicit = re.fullmatch(r'\(([^()]+)\)', nfc(head.get('expansion', '')))
            if explicit and explicit.group(1).lower() in WORD_REGISTER:
                own = [WORD_REGISTER[explicit.group(1).lower()]]
        if own:
            register = 'krama-ngoko' if {'ngoko', 'krama'} <= set(own) else own[0]
        else:
            register = labelled_register(glosses)
        note = ' · '.join(f'{name} {", ".join(registers[name])}' for name in REGISTER_ORDER if name in registers)

        item = {'word': word, 'glosses': glosses, 'gloss_lang': 'en', 'source': 'enwiktionary',
                'url': 'https://en.wiktionary.org/wiki/' + urllib.parse.quote(headword.replace(' ', '_')) + '#Javanese'}
        if entry.get('pos') in POS:
            item['pos'] = POS[entry['pos']]
        if aksara:
            item['aksara'] = aksara
        if register:
            item['register'] = register
        if note and (len(registers) > 1 or not own):
            item['note'] = note
        # Kata yang sama sering punya dua halaman (judul Latin dan judul beraksara) dengan arti yang sama: satukan,
        # tautannya ke halaman berjudul Latin.
        key = (plain(word), item.get('pos'), tuple(glosses))
        if key in merged:
            skipped['kembar digabung'] += 1
            first, second = (merged[key], item) if scripted else (item, merged[key])
            merged[key] = {**second, **first} if not scripted else {**item, **merged[key]}
        else:
            merged[key] = item

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
                text = sense_gloss(sense)
                if text and 'no-gloss' not in sense.get('tags', []):
                    glosses.append(text)
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


# ---------- keluaran ----------

BUILDERS = {'jv': [enwiktionary_javanese], 'id': [enwiktionary_indonesian]}


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
