"""Bangun leksikon pemulih tanda é/è (taling) untuk bahasa Jawa beraksara Latin: database/dictionaries/jv-taling.json.

Mengapa perlu: aksara Jawa membedakan taling (é/è) dari pepet (e), tetapi kebanyakan teks Latin, termasuk keluaran mesin
penerjemah, menulis keduanya "e". Tanpa pemulihan, separuh kata ber-e mendapat sandhangan yang salah saat dialihaksarakan.

Cara: Wikipedia bahasa Jawa menulis é/è secara eksplisit di banyak artikel. Dari artikel yang "bertanda kuat" (>= 10 é/è
dan >= 30% dari semua huruf e), tiap kata dicatat bentuk bertandanya; kata masuk leksikon bila bentuk bertanda terbanyaknya
muncul >= 2 kali dan menguasai >= 60% kemunculan kata itu. Lema kamus (jv.jsonl.gz) yang bertanda ditambahkan bila belum ada
dan tidak punya kembaran tanpa tanda. Ketepatan diukur pada 20% artikel yang ditahan, lalu leksikon akhir dibangun dari semuanya.

Butuh snapshot Wikipedia proyek (data/raw/jvwiki-20231101.parquet, dibuat `python -m src.corpus`) dan pyarrow.
Pakai (dari akar repo): .venv/Scripts/python web/tools/taling_lexicon.py
"""
import argparse
import collections
import gzip
import json
import os
import re
import sys
import unicodedata

MIN_TALING, MIN_RATIO = 10, .30     # artikel "bertanda kuat"
MIN_COUNT, MIN_SHARE = 2, .60       # syarat kata masuk leksikon
WORD = re.compile(r'[a-zéè]+')


def bare(word: str) -> str:
    return word.replace('é', 'e').replace('è', 'e')


def marked_articles(parquet: str) -> list[tuple[int, str]]:
    import pyarrow.parquet as pq

    column = pq.read_table(parquet, columns=['text']).column('text')
    kept = []
    for index in range(len(column)):
        text = unicodedata.normalize('NFC', column[index].as_py()).lower().replace('ê', 'e').replace('ě', 'e')
        plain, taling = text.count('e'), text.count('é') + text.count('è')
        if taling >= MIN_TALING and taling / max(1, plain + taling) >= MIN_RATIO:
            kept.append((index, text))
    return kept


def count_variants(texts) -> dict[str, collections.Counter]:
    variants: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for text in texts:
        for word in WORD.findall(text):
            if 'e' in bare(word):
                variants[bare(word)][word] += 1
    return variants


def lexicon(variants: dict[str, collections.Counter]) -> dict[str, str]:
    out = {}
    for key, counter in variants.items():
        best, count = counter.most_common(1)[0]
        if best != key and count >= MIN_COUNT and count / sum(counter.values()) >= MIN_SHARE:
            out[key] = best
    return out


def accuracy(words: dict[str, str], texts) -> tuple[float, float, int]:
    """(ketepatan dengan leksikon, ketepatan tanpa pemulihan, jumlah kata uji): é dan è dihitung sama, keduanya taling."""
    same = lambda a, b: a.replace('è', 'é') == b.replace('è', 'é')  # noqa: E731
    tokens = [word for text in texts for word in WORD.findall(text) if 'e' in bare(word)]
    restored = sum(1 for token in tokens if same(words.get(bare(token), bare(token)), token))
    untouched = sum(1 for token in tokens if token == bare(token))
    return restored / len(tokens), untouched / len(tokens), len(tokens)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--wiki', default=os.path.join(repo, 'data', 'raw', 'jvwiki-20231101.parquet'))
    parser.add_argument('--dictionary', default=os.path.join(here, '..', 'database', 'dictionaries', 'jv.jsonl.gz'))
    parser.add_argument('--out', default=os.path.join(here, '..', 'database', 'dictionaries', 'jv-taling.json'))
    args = parser.parse_args()

    articles = marked_articles(args.wiki)
    held_out = [text for index, text in articles if index % 5 == 0]
    train = [text for index, text in articles if index % 5 != 0]
    with_lexicon, without, tokens = accuracy(lexicon(count_variants(train)), held_out)
    print(f'{len(articles):,} artikel bertanda kuat; uji tahan {len(held_out):,} artikel, {tokens:,} kata ber-e: '
          f'tanpa pemulihan {without:.1%} benar, dengan leksikon {with_lexicon:.1%}')

    words = lexicon(count_variants(text for _, text in articles))
    from_wiki = len(words)
    lemmas = set()
    with gzip.open(args.dictionary, 'rt', encoding='utf-8') as handle:
        for line in handle:
            lemma = unicodedata.normalize('NFC', json.loads(line)['word']).lower().replace('ê', 'e')
            if WORD.fullmatch(lemma):
                lemmas.add(lemma)
    for lemma in sorted(lemmas):
        if lemma != bare(lemma) and bare(lemma) not in lemmas:
            words.setdefault(bare(lemma), lemma)

    meta = {
        'source': 'Wikipedia bahasa Jawa, snapshot 20231101 (CC BY-SA 4.0) + lema kamus Wiktionary (CC BY-SA 4.0)',
        'articles': len(articles), 'rule': f'bentuk bertanda terbanyak >= {MIN_COUNT} kali dan >= {MIN_SHARE:.0%} kemunculan',
        'from_wikipedia': from_wiki, 'from_dictionary': len(words) - from_wiki,
        'held_out': {'articles': len(held_out), 'words': tokens, 'accuracy_without': round(without, 4), 'accuracy_with': round(with_lexicon, 4)},
    }
    with open(args.out, 'w', encoding='utf-8', newline='\n') as out:
        json.dump({'meta': meta, 'words': dict(sorted(words.items()))}, out, ensure_ascii=False, indent=0)
        out.write('\n')
    print(f'{len(words):,} kata ({from_wiki:,} dari Wikipedia, {len(words) - from_wiki:,} dari kamus) -> {os.path.abspath(args.out)}')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
