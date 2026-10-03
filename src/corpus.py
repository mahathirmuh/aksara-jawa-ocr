"""Fase 2 — korpus teks aksara Jawa dari Wikipedia bahasa Jawa.

Alur:
  artikel Latin -> bersihkan -> transliterasi per kata (honocoroko via Node)
  -> potong jadi baris 20-80 codepoint di batas kata
  -> filter round-trip Latin -> Aksara -> Latin (CER <= 5%) -> dedup
  -> split train/val/test per artikel -> injeksi codepoint langka
  -> data/corpus_jv.txt, data/splits/*.txt, out/corpus_report.md

Label training selalu sama persis dengan string yang dirender, jadi kesalahan
transliterasi tidak membuat label salah — ia membuat teks kurang realistis.
Filter round-trip menjaga realisme itu.

Jalankan dari root repo:  python -m src.corpus
"""

import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

from src.text_augment import insert_codepoint
from src.tokenizer import is_javanese, is_javanese_letter, is_well_formed

ROOT = Path(__file__).resolve().parents[1]
RAW_PARQUET = ROOT / "data" / "raw" / "jvwiki-20231101.parquet"
TRANSLIT_JS = ROOT / "scripts" / "translit" / "translit.js"
CORPUS_PATH = ROOT / "data" / "corpus_jv.txt"
SPLITS_DIR = ROOT / "data" / "splits"
STATS_PATH = ROOT / "data" / "corpus_stats.json"
REPORT_PATH = ROOT / "out" / "corpus_report.md"

MIN_LEN, MAX_LEN = 20, 80
MIN_WORDS_PER_PARAGRAPH = 5
MAX_ROUNDTRIP_CER = 0.05
# Ortografi Jawa paling banyak menumpuk tiga tingkat. Rantai pasangan lebih panjang
# (0,11% baris train) semuanya hasil transliterasi kata asing, dan pernah
# menghentikan training karena tumpukannya terpotong kanvas render.
MAX_PASANGAN_CHAIN = 3
PASANGAN_CHAIN = re.compile("(?:꧀[ꦄ-ꦲ])+")
SPLIT_BUCKETS = {"train": range(0, 90), "val": range(90, 95), "test": range(95, 100)}
MIN_COUNT = {"train": 100, "val": 10, "test": 10}
SEED = 1234

# Wikipedia Jawa memakai å untuk a yang dibaca [ɔ] dan ê untuk pepet; aksara
# menulis keduanya dengan aksara biasa. honocoroko tidak mengenal keduanya.
LATIN_TRANSLATE = str.maketrans({
    "å": "a", "Å": "A", "ê": "e", "Ê": "E", "ô": "o", "Ô": "O", "è": "é", "È": "É",
    "-": " ", "–": " ", "—": " ", "/": " ",
    ";": ",", "!": ".", "?": ".",
    '"': None, "'": None, "“": None, "”": None, "‘": None, "’": None, "«": None, "»": None,
})
WORD_RE = re.compile(r"[(]*[a-zé0-9]+[,.:)]*")
ACRONYM_RE = re.compile(r"[(]*[A-Z]{2,}[,.:)]*")
VOWELS = "aiueoé"
# useSwara:false menulis vokal awal dengan ha, jadi "ana" kembali sebagai
# "hana". h sebelum vokal (di awal kata / setelah vokal) dibuang di KEDUA sisi
# supaya perbedaan yang benar dan sistematis ini tidak dihitung sebagai error.
GLOTTAL_H = re.compile(rf"(?<![^\s{VOWELS}])h(?=[{VOWELS}])")


def clean_words(paragraph: str) -> list[str | None]:
    """Kata Latin siap transliterasi; None menandai kata yang memutus baris."""
    words: list[str | None] = []
    for raw in unicodedata.normalize("NFC", paragraph).translate(LATIN_TRANSLATE).split():
        if ACRONYM_RE.fullmatch(raw):  # KGPAA, IV: transliterasinya tak bermakna
            words.append(None)
            continue
        word = raw.lower()
        words.append(word if WORD_RE.fullmatch(word) else None)
    return words


def iter_paragraphs():
    table = pq.read_table(RAW_PARQUET, columns=["id", "text"])
    for article_id, text in zip(table.column("id").to_pylist(), table.column("text").to_pylist()):
        for paragraph in text.split("\n"):
            words = clean_words(paragraph)
            if sum(w is not None for w in words) >= MIN_WORDS_PER_PARAGRAPH:
                yield article_id, words


def transliterate_words(words: set[str]) -> dict[str, tuple[str, str]]:
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js tidak ditemukan; dibutuhkan untuk honocoroko")
    ordered = sorted(words)
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "words.txt", Path(tmp) / "aks.jsonl"
        src.write_text("\n".join(ordered) + "\n", encoding="utf-8", newline="\n")
        subprocess.run([node, str(TRANSLIT_JS), str(src), str(dst)], check=True)
        results = [json.loads(line) for line in dst.read_text(encoding="utf-8").splitlines()]
    if len(results) != len(ordered):
        raise RuntimeError(f"translit.js mengembalikan {len(results)} baris untuk {len(ordered)} kata")
    return {w: (r["aks"], r["back"]) for w, r in zip(ordered, results)}


def longest_pasangan_chain(text: str) -> int:
    """Jumlah pasangan berturut-turut terpanjang (pangkon + aksara) dalam teks."""
    return max((len(m.group()) // 2 for m in PASANGAN_CHAIN.finditer(text)), default=0)


def valid_aksara(aks: str) -> bool:
    return (
        bool(aks)
        and all(is_javanese(c) for c in aks)
        and is_well_formed(aks)
        and longest_pasangan_chain(aks) <= MAX_PASANGAN_CHAIN
    )


def chunk_paragraph(words, lookup, rng) -> list[list[tuple[str, str, str]]]:
    """Potong satu paragraf jadi baris 20-80 codepoint aksara, di batas kata."""
    lines = []
    current: list[tuple[str, str, str]] = []
    length = 0
    target = rng.randint(MIN_LEN, MAX_LEN)

    def flush():
        nonlocal current, length, target
        if length >= MIN_LEN:
            lines.append(current)
        current, length = [], 0
        target = rng.randint(MIN_LEN, MAX_LEN)

    for word in words:
        entry = lookup.get(word) if word is not None else None
        if entry is None or len(entry[1]) > MAX_LEN:
            flush()  # kata tak bisa ditransliterasi bersih: putus baris di sini
            continue
        added = len(entry[1]) + (1 if current else 0)
        if (length + added > target and length >= MIN_LEN) or length + added > MAX_LEN:
            flush()
            added = len(entry[1])
        current.append(entry)
        length += added
    flush()
    return lines


def roundtrip_cer(latin: str, back: str) -> float:
    ref = GLOTTAL_H.sub("", latin.lower().replace("è", "é"))
    hyp = GLOTTAL_H.sub("", back.lower().replace("è", "é"))
    return Levenshtein.distance(ref, hyp) / max(1, len(ref))


def split_of(article_id) -> str:
    bucket = int(hashlib.md5(str(article_id).encode()).hexdigest(), 16) % 100
    return next(name for name, buckets in SPLIT_BUCKETS.items() if bucket in buckets)


def inject_rare(lines: list[str], min_count: int, rng) -> list[str]:
    """Baris sintetis tambahan sampai setiap codepoint aksara Jawa muncul >= min_count kali."""
    counts = Counter(ch for line in lines for ch in line)
    common_letters = [ch for ch, _ in counts.most_common() if is_javanese_letter(ch)][:20]
    hosts = [line for line in lines if len(line) <= MAX_LEN - 2]
    seen = set(lines)
    injected: list[str] = []
    for cp in range(0xA980, 0xA9E0):
        ch = chr(cp)
        if unicodedata.category(ch) == "Cn":
            continue
        attempts = 0
        while counts[ch] < min_count and attempts < min_count * 50:
            attempts += 1
            new = insert_codepoint(rng.choice(hosts), ch, common_letters, rng)
            if new is None or new in seen or len(new) > MAX_LEN or not is_well_formed(new):
                continue
            seen.add(new)
            injected.append(new)
            counts.update(new)
    return injected


def length_histogram(lines: list[str]) -> dict[str, int]:
    buckets = Counter(min(len(line), MAX_LEN - 1) // 10 * 10 for line in lines)
    return {f"{b}-{b + 9}": buckets[b] for b in range(MIN_LEN // 10 * 10, MAX_LEN, 10)}


def write_report(stats: dict, splits: dict[str, list[str]]) -> None:
    train_counts = Counter(ch for line in splits["train"] for ch in line)
    all_counts = Counter(ch for lines in splits.values() for line in lines for ch in line)
    hist = stats["length_histogram"]
    peak = max(hist.values()) or 1

    out = [
        "# Laporan korpus (Fase 2)",
        "",
        "Sumber: Wikipedia bahasa Jawa, snapshot 20231101 (wikimedia/wikipedia di HuggingFace).",
        "Transliterasi: @naandalist/honocoroko 1.4.0, `useSwara:false`, ditambah pangkon di konsonan",
        "penutup kata (lihat scripts/translit/translit.js).",
        "",
        "## Ringkasan",
        "",
        f"- Artikel: {stats['articles']:,}",
        f"- Kata unik: {stats['vocab']:,} (valid sebagai aksara: {stats['vocab_valid']:,})",
        f"- Baris kandidat: {stats['candidates']:,}",
        f"- **Lolos round-trip (CER <= {MAX_ROUNDTRIP_CER:.0%}): {stats['roundtrip_passed']:,} "
        f"= {stats['roundtrip_pass_rate']:.2%}**",
        f"- Duplikat dibuang: {stats['duplicates']:,}",
        f"- Baris injeksi codepoint langka: {stats['injected_total']:,}",
        f"- **Total baris: {stats['total_lines']:,}** "
        f"(train {stats['split_lines']['train']:,} / val {stats['split_lines']['val']:,} "
        f"/ test {stats['split_lines']['test']:,})",
        f"- Kebocoran teks antar split: {stats['leaked_lines']}",
        "",
        "## Distribusi panjang (codepoint)",
        "",
        "```",
    ]
    for bucket, n in hist.items():
        out.append(f"{bucket:>6} | {'#' * round(40 * n / peak):<40} {n:,}")
    out += [
        "```",
        "",
        "## Distribusi karakter",
        "",
        f"Target: setiap codepoint aksara Jawa >= {MIN_COUNT['train']} kali di train.",
        "",
        "| codepoint | nama | train | semua | status |",
        "|---|---|---:|---:|---|",
    ]
    for ch in sorted(all_counts):
        status = "" if not is_javanese(ch) or train_counts[ch] >= MIN_COUNT["train"] else "**KURANG**"
        name = unicodedata.name(ch, "?")
        out.append(f"| U+{ord(ch):04X} | {name} | {train_counts[ch]:,} | {all_counts[ch]:,} | {status} |")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    rng = random.Random(SEED)

    print("1/6 kosakata ...")
    articles, vocab = set(), set()
    for article_id, words in iter_paragraphs():
        articles.add(article_id)
        vocab.update(w for w in words if w is not None)
    print(f"    {len(articles):,} artikel, {len(vocab):,} kata unik")

    print("2/6 transliterasi (honocoroko) ...")
    lookup = {w: (w, aks, back) for w, (aks, back) in transliterate_words(vocab).items() if valid_aksara(aks)}
    print(f"    {len(lookup):,} kata valid sebagai aksara Jawa")

    print("3/6 potong baris + filter round-trip + dedup ...")
    candidates = passed = duplicates = 0
    seen: set[str] = set()
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_BUCKETS}
    for article_id, words in iter_paragraphs():
        for entries in chunk_paragraph(words, lookup, rng):
            candidates += 1
            latin = " ".join(e[0] for e in entries)
            back = " ".join(e[2] for e in entries)
            if roundtrip_cer(latin, back) > MAX_ROUNDTRIP_CER:
                continue
            passed += 1
            aks = unicodedata.normalize("NFC", " ".join(e[1] for e in entries))
            if aks in seen:
                duplicates += 1
                continue
            seen.add(aks)
            splits[split_of(article_id)].append(aks)
    print(f"    {candidates:,} kandidat, {passed:,} lolos round-trip ({passed / max(1, candidates):.2%})")

    print("4/6 injeksi codepoint langka ...")
    injected = {}
    for name, lines in splits.items():
        extra = inject_rare(lines, MIN_COUNT[name], rng)
        injected[name] = len(extra)
        lines.extend(extra)
        rng.shuffle(lines)
    print(f"    {injected}")

    print("5/6 tulis ...")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, lines in splits.items():
        (SPLITS_DIR / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    all_lines = [line for lines in splits.values() for line in lines]
    CORPUS_PATH.write_text("\n".join(all_lines) + "\n", encoding="utf-8", newline="\n")

    train, val, test = (set(splits[n]) for n in ("train", "val", "test"))
    stats = {
        "articles": len(articles),
        "vocab": len(vocab),
        "vocab_valid": len(lookup),
        "candidates": candidates,
        "roundtrip_passed": passed,
        "roundtrip_pass_rate": passed / max(1, candidates),
        "duplicates": duplicates,
        "injected": injected,
        "injected_total": sum(injected.values()),
        "split_lines": {name: len(lines) for name, lines in splits.items()},
        "total_lines": len(all_lines),
        "leaked_lines": len(train & val) + len(train & test) + len(val & test),
        "length_histogram": length_histogram(all_lines),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=1), encoding="utf-8", newline="\n")

    print("6/6 laporan ...")
    write_report(stats, splits)
    print(f"    total {len(all_lines):,} baris; bocor {stats['leaked_lines']}; laporan {REPORT_PATH}")


if __name__ == "__main__":
    main()
