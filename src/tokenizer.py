"""Tokenizer aksara Jawa — charset, encode/decode, dan reorder logis <-> visual.

Unicode menyimpan taling SESUDAH aksara dasarnya, padahal digambar di KIRI.
CTC butuh target yang monoton terhadap kolom citra, jadi model dilatih pada
urutan VISUAL dan keluarannya dikembalikan ke urutan LOGIS saat decode.

Urutan visual tidak ditulis manual. HarfBuzz membentuk setiap suku kata di
beberapa font; tanda yang pada mayoritas kemunculannya diletakkan sebelum
aksara dasar menjadi "pre-base". Urutan visual kanonis = tanda pre-base di awal
suku kata, sisanya urutan logis. Hasilnya diringkas jadi tabel permutasi per
pola suku kata; training dan inference hanya memakai tabel itu, tanpa font.

Bangun dari korpus:  python -m src.tokenizer
"""

import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import uharfbuzz as hb

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "data" / "tokenizer.json"
SPLITS_DIR = ROOT / "data" / "splits"
CHARSET_DOC_PATH = ROOT / "out" / "charset.md"
WINDOWS_JAVANESE_TEXT = Path("C:/Windows/Fonts/javatext.ttf")

BLANK = 0
PANGKON = "\uA9C0"


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def is_javanese(ch: str) -> bool:
    return "\uA980" <= ch <= "\uA9DF"


def is_javanese_letter(ch: str) -> bool:
    return is_javanese(ch) and unicodedata.category(ch) == "Lo"


def is_mark(ch: str) -> bool:
    # Cf ikut menempel: ZWNJ/ZWJ mengatur pasangan tapi tidak memulai suku kata.
    return unicodedata.category(ch) in ("Mn", "Mc", "Me", "Cf")


def is_well_formed(text: str) -> bool:
    """Setiap tanda harus menempel pada aksara Jawa atau pada tanda lain."""
    prev = ""
    for ch in text:
        if is_mark(ch) and not (prev and (is_javanese_letter(prev) or is_mark(prev))):
            return False
        prev = ch
    return True


def logical_syllables(text: str) -> list[str]:
    """Pecah teks urutan logis jadi suku kata ortografis.

    Suku kata = aksara dasar + semua tanda yang menempel, termasuk rantai
    pasangan: pangkon yang diikuti aksara Jawa menyambung ke suku kata yang sama.
    """
    syllables: list[str] = []
    prev = ""
    for ch in text:
        joins = is_mark(ch) or (prev == PANGKON and is_javanese_letter(ch))
        if syllables and joins:
            syllables[-1] += ch
        else:
            syllables.append(ch)
        prev = ch
    return syllables


def visual_syllables(text: str, prebase: frozenset[str]) -> list[str]:
    """Pecah teks urutan VISUAL jadi suku kata.

    Seperti logical_syllables, kecuali tanda pre-base (mis. taling) sudah
    berada di depan: ia memulai suku kata, dan aksara sesudahnya menyambung.
    """
    syllables: list[str] = []
    prev = ""
    for ch in text:
        if ch in prebase:
            joins = prev in prebase
        else:
            joins = is_mark(ch) or prev in prebase or (prev == PANGKON and is_javanese_letter(ch))
        if syllables and joins:
            syllables[-1] += ch
        else:
            syllables.append(ch)
        prev = ch
    return syllables


def pattern(syllable: str) -> str:
    """Kunci tabel reorder: aksara Jawa diabstraksi jadi 'L', karakter lain tetap."""
    return " ".join("L" if is_javanese_letter(ch) else f"{ord(ch):04X}" for ch in syllable)


def readable_pattern(key: str) -> str:
    def name(elem: str) -> str:
        if elem == "L":
            return "L"
        full = unicodedata.name(chr(int(elem, 16)), elem)
        for prefix in ("JAVANESE ", "VOWEL SIGN ", "CONSONANT SIGN ", "SIGN "):
            full = full.replace(prefix, "")
        return full.lower()

    return " ".join(name(e) for e in key.split(" "))


def inversions(order: Sequence[int]) -> int:
    return sum(order[i] > order[j] for i in range(len(order)) for j in range(i + 1, len(order)))


def apply_permutation(text: str, perm: Sequence[int]) -> str:
    """Susun ulang `text`: posisi visual ke-k diisi karakter logis ke-perm[k]."""
    return "".join(text[i] for i in perm)


def canonical_permutation(syllable: str, prebase: frozenset[str]) -> tuple[int, ...]:
    """Tanda pre-base ke awal suku kata (urutan relatifnya tetap), sisanya urutan logis."""
    front = [i for i, ch in enumerate(syllable) if ch in prebase]
    rest = [i for i, ch in enumerate(syllable) if ch not in prebase]
    return tuple(front + rest)


def default_font_paths() -> list[Path]:
    """Font untuk konsensus urutan visual: font training + Javanese Text bila ada.

    Javanese Text hanya ikut memberi suara soal urutan glyph; ia tidak dipakai
    merender data training.
    """
    paths = sorted(p for p in (ROOT / "fonts").glob("*") if p.suffix.lower() in {".ttf", ".otf"})
    if WINDOWS_JAVANESE_TEXT.exists():
        paths.append(WINDOWS_JAVANESE_TEXT)
    return paths


class HarfBuzzOrder:
    """Urutan visual karakter menurut HarfBuzz, dikonsensuskan di beberapa font."""

    def __init__(self, font_paths: Sequence[str | Path]):
        self.font_paths = [str(p) for p in font_paths]
        if not self.font_paths:
            raise ValueError("Butuh minimal satu font untuk menurunkan urutan visual")
        self._fonts = [hb.Font(hb.Face(hb.Blob.from_file_path(p))) for p in self.font_paths]

    @staticmethod
    def _stream_order(font: hb.Font, text: str) -> tuple[int, ...]:
        buf = hb.Buffer()
        buf.add_codepoints([ord(c) for c in text])
        buf.guess_segment_properties()
        # CHARACTERS: cluster tidak pernah digabung, jadi setiap glyph tetap
        # menunjuk ke karakter asalnya walau glyph-nya dipindah.
        buf.cluster_level = hb.BufferClusterLevel.CHARACTERS
        buf.flags = hb.BufferFlags.DO_NOT_INSERT_DOTTED_CIRCLE
        hb.shape(font, buf, {})

        order: list[int] = []
        for info in buf.glyph_infos:
            if info.cluster not in order:
                order.append(info.cluster)
        # Karakter yang terserap ligatur (mis. ta di dalam glyph pasangan) tidak
        # punya glyph sendiri: taruh tepat setelah pendahulu logisnya.
        for i in range(len(text)):
            if i not in order:
                order.insert(order.index(i - 1) + 1 if i > 0 else 0, i)
        return tuple(order)

    def permutation(self, text: str) -> tuple[tuple[int, ...], bool]:
        """(urutan visual, apakah semua font sepakat)."""
        counts = Counter(self._stream_order(font, text) for font in self._fonts)
        top = max(counts.values())
        # Seri: pilih yang memindahkan karakter paling sedikit. Pemindahan
        # ekstra adalah keanehan satu font, bukan aturan penulisan.
        best = min((order for order, c in counts.items() if c == top), key=inversions)
        return best, len(counts) == 1


class Tokenizer:
    """Pemetaan bijektif string <-> indeks (blank = 0) plus reorder logis <-> visual."""

    def __init__(self, charset: Sequence[str], reorder: dict[str, Sequence[int]]):
        self.charset = list(charset)
        self.char_to_id = {ch: i + 1 for i, ch in enumerate(self.charset)}
        self.reorder: dict[str, tuple[int, ...]] = {}
        self._inverse: dict[str, tuple[int, ...]] = {}
        self.prebase: frozenset[str] = frozenset()
        for key, perm in reorder.items():
            self._add(key, tuple(perm))

    @property
    def n_classes(self) -> int:
        return len(self.charset) + 1  # + blank

    def _add(self, key: str, perm: tuple[int, ...]) -> None:
        elems = key.split(" ")
        visual_key = " ".join(elems[j] for j in perm)
        known = self._inverse.get(visual_key)
        if known is not None and known != perm:
            raise ValueError(f"Tabel reorder tidak bisa dibalik: pola visual {visual_key!r} punya dua asal")
        self.reorder[key] = perm
        self._inverse[visual_key] = perm
        # Tanda yang secara visual mendahului aksara dasar suku katanya.
        before_base = perm[: perm.index(0)]
        moved = {chr(int(elems[j], 16)) for j in before_base if elems[j] != "L"}
        if moved:
            self.prebase = self.prebase | moved

    # --- charset ---------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """String -> indeks 1..N. Tidak me-reorder; panggil to_visual dulu untuk target CTC."""
        text = nfc(text)
        try:
            return [self.char_to_id[ch] for ch in text]
        except KeyError as e:
            raise KeyError(f"Karakter di luar charset: U+{ord(e.args[0]):04X}") from None

    def decode(self, ids: Iterable[int]) -> str:
        """Indeks -> string. Blank dibuang; collapse CTC bukan tugas fungsi ini."""
        return "".join(self.charset[int(i) - 1] for i in ids if int(i) != BLANK)

    # --- reorder ---------------------------------------------------------

    def _permutation_for(self, syllable: str) -> tuple[int, ...]:
        if len(syllable) == 1:
            return (0,)
        perm = self.reorder.get(pattern(syllable))
        # Pola yang belum pernah terlihat (mis. label data nyata): aturan kanonis
        # yang sama, dengan set pre-base hasil turunan HarfBuzz.
        return perm if perm is not None else canonical_permutation(syllable, self.prebase)

    def to_visual(self, text: str) -> str:
        return "".join(
            apply_permutation(syllable, self._permutation_for(syllable))
            for syllable in logical_syllables(nfc(text))
        )

    def to_logical(self, text: str) -> str:
        out = []
        for syllable in visual_syllables(text, self.prebase):
            perm = self._inverse.get(pattern(syllable)) if len(syllable) > 1 else None
            if perm is None:
                # Pola tak dikenal (mis. keluaran model yang rusak): biarkan apa adanya.
                out.append(syllable)
                continue
            logical = [""] * len(syllable)
            for k, j in enumerate(perm):
                logical[j] = syllable[k]
            out.append("".join(logical))
        return nfc("".join(out))

    # --- build / simpan ----------------------------------------------------

    @classmethod
    def build(
        cls, lines: Iterable[str], font_paths: Sequence[str | Path], prebase_threshold: float = 0.5
    ) -> tuple["Tokenizer", dict]:
        """Bangun charset dan tabel reorder dari korpus, bukan dari tabel Unicode manual.

        1. Setiap suku kata dibentuk HarfBuzz di semua font; urutan konsensus dicatat.
        2. Pre-base = tanda yang pada mayoritas kemunculannya diletakkan HarfBuzz
           sebelum aksara dasar suku katanya.
        3. Urutan visual kanonis: tanda pre-base ke awal suku kata, sisanya logis.

        Langkah 3 membuang urutan glyph yang tidak konsisten antar aksara: cakra
        pada pasangan kadang ditaruh sebelum aksara dasar, kadang tidak, tergantung
        ligatur font untuk konsonan tertentu. Cakra bertumpuk di kolom yang sama
        dengan konsonannya, jadi urutan apa pun benar untuk CTC asal SELALU sama.
        Seberapa sering urutan kanonis sama dengan HarfBuzz dicatat di stats.
        """
        hb_order = HarfBuzzOrder(font_paths)
        char_counts: Counter[str] = Counter()
        syllable_counts: Counter[str] = Counter()
        n_lines = 0
        for line in lines:
            line = nfc(line)
            n_lines += 1
            char_counts.update(line)
            syllable_counts.update(s for s in logical_syllables(line) if len(s) > 1)

        # Suku kata dibentuk sendiri-sendiri: reordering tidak melintasi batas
        # suku kata. test_tokenizer membandingkannya dengan shaping satu baris penuh.
        consensus: dict[str, tuple[int, ...]] = {}
        split_syllables = split_occurrences = 0
        mark_total: Counter[str] = Counter()
        mark_before_base: Counter[str] = Counter()
        for syllable, count in syllable_counts.items():
            perm, unanimous = hb_order.permutation(syllable)
            consensus[syllable] = perm
            if not unanimous:
                split_syllables += 1
                split_occurrences += count
            base_position = perm.index(0)
            for k, j in enumerate(perm):
                if j != 0 and not is_javanese_letter(syllable[j]):
                    mark_total[syllable[j]] += count
                    mark_before_base[syllable[j]] += count * (k < base_position)
        rates = {ch: mark_before_base[ch] / mark_total[ch] for ch in mark_total}
        prebase = frozenset(ch for ch, rate in rates.items() if rate > prebase_threshold)

        reorder: dict[str, tuple[int, ...]] = {}
        pattern_counts: Counter[str] = Counter()
        agree = 0
        disagreements: Counter[str] = Counter()
        for syllable, count in syllable_counts.items():
            key = pattern(syllable)
            perm = canonical_permutation(syllable, prebase)
            reorder[key] = perm
            pattern_counts[key] += count
            if perm == consensus[syllable]:
                agree += count
            else:
                disagreements[key] += count

        # Dua pola logis yang jatuh ke pola visual yang sama tidak bisa dibalik.
        by_visual: dict[str, list[str]] = defaultdict(list)
        for key, perm in reorder.items():
            elems = key.split(" ")
            by_visual[" ".join(elems[j] for j in perm)].append(key)
        ambiguous = []
        for visual_key, keys in by_visual.items():
            if len(keys) > 1:
                keys.sort(key=pattern_counts.__getitem__, reverse=True)
                for dropped in keys[1:]:
                    del reorder[dropped]
                ambiguous.append({
                    "visual": readable_pattern(visual_key),
                    "kept": [readable_pattern(keys[0]), pattern_counts[keys[0]]],
                    "dropped": [[readable_pattern(k), pattern_counts[k]] for k in keys[1:]],
                })

        tokenizer = cls(sorted(char_counts), reorder)
        total_occurrences = sum(syllable_counts.values())
        stats = {
            "lines": n_lines,
            "characters": sum(char_counts.values()),
            "classes_including_blank": tokenizer.n_classes,
            "distinct_syllables": len(syllable_counts),
            "reorder_patterns": len(reorder),
            "non_identity_patterns": sum(p != tuple(range(len(p))) for p in reorder.values()),
            "prebase": [f"U+{ord(c):04X}" for c in sorted(prebase)],
            "prebase_threshold": prebase_threshold,
            "before_base_rate": {
                f"U+{ord(ch):04X} {readable_pattern(f'{ord(ch):04X}')}": round(rate, 4)
                for ch, rate in sorted(rates.items(), key=lambda kv: -kv[1])
                if rate > 0
            },
            "fonts": [Path(p).name for p in font_paths],
            "font_disagreement": {"distinct_syllables": split_syllables, "occurrences": split_occurrences},
            "harfbuzz_agreement": agree / max(1, total_occurrences),
            "harfbuzz_disagreements_top": [
                [readable_pattern(k), n] for k, n in disagreements.most_common(20)
            ],
            "ambiguous_patterns": ambiguous,
            "ambiguous_occurrences": sum(n for a in ambiguous for _, n in a["dropped"]),
            "char_counts": {f"U+{ord(c):04X}": n for c, n in sorted(char_counts.items())},
        }
        return tokenizer, stats

    def save(self, path: str | Path, stats: dict | None = None) -> None:
        data = {
            "blank": BLANK,
            "charset": [f"U+{ord(c):04X}" for c in self.charset],
            "reorder": {k: list(v) for k, v in sorted(self.reorder.items())},
            "stats": stats or {},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["blank"] != BLANK:
            raise ValueError(f"{path}: blank = {data['blank']}, harus {BLANK}")
        charset = [chr(int(cp[2:], 16)) for cp in data["charset"]]
        return cls(charset, data["reorder"])


def write_charset_doc(tokenizer: Tokenizer, stats: dict, path: Path) -> None:
    counts = stats["char_counts"]
    lines = [
        "# Charset tokenizer",
        "",
        f"- Kelas: **{tokenizer.n_classes}** (indeks 0 = blank CTC, 1..{len(tokenizer.charset)} = karakter)",
        f"- Dibangun dari {stats['lines']:,} baris korpus, {stats['characters']:,} karakter",
        f"- Pola suku kata di tabel reorder: {stats['reorder_patterns']} "
        f"({stats['non_identity_patterns']} di antaranya urutan visual berbeda dari logis)",
        f"- Tanda pre-base (diturunkan dari HarfBuzz, ambang {stats['prebase_threshold']:.0%}): "
        f"{', '.join(stats['prebase']) or '-'}",
        f"- Font konsensus: {', '.join(stats['fonts'])}",
        f"- Urutan kanonis sama dengan HarfBuzz pada {stats['harfbuzz_agreement']:.2%} kemunculan suku kata",
        f"- Pola ambigu yang dibuang: {len(stats['ambiguous_patterns'])} "
        f"({stats['ambiguous_occurrences']:,} kemunculan)",
        "",
        "## Laju 'sebelum aksara dasar' per tanda (konsensus HarfBuzz)",
        "",
        *[f"- {name}: {rate:.2%}" for name, rate in stats["before_base_rate"].items()],
        "",
        "## Charset",
        "",
        "| id | codepoint | nama | jumlah |",
        "|---:|---|---|---:|",
    ]
    for i, ch in enumerate(tokenizer.charset, start=1):
        cp = f"U+{ord(ch):04X}"
        lines.append(f"| {i} | {cp} | {unicodedata.name(ch, '?')} | {counts.get(cp, 0):,} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    lines: list[str] = []
    for name in ("train", "val", "test"):
        lines += (SPLITS_DIR / f"{name}.txt").read_text(encoding="utf-8").splitlines()
    fonts = default_font_paths()
    tokenizer, stats = Tokenizer.build(lines, fonts)
    tokenizer.save(TOKENIZER_PATH, stats)
    write_charset_doc(tokenizer, stats, CHARSET_DOC_PATH)

    print(f"baris                 : {stats['lines']:,}")
    print(f"kelas (termasuk blank): {stats['classes_including_blank']}")
    print(f"pola reorder          : {stats['reorder_patterns']} ({stats['non_identity_patterns']} non-identitas)")
    print(f"pre-base              : {', '.join(stats['prebase'])}")
    print(f"laju sebelum-dasar    : {stats['before_base_rate']}")
    print(f"font tidak sepakat    : {stats['font_disagreement']}")
    print(f"kecocokan HarfBuzz    : {stats['harfbuzz_agreement']:.4%}")
    print(f"pola ambigu dibuang   : {len(stats['ambiguous_patterns'])} ({stats['ambiguous_occurrences']:,} kemunculan)")
    for a in stats["ambiguous_patterns"][:10]:
        print(f"   {a}")
    print(f"tersimpan             : {TOKENIZER_PATH}, {CHARSET_DOC_PATH}")


if __name__ == "__main__":
    main()
