"""Metrik tingkat karakter dari penjajaran referensi-hipotesis: recall, presisi, F1, akurasi, dan per kelas.

CER (`src.decode.cer`) tetap metrik gerbang PLAN.md §1.2. Metrik di sini pelengkapnya, dihitung dari penjajaran
yang SAMA dengan yang dipakai `scripts/compare_runs.py` untuk recall aksara langka: biaya Levenshtein minimum, lalu
kecocokan terbanyak, lalu substitusi sekelas terbanyak (`alignment`). Dari satu penjajaran diperoleh M cocok,
S substitusi, D hapus, I sisip, dengan |referensi| = M + S + D dan |hipotesis| = M + S + I, sehingga

    CER       = (S + D + I) / |referensi|          (bisa > 1)
    recall    = M / |referensi|                     bagian referensi yang terbaca benar
    presisi   = M / |hipotesis|                     bagian keluaran yang benar; turun bila model mengarang
    F1        = 2 x presisi x recall / (presisi + recall)
    akurasi   = M / (M + S + D + I)                 porsi langkah penjajaran yang benar, selalu 0..1

Semua agregat mikro (dijumlahkan atas semua baris). Per kelas (tiap karakter c): TP = M pada c, FN = S + D dengan
referensi c, FP = S + I dengan hipotesis c. Modul ini tanpa torch supaya bisa dipakai skrip apa pun.
"""

import unicodedata
from collections import Counter
from typing import Iterable, Sequence

Op = tuple[str, int | None, int | None]


def glyph_class(ch: str) -> int:
    """Kelas karakter untuk memutus seri penjajaran: 0 spasi, 1 tanda gabung (sandhangan, pangkon: kategori Unicode M),
    2 lainnya (aksara, angka, pada)."""
    if ch.isspace():
        return 0
    return 1 if unicodedata.category(ch).startswith("M") else 2


def alignment(reference: str, hypothesis: str) -> list[Op]:
    """Penjajaran karakter berbiaya Levenshtein minimum; di antara yang seri, yang kecocokannya terbanyak, lalu yang
    substitusinya paling banyak sekelas (`glyph_class`).

    Hasil: (tag, i, j) per langkah, urut dari kiri; tag equal/replace/delete/insert, i indeks referensi dan j indeks
    hipotesis (None bila langkah itu tidak memakainya). Opcodes rapidfuzz memilih di antara penjajaran seri tanpa
    melihat kecocokan: "ꦢꦪ" vs "ꦪ " (ꦢ tidak terbaca, ada spasi tambahan) dijajarkan sebagai dua substitusi, sehingga
    ꦪ yang terbaca benar dihitung salah. Pada keluaran nyata 1-2% karakter benar hilang dengan cara itu (2026-10-03).
    Kriteria ketiga memasangkan glyph dengan glyph: tanpa itu "ꦏꦊ" vs "ꦏ꧒ " (nga lelet terbaca angka dua, lalu spasi
    tambahan) dijajarkan sebagai ꦊ -> spasi, dan "ꦦꦂ" vs "꧘" (pa murda terbaca angka delapan, layar hilang) sebagai
    ꦂ -> ꧘, sehingga kebingungan homoglif hilang dari hitungan (crnn_fonts, nga lelet -> angka dua: hanya 12 dari 21
    tercatat; 2026-10-03). Jumlah edit dan kecocokan tidak berubah olehnya. Bila masih seri, pilihannya deterministik
    (diagonal, lalu hapus, lalu sisip, dari ujung kanan).
    """
    n, m = len(reference), len(hypothesis)
    big = n + m + 1  # lebih besar dari jumlah kecocokan maupun jumlah substitusi yang mungkin
    edit = big * big  # nilai = edit * jumlah edit - big * kecocokan - substitusi sekelas: urutan leksikografis
    ref_class = [glyph_class(ch) for ch in reference]
    hyp_class = [glyph_class(ch) for ch in hypothesis]

    def diagonal(i: int, j: int) -> int:
        return -big if reference[i] == hypothesis[j] else edit - (ref_class[i] == hyp_class[j])

    table = [list(range(0, (m + 1) * edit, edit))]
    for i in range(1, n + 1):
        ch, cls, prev, row = reference[i - 1], ref_class[i - 1], table[-1], [i * edit]
        for j in range(1, m + 1):
            diag = prev[j - 1] + (-big if ch == hypothesis[j - 1] else edit - (cls == hyp_class[j - 1]))
            row.append(min(diag, prev[j] + edit, row[j - 1] + edit))
        table.append(row)
    ops: list[Op] = []
    i, j = n, m
    while i or j:
        value = table[i][j]
        if i and j and value == table[i - 1][j - 1] + diagonal(i - 1, j - 1):
            i, j = i - 1, j - 1
            ops.append(("equal" if reference[i] == hypothesis[j] else "replace", i, j))
            continue
        if i and value == table[i - 1][j] + edit:
            i -= 1
            ops.append(("delete", i, None))
        else:
            j -= 1
            ops.append(("insert", None, j))
    ops.reverse()
    return ops


def char_counts(reference: str, hypothesis: str) -> dict[str, int]:
    """Jumlah langkah penjajaran satu baris: match, sub, del, ins. match + sub + del = |referensi|."""
    counts = Counter(tag for tag, _, _ in alignment(reference, hypothesis))
    return {"match": counts["equal"], "sub": counts["replace"], "del": counts["delete"], "ins": counts["insert"]}


def rates(match: int, sub: int, dele: int, ins: int) -> dict[str, float | None]:
    """recall, presisi, F1, akurasi dari jumlah langkah; None bila penyebutnya 0 (bukan 0%)."""
    ref_len, hyp_len, steps = match + sub + dele, match + sub + ins, match + sub + dele + ins
    recall = match / ref_len if ref_len else None
    precision = match / hyp_len if hyp_len else None
    f1 = None
    if recall is not None and precision is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "char_accuracy": match / steps if steps else None}


def char_metrics(pairs: Iterable[tuple[str, str]]) -> dict:
    """Metrik mikro semua baris: counts (match, sub, del, ins), precision, recall, f1, char_accuracy.

    Dihitung pada string apa adanya (termasuk spasi), sama dengan CER. Penjajaran = `alignment`, jadi recall di sini
    sama dengan rata-rata `matched_mask` scripts/compare_runs.py.
    """
    total = Counter()
    for reference, hypothesis in pairs:
        total.update(char_counts(reference, hypothesis))
    counts = {key: int(total[key]) for key in ("match", "sub", "del", "ins")}
    return {"counts": counts, **rates(counts["match"], counts["sub"], counts["del"], counts["ins"])}


def class_metrics(pairs: Sequence[tuple[str, str]]) -> list[dict]:
    """Recall, presisi, F1 per karakter, dari penjajaran yang sama: satu entri per karakter yang muncul di referensi
    atau hipotesis, diurutkan dari yang paling sering di referensi.

    tp = cocok; fn = substitusi/hapus dengan referensi karakter itu; fp = substitusi/sisip dengan hipotesis karakter
    itu. Sebuah substitusi c -> d menambah fn untuk c dan fp untuk d.
    """
    tp, fn, fp = Counter(), Counter(), Counter()
    for reference, hypothesis in pairs:
        for tag, i, j in alignment(reference, hypothesis):
            if tag == "equal":
                tp[reference[i]] += 1
            elif tag == "replace":
                fn[reference[i]] += 1
                fp[hypothesis[j]] += 1
            elif tag == "delete":
                fn[reference[i]] += 1
            else:
                fp[hypothesis[j]] += 1
    chars = set(tp) | set(fn) | set(fp)
    out = []
    for ch in sorted(chars, key=lambda c: (-(tp[c] + fn[c]), c)):
        ref_total, hyp_total = tp[ch] + fn[ch], tp[ch] + fp[ch]
        recall = tp[ch] / ref_total if ref_total else None
        precision = tp[ch] / hyp_total if hyp_total else None
        f1 = None
        if recall is not None and precision is not None:
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append({"char": ch, "code": f"U+{ord(ch):04X}", "name": unicodedata.name(ch, "?"),
                    "ref": ref_total, "hyp": hyp_total, "tp": tp[ch], "fn": fn[ch], "fp": fp[ch],
                    "precision": precision, "recall": recall, "f1": f1})
    return out


def macro_f1(classes: Iterable[dict]) -> float | None:
    """Rata-rata F1 kelas yang muncul di referensi (ref > 0); None bila tidak ada kelas."""
    values = [c["f1"] for c in classes if c["ref"] and c["f1"] is not None]
    return sum(values) / len(values) if values else None
