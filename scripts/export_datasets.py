"""Ekspor kartu data untuk halaman Dataset di web (kontrak data dataset, skema 1).

  .venv/Scripts/python scripts/export_datasets.py          # ~20 dtk di CPU, tanpa memuat model

Keluaran: out/results/datasets.json (ditimpa). Web mengimpornya dengan `php artisan aksara:datasets`;
`php artisan aksara:import` ikut mengimpornya bila berkasnya ada di folder hasil.

Isi:
  corpus, splits   korpus teks dan pembagian latih/validasi/uji: jumlah baris dihitung ulang dari data/splits dan
                   dicocokkan dengan data/corpus_stats.json (berhenti kalau berbeda)
  usage, lineage   pemakaian oleh run resmi (OFFICIAL_RUN di scripts/export_results.py) dan rantai checkpoint yang
                   dilanjutkannya: argumen dan langkah dibaca dari checkpoint, proses training dari log.jsonl, baris
                   yang dijadwalkan dari urutan batch DataLoader src.train (lihat Schedule.batches)
  rare             karakter langka di bagian latih (aturan src.text_augment) dan seberapa sering run resmi melihatnya
  datasets         daftar dataset: jumlah dihitung, sumber dan lisensi dari catatan di repo ini
  fonts            font per peran, lebar aksara yang sama dengan font uji, glyph yang tidak ada
  support          data pendukung milik repo OCR (model bahasa karakter, charset, uji buta VLM)

Teks label data nyata tidak pernah ditulis ke keluaran: hanya jumlah. Semua angka dihitung di sini; web hanya
menampilkan.
"""

import argparse
import csv
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import uharfbuzz as hb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export_results import (  # noqa: E402  (scripts/ bukan paket: dijalankan dari scripts/ atau lewat sys.path test)
    CHECKPOINTS,
    LEGACY_REPORTS,
    LM_PATH,
    OFFICIAL_LINES,
    OFFICIAL_RUN,
    OPTIONAL_CHECKPOINTS,
    checkpoint_meta,
    gate_problem,
    read_json,
    write_atomic,
)
from src.corpus import MAX_LEN, MAX_ROUNDTRIP_CER, MIN_COUNT, SPLIT_BUCKETS  # noqa: E402
from src.dataset import MIN_SPACE_RATIO, LengthBucketSampler, space_ratio  # noqa: E402
from src.text_augment import RARE_MAX_LINE_FRACTION  # noqa: E402
from src.tokenizer import is_javanese, nfc  # noqa: E402

SCHEMA = 1
SPLITS = ("train", "val", "test")
FONT_SUFFIXES = {".ttf", ".otf"}
# Font uji G1/G2 dicari di sini menurut nama berkas di laporan src.evaluate (javatext.ttf bawaan Windows).
TEST_FONT_DIR = Path("C:/Windows/Fonts")
# Aksara, angka, dan pada: codepoint blok Jawa yang bukan tanda (sandhangan menempel pada aksara lain, jadi lebar
# majunya tidak bisa dibandingkan antar font).
SPACING = tuple(chr(cp) for cp in range(0xA980, 0xA9E0) if unicodedata.category(chr(cp)) not in ("Mn", "Mc", "Cn"))
# Font latih dianggap sekeluarga dengan font uji bila lebar maju sebanyak ini dari SPACING persis sama. Terukur
# 2026-10-05: CarakanJawa 72 dari 73 (satu-satunya beda: aksara O yang tidak dimilikinya), font latih lain 0 dari 73.
FAMILY_SHARE = 0.9

FONT_GROUPS = (("core", "fonts"), ("extra", "fonts/extra"), ("review", "fonts/extra_review"),
               ("rejected", "fonts/extra_rejected"))
# Font inti ikut repo; sumber dan lisensinya sama dengan README bagian "Pihak ketiga".
CORE_FONTS = {
    "NotoSansJavanese-Regular.ttf": {"license": "SIL OFL 1.1", "source": "https://github.com/notofonts/javanese"},
    "TuladhaJejegOT-Regular.ttf": {"license": "SIL OFL 1.1", "source": "https://github.com/akufadhl/Tuladha-Jejeg-OT"},
}
TEST_FONTS = {
    "javatext.ttf": {"license": "bawaan Windows (Microsoft), tidak boleh disebarkan",
                     "source": "C:/Windows/Fonts/javatext.ttf"},
}
# Cacat yang tidak bisa dihitung skrip ini (ditemukan tinjauan 2026-10-03, dicatat di CLAUDE.md bagian font).
FONT_NOTES = {
    "TuladhaJejegOT-Regular.ttf": ["Konversi OpenType; versi aslinya hanya punya tabel Graphite, yang tidak dibaca "
                                   "HarfBuzz."],
    "BasaJan.ttf": ["Di lingkungan training (Pillow 12.3.0, HarfBuzz 14.2.1) aturan GSUB font ini tidak diterapkan: "
                    "pasangan tidak menumpuk pada sekitar 10% sampel sintetis. Ditemukan 2026-10-03, belum "
                    "diperbaiki."],
    "NewKramawirya.ttf": ["Pada lungsi sesudah pangkon digambar dengan glyph pada lingsa, jadi dua label untuk "
                          "gambar yang sama (pangkon + lungsi ada di 12% baris latih)."],
    "javatext.ttf": ["Javanese Text bawaan Windows; tidak dipakai merender data latih."],
}
# Font yang catatannya di atas adalah cacat (citra tidak sepadan dengan labelnya, atau tidak dirender dengan benar di
# lingkungan training). Kartu metode menghitung font latih bercacat dari daftar ini ditambah font yang tidak punya
# glyph sebuah karakter charset, supaya tidak menyebut label sintetis "pasti benar".
FONT_DEFECTS = ("BasaJan.ttf", "NewKramawirya.ttf")


def count_lines(path: Path) -> int:
    """Jumlah baris berkas teks tanpa memuatnya ke memori (baris terakhir boleh tanpa newline)."""
    count, last = 0, b"\n"
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            count += chunk.count(b"\n")
            last = chunk[-1:]
    return count + (last != b"\n")


def relative(path: Path, root: Path) -> str:
    """Jalur relatif terhadap repo. Berkas di luar repo hanya disebut dua bagian terakhirnya: jalur utuhnya bisa
    memuat nama folder pengguna, dan kartu ini disimpan web lalu bisa ditampilkan."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return "(luar repo)/" + "/".join(path.parts[-2:])


def day(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def pipeline_key(run: str) -> str | None:
    """Kunci pipeline web yang membaca checkpoint run ini: aturan `official_pipeline` di export_results.py, tanpa
    membuka checkpoint. Web membandingkannya dengan pipeline resminya untuk tahu kartu data ini masih berlaku."""
    paths = {**CHECKPOINTS, **{spec["key"]: Path(spec["run"]) / "last_snapshot.pt" for spec in OPTIONAL_CHECKPOINTS}}
    keys = [key for key, path in paths.items() if path.parent.name == run]
    return keys[0] if len(keys) == 1 else None


def read_charset(path: Path) -> list[str]:
    """Karakter charset tokenizer (tanpa blank); berkasnya menyimpan "U+XXXX", seperti Tokenizer.load."""
    return [chr(int(code[2:], 16)) for code in read_json(path)["charset"]]


def corpus_card(root: Path) -> tuple[dict, list[dict]]:
    """Korpus dan pembagiannya. Jumlah baris dihitung dari berkas split, lalu wajib sama dengan corpus_stats.json."""
    stats_path = root / "data/corpus_stats.json"
    if not stats_path.exists():
        raise SystemExit(f"{stats_path} tidak ada; bangun korpus dulu: python -m src.corpus")
    stats = read_json(stats_path)
    splits = []
    for key in SPLITS:
        path = root / "data/splits" / f"{key}.txt"
        if not path.exists():
            raise SystemExit(f"{path} tidak ada; bangun korpus dulu: python -m src.corpus")
        lines = count_lines(path)
        if lines != stats["split_lines"][key]:
            raise SystemExit(f"{path}: {lines} baris, tetapi corpus_stats.json mencatat {stats['split_lines'][key]}; "
                             "berkas split dan statistiknya bukan dari pembangunan korpus yang sama")
        splits.append({"key": key, "file": f"data/splits/{key}.txt", "lines": lines, "injected": stats["injected"][key]})
    total = sum(s["lines"] for s in splits)
    if total != stats["total_lines"]:
        raise SystemExit(f"jumlah baris split {total} berbeda dari total_lines {stats['total_lines']} di corpus_stats.json")
    for s in splits:
        s["share"] = s["lines"] / total
    passed, duplicates, injected = stats["roundtrip_passed"], stats["duplicates"], stats["injected_total"]
    corpus = {
        "articles": stats["articles"],
        "candidates": stats["candidates"],
        "roundtrip_passed": passed,
        "roundtrip_pass_rate": stats["roundtrip_pass_rate"],
        "duplicates": duplicates,
        "injected": injected,
        # Selisih yang tidak dijelaskan langkah di atas; 0 pada korpus sekarang (lolos - duplikat + sisipan = total).
        "other": total - (passed - duplicates + injected),
        "lines": total,
        "leaked_lines": stats["leaked_lines"],
        # Aturan pembangunan korpus (src/corpus.py): keranjang hash id artikel per bagian dari 100, batas CER uji
        # bolak-balik, dan kemunculan minimum tiap codepoint aksara Jawa yang dikejar baris sisipan.
        "split_buckets": {key: len(SPLIT_BUCKETS[key]) for key in SPLITS},
        "max_roundtrip_cer": MAX_ROUNDTRIP_CER,
        "min_count": {key: MIN_COUNT[key] for key in SPLITS},
        "length_histogram": histogram(stats["length_histogram"]),
        "built": day(stats_path),
    }
    return corpus, splits


def histogram(bins: dict[str, int]) -> list[dict]:
    """Sebaran panjang baris. Keranjang terakhir corpus_stats.json berlabel "70-79" tetapi juga memuat baris sepanjang
    MAX_LEN (src.corpus.length_histogram memotong panjang ke MAX_LEN - 1), jadi labelnya dibetulkan di sini."""
    rows = [{"range": key, "lines": value} for key, value in bins.items()]
    if rows and rows[-1]["range"].endswith(f"-{MAX_LEN - 1}"):
        rows[-1]["range"] = rows[-1]["range"].rsplit("-", 1)[0] + f"-{MAX_LEN}"
    return rows


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def effective_segments(rows: list[dict], upto: int) -> list[tuple[int, int]]:
    """Rentang langkah tiap proses training yang ikut membentuk checkpoint di langkah `upto`.

    Setiap proses src.train mengulang urutan batch LengthBucketSampler dari awal (epoch tidak disimpan di checkpoint),
    jadi proses yang mulai di langkah s dan sampai langkah e melihat batch ke-0 sampai ke-(e-s-1). Langkah sesudah
    titik lanjut proses berikutnya hilang bersama prosesnya, begitu juga langkah sesudah `upto`.
    """
    kept: list[list[int]] = []
    for row in rows:
        step = row.get("step")
        if not isinstance(step, int):
            continue
        if row.get("event") == "start":
            kept = [[a, min(b, step)] for a, b in kept if a < step]
            kept.append([step, step])
        elif kept:
            kept[-1][1] = max(kept[-1][1], step)
    return [(a, min(b, upto)) for a, b in kept if min(b, upto) > a]


def chain(run: str, root: Path) -> tuple[list[dict], bool]:
    """Rantai checkpoint dari bobot awal sampai run resmi (urutan waktu), mengikuti args.init tiap checkpoint.

    Checkpoint run resmi wajib ada. Bila checkpoint leluhur sudah dihapus, rantai berhenti di sana dan nilai kedua
    False (jumlah langkah dan baris di rantai lalu hanya batas bawah).
    """
    path = root / "out/checkpoints" / run / "last_snapshot.pt"
    links, seen, complete = [], set(), True
    while True:
        if path in seen:
            raise SystemExit(f"rantai checkpoint berputar di {path}")
        seen.add(path)
        if links and not path.exists():
            complete = False
            break
        step, args = checkpoint_meta(path)
        links.append({"run": path.parent.name, "path": path, "step": step, "args": args})
        init = args.get("init") or ""
        if not init:
            break
        path = Path(init) if Path(init).is_absolute() else root / init
    return links[::-1], complete


def font_files(folder: Path) -> list[Path]:
    """Berkas font di sebuah folder, dengan aturan yang sama dengan TRAIN_FONTS di src/dataset.py."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in FONT_SUFFIXES)


class Schedule:
    """Baris bagian latih yang dijadwalkan sebuah run, dengan urutan yang sama dengan src.train.

    src.train mengacak daftar baris dengan random.Random(seed) lalu mengambil `train_lines` baris pertama; hasil acak
    itu hanya bergantung pada jumlah baris dan seed, jadi kumpulan run yang lebih kecil selalu awalan kumpulan run yang
    lebih besar (seed sama). Urutan batch = urutan DataLoader di awal sebuah proses, yang diulang tiap proses.
    """

    def __init__(self, lines: list[str]):
        self.lines = lines
        self._order: dict[int, list[int]] = {}
        self._epochs: dict[tuple[int, int, int, bool], tuple[LengthBucketSampler, list[list[list[int]]]]] = {}

    def order(self, seed: int) -> list[int]:
        if seed not in self._order:
            order = list(range(len(self.lines)))
            random.Random(seed).shuffle(order)
            self._order[seed] = order
        return self._order[seed]

    def pool(self, size: int, seed: int) -> list[int]:
        """Indeks baris (di berkas split) yang masuk kumpulan run."""
        return self.order(seed)[:size]

    def batches(self, size: int, seed: int, batch_size: int, steps: int, workers: int) -> list[list[int]]:
        """`steps` batch pertama sebuah proses training, tiap batch berupa indeks baris di berkas split.

        Urutannya urutan DataLoader yang sebenarnya, bukan panggilan pertama sampler. Dengan worker (num_workers > 0)
        DataLoader PyTorch memanggil iter(sampler) dua kali sebelum batch pertama (sekali saat iteratornya dibuat,
        sekali lagi di _reset), dan LengthBucketSampler menaikkan epoch-nya di tiap panggilan, jadi epoch pertama
        memakai random.Random(seed + 1), bukan seed + 0. Ditemukan tinjauan 2026-10-05: terukur dengan
        src.train.make_loader sungguhan, dan jumlah split_batches di log training hanya cocok dengan urutan ini.
        Batch terakhir sebuah kelompok panjang bisa lebih kecil dari batch_size. Lewat satu epoch, urutan diacak lagi.
        """
        pool = self.pool(size, seed)
        key = (len(pool), seed, batch_size, workers > 0)
        if key not in self._epochs:
            sampler = LengthBucketSampler([self.lines[i] for i in pool], batch_size, shuffle=True, seed=seed)
            if workers > 0:
                iter(sampler)
            self._epochs[key] = (sampler, [])
        sampler, epochs = self._epochs[key]
        taken: list[list[int]] = []
        epoch = 0
        while len(taken) < steps:
            if epoch == len(epochs):
                epochs.append(list(iter(sampler)))
            if not epochs[epoch]:
                break
            taken.extend(epochs[epoch][: steps - len(taken)])
            epoch += 1
        return [[pool[i] for i in batch] for batch in taken]


def logged_fonts(rows: list[dict], step: int) -> list[str] | None:
    """Nama font perender data latih seperti dicatat src.train di event start, dari proses terakhir yang mulai SEBELUM
    langkah checkpoint: proses itulah yang menulis checkpoint-nya (proses yang mulai tepat di langkah itu baru
    melanjutkannya, dan fontnya bisa lain). None = log run ini belum mencatat daftar font (run sebelum 2026-10-06)."""
    for row in reversed(rows):
        if (row.get("event") == "start" and isinstance(row.get("step"), int) and row["step"] < step
                and isinstance(row.get("fonts"), list)):
            return [str(name) for name in row["fonts"]]
    return None


def run_usage(link: dict, schedule: Schedule, root: Path, warnings: list[str]) -> tuple[dict, set[int] | None]:
    """Data latih yang dipakai satu run di rantai: kumpulan baris, sampel, baris berbeda yang dijadwalkan."""
    args, step, run = link["args"], link["step"], link["run"]
    batch_size, seed = args["batch_size"], args.get("seed", 0)
    pool = min(args["train_lines"], len(schedule.lines))
    rows = read_log(link["path"].parent / "log.jsonl")
    segments = effective_segments(rows, step)
    if sum(b - a for a, b in segments) != step:
        warnings.append(f"Log run {run} tidak menjelaskan semua {step} langkahnya; baris berbeda dihitung seolah "
                        "satu proses tanpa lanjutan (batas atas).")
        segments = [(0, step)]
    starts = [row for row in rows if row.get("event") == "start" and isinstance(row.get("step"), int)]
    # Run yang dimundurkan: sebuah proses mulai di langkah sebelum langkah checkpoint padahal proses sebelumnya sudah
    # melewati langkah itu. Log tidak mencatat proses mana yang menulis berkasnya (best.pt bisa milik silsilah yang
    # ditinggalkan), jadi hitungan dari proses-proses terakhir belum tentu benar. Melanjutkan tepat di langkah
    # checkpoint, atau sebelum proses mana pun mencapainya, bukan pemunduran.
    reached, rollback = -1, None
    for row in rows:
        at = row.get("step")
        if not isinstance(at, int):
            continue
        if row.get("event") == "start" and at < step <= reached:
            rollback = at
        reached = max(reached, at)
    if rollback is not None:
        warnings.append(f"Log run {run}: sebuah proses mulai lagi di langkah {rollback} sesudah proses sebelumnya melewati "
                        f"langkah checkpoint {step}, jadi tidak pasti proses mana yang menulis checkpoint-nya; baris "
                        "berbeda dihitung dari proses-proses terakhir.")
    data_args = ("train_lines", "seed", "batch_size", "workers")
    if any(row.get("args", {}).get(key, args.get(key)) != args.get(key) for row in starts for key in data_args):
        warnings.append(f"Run {run} dilanjutkan dengan argumen data yang berbeda dari checkpoint-nya; baris berbeda "
                        "dihitung dengan argumen checkpoint untuk semua prosesnya.")
    scheduled: set[int] | None = set()
    samples = step * batch_size
    if args.get("overfit") or args.get("real_train"):
        warnings.append(f"Run {run} memakai --overfit atau --real-train; baris berbeda tidak dihitung.")
        scheduled = None
    else:
        # Tiap proses mengulang urutan dari awal: sampel dijumlahkan per proses, baris berbeda digabung.
        samples = 0
        for a, b in segments:
            for batch in schedule.batches(pool, seed, batch_size, b - a, args.get("workers") or 0):
                samples += len(batch)
                scheduled.update(batch)
    extra = len(font_files(root / args["extra_fonts"])) if args.get("extra_fonts") else 0
    if args.get("extra_fonts") and not extra:
        warnings.append(f"Run {run} dilatih dengan font tambahan dari {args['extra_fonts']}, tetapi folder itu kosong "
                        "atau tidak ada di mesin ini; jumlah dan daftar font di kartu ini tidak lengkap.")
    core = len(font_files(root / "fonts"))
    # Daftar font yang sebenarnya dipakai hanya ada di log run yang lebih baru (src.train mencatatnya di event start).
    # Tanpa itu jumlahnya dihitung dari folder font SEKARANG, yang bisa sudah berubah sejak run itu dilatih.
    logged = logged_fonts(rows, step)
    if logged is not None:
        fonts, extra = len(logged), sum(1 for name in logged if name not in CORE_FONTS)
    else:
        fonts = core + extra
    return {
        "run": run,
        "checkpoint": relative(link["path"], root),
        "steps": step,
        "batch_size": batch_size,
        "samples": samples,
        "pool": pool,
        "seed": seed,
        "processes": len(segments),
        "segments": [list(s) for s in segments],
        "distinct_lines": None if scheduled is None else len(scheduled),
        # Data nyata ikut dilatih atau dipakai sebagai validasi (Fase 6): jadwal dan validasi sintetis tidak berlaku.
        "real_train": bool(args.get("real_train")),
        "real_val": bool(args.get("real_val")),
        "fonts": fonts,
        "fonts_source": "log" if logged is not None else "folder",
        "font_names": logged,
        "extra_fonts": extra,
        "augment": args.get("augment") or "none",
        "drop_space_prob": args.get("drop_space_prob") or 0.0,
        "track_prob": args.get("track_prob") or 0.0,
        "track_max": args.get("track_max") or 0.0,
        "rare_insert_prob": args.get("rare_insert_prob") or 0.0,
        "rare_opener_prob": args.get("rare_opener_prob") or 0.0,
        "val_lines": args.get("val_lines") or 0,
        "val_steps": sorted({row["step"] for row in rows if "val_cer" in row and row.get("step", 0) <= step}),
    }, scheduled


def gate_reports(run: str, root: Path, real_lines: int) -> list[dict]:
    """Laporan gerbang resmi run: berapa baris dan font apa yang dipakai. Berhenti bila sebuah laporan tidak sah."""
    prefix = LEGACY_REPORTS.get(run, run)
    reports = []
    for code, name in (("G1", f"{prefix}_G1_10k"), ("G2", f"{prefix}_G2_10k"), ("G3", f"{prefix}_G3_full")):
        path = root / "out/eval" / f"{name}.json"
        if not path.exists():
            raise SystemExit(f"run resmi {run}: laporan {code} {path} tidak ada (G1/G2: scripts/make_official.sh {run})")
        report = read_json(path)
        problem = gate_problem(code, report, run, real_lines)
        if problem:
            raise SystemExit(f"run resmi {run}: laporan {code} {path} tidak sah ({problem})")
        results = report["results"]
        fonts = [] if code == "G3" else [font for font in results if font != "semua"]
        lines = results["semua"]["lines"] + sum(results[f].get("rejected_by_min_frames", 0) for f in fonts)
        reports.append({"code": code, "lines": lines, "fonts": fonts, "split": report.get("split"),
                        "augment": report.get("augment"), "source": f"out/eval/{name}.json"})
    return reports


def other_reports(root: Path, official: set[str]) -> dict:
    """Laporan src.evaluate di luar gerbang resmi yang juga membaca data uji: bagian uji korpus tidak "disimpan sampai
    akhir" secara harfiah. quick = tes cepat penentu arah (kurang dari OFFICIAL_LINES baris pertama), full = evaluasi
    penuh run lain, real = semua laporan gerbang G3 pada data nyata, termasuk yang resmi."""
    found = {"quick": 0, "quick_max_lines": 0, "full": 0, "real": 0}
    for path in sorted((root / "out/eval").glob("*.json")):
        try:
            report = read_json(path)
        except (OSError, ValueError):
            continue
        results = report.get("results") if isinstance(report, dict) else None
        if not isinstance(results, dict) or not isinstance(results.get("semua"), dict):
            continue
        # Baris yang dibaca = baris per font: laporan berfont banyak merender baris yang SAMA dengan tiap font, dan
        # results.semua.lines menjumlahkan pasangan baris x font (100 baris x 2 font = 200). Baris yang ditolak
        # (terlalu padat) tetap dibaca.
        per_font = [value.get("lines", 0) + value.get("rejected_by_min_frames", 0)
                    for name, value in results.items() if name != "semua" and isinstance(value, dict)]
        lines = max(per_font) if per_font else results["semua"].get("lines") or 0
        if report.get("goal") == "G3":
            found["real"] += 1
        elif report.get("split") == "test" and relative(path, root) not in official:
            if lines < OFFICIAL_LINES:
                found["quick"] += 1
                found["quick_max_lines"] = max(found["quick_max_lines"], lines)
            else:
                found["full"] += 1
    return found


def other_readers(root: Path) -> list[dict]:
    """Evaluasi di luar out/eval yang juga membaca bagian uji korpus: scripts/eval_rare.py (aksara langka sintetis,
    out/compare/**/rare_synthetic.json) dan scripts/beam_eval.py (render bersih font uji, out/beam/*.json). Keduanya
    mengambil barisnya secara acak dari SELURUH bagian uji, jadi baris di luar N pertama gerbang resmi ikut terbaca."""
    readers = []
    reports = [("rare_synthetic", path) for path in sorted((root / "out/compare").rglob("rare_synthetic.json"))]
    reports += [("beam", path) for path in sorted((root / "out/beam").glob("*.json"))]
    for kind, path in reports:
        try:
            report = read_json(path)
        except (OSError, ValueError):
            continue
        if not isinstance(report, dict):
            continue
        if kind == "rare_synthetic":
            settings = report.get("settings") or {}
            lines = settings.get("lines") if Path(str(settings.get("split", ""))).name == "test.txt" else None
            checkpoints = len(report.get("checkpoints") or [])
        else:
            lines, checkpoints = (report.get("clean_heldout_font") or {}).get("lines"), 1
        if isinstance(lines, int) and lines > 0:
            readers.append({"kind": kind, "file": relative(path, root), "lines": lines, "checkpoints": checkpoints})
    return readers


def other_real_readers(root: Path) -> list[dict]:
    """Evaluasi di luar out/eval yang juga membaca data uji nyata: scripts/beam_eval.py menyimpan hasil seluruh baris
    nyata di blok "real" laporannya (out/beam/*.json). Tanpa ini "sudah dievaluasi N kali" hanya menghitung out/eval."""
    readers = []
    for path in sorted((root / "out/beam").glob("*.json")):
        try:
            report = read_json(path)
        except (OSError, ValueError):
            continue
        lines = (report.get("real") or {}).get("lines") if isinstance(report, dict) else None
        if isinstance(lines, int) and lines > 0:
            readers.append({"kind": "beam", "file": relative(path, root), "lines": lines})
    return readers


def recorded_fonts(root: Path) -> dict[str, int]:
    """Jumlah font latih tiap run seperti tercatat di ekspor hasil ("... · 10 font · ..." di konfigurasi pipeline),
    bila out/results/manifest.json ada. Dipakai untuk memperingatkan bila folder font sudah berubah sejak run dilatih."""
    path = root / "out/results/manifest.json"
    if not path.exists():
        return {}
    counts = {}
    for pipeline in read_json(path).get("pipelines") or []:
        config = str(pipeline.get("config") or "")
        run = re.match(r"crnn:([A-Za-z0-9_]+)", config)
        fonts = re.search(r"(\d+) font\b", config)
        if run and fonts:
            counts[run[1]] = int(fonts[1])
    return counts


def shared_advances(own: dict[str, float | None], reference: dict[str, float | None]) -> dict:
    """Berapa codepoint SPACING yang lebar majunya persis sama di dua font, dan apakah itu satu keluarga huruf."""
    same = sum(1 for ch in SPACING if own[ch] is not None and own[ch] == reference[ch])
    return {"same": same, "of": len(SPACING), "family": same / len(SPACING) >= FAMILY_SHARE}


def rare_coverage(lines: list[str], charset: list[str], pool: list[int], scheduled: set[int] | None) -> dict:
    """Karakter langka di bagian latih, dan berapa baris yang memuatnya di kumpulan dan jadwal run resmi.

    Langka = aturan src.text_augment: codepoint aksara Jawa di charset yang ada di < RARE_MAX_LINE_FRACTION baris.
    `scheduled` None = jadwal run resmi tidak dihitung: scheduled_lines kosong dan never_scheduled null.
    """
    per_line: Counter = Counter()
    for line in lines:
        per_line.update(set(line))
    javanese = [ch for ch in charset if is_javanese(ch)]
    limit = RARE_MAX_LINE_FRACTION * max(1, len(lines))
    rare = [ch for ch in javanese if per_line[ch] < limit]
    in_pool, in_schedule = Counter(), Counter()
    if rare:
        pattern = re.compile("[" + "".join(rare) + "]")
        for i in pool:
            found = set(pattern.findall(lines[i]))
            if found:
                in_pool.update(found)
                if scheduled is not None and i in scheduled:
                    in_schedule.update(found)

    def spread(counts: Counter) -> dict:
        values = [counts[ch] for ch in rare]
        return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)} if values else {}

    return {
        "codepoints": len(rare),
        "javanese": len(javanese),
        "charset": len(charset),
        "max_line_fraction": RARE_MAX_LINE_FRACTION,
        "train_lines": spread(per_line),
        "pool_lines": spread(in_pool),
        "scheduled_lines": spread(in_schedule) if scheduled is not None else {},
        "never_scheduled": sum(1 for ch in rare if not in_schedule[ch]) if scheduled is not None else None,
    }


def nusaaksara_card(path: Path) -> dict:
    """Jumlah baris, halaman, dan kondisi NusaAksara. Teks label hanya dihitung, tidak pernah ditulis ke keluaran."""
    if not path.exists():
        raise SystemExit(f"{path} tidak ada (data uji G3)")
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {
        "lines": len(rows),
        "pages": len({row["source_id"] for row in rows}),
        "conditions": dict(Counter(row["condition"] for row in rows).most_common()),
        "without_space": sum(" " not in nfc(row["text"]) for row in rows),
    }


def commons_card(folder: Path) -> dict | None:
    """Papan Commons: draf baris, berkas sumber, dan baris yang sudah diverifikasi (labels.tsv dari import_review)."""
    drafts, sources, labels = folder / "drafts.jsonl", folder / "sources.jsonl", folder / "labels.tsv"
    if not drafts.exists():
        return None
    rows = [json.loads(line) for line in drafts.read_text(encoding="utf-8").splitlines() if line.strip()]
    verified = 0
    if labels.exists():
        with labels.open(encoding="utf-8", newline="") as f:
            verified = sum(1 for _ in csv.DictReader(f, delimiter="\t"))
    return {
        "drafts": len(rows),
        "files": len({row.get("source_file") for row in rows}),
        "candidates": count_lines(sources) if sources.exists() else 0,
        "verified": verified,
        "licenses": dict(Counter(row.get("license") or "tidak tercatat" for row in rows).most_common()),
    }


def font_sources(path: Path) -> dict[str, dict]:
    """Tabel fonts/extra/SOURCES.md: berkas -> folder, sumber, lisensi, status pemeriksaan visual."""
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].startswith("`") and cells[0].endswith("`"):
            rows[cells[0].strip("`")] = {"folder": cells[1].strip("`"), "source": cells[2], "license": cells[3],
                                         "status": cells[4]}
    return rows


def advances(path: Path) -> dict[str, float | None]:
    """Lebar maju (dalam em) tiap codepoint SPACING; None bila font tidak punya glyph-nya."""
    face = hb.Face(hb.Blob.from_file_path(str(path)))
    font = hb.Font(face)
    out = {}
    for ch in SPACING:
        glyph = font.get_nominal_glyph(ord(ch))
        out[ch] = round(font.get_glyph_h_advance(glyph) / face.upem, 4) if glyph else None
    return out


def missing_glyphs(path: Path, charset: list[str]) -> list[str]:
    covered = set(hb.Face(hb.Blob.from_file_path(str(path))).unicodes)
    return [f"U+{ord(ch):04X}" for ch in charset if is_javanese(ch) and ord(ch) not in covered]


def font_cards(root: Path, official_args: dict, test_fonts: list[str], charset: list[str], test_font_dir: Path,
               warnings: list[str], logged: list[str] | None = None) -> list[dict]:
    """Font per peran untuk run resmi: latih + validasi (inti), latih (tambahan), uji, dan yang tidak dipakai.
    `logged` = nama font yang dicatat log run resmi (None bila log-nya belum mencatat): bila ada, peran latih hanya
    diberikan ke nama-nama itu, bukan ke semua isi folder font sekarang."""
    sources = font_sources(root / "fonts/extra/SOURCES.md")
    extra_dir = (root / official_args["extra_fonts"]).resolve() if official_args.get("extra_fonts") else None
    reference = {}
    cards = []
    for name in test_fonts:
        path = test_font_dir / name
        info = TEST_FONTS.get(name, {"license": "tidak tercatat", "source": path.as_posix()})
        card = {"file": name, "group": "test", "roles": ["test"], "license": info["license"], "source": info["source"],
                "in_repo": False, "available": path.exists(), "notes": list(FONT_NOTES.get(name, []))}
        if path.exists():
            reference[name] = advances(path)
            card["missing"] = missing_glyphs(path, charset)
        else:
            warnings.append(f"Font uji {name} tidak ditemukan di {test_font_dir}; kemiripan dengan font latih tidak dihitung.")
        cards.append(card)

    for group, folder in FONT_GROUPS:
        for path in font_files(root / folder):
            name = path.name
            if group == "core":
                info = CORE_FONTS.get(name)
                if info is None:
                    raise SystemExit(f"font inti {name} belum punya catatan lisensi di CORE_FONTS (scripts/export_datasets.py)")
                roles = ["train", "val"] if logged is None or name in logged else ["val"]
            else:
                info = sources.get(name)
                if info is None:
                    raise SystemExit(f"font {folder}/{name} tidak ada di tabel fonts/extra/SOURCES.md")
                if info["folder"] != Path(folder).name:
                    raise SystemExit(f"font {name} ada di {folder}, tetapi SOURCES.md mencatat folder {info['folder']}")
                in_folder = bool(extra_dir and path.parent.resolve() == extra_dir)
                roles = ["train"] if in_folder and (logged is None or name in logged) else []
            card = {"file": name, "group": group, "roles": roles, "license": info["license"], "source": info["source"],
                    "in_repo": group == "core", "available": True, "notes": list(FONT_NOTES.get(name, []))}
            if info.get("status") and group != "core":
                card["review"] = info["status"]
            if roles:
                card["missing"] = missing_glyphs(path, charset)
                card["drops_space"] = space_ratio(str(path)) < MIN_SPACE_RATIO
                own = advances(path)
                for test_name, ref in reference.items():
                    card["shared_with_test"] = {"font": test_name, **shared_advances(own, ref)}
            cards.append(card)
    order = {"core": 0, "extra": 1, "test": 2, "review": 3, "rejected": 4}
    return sorted(cards, key=lambda card: (order[card["group"]], card["file"].lower()))


def support_cards(root: Path, charset: list[str]) -> list[dict]:
    """Data pendukung milik repo OCR; data pendukung milik web (kamus, leksikon, anotasi) dihitung web sendiri."""
    items = [{"key": "charset", "file": "data/tokenizer.json", "characters": len(charset), "classes": len(charset) + 1}]
    lm = root / LM_PATH.relative_to(ROOT)
    match = re.fullmatch(r"o(\d+)_n(\d+)_ds([\d.]+)\.pkl", lm.name)
    if lm.exists() and match:
        items.append({"key": "charlm", "file": relative(lm, root), "order": int(match[1]), "lines": int(match[2]),
                      "drop_space_prob": float(match[3]), "split": "train"})
    blind = root / "out/eval/vlm_blind_50.json"
    if blind.exists():
        items.append({"key": "vlm_blind", "file": relative(blind, root), "lines": len(read_json(blind).get("items", []))})
    return items


def label_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f, delimiter="\t"))


def dataset_cards(corpus: dict, splits: list[dict], real: dict, commons: dict | None, gates: list[dict],
                  official_args: dict, root: Path) -> list[dict]:
    """Daftar dataset. Jumlah dari hitungan di atas; isi, sumber, dan lisensi dari catatan repo (README, SOURCE.md)."""
    lines = {s["key"]: s["lines"] for s in splits}
    synthetic = [g["code"] for g in gates if g["code"] != "G3"]
    cards = [
        {
            "key": "corpus",
            "name": "Korpus teks sintetis",
            "content": "Paragraf Wikipedia bahasa Jawa yang dialihaksarakan ke aksara Jawa. Citra baris dirender dari "
                       "teks ini saat training dan evaluasi; citranya tidak disimpan.",
            "count": corpus["lines"],
            "unit": "baris teks",
            "articles": corpus["articles"],
            "roles": {"train": lines["train"], "val": lines["val"], "test": lines["test"]},
            "gates": synthetic,
            "status": "used",
            "source": "Wikipedia bahasa Jawa, snapshot 20231101 (wikimedia/wikipedia di HuggingFace)",
            "source_url": "https://huggingface.co/datasets/wikimedia/wikipedia",
            "license": "CC BY-SA 4.0",
            "shareable": True,
            "in_repo": False,
            "share_note": "Boleh disebar dengan atribusi dan lisensi yang sama. Tidak ikut repo karena besar; dibangun "
                          "ulang dengan python -m src.corpus.",
        },
        {
            "key": "nusaaksara",
            "name": "NusaAksara, bagian aksara Jawa",
            "content": "Potongan baris dari pindaian buku cetak, ditranskripsi penutur asli dan komunitas aksara.",
            "count": real["lines"],
            "unit": "baris citra",
            "pages": real["pages"],
            "without_space": real["without_space"],
            "roles": {"test": real["lines"]},
            "gates": ["G3"],
            "status": "used",
            "source": "NusaAksara, ACL 2025 (HuggingFace NusaAksara/NusaAksara, config Image Transcription)",
            "source_url": "https://huggingface.co/datasets/NusaAksara/NusaAksara",
            "license": "non-komersial",
            "shareable": False,
            "in_repo": False,
            "share_note": "Lisensi non-komersial. Tidak disebarkan repo ini; dipakai lokal, hanya sebagai data uji.",
        },
    ]
    if commons:
        licenses = ", ".join(commons["licenses"])
        # Peran Commons mengikuti run resmi: baris yang dipakai --real-train / --real-val dari folder Commons.
        folder = (root / "data/real/commons").resolve()
        roles = {}
        for role, key in (("train", "real_train"), ("val", "real_val")):
            path = (root / official_args[key]).resolve() if official_args.get(key) else None
            if path and path.exists() and folder in path.parents:
                roles[role] = label_rows(path)
        pending = max(0, commons["drafts"] - commons["verified"])
        cards.append({
            "key": "commons",
            "name": "Papan nama Wikimedia Commons",
            "content": "Foto papan nama dan penanda beraksara Jawa di ruang publik. Potongan barisnya baru punya label "
                       "draf yang harus diperiksa pembaca aksara.",
            "count": commons["drafts"],
            "unit": "baris citra",
            "files": commons["files"],
            "verified": commons["verified"],
            "pending": pending,
            "roles": roles,
            "planned": [role for role in ("train", "val") if role not in roles],
            "gates": [],
            "status": "pending" if pending else "used" if roles else "verified",
            "source": f"Wikimedia Commons, {commons['files']} berkas foto",
            "source_url": "https://commons.wikimedia.org/wiki/Category:Javanese_script",
            "license": licenses,
            "shareable": True,
            "in_repo": False,
            "share_note": "Lisensi bebas per berkas, atribusi wajib. Belum ikut repo: labelnya belum diverifikasi.",
        })
    return cards


def build(root: Path = ROOT, run: str = OFFICIAL_RUN, test_font_dir: Path = TEST_FONT_DIR) -> dict:
    warnings: list[str] = []
    corpus, splits = corpus_card(root)
    charset = read_charset(root / "data/tokenizer.json")

    links, complete = chain(run, root)
    official = links[-1]
    train_path = root / "data/splits/train.txt"
    # Waktu berkas hanya petunjuk: last_snapshot.pt disalin ulang tiap evaluasi (scripts/fase6_common.sh), jadi log
    # training ikut dilihat. Tanpa hash korpus di checkpoint, korpus yang ditulis ulang tidak selalu terdeteksi.
    stamps = [path.stat().st_mtime for link in links for path in (link["path"], link["path"].parent / "log.jsonl")
              if path.exists()]
    oldest = min(stamps)
    if train_path.stat().st_mtime > oldest:
        warnings.append("data/splits/train.txt lebih baru daripada checkpoint di rantai run resmi: pemakaian di bawah "
                        "dihitung dari berkas yang sekarang, bukan yang dipakai saat training.")
    train_lines = train_path.read_text(encoding="utf-8").splitlines()
    if len(train_lines) != splits[0]["lines"]:
        raise SystemExit(f"{train_path}: splitlines memberi {len(train_lines)} baris, hitungan newline {splits[0]['lines']}")
    schedule = Schedule(train_lines)
    runs, seen, scheduled, known = [], set(), set(), True
    for link in links:  # sesudah putaran terakhir, `scheduled` = jadwal run resmi (None bila tidak dihitung)
        usage, scheduled = run_usage(link, schedule, root, warnings)
        usage["init"] = runs[-1]["run"] if runs else None
        runs.append(usage)
        known = known and scheduled is not None
        seen |= scheduled or set()
    train = runs[-1]
    # Jumlah font tiap run dihitung dari folder font sekarang kecuali log-nya mencatat daftarnya. Ekspor hasil
    # mencatat jumlahnya sendiri; bila berbeda, folder sudah berubah sejak run dilatih dan itu harus terlihat.
    recorded = recorded_fonts(root)
    for usage in runs:
        then = recorded.get(usage["run"])
        if usage["fonts_source"] == "folder" and then is not None and then != usage["fonts"]:
            warnings.append(f"Run {usage['run']}: ekspor hasil mencatat {then} font latih, tetapi folder font sekarang berisi "
                            f"{usage['fonts']}; jumlah dan daftar font di kartu ini mengikuti folder sekarang, bukan font "
                            "yang dipakai saat run itu dilatih.")

    # Log run resmi mencatat NAMA fontnya: tabel font di kartu ini memberi peran latih ke nama-nama itu saja, dan
    # selisihnya dengan folder sekarang (font yang sudah dihapus, font yang ditambahkan sesudahnya) harus terlihat.
    logged = train["font_names"]
    if logged:  # daftar kosong = run tanpa baris sintetis (--train-lines 0 dengan --real-train): tidak ada yang dibandingkan
        folder = {path.name for path in font_files(root / "fonts")}
        if official["args"].get("extra_fonts"):
            folder |= {path.name for path in font_files(root / official["args"]["extra_fonts"])}
        gone, added = sorted(set(logged) - folder), sorted(folder - set(logged))
        if gone:
            warnings.append(f"Run {run}: log mencatat {len(logged)} font latih, tetapi {len(gone)} di antaranya tidak ada "
                            f"lagi di folder font ({', '.join(gone)}); font itu tidak ada di tabel font kartu ini.")
        if added:
            warnings.append(f"Run {run}: folder font sekarang memuat {len(added)} font yang tidak tercatat di log run itu "
                            f"({', '.join(added)}); font itu tidak diberi peran latih.")

    real = nusaaksara_card(root / "data/real/nusaaksara/labels.tsv")
    gates = gate_reports(run, root, real["lines"])
    synthetic = [g for g in gates if g["code"] != "G3"]
    others = other_reports(root, {g["source"] for g in gates})
    test_fonts = sorted({name for g in synthetic for name in g["fonts"]})
    commons = commons_card(root / "data/real/commons")

    return {
        "schema": SCHEMA,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "metopenv5",
        "official": {"run": run, "pipeline": pipeline_key(run), "checkpoint": train["checkpoint"], "step": train["steps"]},
        "corpus": corpus,
        "splits": splits,
        "usage": {
            "train": train,
            "val": {"lines": min(train["val_lines"], splits[1]["lines"]), "steps": train["val_steps"],
                    "fonts": len(font_files(root / "fonts"))},
            # G1 dan G2 membaca baris yang sama: N baris pertama bagian uji (src.evaluate).
            "test": {"lines": max((g["lines"] for g in synthetic), default=0), "gates": synthetic,
                     "quick": others["quick"], "quick_max_lines": others["quick_max_lines"], "full": others["full"],
                     "others": other_readers(root)},
            "real": {"lines": real["lines"], "gates": [g for g in gates if g["code"] == "G3"], "reports": others["real"],
                     "others": other_real_readers(root)},
        },
        "lineage": {
            "runs": runs,
            "complete": complete,
            "steps": sum(r["steps"] for r in runs),
            "samples": sum(r["samples"] for r in runs),
            "pool": max(r["pool"] for r in runs),
            "distinct_lines": len(seen) if known else None,
        },
        "rare": rare_coverage(train_lines, charset, schedule.pool(train["pool"], train["seed"]), scheduled),
        "datasets": dataset_cards(corpus, splits, real, commons, gates, official["args"], root),
        "fonts": font_cards(root, official["args"], test_fonts, charset, test_font_dir, warnings, logged),
        "support": support_cards(root, charset),
        "warnings": warnings,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "out/results"))
    args = parser.parse_args(argv)
    card = build()
    if card["official"]["pipeline"] is None:
        raise SystemExit(f"run resmi {card['official']['run']} tidak dikenal scripts/export_results.py (CHECKPOINTS / "
                         "OPTIONAL_CHECKPOINTS); web tidak bisa mencocokkan kartu ini dengan pipeline resminya")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(out_dir / "datasets.json", json.dumps(card, ensure_ascii=False, indent=1, allow_nan=False) + "\n")

    train, lineage = card["usage"]["train"], card["lineage"]
    shares = " / ".join(f"{s['share']:.1%}" for s in card["splits"])
    print(f"korpus: {card['corpus']['lines']:,} baris ({shares})")
    def count(value) -> str:
        return "tidak dihitung" if value is None else f"{value:,}"

    print(f"run resmi {card['official']['run']}: kumpulan {train['pool']:,} baris, {train['samples']:,} sampel, "
          f"{count(train['distinct_lines'])} baris berbeda; rantai {len(lineage['runs'])} run, {lineage['steps']:,} langkah, "
          f"{count(lineage['distinct_lines'])} baris berbeda")
    print(f"font: {sum(1 for f in card['fonts'] if 'train' in f['roles'])} latih, "
          f"{sum(1 for f in card['fonts'] if 'test' in f['roles'])} uji; "
          f"sekeluarga dengan font uji: {[f['file'] for f in card['fonts'] if f.get('shared_with_test', {}).get('family')]}")
    for warning in card["warnings"]:
        print(f"PERINGATAN: {warning}")
    print(f"ditulis: {out_dir / 'datasets.json'}")


if __name__ == "__main__":
    main()
