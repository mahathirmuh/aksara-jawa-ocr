"""Ekspor kartu metode untuk halaman Metode di web (kontrak metode, skema 1).

  .venv/Scripts/python scripts/export_methods.py           # beberapa detik di CPU; model dibuat tanpa inferensi

Keluaran: out/results/methods.json (ditimpa). Web mengimpornya dengan `php artisan aksara:methods`;
`php artisan aksara:import` ikut mengimpornya bila berkasnya ada di folder hasil.

Kartu ini mendaftar metode dan machine learning yang dipakai repo OCR, dikelompokkan menurut tahap. Tiap butir punya
penjelasan singkat, pengaturan, dan bila ada bukti terukur serta catatan batasannya. Asal isinya:
  model        dibangun dari model_config checkpoint run resmi (OFFICIAL_RUN) dan dimuati bobotnya: jumlah parameter,
               lapis, kanal, dan langkah lebar dihitung dari modul yang sebenarnya
  pelatihan    argumen dari checkpoint tiap run di rantai
  data         preset augmentasi dari src/augment.py, ukuran render dari src/dataset.py, font dari log run (bila
               mencatatnya), ekspor hasil, atau folder font sekarang
  bukti        gerbang, metrik, dan ablasi dari out/results/manifest.json; selisih dan selang kepercayaan dari
               out/compare/<A>_vs_<B>.json (scripts/compare_runs.py), hanya bila CER di berkas itu sama dengan manifest;
               evaluasi sintetis dari out/compare/spacing_synthetic.json. Bukti yang berkasnya tidak ada dilewati.
Kalimat yang menyebut cara kerja kode (pengoptimal, jadwal, rugi, normalisasi kontras, penghalusan model bahasa) dijaga
CODE_FACTS: skrip berhenti bila potongan kodenya sudah tidak ada. Dua angka yang tidak punya berkas untuk dihitung ulang
diberi sumber "catatan pengukuran di CLAUDE.md", dan nilai gerbang G4 dikutip dari manifest dengan sumbernya.
Metode tahap lanjutan alur (alih aksara, terjemahan, tingkat tutur) milik web dan ditambahkan web sendiri.
"""

import argparse
import inspect
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from export_datasets import (  # noqa: E402
    FONT_DEFECTS,
    TEST_FONT_DIR,
    advances,
    chain,
    font_files,
    gate_reports,
    missing_glyphs,
    nusaaksara_card,
    pipeline_key,
    read_log,
    recorded_fonts,
    relative,
    shared_advances,
)
from export_results import (  # noqa: E402
    ALPHA,
    BETA,
    LM_PATH,
    OFFICIAL_LINES,
    OFFICIAL_RUN,
    WIDTH,
    checkpoint_meta,
    read_json,
    write_atomic,
)
from src import charlm, train  # noqa: E402
from src.augment import PRESET_OPS, build_augment  # noqa: E402
from src.dataset import (  # noqa: E402
    DOWNSAMPLE,
    MIN_FRAMES_PER_TARGET,
    MIN_SPACE_RATIO,
    RENDER_SIZE_RANGE,
    H,
    space_ratio,
    to_tensor,
)
from src.evaluate import TARGETS  # noqa: E402
from src.model import CRNN  # noqa: E402
from src.tokenizer import BLANK  # noqa: E402

SCHEMA = 1
# Jenis butir: model = model deep learning, training = teknik pelatihan, data = teknik data sintetis,
# statistic = metode statistik, rule = aturan atau algoritme tanpa pembelajaran, evaluation = cara mengukur.
KINDS = ("model", "training", "data", "statistic", "rule", "evaluation")
# Status: official = bagian dari model resmi (arsitektur, data, pelatihan, cara baca), used = dipakai untuk mengukur,
# available = ada tetapi bukan jalur resmi, tested = diuji lalu tidak dipakai, comparator = pembanding.
STATUSES = ("official", "used", "available", "tested", "comparator")
# Potongan kode yang harus tetap ada supaya kalimat di kartu benar: nama -> (tempatnya, teksnya). Bila salah satu
# hilang, kodenya sudah berubah: perbarui butir yang menyebutnya, baru jalankan ulang.
CODE_FACTS = {
    "optimizer": ("src.train.main", "torch.optim.AdamW("),
    "scheduler": ("src.train.main", "torch.optim.lr_scheduler.OneCycleLR("),
    "loss": ("src.train.main", "nn.CTCLoss(blank=BLANK, zero_infinity=True)"),
    "clip": ("src.train.main", "nn.utils.clip_grad_norm_(model.parameters(), args.clip)"),
    "background": ("src.dataset.to_tensor", "background = float(np.percentile(arr, 50))"),
    "ink": ("src.dataset.to_tensor", "ink = float(np.percentile(arr, 1))"),
    "smoothing": ("src.charlm", "Witten-Bell"),
}
# Run kontrol tiap perlakuan (CLAUDE.md, aturan analisis fase 5 sampai 7): perlakuan -> {run perlakuan: run kontrol}.
# Kontrol sama dengan run perlakuannya kecuali SATU perlakuan itu, jadi sebuah run hanya boleh disebut kontrol untuk
# perlakuan di barisnya (fase7_track adalah kontrol sisipan aksara langka, bukan kontrol jarak).
CONTROLS = {
    "fonts": {"fase5_fonts": "fase5_core"},
    "tracking": {"fase7_track": "fase7_ctrl"},
    "rare": {"fase7_track_rare": "fase7_track", "fase6_rare": "fase6_ctrl"},
}
TREATMENTS = {"fonts": "font tambahan", "tracking": "jarak antar suku kata", "rare": "sisipan aksara langka"}
# Pipeline resmi bila manifest hasil tidak menyebutnya (ekspor sebelum kunci "official" ada); sama dengan web.
LEGACY_OFFICIAL = "crnn_fonts"
# Operasi augmentasi src/augment.py dengan kata sehari-hari, supaya kalimat di kartu hanya menyebut operasi yang
# memang ada di preset run resmi. Operasi baru di src.augment.OPS wajib ditambahkan di sini (dijaga test).
AUGMENT_WORDS = {
    "margin": "diberi tepi kosong",
    "tight": "dipotong rapat sampai tinta menyentuh tepi",
    "rotate": "dimiringkan sedikit",
    "elastic": "dilengkungkan halus",
    "stroke": "ditebal-tipiskan goresannya",
    "contrast": "diubah kontrasnya",
    "texture": "diberi bayangan dan butiran kertas",
    "blur": "diburamkan",
    "noise": "diberi bising",
    "binarize": "dijadikan hitam-putih",
    "speckle": "diberi bintik",
    "jpeg": "dikompresi",
}


def num(value: int) -> str:
    """Bilangan bulat gaya Indonesia: 4590909 -> "4.590.909"."""
    return f"{value:,}".replace(",", ".")


def dec(value: float, digits: int = 4) -> str:
    """Pecahan gaya Indonesia tanpa nol di belakang: 0.5 -> "0,5", 0.0003 -> "0,0003", 16.0 -> "16"."""
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}".replace(".", ",") + "%"


def points(value: float, digits: int = 2) -> str:
    """Selisih dua proporsi dalam poin persentase, bertanda: -0.0929 -> "−9,29 poin"."""
    sign = "−" if value < 0 else "+"
    return f"{sign}{abs(value) * 100:.{digits}f}".replace(".", ",") + " poin"


def interval(low: float, high: float, digits: int = 2) -> str:
    """Selang dua proporsi dalam poin persentase: "−11,92 sampai −6,83"."""
    def one(value: float) -> str:
        return ("−" if value < 0 else "+" if value > 0 else "") + f"{abs(value) * 100:.{digits}f}".replace(".", ",")
    return f"{one(low)} sampai {one(high)}"


def listing(items: list[str]) -> str:
    """Rincian dalam kalimat: "a", "a dan b", "a, b, dan c"."""
    if len(items) <= 2:
        return " dan ".join(items)
    return ", ".join(items[:-1]) + ", dan " + items[-1]


def entry(key: str, name: str, kind: str, status: str, summary: str, settings: list[tuple[str, str]] | None = None,
          evidence: str | None = None, evidence_source: str | None = None, files: list[str] | None = None,
          note: str | None = None) -> dict:
    """Satu butir metode. `evidence` = hasil terukur dengan sumbernya; `note` = batasan atau keterangan yang harus
    dibaca bersama butir itu (bukan hasil ukur)."""
    if kind not in KINDS or status not in STATUSES:
        raise ValueError(f"butir {key}: jenis {kind!r} atau status {status!r} tidak dikenal")
    return {"key": key, "name": name, "kind": kind, "status": status, "summary": summary,
            "settings": [list(pair) for pair in settings or []], "evidence": evidence,
            "evidence_source": evidence_source if evidence else None, "note": note, "files": files or []}


def model_card(checkpoint: Path) -> dict:
    """Arsitektur dan jumlah parameter, dihitung dari model yang dibangun dari checkpoint (bobot dimuat ketat)."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["model_config"]
    model = CRNN(config["n_classes"], channels=tuple(config["channels"]), hidden=config["hidden"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    def count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    frames = 64
    with torch.no_grad():
        out = model(torch.zeros(1, 1, model.height, frames * CRNN.WIDTH_STRIDE))
    if tuple(out.shape) != (1, frames, config["n_classes"]):
        raise SystemExit(f"keluaran model {tuple(out.shape)} tidak sesuai langkah lebar {CRNN.WIDTH_STRIDE}")
    if model.height != H or CRNN.WIDTH_STRIDE != DOWNSAMPLE:
        raise SystemExit("tinggi atau langkah lebar model berbeda dari konstanta src/dataset.py")
    convs = [m for m in model.cnn.modules() if isinstance(m, nn.Conv2d)]
    return {
        "classes": config["n_classes"],
        "channels": list(config["channels"]),
        "hidden": config["hidden"],
        "height": model.height,
        "feature_height": model.feat_height,
        "width_stride": CRNN.WIDTH_STRIDE,
        "conv_layers": len(convs),
        "kernel": convs[0].kernel_size[0],
        "lstm_layers": model.rnn.num_layers,
        "bidirectional": model.rnn.bidirectional,
        "parameters": {"total": count(model), "cnn": count(model.cnn), "proj": count(model.proj),
                       "rnn": count(model.rnn), "head": count(model.head)},
    }


def code_facts() -> dict:
    """Cara kerja kode yang disebut kartu, seperti tertulis di sumbernya; berhenti bila salah satu sudah berubah."""
    sources = {"src.train.main": inspect.getsource(train.main), "src.dataset.to_tensor": inspect.getsource(to_tensor),
               "src.charlm": inspect.getsource(charlm)}
    missing = [name for name, (where, text) in CODE_FACTS.items() if text not in sources[where]]
    start = re.search(r"OneCycleLR\([^)]*pct_start=([0-9.]+)", sources["src.train.main"])
    if missing or not start:
        raise SystemExit(f"kode tidak lagi cocok dengan kartu metode ({', '.join(missing) or 'pct_start'}); perbarui "
                         "butirnya dan CODE_FACTS di scripts/export_methods.py")
    return {"optimizer": "AdamW", "scheduler": "OneCycleLR", "pct_start": float(start[1]), "loss": "CTC"}


class Evidence:
    """Bukti terukur dari manifest hasil dan berkas pembanding; yang berkasnya tidak ada mengembalikan None."""

    SCOPE = "nusaaksara_745"  # baris nyata: cakupan berkas pembanding scripts/compare_runs.py

    def __init__(self, root: Path):
        self.root = root
        path = root / "out/results/manifest.json"
        if not path.exists():
            raise SystemExit(f"{path} tidak ada; jalankan dulu scripts/export_results.py")
        self.manifest = read_json(path)
        if self.manifest.get("limited"):
            raise SystemExit("manifest hasil dibuat dengan --limit; kartu metode butuh ekspor penuh")
        self.metrics = {(m["scope"], m["pipeline"]): m for m in self.manifest["metrics"]}
        self.gates = {g["code"]: g for g in self.manifest["gates"]}
        self.official = self.manifest.get("official") or LEGACY_OFFICIAL
        self.done = {p["key"] for p in self.manifest["pipelines"] if p.get("status") == "done"}

    def cer(self, scope: str, pipeline: str) -> float | None:
        return self.metrics.get((scope, pipeline), {}).get("cer")

    def compare(self, a: str, b: str) -> dict | None:
        """Laporan scripts/compare_runs.py untuk pipeline A dan B (selisih = A - B), bila ada, memang untuk keduanya,
        dan dihitung dari prediksi yang sama dengan manifest ini: CER kedua sisi harus sama dengan metrik manifest.
        Berkas dari ekspor hasil yang lain tidak dipakai, supaya selisih dan selang kepercayaannya tidak berasal dari
        dua ekspor yang berbeda."""
        path = self.root / "out/compare" / f"{a}_vs_{b}.json"
        if not path.exists():
            return None
        report = read_json(path)
        if report.get("a", {}).get("key") != a or report.get("b", {}).get("key") != b:
            return None
        for side, key in (("a", a), ("b", b)):
            now = self.cer(self.SCOPE, key)
            if now is None or abs(report["cer"][side] - now) > 1e-9:
                return None
        return report

    def g3_change(self, treated: str, control: str) -> tuple[str, str] | None:
        """Kalimat "G3 kontrol -> perlakuan (selisih; SK 95% per halaman)" dan berkas sumbernya."""
        after, before = self.cer(self.SCOPE, treated), self.cer(self.SCOPE, control)
        if after is None or before is None:
            return None
        text = f"G3 {pct(before)} → {pct(after)} ({points(after - before)}"
        source = "out/results/manifest.json"
        report = self.compare(treated, control)
        sign = 1.0
        if report is None:
            report, sign = self.compare(control, treated), -1.0
        if report:
            low, high = sorted(sign * edge for edge in report["cer"]["bootstrap_clusters"]["ci95"])
            text += f"; selang kepercayaan 95% per halaman {interval(low, high)}"
            source = relative(self.root / "out/compare" / f"{report['a']['key']}_vs_{report['b']['key']}.json", self.root)
        return text + ")", source

    def bootstrap(self) -> dict | None:
        """Pengaturan bootstrap seperti tercatat di berkas pembanding yang sah (jumlah pengambilan ulang, selang)."""
        for path in sorted((self.root / "out/compare").glob("*_vs_*.json")):
            a, _, b = path.stem.partition("_vs_")
            report = self.compare(a, b)
            if report and isinstance(report.get("bootstrap"), dict):
                return report["bootstrap"]
        return None


def run_args(root: Path, run: str) -> dict | None:
    """Argumen training sebuah run dari checkpoint-nya, atau None bila checkpoint-nya tidak ada di mesin ini."""
    path = root / "out/checkpoints" / run / "last_snapshot.pt"
    return checkpoint_meta(path)[1] if path.exists() else None


def tracking_control(root: Path, run: str) -> str | None:
    """Run kontrol perlakuan jarak untuk `run`: terdaftar di CONTROLS dan checkpoint-nya memang dilatih tanpa jarak."""
    control = CONTROLS["tracking"].get(run)
    args = run_args(root, control) if control else None
    return control if args is not None and not args.get("track_prob") else None


def font_facts(root: Path, link: dict, charset: list[str], recorded: dict[str, int]) -> dict:
    """Font latih sebuah run. Jumlahnya dari log run (bila mencatat daftarnya), kalau tidak dari ekspor hasil, kalau
    tidak dari folder font sekarang. Cacat dan lebar spasi hanya bisa diukur pada berkas yang ada di folder sekarang;
    `current` False berarti folder itu sudah tidak sama banyaknya dengan saat run dilatih."""
    args = link["args"]
    core = font_files(root / "fonts")
    paths = core + (font_files(root / args["extra_fonts"]) if args.get("extra_fonts") else [])
    logged = next((row["fonts"] for row in reversed(read_log(link["path"].parent / "log.jsonl"))
                   if row.get("event") == "start" and isinstance(row.get("fonts"), list)), None)
    if logged is not None:
        train_count, source = len(logged), "log"
    elif link["run"] in recorded:
        train_count, source = recorded[link["run"]], "results"
    else:
        train_count, source = len(paths), "folder"
    return {
        "train": train_count, "core": len(core), "extra": train_count - len(core), "source": source,
        "measured": len(paths), "current": len(paths) == train_count, "paths": paths,
        "defects": sorted(p.stem for p in paths if p.name in FONT_DEFECTS or missing_glyphs(p, charset)),
        "narrow": sum(1 for p in paths if space_ratio(str(p)) < MIN_SPACE_RATIO),
    }


def family_twins(paths: list[Path], gates: list[dict], test_font_dir: Path) -> list[dict]:
    """Font uji gerbang sintetis yang sekeluarga dengan sebuah font latih (aturan kartu data: lebar maju yang sama)."""
    twins = []
    for name in sorted({font for gate in gates if gate["code"] != "G3" for font in gate["fonts"]}):
        path = test_font_dir / name
        if not path.exists():
            continue
        reference = advances(path)
        for train_path in paths:
            shared = shared_advances(advances(train_path), reference)
            if shared["family"]:
                twins.append({"test": Path(name).stem, "train": train_path.stem, **shared})
    return twins



def family_note(twins: list[dict], subject: str) -> str | None:
    """Batasan G1/G2 dan evaluasi sintetis lain: font ujinya sekeluarga dengan font latih."""
    if not twins:
        return None
    parts = []
    for test in sorted({twin["test"] for twin in twins}):
        mine = [twin for twin in twins if twin["test"] == test]
        widest = max(mine, key=lambda twin: twin["same"])
        parts.append(f"font uji {test} sekeluarga dengan font latih {listing([twin['train'] for twin in mine])} (lebar "
                     f"{widest['same']} dari {widest['of']} aksara, angka, dan pada persis sama)")
    text = listing(parts)
    return (f"{text[0].upper()}{text[1:]}, jadi {subject} mengukur generalisasi di dalam keluarga huruf itu, bukan ke "
            "font yang belum pernah dilihat model.")


def probe_note(twins: list[dict]) -> str | None:
    """Batasan evaluasi sintetis tertarget: citranya dirender dengan font uji yang sekeluarga dengan font latih."""
    if not twins:
        return None
    tests = sorted({twin["test"] for twin in twins})
    trains = sorted({twin["train"] for twin in twins})
    return (f"Citranya dirender dengan font uji {listing(tests)}, yang sekeluarga dengan font latih {listing(trains)}, jadi "
            "hasilnya belum tentu berlaku untuk bentuk huruf lain.")


def spacing_note(report: dict, run: str, control: str | None, track_max: float) -> str | None:
    """Hasil scripts/eval_spacing.py untuk run resmi dan run kontrolnya: spasi palsu per 100 batas suku kata pada baris
    tanpa spasi yang direnggangkan sejauh jarak terbesar yang masih di dalam rentang latih run resmi, lalu pada jarak
    terbesar yang diuji di luar rentang itu (di sana gejalanya kembali)."""
    for font in (report.get("fonts") or {}).values():
        results = font.get("results") or {}

        def rows(name: str | None) -> list[dict]:
            return (results.get(name) or {}).get("tanpa-spasi", [])

        def widest(name: str | None) -> dict | None:
            inside = [row for row in rows(name) if 0 < row["tracking"] <= track_max + 1e-9]
            return max(inside, key=lambda row: row["tracking"]) if inside else None

        def per_hundred(row: dict) -> str:
            value = row["false_per_boundary"] * 100
            return dec(value, 2 if 0 < value < 0.05 else 1)  # nilai kecil yang bukan nol tidak ditulis "0"

        ours, theirs = widest(run), widest(control)
        if ours and theirs and ours["tracking"] == theirs["tracking"]:
            text = (f"Baris tanpa spasi yang direnggangkan {dec(ours['tracking'])} em: {per_hundred(ours)} spasi palsu per "
                    f"100 batas suku kata pada model resmi, {per_hundred(theirs)} pada run kontrol {control}.")
            outside = [row for row in rows(run) if row["tracking"] > track_max + 1e-9]
            if outside:
                far = max(outside, key=lambda row: row["tracking"])
                text += (f" Di luar rentang jarak yang dilatihkan ({dec(far['tracking'])} em) model resmi kembali "
                         f"mengeluarkan {per_hundred(far)}.")
            return text
    return None


def ablation_evidence(rows: list[dict], seeds: set) -> tuple[str, str] | None:
    """(bukti, catatan) dari ablasi kumulatif. Yang dikutip jumlah per kelompok operasi, bukan "penurunan terbesar":
    langkah-langkahnya saling menutupi (satu langkah naik, langkah berikutnya turun kembali) dan tiap run satu kali."""
    if len(rows) < 2 or any(row.get("G3") is None for row in rows):
        return None
    steps = [(b["added"], b["G3"] - a["G3"]) for a, b in zip(rows, rows[1:])]
    text = f"Ablasi {len(rows)} run: G3 {pct(rows[0]['G3'])} → {pct(rows[-1]['G3'])}."
    preset = [(name, delta) for name, delta in steps if name in PRESET_OPS]
    scan = [(name, delta) for name, delta in steps if name not in PRESET_OPS]
    if preset and scan and steps == preset + scan:
        text += (f" {len(preset)} operasi preset bersama mengubahnya {points(sum(delta for _, delta in preset))}; "
                 f"{len(scan)} operasi tiruan pindaian sesudahnya {points(sum(delta for _, delta in scan))} ("
                 + ", ".join(f"{name} {points(delta)}" for name, delta in scan) + ").")
    else:
        text += " Selisih tiap langkah: " + ", ".join(f"{name} {points(delta)}" for name, delta in steps) + "."
    name, rise = max(steps, key=lambda step: step[1])
    note = ("Operasi ditambahkan berurutan dan tiap run dilatih satu kali" + (" dengan seed yang sama" if len(seeds) == 1 else "")
            + ", jadi selisih satu langkah mencampur pengaruh operasinya dengan derau antar-run"
            + (f" (menambahkan {name} malah menaikkan G3 {points(rise).lstrip('+')})" if rise > 0 else "")
            + ". G3 adalah data uji: angka ini dibaca sebagai arah, bukan dasar memilih operasi.")
    return text, note


def ocr_group(model: dict, tokenizer: dict, evidence: Evidence) -> dict:
    params = model["parameters"]
    stats = tokenizer["stats"]
    gate = evidence.gates.get("G4")
    codes = [int(code[2:], 16) for code in tokenizer["charset"]]
    javanese = sum(0xA980 <= code <= 0xA9DF for code in codes)
    others = len(codes) - javanese  # karakter di luar blok Jawa: spasi
    names = dict(key.split(" ", 1) for key in stats["before_base_rate"])
    prebase = ", ".join(f"{names[code]} ({code})" if code in names else code for code in stats["prebase"])
    return {
        "key": "ocr",
        "title": "Pembaca aksara",
        "intro": "Satu model membaca seluruh baris sekaligus, tanpa memotong per karakter. Arsitekturnya baku (CRNN, "
                 "Shi dkk. 2015; CTC, Graves dkk. 2006) dan dilatih dari nol, bukan dari model pralatih.",
        "methods": [
            entry("cnn", "CNN (jaringan konvolusi)", "model", "official",
                  f"Mengubah piksel citra baris menjadi ciri visual. Tinggi citra {model['height']} piksel dipampatkan "
                  f"menjadi {model['feature_height']}, sedangkan lebarnya hanya diperkecil {model['width_stride']} kali "
                  "karena lebar adalah sumbu waktu untuk CTC.",
                  [("Lapis konvolusi", f"{model['conv_layers']} ({model['kernel']}×{model['kernel']}, BatchNorm, ReLU)"),
                   ("Kanal", ", ".join(str(c) for c in model["channels"])),
                   ("Parameter", f"{num(params['cnn'] + params['proj'])} (termasuk proyeksi ke {model['hidden']} dimensi)")],
                  files=["src/model.py"]),
            entry("bilstm", "BiLSTM (LSTM dua arah)" if model["bidirectional"] else "LSTM", "model", "official",
                  "Membaca deretan ciri dari kiri ke kanan dan dari kanan ke kiri, sehingga tiap kolom citra dibaca "
                  "bersama konteks di kedua sisinya." if model["bidirectional"] else
                  "Membaca deretan ciri dari kiri ke kanan, sehingga tiap kolom citra dibaca bersama konteks di kirinya.",
                  [("Lapis", str(model["lstm_layers"])), ("Unit per arah", str(model["hidden"])),
                   ("Parameter", num(params["rnn"]))],
                  files=["src/model.py"]),
            entry("head", "Lapisan keluaran", "model", "official",
                  f"Memberi skor {model['classes']} kelas untuk tiap kolom: {javanese} karakter aksara Jawa (aksara, "
                  "sandhangan, angka, pada), " + ("spasi, " if others == 1 else f"{others} karakter lain, " if others else "")
                  + "dan satu kelas kosong.",
                  [("Kelas", str(model["classes"])), ("Parameter", num(params["head"]))],
                  files=["src/model.py", "data/tokenizer.json"]),
            entry("ctc", "CTC (Connectionist Temporal Classification)", "training", "official",
                  "Fungsi rugi yang melatih model hanya dari teks satu baris, tanpa perlu tahu letak tiap karakter di "
                  "citra. Syaratnya jumlah kolom keluaran cukup untuk teksnya, jadi baris yang terlalu padat dilewati.",
                  [("Kelas kosong", f"indeks {BLANK}"), ("Kolom per karakter, paling sedikit", dec(MIN_FRAMES_PER_TARGET))],
                  files=["src/train.py", "src/dataset.py"]),
            entry("greedy", "Decoding greedy", "rule", "official",
                  "Mengambil kelas dengan skor tertinggi di tiap kolom, membuang pengulangan dan kelas kosong, lalu "
                  "mengembalikan urutan visual ke urutan Unicode. Semua angka resmi dibaca dengan cara ini.",
                  files=["src/decode.py"]),
            entry("visual_order", "Tokenizer urutan visual", "rule", "official",
                  "CTC hanya bisa menyelaraskan teks dari kiri ke kanan, padahal taling digambar di kiri aksara tetapi "
                  "disimpan sesudahnya di Unicode. Model dilatih pada urutan seperti yang terlihat. Aturan pindahnya "
                  f"diturunkan dari mesin tata huruf HarfBuzz pada {len(stats['fonts'])} font, bukan ditulis tangan.",
                  [("Tanda yang dipindah ke depan", prebase),
                   ("Pola suku kata", f"{num(stats['reorder_patterns'])} ({num(stats['non_identity_patterns'])} berubah urutan)"),
                   ("Sama dengan urutan HarfBuzz", f"{pct(stats['harfbuzz_agreement'])} kemunculan suku kata")],
                  # Nilai gerbang G4 ditetapkan scripts/export_results.py dari hasil test bolak-balik; yang dikutip
                  # di sini nilai, dasar, dan sumber gerbang itu sendiri.
                  f"Gerbang G4, {pct(gate['value'], 0)}: bolak-balik urutan visual dan urutan Unicode diuji pada "
                  f"{gate['basis']}." if gate else None,
                  gate.get("source") or "out/results/manifest.json" if gate else None, files=["src/tokenizer.py"]),
            entry("contrast", "Normalisasi kontras per citra", "rule", "official",
                  f"Sebelum masuk model, tiap citra disetel ke tinggi {model['height']} piksel lalu diskalakan: latar "
                  "(median piksel) menjadi 0 dan tinta (persentil 1) menjadi 1. Kertas abu-abu atau cetakan pudar jadi "
                  "tidak mengubah masukan model.",
                  evidence="Tanpa langkah ini model dasar jatuh dari CER 0,9% ke 74,5% hanya karena latar abu-abu.",
                  evidence_source="catatan pengukuran di CLAUDE.md", files=["src/dataset.py"]),
        ],
    }


def data_group(run: str, args: dict, fonts: dict, real: dict, evidence: Evidence, root: Path) -> dict:
    augment = build_augment(args.get("augment") or "none")
    low, high = RENDER_SIZE_RANGE
    measured = "" if fonts["current"] else f" yang ada di folder sekarang (run ini dilatih dengan {fonts['train']} font)"
    methods = [
        entry("render", "Render teks dengan shaping OpenType", "data", "official",
              "Teks aksara Jawa dirender dengan font lewat mesin tata huruf HarfBuzz (RAQM), supaya pasangan menumpuk "
              "dan taling pindah ke kiri seperti pada tulisan aksara Jawa yang benar. Tanpa RAQM gambar tetap keluar "
              "tetapi salah, jadi keberadaannya diperiksa di kode.",
              [("Ukuran huruf", f"{low} sampai {high} piksel, diundi per baris"), ("Tinggi citra akhir", f"{H} piksel")],
              files=["src/render.py", "src/dataset.py"],
              note=f"{len(fonts['defects'])} dari {fonts['measured']} font latih{measured} punya cacat yang diketahui "
                   f"({listing(fonts['defects'])}): pada font itu sebagian citra tidak sepadan dengan labelnya, atau "
                   "pasangannya tidak menumpuk di lingkungan training. Rinciannya di halaman Dataset, bagian Font per "
                   "peran." if fonts["defects"] else None),
    ]

    # Bukti variasi font berasal dari run perlakuan dan kontrolnya sendiri (jumlah font keduanya dari ekspor hasil),
    # bukan dari model resmi: itu disebut di kalimatnya.
    font_evidence = None
    if fonts["extra"] > 0:
        recorded = fonts["recorded"]
        for treated, control in CONTROLS["fonts"].items():
            change = evidence.g3_change(pipeline_key(treated) or "", pipeline_key(control) or "")
            if change and treated in recorded and control in recorded:
                own = "" if treated == run else ", bukan model resmi"
                font_evidence = (f"Pada run {control} lawan {treated} ({recorded[control]} lawan {recorded[treated]} "
                                 f"font{own}): {change[0]}.", change[1])
                break
    methods.append(entry(
        "fonts", "Variasi font", "data", "official",
        f"Tiap baris dirender dengan satu font yang diundi dari {fonts['train']} font. Bentuk huruf cetakan nyata "
        "berbeda dari font mana pun, jadi model perlu melihat banyak ragam bentuk huruf.",
        [("Font latih", f"{fonts['train']} ({fonts['core']} inti, {fonts['extra']} tambahan)" if fonts["current"]
          else f"{fonts['train']} (tercatat saat run dilatih; folder font sekarang berisi {fonts['measured']})")],
        font_evidence[0] if font_evidence else None, font_evidence[1] if font_evidence else None,
        files=["src/dataset.py", "fonts/extra/SOURCES.md"]))
    if augment:
        ablation = evidence.manifest.get("ablation") or []
        seeds = set()
        for k in range(len(ablation)):
            for row in read_log(root / f"out/checkpoints/ablation_{k}/log.jsonl"):
                if row.get("event") == "start":
                    seeds.add(row.get("args", {}).get("seed"))
        measured_ablation = ablation_evidence(ablation, seeds)
        methods.append(entry(
            "augment", "Augmentasi degradasi citra", "data", "official",
            "Citra bersih hasil render dirusak secara acak supaya mirip pindaian dan foto: "
            + listing([AUGMENT_WORDS.get(name, name) for name in augment.names])
            + (". Tiap operasi diundi sendiri-sendiri, jadi tiap citra mendapat campuran yang berbeda." if augment.p < 1
               else ". Semua operasi dikenakan pada tiap citra."),
            [("Preset", args["augment"]), ("Operasi", f"{len(augment.names)}: {', '.join(augment.names)}"),
             ("Peluang tiap operasi", dec(augment.p))],
            measured_ablation[0] if measured_ablation else None, "out/results/manifest.json",
            files=["src/augment.py", "scripts/ablation.py"], note=measured_ablation[1] if measured_ablation else None))
    if args.get("track_prob"):
        control = tracking_control(root, run)
        change = evidence.g3_change(pipeline_key(run) or "", pipeline_key(control) or "") if control else None
        methods.append(entry(
            "tracking", "Jarak antar suku kata acak", "data", "official",
            "Sebagian baris dirender dengan jarak tambahan yang seragam di antara suku kata. Cetakan nyata sering "
            "renggang, dan tanpa ini model membaca renggang itu sebagai spasi kata.",
            [("Peluang", dec(args["track_prob"])), ("Jarak tambahan", f"0 sampai {dec(args.get('track_max') or 0)} em")],
            f"Terhadap run kontrol {control} (jumlah langkah sama, tanpa jarak): {change[0]}." if change else None,
            change[1] if change else None, files=["src/render.py", "src/dataset.py"]))
    if args.get("drop_space_prob"):
        # Spasi dibuang bila undiannya kena ATAU font terpilih berspasi terlalu sempit (src/dataset.py), jadi porsi
        # baris tanpa spasi lebih besar dari peluangnya: p + (1 - p) x porsi font berspasi sempit.
        chance = args["drop_space_prob"]
        narrow = fonts["narrow"] / fonts["measured"] if fonts["measured"] else 0.0
        settings = [("Peluang", dec(chance))]
        if fonts["narrow"]:
            settings += [("Selalu, pada font berspasi sempit", f"{fonts['narrow']} dari {fonts['measured']} font{measured}"),
                         ("Porsi baris tanpa spasi", "sekitar " + pct(chance + (1 - chance) * narrow, 0))]
        methods.append(entry(
            "drop_space", "Membuang spasi", "data", "official",
            "Pada sebagian baris, spasi dihapus dari teks sebelum dirender, sehingga konsonan penutup kata bertemu kata "
            "berikutnya sebagai pasangan. Begitulah buku cetak beraksara Jawa umumnya ditulis."
            + (" Pada font yang spasinya nyaris tak tampak, spasi selalu dibuang supaya label tidak memuat spasi yang "
               "tidak terlihat di citra." if fonts["narrow"] else ""),
            settings,
            f"{num(real['without_space'])} dari {num(real['lines'])} baris cetak nyata "
            f"({pct(real['without_space'] / real['lines'], 1)}) ditulis tanpa spasi.",
            "data/real/nusaaksara/labels.tsv", files=["src/dataset.py"]))

    rare_used = bool(args.get("rare_insert_prob") or args.get("rare_opener_prob"))
    # Sisipan aksara langka yang diuji dengan run resmi sebagai kontrolnya (run perlakuan = run resmi + sisipan).
    report, source = None, None
    if not rare_used:
        for treated, control in CONTROLS["rare"].items():
            a, b = pipeline_key(treated) or "", pipeline_key(control) or ""
            if control == run and (report := evidence.compare(a, b)):
                source = f"out/compare/{a}_vs_{b}.json"
                break
    if rare_used or report:
        measured_rare = None
        if report:
            group = report["rare"]["groups"]["langka"]
            cer = report["cer"]
            low, high = cer["bootstrap_clusters"]["ci95"]
            same = low <= 0 <= high  # selang kepercayaan per halaman memuat nol
            measured_rare = (f"Pada {num(report['lines'])} baris nyata, aksara langka yang terbaca benar naik dari "
                             f"{pct(group['recall']['b'], 1)} ke {pct(group['recall']['a'], 1)}, tetapi hanya "
                             f"{pct(group['precision']['a'], 1)} dari keluarannya benar, dan G3 "
                             + ("tidak berbeda nyata" if same else f"berubah {points(cer['a'] - cer['b'])}")
                             + f" ({pct(cer['a'])} melawan {pct(cer['b'])}). Karena itu tidak dipakai di run resmi.")
        methods.append(entry(
            "rare", "Sisipan aksara langka", "data", "official" if rare_used else "tested",
            "Menyisipkan satu aksara langka (murda, aksara swara, pada) atau pembuka baris adeg-adeg ke teks sebelum "
            "dirender, karena karakter itu hampir tidak ada di korpus.",
            [("Peluang sisip", dec(args.get("rare_insert_prob") or 0)), ("Peluang pembuka", dec(args.get("rare_opener_prob") or 0))]
            if rare_used else [],
            measured_rare, source, files=["src/text_augment.py"]))
    return {
        "key": "data",
        "title": "Data latih sintetis",
        "intro": "Model tidak pernah dilatih pada foto. Semua citra latih dibuat dari teks korpus saat dibutuhkan dan "
                 "tidak disimpan; labelnya teks yang dirender itu sendiri, bukan anotasi manusia.",
        "methods": methods,
    }


def recipe_changes(links: list[dict], counts: list[int]) -> str:
    """Apa yang berubah di argumen data tiap run dibanding run sebelumnya, dari argumen di checkpoint-nya. `counts` =
    jumlah font latih tiap run. Yang tidak tercatat di argumen (mis. aturan font berspasi sempit) tidak terlihat di
    sini, jadi run tanpa perubahan disebut "argumen sama", bukan "resep sama"."""
    def recipe(args: dict) -> dict:
        return {"augment": args.get("augment") or "none",
                "drop": bool(args.get("drop_space_prob")), "track": bool(args.get("track_prob")),
                "rare": bool(args.get("rare_insert_prob") or args.get("rare_opener_prob"))}

    parts = []
    for index, (before, link) in enumerate(zip(links, links[1:])):
        old, new = recipe(before["args"]), recipe(link["args"])
        changed = []
        if new["augment"] != old["augment"]:
            changed.append("tanpa augmentasi" if new["augment"] == "none" else f"augmentasi {new['augment']}")
        if counts[index + 1] != counts[index]:
            changed.append(f"{counts[index + 1]} font (dari {counts[index]})")
        for key, label in (("drop", "buang spasi"), ("track", "jarak antar suku kata"), ("rare", "sisipan aksara langka")):
            if new[key] != old[key]:
                changed.append(label if new[key] else f"tanpa {label}")
        parts.append(f"{link['run']}: {', '.join(changed) if changed else 'argumen sama, langkah tambahan'}")
    return "; ".join(parts)


def training_group(links: list[dict], complete: bool, code: dict, device: str | None, counts: list[int]) -> dict:
    """Butir pelatihan. `complete` False = checkpoint leluhur sudah tidak ada: rantai di `links` hanya ujungnya."""
    args = links[-1]["args"]
    packed = args.get("packed", True)
    steps = sum(link["step"] for link in links)
    changes = recipe_changes(links, counts)
    # Kalimat pembuka mengikuti rantai dan perangkat yang tercatat, bukan keadaan mesin tempat ekspor dijalankan.
    where = {"xpu": " di laptop tanpa kartu grafis NVIDIA", "cpu": " di CPU, tanpa kartu grafis"}.get(device or "", "")
    staged = len(links) > 1 or not complete or bool(links[0]["args"].get("init"))
    # Siklus laju belajar dihitung untuk seluruh langkah yang direncanakan (--steps); run yang dihentikan lebih awal
    # tidak menyelesaikannya.
    planned = [link["args"].get("steps") or link["step"] for link in links]
    unfinished = [link["run"] for link, total in zip(links[:-1], planned) if link["step"] < total]
    cycle = (f"Run resmi menyelesaikan siklusnya ({num(links[-1]['step'])} langkah)." if links[-1]["step"] >= planned[-1]
             else f"Run resmi dihentikan di langkah {num(links[-1]['step'])} dari {num(planned[-1])} yang direncanakan, "
                  "sebelum siklusnya selesai.")
    if unfinished:
        cycle += (f" {len(unfinished)} dari {len(links) - 1} run sebelumnya di rantai ({listing(unfinished)}) dihentikan "
                  "sebelum siklusnya selesai.")
    if not complete:
        chain_text = ("Model resmi bukan hasil satu kali latih: checkpoint-nya melanjutkan run lain, tetapi checkpoint "
                      "awal rantainya tidak ada di mesin ini, jadi rantai dan jumlah langkah di samping tidak lengkap.")
    elif len(links) > 1:
        chain_text = (f"Model resmi bukan hasil satu kali latih. {len(links)} run berurutan masing-masing memulai dari "
                      "bobot run sebelumnya, dan resep datanya bertambah di sepanjang rantai.")
    elif links[0]["args"].get("init"):
        chain_text = "Model resmi dilatih dalam satu run, dimulai dari bobot yang sudah ada."
    else:
        chain_text = "Model resmi dilatih dalam satu run dari bobot acak."
    return {
        "key": "training",
        "title": "Pelatihan",
        "intro": f"Model dilatih {'bertahap' if staged else 'dalam satu run'}{where}. Hitungannya langkah, "
                 "bukan epoch, karena citra selalu dibuat baru.",
        "methods": [
            entry("optimizer", "AdamW", "training", "official",
                  "Pengoptimal yang memperbarui bobot model di tiap langkah, dengan peluruhan bobot yang dipisah dari "
                  "gradien.",
                  [("Laju belajar puncak", dec(args["lr"], 6)), ("Peluruhan bobot", dec(args.get("weight_decay") or 0, 6)),
                   ("Ukuran batch", str(args["batch_size"])), ("Langkah run resmi", num(links[-1]["step"]))],
                  files=["src/train.py"]),
            entry("onecycle", "Jadwal laju belajar satu siklus (OneCycle)", "training", "official",
                  f"Laju belajar naik selama {pct(code['pct_start'], 0)} pertama dari langkah yang direncanakan sebuah "
                  f"run, lalu turun sampai langkah terakhirnya. {cycle}",
                  [("Bagian naik", pct(code["pct_start"], 0)), ("Puncak", dec(args["lr"], 6)),
                   ("Langkah direncanakan run resmi", num(planned[-1]))],
                  files=["src/train.py"]),
            entry("clip", "Pemotongan gradien", "training", "official",
                  "Norma gradien dibatasi supaya satu batch yang sulit tidak merusak bobot.",
                  [("Batas norma", dec(args.get("clip") or 0))], files=["src/train.py"]),
            entry("batching", "Batch per panjang teks dan pengemasan LSTM", "training", "official",
                  "Baris dengan panjang mirip dikumpulkan dalam satu batch supaya sedikit padding, dan LSTM hanya "
                  "membaca kolom yang berisi citra. Tanpa pengemasan, model belajar mengeluarkan karakter di daerah "
                  "padding lalu menambah tanda palsu di tepi kanan saat membaca satu baris.",
                  [("Pengemasan LSTM", "ya" if packed else "tidak"),
                   ("Batas kolom per forward", num(args.get("max_batch_columns") or 0))],
                  "Pada uji menghafal 32 baris: CER 1,12% tanpa pengemasan, 0,07% dengan pengemasan.",
                  "catatan pengukuran di CLAUDE.md", files=["src/dataset.py", "src/model.py"]),
            entry("staged", "Pelatihan bertahap", "training", "official", chain_text,
                  [("Rantai run", ("… → " if not complete else "") + " → ".join(link["run"] for link in links)),
                   ("Jumlah langkah", num(steps) if complete else f"paling sedikit {num(steps)}")]
                  + ([("Yang berubah", changes)] if changes else []) + [
                   ("Perangkat", {"xpu": "grafis bawaan Intel (PyTorch XPU)", "cuda": "GPU NVIDIA", "cpu": "CPU"}
                    .get(device or "", device or "tidak tercatat"))],
                  files=["src/train.py"]),
        ],
    }


def decoding_group(official: str, evidence: Evidence, root: Path) -> dict:
    lm = re.fullmatch(r"o(\d+)_n(\d+)_ds([\d.]+)\.pkl", LM_PATH.name)

    def measured(key: str, name: str) -> str | None:
        """Greedy lawan beam + LM pada checkpoint yang sama (pipeline `key` dan `key`_beam di manifest)."""
        base, beam = evidence.metrics.get((Evidence.SCOPE, key)), evidence.metrics.get((Evidence.SCOPE, f"{key}_beam"))
        if not (base and beam):
            return None
        paired = (f" ({num(beam['better'])} baris membaik, {num(beam['worse'])} memburuk)"
                  if "better" in beam and "worse" in beam else "")
        return f"Pada checkpoint {name}: G3 {pct(base['cer'])} → {pct(beam['cer'])}{paired}."

    # Angka checkpoint resmi didahulukan; selama belum ada, yang dikutip hasil pada fase5_fonts dan itu disebut.
    note = measured(official, "resmi") if official else None
    if note is None:
        note = measured("crnn_fonts", "fase5_fonts")
        if note:
            note += " Belum diuji pada checkpoint resmi."
    settings = [("Lebar beam", str(WIDTH)), ("Bobot model bahasa", dec(ALPHA)), ("Bonus panjang", dec(BETA))]
    if lm and (root / LM_PATH.relative_to(ROOT)).exists():
        settings.append(("Model bahasa", f"n-gram karakter order {lm[1]}, penghalusan Witten-Bell, {num(int(lm[2]))} baris latih"))
    return {
        "key": "decoding",
        "title": "Koreksi sesudah baca",
        "intro": "Cara baca alternatif yang memakai pengetahuan tentang urutan karakter bahasa Jawa. Tersedia di halaman "
                 "Demo, tetapi bukan jalur angka resmi.",
        "methods": [
            entry("beam_lm", "Beam search dengan model bahasa karakter", "statistic", "available",
                  "Alih-alih mengambil kelas terbaik per kolom, beberapa kandidat teks dipertahankan dan dinilai dengan "
                  "gabungan skor CTC dan model bahasa n-gram tingkat karakter. Model bahasanya dihitung dari bagian "
                  "latih korpus saja.",
                  settings, note, "out/results/manifest.json", files=["src/beam.py", "src/charlm.py"]),
        ],
    }


def evaluation_group(run: str, args: dict, real: dict, evidence: Evidence, root: Path, twins: list[dict],
                     test_fonts: list[str]) -> dict:
    import compare_runs  # impor di sini: modulnya ikut memuat src.train dan numpy, hanya dibutuhkan untuk satu angka

    # Pengaturan bootstrap dari berkas pembanding yang sah; tanpa berkas, bawaan scripts/compare_runs.py.
    boot = evidence.bootstrap() or {}
    resamples = boot.get("resamples") or inspect.signature(compare_runs.compare).parameters["resamples"].default
    gates = evidence.gates
    order = [code for code in ("G1", "G2", "G3", "G4") if code in gates]
    status = " · ".join(f"{code} {pct(gates[code]['value'], 0 if code == 'G4' else 2)}"
                        + ("" if gates[code]["passed"] else " (belum tercapai)") for code in order)
    font_text = f", font {listing(test_fonts)}" if test_fonts else ""
    pairs = [(kind, treated, control) for kind, table in CONTROLS.items() for treated, control in table.items()
             if pipeline_key(treated) in evidence.done and pipeline_key(control) in evidence.done]
    kinds = [TREATMENTS[kind] for kind in CONTROLS if any(pair[0] == kind for pair in pairs)]
    blind, then = evidence.cer("blind_50", "vlm_zeroshot"), evidence.cer("blind_50", "crnn_fonts")
    ablation = evidence.manifest.get("ablation") or []
    ablation_steps = None
    log = read_log(root / "out/checkpoints/ablation_0/log.jsonl")
    for row in log:
        if row.get("event") == "start":
            ablation_steps = row["args"].get("steps")
    spacing = root / "out/compare/spacing_synthetic.json"
    report = read_json(spacing) if spacing.exists() else None
    probe = report["settings"] if report else None
    probe_lines = None
    if probe:
        # Baris yang benar-benar dibaca per font bisa lebih sedikit daripada yang diambil (baris yang gagal dirender
        # dibuang): terukur 2026-10-04, 301 diambil dan 300 dibaca.
        read = [font["lines"] for font in (report.get("fonts") or {}).values() if isinstance(font.get("lines"), int)]
        taken = num(probe["lines"])
        probe_lines = taken if not read or min(read) == probe["lines"] else f"{num(min(read))} (dari {taken} yang diambil)"
    methods = [
        entry("cer", "CER (character error rate)", "evaluation", "used",
              "Jumlah suntingan (ganti, hapus, sisip) untuk mengubah keluaran menjadi jawaban benar, dibagi panjang "
              "jawaban benar, dihitung atas semua baris sekaligus. Teks dinormalkan (NFC) dan dibandingkan dalam urutan "
              "Unicode.",
              [("Jarak", "Levenshtein per karakter")], files=["src/decode.py"]),
        entry("gates", "Gerbang dengan target tetap", "evaluation", "used",
              "Empat ukuran dengan target yang ditetapkan sebelum melatih: render bersih, render rusak berat, cetakan "
              "nyata, dan bolak-balik tokenizer. Target tidak dilonggarkan bila belum tercapai.",
              [("G1, render bersih", f"CER < {pct(TARGETS['G1'], 0)} pada {num(OFFICIAL_LINES)} baris uji{font_text}"),
               ("G2, render rusak berat", f"CER < {pct(TARGETS['G2'], 0)} pada {num(OFFICIAL_LINES)} baris uji{font_text}"),
               ("G3, cetakan nyata", f"CER < {pct(TARGETS['G3'], 0)} pada {num(real['lines'])} baris"),
               ("G4, tokenizer", "bolak-balik 100%")],
              status + "." if order else None, "out/results/manifest.json", files=["src/evaluate.py"],
              note=family_note(twins, "G1 dan G2")),
        entry("bootstrap", "Selang kepercayaan bootstrap berpasangan", "statistic", "used",
              f"Selisih dua model diberi selang kepercayaan 95% dengan mengambil ulang baris secara acak {num(resamples)} "
              "kali. Baris dari halaman yang sama saling mirip, jadi pengambilan ulang juga dilakukan per halaman, dan "
              "selang itulah yang dilaporkan. Jumlah baris yang membaik dan memburuk diuji dengan uji tanda. Tiap model "
              "dilatih satu kali, jadi selang ini hanya memuat keragaman baris dan halaman, bukan keragaman antar-run.",
              [("Pengambilan ulang", num(resamples))]
              + ([("Selang", boot["interval"].replace("-", " sampai "))] if boot.get("interval") else [])
              + [("Satuan", f"baris, dan halaman ({num(real['pages'])} halaman)")],
              files=["scripts/compare_runs.py"]),
    ]
    if pairs:
        methods.append(entry(
            "control", "Run kontrol", "evaluation", "used",
            f"Untuk perlakuan yang diuji tersendiri ({listing(kinds)}) dilatih model pembanding dengan jumlah langkah, "
            "data, dan titik lanjut yang sama, tanpa perlakuan itu. Selisihnya memisahkan pengaruh perlakuan dari "
            "pengaruh tambahan pelatihan.",
            [(f"Kontrol {TREATMENTS[kind]}", "; ".join(f"{control} (untuk {treated})" for k, treated, control in pairs if k == kind))
             for kind in CONTROLS if any(pair[0] == kind for pair in pairs)],
            files=["scripts/run_fase6_ctrl.sh", "scripts/run_fase7_ctrl.sh"]))
    if ablation:
        methods.append(entry(
            "ablation", "Ablasi kumulatif", "evaluation", "used",
            "Operasi augmentasi ditambahkan satu per satu, dan model dilatih ulang singkat tiap kali, untuk melihat "
            "sumbangan masing-masing. Kolom G3-nya memakai data uji, jadi dibaca sebagai arah, bukan dasar memilih.",
            [("Run", str(len(ablation)))] + ([("Langkah per run", num(ablation_steps))] if ablation_steps else []),
            files=["scripts/ablation.py"]))
    if probe:
        probe_fonts = {Path(name).stem for name in (report.get("fonts") or {})}
        methods.append(entry(
            "probe", "Evaluasi sintetis tertarget", "evaluation", "used",
            "Citra uji dibuat sengaja untuk satu pertanyaan, misalnya baris tanpa spasi yang direnggangkan sedikit demi "
            "sedikit, lalu dibaca semua model pada citra yang sama. Sebuah gejala jadi bisa diuji tanpa menyentuh data "
            "uji nyata.",
            [("Jarak yang diuji", "; ".join(dec(t) for t in probe["tracking"]) + " em"), ("Baris", probe_lines)],
            spacing_note(report, run, tracking_control(root, run), args.get("track_max") or 0.0),
            "out/compare/spacing_synthetic.json", files=["scripts/eval_spacing.py", "scripts/eval_rare.py"],
            note=probe_note([twin for twin in twins if twin["test"] in probe_fonts])))
    if blind is not None and then is not None:
        methods.append(entry(
            "vlm_blind", "Uji buta terhadap model bahasa-visual", "evaluation", "comparator",
            "Sebuah model bahasa-visual umum membaca baris nyata tanpa pelatihan khusus dan tanpa melihat jawaban, "
            "sebagai pembanding terhadap model proyek ini.",
            [("Baris", num(evidence.metrics[("blind_50", "vlm_zeroshot")]["lines"]))],
            f"CER {pct(blind)} melawan {pct(then)} untuk CRNN saat itu (fase5_fonts).",
            "out/results/manifest.json", files=["out/eval/vlm_blind_50.json"]))
    return {
        "key": "evaluation",
        "title": "Evaluasi dan statistik",
        "intro": "Angka dilaporkan apa adanya terhadap target yang ditetapkan lebih dulu, dan dua model selalu "
                 "dibandingkan pada baris yang sama.",
        "methods": methods,
    }


def build(root: Path = ROOT, run: str = OFFICIAL_RUN, test_font_dir: Path = TEST_FONT_DIR) -> dict:
    links, complete = chain(run, root)
    official = links[-1]
    args = official["args"]
    key = pipeline_key(run)
    evidence = Evidence(root)
    if key and evidence.official != key:
        raise SystemExit(f"manifest hasil menetapkan {evidence.official} sebagai resmi, bukan {key}; "
                         "jalankan ulang scripts/export_results.py")
    model = model_card(official["path"])
    tokenizer = read_json(root / "data/tokenizer.json")
    if model["classes"] != len(tokenizer["charset"]) + 1:
        raise SystemExit("jumlah kelas checkpoint berbeda dari data/tokenizer.json")
    charset = [chr(int(code[2:], 16)) for code in tokenizer["charset"]]
    recorded = recorded_fonts(root)
    fonts = [font_facts(root, link, charset, recorded) for link in links]
    real = nusaaksara_card(root / "data/real/nusaaksara/labels.tsv")
    gates = gate_reports(run, root, real["lines"])
    test_fonts = sorted({Path(name).stem for gate in gates if gate["code"] != "G3" for name in gate["fonts"]})
    twins = family_twins(fonts[-1]["paths"], gates, test_font_dir)
    device = None
    for row in read_log(official["path"].parent / "log.jsonl"):
        if row.get("event") == "start":
            device = row.get("device")
            if row.get("params") not in (None, model["parameters"]["total"]):
                raise SystemExit(f"log run {run} mencatat {row['params']} parameter, model {model['parameters']['total']}")
    groups = [
        ocr_group(model, tokenizer, evidence),
        data_group(run, args, {**fonts[-1], "recorded": recorded}, real, evidence, root),
        training_group(links, complete, code_facts(), device, [facts["train"] for facts in fonts]),
        decoding_group(key or "", evidence, root),
        evaluation_group(run, args, real, evidence, root, twins, test_fonts),
    ]
    return {
        "schema": SCHEMA,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "metopenv5",
        "official": {"run": run, "pipeline": key, "checkpoint": relative(official["path"], root), "step": official["step"]},
        # Bukti di kartu dibaca dari ekspor hasil ini; web membandingkannya dengan hasil yang sedang diimpor.
        "results_generated": evidence.manifest["generated"],
        "chain_complete": complete,
        "model": model,
        "groups": groups,
        "planned": [{"key": p["key"], "label": p["label"], "config": p["config"]}
                    for p in evidence.manifest["pipelines"] if p["status"] == "planned"],
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
    write_atomic(out_dir / "methods.json", json.dumps(card, ensure_ascii=False, indent=1, allow_nan=False) + "\n")

    total = sum(len(group["methods"]) for group in card["groups"])
    print(f"run resmi {card['official']['run']}: model {num(card['model']['parameters']['total'])} parameter, "
          f"{card['model']['classes']} kelas")
    for group in card["groups"]:
        print(f"  {group['title']}: {', '.join(m['name'] for m in group['methods'])}")
    print(f"{total} butir metode, {len(card['planned'])} direncanakan; ditulis: {out_dir / 'methods.json'}")


if __name__ == "__main__":
    main()
