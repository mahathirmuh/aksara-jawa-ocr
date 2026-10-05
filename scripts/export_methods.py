"""Ekspor kartu metode untuk halaman Metode di web (kontrak metode, skema 1).

  .venv/Scripts/python scripts/export_methods.py           # beberapa detik di CPU; model dibuat tanpa inferensi

Keluaran: out/results/methods.json (ditimpa). Web mengimpornya dengan `php artisan aksara:methods`;
`php artisan aksara:import` ikut mengimpornya bila berkasnya ada di folder hasil.

Kartu ini mendaftar metode dan machine learning yang dipakai repo OCR, dikelompokkan menurut tahap. Tiap butir punya
penjelasan singkat, pengaturan, dan bukti terukur. Tidak ada angka yang diketik tangan di sini kecuali yang ditandai
"catatan":
  model        dibangun dari model_config checkpoint run resmi (OFFICIAL_RUN) dan dimuati bobotnya: jumlah parameter,
               lapis, kanal, dan langkah lebar dihitung dari modul yang sebenarnya
  pelatihan    argumen dari checkpoint; pengoptimal, jadwal, dan fungsi rugi dicocokkan dengan sumber src/train.py
               (skrip berhenti bila kodenya sudah berbeda dari yang ditulis di kartu)
  data         preset augmentasi dari src/augment.py, ukuran render dari src/dataset.py, font dari folder font
  bukti        gerbang, metrik, dan ablasi dari out/results/manifest.json; selisih dan selang kepercayaan dari
               out/compare/<A>_vs_<B>.json (scripts/compare_runs.py). Bukti yang berkasnya tidak ada dilewati.
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

from export_datasets import chain, font_files, nusaaksara_card, pipeline_key, read_log, relative  # noqa: E402
from export_results import ALPHA, BETA, LM_PATH, OFFICIAL_LINES, OFFICIAL_RUN, WIDTH, read_json, write_atomic  # noqa: E402
from src import train  # noqa: E402
from src.augment import build_augment  # noqa: E402
from src.dataset import DOWNSAMPLE, H, MIN_FRAMES_PER_TARGET, RENDER_SIZE_RANGE  # noqa: E402
from src.evaluate import TARGETS  # noqa: E402
from src.model import CRNN  # noqa: E402

SCHEMA = 1
# Jenis butir: model = model deep learning, training = teknik pelatihan, data = teknik data sintetis,
# statistic = metode statistik, rule = aturan atau algoritme tanpa pembelajaran, evaluation = cara mengukur.
KINDS = ("model", "training", "data", "statistic", "rule", "evaluation")
# Status: official = bagian dari model resmi (arsitektur, data, pelatihan, cara baca), used = dipakai untuk mengukur,
# available = ada tetapi bukan jalur resmi, tested = diuji lalu tidak dipakai, comparator = pembanding.
STATUSES = ("official", "used", "available", "tested", "comparator")
# Yang harus tetap ada di src.train.main supaya kalimat di kartu benar. Bila salah satu hilang, kodenya sudah
# berubah: perbarui butir pelatihan di bawah, baru jalankan ulang.
TRAIN_CODE = {
    "optimizer": "torch.optim.AdamW(",
    "scheduler": "torch.optim.lr_scheduler.OneCycleLR(",
    "loss": "nn.CTCLoss(blank=BLANK, zero_infinity=True)",
    "clip": "nn.utils.clip_grad_norm_(model.parameters(), args.clip)",
}
# Run kontrol tiap perlakuan (CLAUDE.md, aturan analisis fase 6 dan 7): sama kecuali perlakuannya.
CONTROL_OF = {"fase7_track": "fase7_ctrl", "fase7_track_rare": "fase7_track", "fase6_rare": "fase6_ctrl"}
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
          evidence: str | None = None, evidence_source: str | None = None, files: list[str] | None = None) -> dict:
    if kind not in KINDS or status not in STATUSES:
        raise ValueError(f"butir {key}: jenis {kind!r} atau status {status!r} tidak dikenal")
    return {"key": key, "name": name, "kind": kind, "status": status, "summary": summary,
            "settings": [list(pair) for pair in settings or []], "evidence": evidence,
            "evidence_source": evidence_source if evidence else None, "files": files or []}


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


def train_code() -> dict:
    """Pengoptimal, jadwal, rugi, dan pemotongan gradien seperti tertulis di src.train.main; berhenti bila berubah."""
    source = inspect.getsource(train.main)
    missing = [name for name, text in TRAIN_CODE.items() if text not in source]
    start = re.search(r"OneCycleLR\([^)]*pct_start=([0-9.]+)", source)
    if missing or not start:
        raise SystemExit(f"src/train.py tidak lagi cocok dengan kartu metode ({', '.join(missing) or 'pct_start'}); "
                         "perbarui butir pelatihan di scripts/export_methods.py")
    return {"optimizer": "AdamW", "scheduler": "OneCycleLR", "pct_start": float(start[1]), "loss": "CTC"}


class Evidence:
    """Bukti terukur dari manifest hasil dan berkas pembanding; yang berkasnya tidak ada mengembalikan None."""

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

    def cer(self, scope: str, pipeline: str) -> float | None:
        return self.metrics.get((scope, pipeline), {}).get("cer")

    def compare(self, a: str, b: str) -> dict | None:
        """Laporan scripts/compare_runs.py untuk pipeline A dan B (selisih = A - B), bila ada dan memang untuk keduanya."""
        path = self.root / "out/compare" / f"{a}_vs_{b}.json"
        if not path.exists():
            return None
        report = read_json(path)
        return report if report.get("a", {}).get("key") == a and report.get("b", {}).get("key") == b else None

    def g3_change(self, treated: str, control: str) -> tuple[str, str] | None:
        """Kalimat "G3 kontrol -> perlakuan (selisih; SK 95% per halaman)" dan berkas sumbernya."""
        after, before = self.cer("nusaaksara_745", treated), self.cer("nusaaksara_745", control)
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


def spacing_note(report: dict, run: str, control: str | None, track_max: float) -> str | None:
    """Hasil scripts/eval_spacing.py untuk run resmi dan run kontrolnya: spasi palsu per 100 batas suku kata pada baris
    tanpa spasi yang direnggangkan sejauh jarak terbesar yang masih di dalam rentang latih run resmi."""
    for font in (report.get("fonts") or {}).values():
        results = font.get("results") or {}

        def widest(name: str | None) -> dict | None:
            rows = [row for row in (results.get(name) or {}).get("tanpa-spasi", []) if 0 < row["tracking"] <= track_max + 1e-9]
            return max(rows, key=lambda row: row["tracking"]) if rows else None

        def per_hundred(row: dict) -> str:
            value = row["false_per_boundary"] * 100
            return dec(value, 2 if 0 < value < 0.05 else 1)  # nilai kecil yang bukan nol tidak ditulis "0"

        ours, theirs = widest(run), widest(control)
        if ours and theirs and ours["tracking"] == theirs["tracking"]:
            return (f"Baris tanpa spasi yang direnggangkan {dec(ours['tracking'])} em: {per_hundred(ours)} spasi palsu per "
                    f"100 batas suku kata pada model resmi, {per_hundred(theirs)} pada run kontrol {control}.")
    return None


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
            entry("bilstm", "BiLSTM (LSTM dua arah)", "model", "official",
                  "Membaca deretan ciri dari kiri ke kanan dan dari kanan ke kiri, sehingga tiap kolom citra dibaca "
                  "bersama konteks di kedua sisinya.",
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
                  [("Kelas kosong", "indeks 0"), ("Kolom per karakter, paling sedikit", dec(MIN_FRAMES_PER_TARGET))],
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
                  f"Bolak-balik urutan visual dan urutan Unicode benar pada {num(stats['lines'])} baris korpus "
                  f"({pct(gate['value'], 0)}, gerbang G4)." if gate else None,
                  "data/tokenizer.json", files=["src/tokenizer.py"]),
            entry("contrast", "Normalisasi kontras per citra", "rule", "official",
                  f"Sebelum masuk model, tiap citra disetel ke tinggi {model['height']} piksel lalu diskalakan: latar "
                  "(median piksel) menjadi 0 dan tinta (persentil 1) menjadi 1. Kertas abu-abu atau cetakan pudar jadi "
                  "tidak mengubah masukan model.",
                  evidence="Tanpa langkah ini model dasar jatuh dari CER 0,9% ke 74,5% hanya karena latar abu-abu.",
                  evidence_source="catatan pengukuran di CLAUDE.md", files=["src/dataset.py"]),
        ],
    }


def data_group(run: str, args: dict, fonts: dict, real: dict, evidence: Evidence) -> dict:
    augment = build_augment(args.get("augment") or "none")
    low, high = RENDER_SIZE_RANGE
    methods = [
        entry("render", "Render teks dengan shaping OpenType", "data", "official",
              "Teks aksara Jawa dirender dengan font lewat mesin tata huruf HarfBuzz (RAQM), supaya pasangan menumpuk "
              "dan taling pindah ke kiri seperti di cetakan. Tanpa RAQM gambar tetap keluar tetapi salah, jadi "
              "keberadaannya diperiksa di kode.",
              [("Ukuran huruf", f"{low} sampai {high} piksel, diundi per baris"), ("Tinggi citra akhir", f"{H} piksel")],
              files=["src/render.py", "src/dataset.py"]),
    ]
    change = evidence.g3_change("crnn_fonts", "crnn_core")
    methods.append(entry(
        "fonts", "Variasi font", "data", "official",
        f"Tiap baris dirender dengan satu font yang diundi dari {fonts['train']} font. Bentuk huruf cetakan nyata "
        "berbeda dari font mana pun, jadi model perlu melihat banyak ragam bentuk huruf.",
        [("Font latih", f"{fonts['train']} ({fonts['core']} inti, {fonts['extra']} tambahan)")],
        f"Dari {fonts['core']} font ke {fonts['train']} font: {change[0]}." if change and fonts["extra"] else None,
        change[1] if change else None, files=["src/dataset.py", "fonts/extra/SOURCES.md"]))
    if augment:
        drops = None
        ablation = evidence.manifest.get("ablation") or []
        if len(ablation) > 1 and all(row.get("G3") is not None for row in ablation):
            falls = sorted((b["G3"] - a["G3"], b["added"]) for a, b in zip(ablation, ablation[1:]))
            falls = [(delta, name) for delta, name in falls if delta < 0][:3]
            drops = f"Ablasi {len(ablation)} run: G3 {pct(ablation[0]['G3'])} → {pct(ablation[-1]['G3'])}."
            if falls:
                drops += (" Penurunan terbesar saat menambahkan "
                          + ", ".join(f"{name} ({points(delta)})" for delta, name in falls) + ".")
        methods.append(entry(
            "augment", "Augmentasi degradasi citra", "data", "official",
            "Citra bersih hasil render dirusak secara acak supaya mirip pindaian dan foto: "
            + listing([AUGMENT_WORDS.get(name, name) for name in augment.names])
            + (". Tiap operasi diundi sendiri-sendiri, jadi tiap citra mendapat campuran yang berbeda." if augment.p < 1
               else ". Semua operasi dikenakan pada tiap citra."),
            [("Preset", args["augment"]), ("Operasi", f"{len(augment.names)}: {', '.join(augment.names)}"),
             ("Peluang tiap operasi", dec(augment.p))],
            drops, "out/results/manifest.json", files=["src/augment.py", "scripts/ablation.py"]))
    if args.get("track_prob"):
        control = CONTROL_OF.get(run)
        change = evidence.g3_change(pipeline_key(run) or "", pipeline_key(control) or "") if control else None
        methods.append(entry(
            "tracking", "Jarak antar suku kata acak", "data", "official",
            "Sebagian baris dirender dengan jarak tambahan yang seragam di antara suku kata. Cetakan nyata sering "
            "renggang, dan tanpa ini model membaca renggang itu sebagai spasi kata.",
            [("Peluang", dec(args["track_prob"])), ("Jarak tambahan", f"0 sampai {dec(args.get('track_max') or 0)} em")],
            f"Terhadap run kontrol {control} (jumlah langkah sama, tanpa jarak): {change[0]}." if change else None,
            change[1] if change else None, files=["src/render.py", "src/dataset.py"]))
    if args.get("drop_space_prob"):
        methods.append(entry(
            "drop_space", "Membuang spasi", "data", "official",
            "Pada sebagian baris, spasi dihapus dari teks sebelum dirender, sehingga konsonan penutup kata bertemu kata "
            "berikutnya sebagai pasangan. Begitulah buku cetak beraksara Jawa umumnya ditulis.",
            [("Peluang", dec(args["drop_space_prob"]))],
            f"{num(real['without_space'])} dari {num(real['lines'])} baris cetak nyata "
            f"({pct(real['without_space'] / real['lines'], 1)}) ditulis tanpa spasi.",
            "data/real/nusaaksara/labels.tsv", files=["src/dataset.py"]))

    rare_used = bool(args.get("rare_insert_prob") or args.get("rare_opener_prob"))
    report = None if rare_used else evidence.compare("crnn_fase7_track_rare", "crnn_fase7_track")
    if rare_used or report:
        note = None
        if report:
            group = report["rare"]["groups"]["langka"]
            cer = report["cer"]
            low, high = cer["bootstrap_clusters"]["ci95"]
            same = low <= 0 <= high  # selang kepercayaan per halaman memuat nol
            note = (f"Pada {num(report['lines'])} baris nyata, aksara langka yang terbaca benar naik dari "
                    f"{pct(group['recall']['b'], 1)} ke {pct(group['recall']['a'], 1)}, tetapi hanya "
                    f"{pct(group['precision']['a'], 1)} dari keluarannya benar, dan G3 "
                    + ("tidak berbeda nyata" if same else f"berubah {points(cer['a'] - cer['b'])}")
                    + f" ({pct(cer['a'])} melawan {pct(cer['b'])}). Karena itu tidak dipakai di run resmi.")
        methods.append(entry(
            "rare", "Sisipan aksara langka", "data", "official" if rare_used else "tested",
            "Menyisipkan satu aksara langka (murda, aksara swara, pada) atau pembuka baris adeg-adeg ke teks sebelum "
            "dirender, karena karakter itu hampir tidak ada di korpus.",
            [("Peluang sisip", dec(args["rare_insert_prob"])), ("Peluang pembuka", dec(args["rare_opener_prob"]))]
            if rare_used else [],
            note, "out/compare/crnn_fase7_track_rare_vs_crnn_fase7_track.json", files=["src/text_augment.py"]))
    return {
        "key": "data",
        "title": "Data latih sintetis",
        "intro": "Model tidak pernah dilatih pada foto. Semua citra latih dibuat dari teks korpus saat dibutuhkan dan "
                 "tidak disimpan; labelnya pasti benar karena berasal dari teks yang dirender.",
        "methods": methods,
    }


def recipe_changes(links: list[dict], root: Path) -> str:
    """Apa yang berubah di resep data tiap run dibanding run sebelumnya, dari argumen di checkpoint-nya."""
    def recipe(args: dict) -> dict:
        return {"augment": args.get("augment") or "none",
                "extra": len(font_files(root / args["extra_fonts"])) if args.get("extra_fonts") else 0,
                "drop": bool(args.get("drop_space_prob")), "track": bool(args.get("track_prob")),
                "rare": bool(args.get("rare_insert_prob") or args.get("rare_opener_prob"))}

    parts = []
    for before, link in zip(links, links[1:]):
        old, new = recipe(before["args"]), recipe(link["args"])
        added = []
        if new["augment"] != old["augment"]:
            added.append("tanpa augmentasi" if new["augment"] == "none" else f"augmentasi {new['augment']}")
        if new["extra"] != old["extra"]:
            added.append(f"{new['extra']} font tambahan")
        for key, label in (("drop", "buang spasi"), ("track", "jarak antar suku kata"), ("rare", "sisipan aksara langka")):
            if new[key] != old[key]:
                added.append(label if new[key] else f"tanpa {label}")
        parts.append(f"{link['run']}: {', '.join(added) if added else 'resep sama, langkah tambahan'}")
    return "; ".join(parts)


def training_group(links: list[dict], code: dict, device: str | None, root: Path) -> dict:
    args = links[-1]["args"]
    packed = args.get("packed", True)
    steps = sum(link["step"] for link in links)
    changes = recipe_changes(links, root)
    # Kalimat pembuka mengikuti rantai dan perangkat yang tercatat, bukan keadaan mesin tempat ekspor dijalankan.
    where = {"xpu": " di laptop tanpa kartu grafis NVIDIA", "cpu": " di CPU, tanpa kartu grafis"}.get(device or "", "")
    return {
        "key": "training",
        "title": "Pelatihan",
        "intro": f"Model dilatih {'bertahap' if len(links) > 1 else 'dalam satu run'}{where}. Hitungannya langkah, "
                 "bukan epoch, karena citra selalu dibuat baru.",
        "methods": [
            entry("optimizer", "AdamW", "training", "official",
                  "Pengoptimal yang memperbarui bobot model di tiap langkah, dengan peluruhan bobot yang dipisah dari "
                  "gradien.",
                  [("Laju belajar puncak", dec(args["lr"], 6)), ("Peluruhan bobot", dec(args.get("weight_decay") or 0, 6)),
                   ("Ukuran batch", str(args["batch_size"])), ("Langkah run resmi", num(links[-1]["step"]))],
                  files=["src/train.py"]),
            entry("onecycle", "Jadwal laju belajar satu siklus (OneCycle)", "training", "official",
                  f"Laju belajar naik selama {pct(code['pct_start'], 0)} langkah pertama lalu turun sampai akhir run. "
                  "Siklusnya selesai tepat di langkah terakhir, jadi tiap run punya siklusnya sendiri.",
                  [("Bagian naik", pct(code["pct_start"], 0)), ("Puncak", dec(args["lr"], 6))],
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
            entry("staged", "Pelatihan bertahap", "training", "official",
                  f"Model resmi bukan hasil satu kali latih. {len(links)} run berurutan masing-masing memulai dari "
                  "bobot run sebelumnya, dan resep datanya bertambah di sepanjang rantai."
                  if len(links) > 1 else "Model resmi dilatih dalam satu run dari bobot acak.",
                  [("Rantai run", " → ".join(link["run"] for link in links)), ("Jumlah langkah", num(steps))]
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
        base, beam = evidence.metrics.get(("nusaaksara_745", key)), evidence.metrics.get(("nusaaksara_745", f"{key}_beam"))
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


def evaluation_group(run: str, args: dict, real: dict, evidence: Evidence, root: Path) -> dict:
    import compare_runs  # impor di sini: modulnya ikut memuat src.train dan numpy, hanya dibutuhkan untuk satu angka

    resamples = inspect.signature(compare_runs.compare).parameters["resamples"].default
    gates = evidence.gates
    order = [code for code in ("G1", "G2", "G3", "G4") if code in gates]
    status = " · ".join(f"{code} {pct(gates[code]['value'], 0 if code == 'G4' else 2)}"
                        + ("" if gates[code]["passed"] else " (belum tercapai)") for code in order)
    controls = sorted(p["key"].removeprefix("crnn_") for p in evidence.manifest["pipelines"] if p["key"].endswith("_ctrl"))
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
              [("G1, render bersih", f"CER < {pct(TARGETS['G1'], 0)} pada {num(OFFICIAL_LINES)} baris uji"),
               ("G2, render rusak berat", f"CER < {pct(TARGETS['G2'], 0)} pada {num(OFFICIAL_LINES)} baris uji"),
               ("G3, cetakan nyata", f"CER < {pct(TARGETS['G3'], 0)} pada {num(real['lines'])} baris"),
               ("G4, tokenizer", "bolak-balik 100%")],
              status + "." if order else None, "out/results/manifest.json", files=["src/evaluate.py"]),
        entry("bootstrap", "Selang kepercayaan bootstrap berpasangan", "statistic", "used",
              f"Selisih dua model diberi selang kepercayaan 95% dengan mengambil ulang baris secara acak {num(resamples)} "
              "kali. Baris dari halaman yang sama saling mirip, jadi pengambilan ulang juga dilakukan per halaman, dan "
              "selang itulah yang dilaporkan. Jumlah baris yang membaik dan memburuk diuji dengan uji tanda. Tiap model "
              "dilatih satu kali, jadi selang ini hanya memuat keragaman baris dan halaman, bukan keragaman antar-run.",
              [("Pengambilan ulang", num(resamples)), ("Selang", "persentil 2,5 sampai 97,5"),
               ("Satuan", f"baris, dan halaman ({num(real['pages'])} halaman)")],
              files=["scripts/compare_runs.py"]),
        entry("control", "Run kontrol", "evaluation", "used",
              "Untuk tiap perlakuan penting dilatih model pembanding dengan jumlah langkah, data, dan titik lanjut yang "
              "sama, tanpa perlakuan itu. Selisihnya memisahkan pengaruh perlakuan dari pengaruh tambahan pelatihan.",
              [("Run kontrol", ", ".join(controls))] if controls else [], files=["scripts/run_fase7_ctrl.sh"]),
    ]
    if ablation:
        methods.append(entry(
            "ablation", "Ablasi kumulatif", "evaluation", "used",
            "Operasi augmentasi ditambahkan satu per satu, dan model dilatih ulang singkat tiap kali, untuk melihat "
            "sumbangan masing-masing. Kolom G3-nya memakai data uji, jadi dibaca sebagai arah, bukan dasar memilih.",
            [("Run", str(len(ablation)))] + ([("Langkah per run", num(ablation_steps))] if ablation_steps else []),
            files=["scripts/ablation.py"]))
    if probe:
        methods.append(entry(
            "probe", "Evaluasi sintetis tertarget", "evaluation", "used",
            "Citra uji dibuat sengaja untuk satu pertanyaan, misalnya baris tanpa spasi yang direnggangkan sedikit demi "
            "sedikit, lalu dibaca semua model pada citra yang sama. Sebuah gejala jadi bisa diuji tanpa menyentuh data "
            "uji nyata.",
            [("Jarak yang diuji", "; ".join(dec(t) for t in probe["tracking"]) + " em"), ("Baris", probe_lines)],
            spacing_note(report, run, CONTROL_OF.get(run), args.get("track_max") or 0.0),
            "out/compare/spacing_synthetic.json", files=["scripts/eval_spacing.py", "scripts/eval_rare.py"]))
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


def build(root: Path = ROOT, run: str = OFFICIAL_RUN) -> dict:
    links, _ = chain(run, root)
    official = links[-1]
    args = official["args"]
    key = pipeline_key(run)
    evidence = Evidence(root)
    if key and evidence.manifest.get("official") not in (None, key):
        raise SystemExit(f"manifest hasil menetapkan {evidence.manifest['official']} sebagai resmi, bukan {key}; "
                         "jalankan ulang scripts/export_results.py")
    model = model_card(official["path"])
    tokenizer = read_json(root / "data/tokenizer.json")
    if model["classes"] != len(tokenizer["charset"]) + 1:
        raise SystemExit("jumlah kelas checkpoint berbeda dari data/tokenizer.json")
    core = len(font_files(root / "fonts"))
    extra = len(font_files(root / args["extra_fonts"])) if args.get("extra_fonts") else 0
    real = nusaaksara_card(root / "data/real/nusaaksara/labels.tsv")
    device = None
    for row in read_log(official["path"].parent / "log.jsonl"):
        if row.get("event") == "start":
            device = row.get("device")
            if row.get("params") not in (None, model["parameters"]["total"]):
                raise SystemExit(f"log run {run} mencatat {row['params']} parameter, model {model['parameters']['total']}")
    groups = [
        ocr_group(model, tokenizer, evidence),
        data_group(run, args, {"train": core + extra, "core": core, "extra": extra}, real, evidence),
        training_group(links, train_code(), device, root),
        decoding_group(key or "", evidence, root),
        evaluation_group(run, args, real, evidence, root),
    ]
    return {
        "schema": SCHEMA,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "metopenv5",
        "official": {"run": run, "pipeline": key, "checkpoint": relative(official["path"], root), "step": official["step"]},
        # Bukti di kartu dibaca dari ekspor hasil ini; web membandingkannya dengan hasil yang sedang diimpor.
        "results_generated": evidence.manifest["generated"],
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
