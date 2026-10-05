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
  bukti        gerbang, metrik, ablasi, dan selisih antar-run dari out/results/manifest.json; selang kepercayaan
               selisih itu dari out/compare/<A>_vs_<B>.json (scripts/compare_runs.py), hanya bila CER di berkas itu
               sama dengan manifest; evaluasi sintetis dari out/compare/spacing_synthetic.json, hanya bila laporan itu
               membaca checkpoint yang sekarang. Pasangan perlakuan-kontrol dikutip hanya bila `control_run` lolos.
               Bukti yang berkasnya tidak ada dilewati; batasan yang tidak bisa dihitung ditulis sebagai catatan.
Kalimat yang menyebut cara kerja kode (pengoptimal, jadwal, rugi, normalisasi kontras, penghalusan model bahasa) dijaga
CODE_FACTS: skrip berhenti bila potongan kodenya sudah tidak ada. Dua angka yang tidak punya berkas untuk dihitung ulang
diberi sumber "catatan pengukuran di CLAUDE.md", dan nilai gerbang G4 dikutip dari manifest dengan sumbernya.
Metode tahap lanjutan alur (alih aksara, terjemahan, tingkat tutur) milik web dan ditambahkan web sendiri.
"""

import argparse
import hashlib
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
    logged_fonts,
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
    # Rumus penghalusan Witten-Bell itu sendiri (bukan kata di docstring): P = (hitungan + jenis x P orde lebih rendah)
    # / (total + jenis).
    "smoothing": ("src.charlm.CharLM.prob", "p = (self.grams[k].get(h + ch, 0) + types * p) / (total + types)"),
}
# Run kontrol tiap perlakuan (CLAUDE.md, aturan analisis fase 5 sampai 7): perlakuan -> {run perlakuan: run kontrol}.
# Kontrol sama dengan run perlakuannya kecuali SATU perlakuan itu, jadi sebuah run hanya boleh disebut kontrol untuk
# perlakuan di barisnya (fase7_track adalah kontrol sisipan aksara langka, bukan kontrol jarak). Daftar ini hanya
# calon: sebuah pasangan baru dikutip sesudah `control_run` memeriksanya pada kedua checkpoint.
CONTROLS = {
    "fonts": {"fase5_fonts": "fase5_core"},
    "tracking": {"fase7_track": "fase7_ctrl"},
    "rare": {"fase7_track_rare": "fase7_track", "fase6_rare": "fase6_ctrl"},
}
TREATMENTS = {"fonts": "font tambahan", "tracking": "jarak antar suku kata", "rare": "sisipan aksara langka"}
# Argumen src.train milik tiap perlakuan (boleh berbeda antara run perlakuan dan kontrolnya), dan argumen pelatihan
# yang harus sama di keduanya di samping argumen data (`data_recipe`).
TREATMENT_KEYS = {"fonts": ("extra_fonts",), "tracking": ("track_prob", "track_max"),
                  "rare": ("rare_insert_prob", "rare_opener_prob", "rare_max_similarity", "rare_attach")}
TRAINING_KEYS = ("batch_size", "lr", "weight_decay", "steps")
# Aturan "spasi selalu dibuang pada font berspasi sempit" (src.dataset.MIN_SPACE_RATIO) ditambahkan 2026-09-14, sesudah
# run-run ini dilatih (CLAUDE.md, "Lebar spasi font tambahan"): di font berspasi sempit, labelnya masih memuat spasi
# yang nyaris tidak tampak di citra. Hanya berakibat pada run yang memakai font semacam itu (font tambahan).
RUNS_BEFORE_NARROW_SPACE_RULE = ("base", "fase5_quick", "fase5_core", "fase5_fonts")
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
    dibaca bersama butir itu (bukan hasil ukur). Bukti tanpa sumber ditolak: halaman selalu menulis "Sumber:" di bawah
    kotak "Terukur", dan web menolak kartu yang butirnya tidak begitu."""
    if kind not in KINDS or status not in STATUSES:
        raise ValueError(f"butir {key}: jenis {kind!r} atau status {status!r} tidak dikenal")
    if evidence and not evidence_source:
        raise ValueError(f"butir {key}: bukti tanpa sumber")
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
               "src.charlm.CharLM.prob": inspect.getsource(charlm.CharLM.prob)}
    missing = [name for name, (where, text) in CODE_FACTS.items() if text not in sources[where]]
    start = re.search(r"OneCycleLR\([^)]*pct_start=([0-9.]+)", sources["src.train.main"])
    if missing or not start:
        raise SystemExit(f"kode tidak lagi cocok dengan kartu metode ({', '.join(missing) or 'pct_start'}); perbarui "
                         "butirnya dan CODE_FACTS di scripts/export_methods.py")
    return {"optimizer": "AdamW", "scheduler": "OneCycleLR", "pct_start": float(start[1]), "loss": "CTC"}


class Evidence:
    """Bukti terukur dari manifest hasil dan berkas pembanding; yang berkasnya tidak ada mengembalikan None. Selisih dua
    run selalu dihitung dari CER manifest; berkas pembanding hanya menyumbang selang kepercayaannya, dan hanya bila
    CER di berkas itu sama dengan manifest."""

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


def run_meta(root: Path, run: str) -> tuple[int, dict] | None:
    """Langkah dan argumen training sebuah run dari checkpoint-nya, atau None bila checkpoint-nya tidak ada di mesin ini."""
    path = root / "out/checkpoints" / run / "last_snapshot.pt"
    return checkpoint_meta(path) if path.exists() else None


def planned_steps(path: Path) -> int | None:
    """Jumlah langkah yang direncanakan sebuah run, dari jadwal laju belajar di checkpoint-nya: siklus OneCycle dihitung
    untuk total itu, juga bila run dimulai dengan --epochs (args.steps 0). None bila checkpoint tidak menyimpannya."""
    total = (torch.load(path, map_location="cpu", weights_only=False).get("scheduler") or {}).get("total_steps")
    return int(total) if total else None


def file_sha(path: Path) -> str:
    """12 heksadesimal pertama SHA-1 berkas: identitas checkpoint yang sama dengan laporan scripts/eval_spacing.py."""
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha1").hexdigest()[:12]


def data_recipe(args: dict) -> dict:
    """Argumen data sebuah run dalam bentuk yang bisa dibandingkan: yang mati (None, 0, "", False) disamakan, dan
    pengaturan yang hanya berlaku bila induknya hidup (jarak terbesar, saringan dan tempelan sisipan) dikosongkan bila
    induknya mati."""
    track = args.get("track_prob") or 0
    rare = bool(args.get("rare_insert_prob") or args.get("rare_opener_prob"))
    return {
        "augment": args.get("augment") or "none",
        "extra_fonts": args.get("extra_fonts") or "",
        "drop_space_prob": args.get("drop_space_prob") or 0,
        "track_prob": track,
        "track_max": (args.get("track_max") or 0) if track else 0,
        "rare_insert_prob": args.get("rare_insert_prob") or 0,
        "rare_opener_prob": args.get("rare_opener_prob") or 0,
        "rare_max_similarity": (args.get("rare_max_similarity") or 0) if rare else 0,
        "rare_attach": bool(args.get("rare_attach")) if rare else False,
        "train_lines": args.get("train_lines") or 0,
        "seed": args.get("seed") or 0,
        "real_train": args.get("real_train") or "",
        "real_val": args.get("real_val") or "",
    }


def is_treated(kind: str, recipe: dict) -> bool:
    """Apakah perlakuan `kind` hidup di resep data ini."""
    return {"fonts": bool(recipe["extra_fonts"]), "tracking": recipe["track_prob"] > 0,
            "rare": recipe["rare_insert_prob"] > 0 or recipe["rare_opener_prob"] > 0}[kind]


def control_run(root: Path, kind: str, run: str) -> str | None:
    """Run kontrol perlakuan `kind` untuk `run`, bila pasangan itu memang sah: terdaftar di CONTROLS, kedua checkpoint
    ada di mesin ini, perlakuannya hidup di `run` dan mati di kontrol, jumlah langkahnya sama, dan argumen data serta
    pelatihannya sama kecuali argumen perlakuan itu sendiri. Pasangan yang tidak lolos tidak dikutip di kartu."""
    control = CONTROLS[kind].get(run)
    ours, theirs = run_meta(root, run), (run_meta(root, control) if control else None)
    if ours is None or theirs is None or ours[0] != theirs[0]:
        return None
    a, b = data_recipe(ours[1]), data_recipe(theirs[1])
    same_data = all(a[key] == b[key] for key in a if key not in TREATMENT_KEYS[kind])
    same_training = all(ours[1].get(key) == theirs[1].get(key) for key in TRAINING_KEYS)
    return control if is_treated(kind, a) and not is_treated(kind, b) and same_data and same_training else None


def font_facts(root: Path, link: dict, charset: list[str], recorded: dict[str, int]) -> dict:
    """Font latih sebuah run. Sumbernya, berurutan: daftar NAMA di log run (bila src.train sudah mencatatnya), jumlah di
    ekspor hasil (teks konfigurasi pipeline; hanya dipakai bila tidak bertentangan dengan argumen checkpoint), lalu
    folder font sekarang. Cacat dan lebar spasi hanya bisa diukur pada berkas yang ada di folder sekarang (`paths`);
    `current` False berarti berkas-berkas itu bukan persis font yang dipakai run."""
    args = link["args"]
    core = font_files(root / "fonts")
    folder = core + (font_files(root / args["extra_fonts"]) if args.get("extra_fonts") else [])
    logged = logged_fonts(read_log(link["path"].parent / "log.jsonl"), link["step"])
    then = recorded.get(link["run"])
    core_count = len(core)
    if logged is not None:
        # Hanya berkas yang namanya dicatat log yang diukur: font yang ditambahkan ke folder sesudahnya bukan font run ini.
        paths = [path for path in folder if path.name in logged]
        train_count, source, current = len(logged), "log", len(paths) == len(set(logged))
        core_count = sum(1 for path in core if path.name in logged)
    elif then is not None and (args.get("extra_fonts") or then == len(core)):
        paths, train_count, source, current = folder, then, "results", len(folder) == then
    else:
        paths, train_count, source, current = folder, len(folder), "folder", True
    return {
        "run": link["run"], "train": train_count, "core": core_count, "extra": train_count - core_count, "source": source,
        "measured": len(paths), "current": current, "paths": paths,
        "defects": sorted(p.stem for p in paths if p.name in FONT_DEFECTS or missing_glyphs(p, charset)),
        "narrow": sum(1 for p in paths if space_ratio(str(p)) < MIN_SPACE_RATIO),
    }


def chain_fonts(facts: list[dict]) -> dict[str, dict]:
    """Font latih di sepanjang rantai run (yang berkasnya ada di folder sekarang): nama berkas -> berkas dan run yang
    memakainya. Model resmi mewarisi bobot semua run itu, jadi "font yang pernah dilihat model" adalah gabungannya,
    bukan font run terakhir saja."""
    owners: dict[str, dict] = {}
    for fact in facts:
        for path in fact["paths"]:
            owners.setdefault(path.name, {"path": path, "runs": []})["runs"].append(fact["run"])
    return owners


def family_twins(owners: dict[str, dict], fonts: list[str], test_font_dir: Path) -> tuple[list[dict], list[str]]:
    """Font uji `fonts` yang sekeluarga dengan sebuah font latih di rantai (aturan kartu data: lebar maju yang sama), dan
    font uji yang berkasnya tidak ada di mesin ini (kemiripannya tidak bisa dihitung)."""
    twins, unknown = [], []
    for name in sorted(set(fonts)):
        path = test_font_dir / name
        if not path.exists():
            unknown.append(Path(name).stem)
            continue
        reference = advances(path)
        for owner in owners.values():
            shared = shared_advances(advances(owner["path"]), reference)
            if shared["family"]:
                twins.append({"test": Path(name).stem, "train": owner["path"].stem, "runs": owner["runs"], **shared})
    return twins, unknown


def family_note(twins: list[dict], subject: str, official: str, unknown: list[str], incomplete: bool) -> str | None:
    """Batasan G1/G2: font ujinya sekeluarga dengan font latih di rantai. `official` = run resmi (font yang hanya dipakai
    run sebelumnya di rantai disebut begitu); `unknown` = font uji yang berkasnya tidak ada di mesin ini; `incomplete`
    = ada font latih yang berkasnya tidak ada lagi, jadi tidak ikut diperiksa. Yang tidak bisa dihitung DISEBUT, bukan
    didiamkan."""
    sentences = []
    if twins:
        parts = []
        for test in sorted({twin["test"] for twin in twins}):
            mine = [twin for twin in twins if twin["test"] == test]
            names = [twin["train"] + ("" if official in twin["runs"] else f" (font run {listing(twin['runs'])} di rantai)")
                     for twin in mine]
            overlaps = [str(twin["same"]) for twin in mine]
            overlap = overlaps[0] if len(set(overlaps)) == 1 else "berturut-turut " + listing(overlaps)
            parts.append(f"font uji {test} sekeluarga dengan font latih {listing(names)} (lebar {overlap} dari "
                         f"{mine[0]['of']} aksara, angka, dan pada persis sama)")
        text = listing(parts)
        sentences.append(f"{text[0].upper()}{text[1:]}, jadi {subject} mengukur generalisasi di dalam keluarga huruf itu, "
                         "bukan ke font yang belum pernah dilihat model.")
    if unknown:
        sentences.append(f"Kemiripan font uji {listing(unknown)} dengan font latih tidak bisa dihitung di mesin ini: "
                         "berkas fontnya tidak ada.")
    if incomplete:
        sentences.append("Sebagian font latih tidak ada lagi di folder font, jadi kemiripannya dengan font uji tidak ikut "
                         "diperiksa.")
    return " ".join(sentences) or None


def probe_note(twins: list[dict], unknown: list[str]) -> str | None:
    """Batasan evaluasi sintetis tertarget: citranya dirender dengan font yang sekeluarga dengan font latih di rantai."""
    sentences = []
    if twins:
        tests = sorted({twin["test"] for twin in twins})
        trains = sorted({twin["train"] for twin in twins})
        sentences.append(f"Citranya dirender dengan font uji {listing(tests)}, yang sekeluarga dengan font latih "
                         f"{listing(trains)}, jadi hasilnya belum tentu berlaku untuk bentuk huruf lain.")
    if unknown:
        sentences.append(f"Kemiripan font perender {listing(unknown)} dengan font latih tidak bisa dihitung di mesin ini: "
                         "berkas fontnya tidak ada.")
    return " ".join(sentences) or None


def report_matches(report: dict, root: Path, run: str) -> bool:
    """Apakah laporan evaluasi sintetis memang membaca checkpoint `run` yang sekarang: langkahnya sama, dan bila laporan
    mencatat sidik berkasnya, sidik itu sama. Laporan dari checkpoint lain tidak dikutip sebagai "model resmi"."""
    info = (report.get("checkpoints") or {}).get(run)
    meta = run_meta(root, run)
    if not isinstance(info, dict) or meta is None or info.get("step") != meta[0]:
        return False
    return not info.get("sha1") or info["sha1"] == file_sha(root / "out/checkpoints" / run / "last_snapshot.pt")


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


def ablation_evidence(rows: list[dict], seeds: list) -> tuple[str, str] | None:
    """(bukti, catatan) dari ablasi kumulatif. Yang dikutip jumlah per kelompok operasi, bukan "penurunan terbesar":
    langkah-langkahnya saling menutupi (satu langkah naik, langkah berikutnya turun kembali) dan tiap run satu kali.
    `seeds` = seed tiap run ablasi menurut log-nya, None bila log run itu tidak ada: "seed yang sama" hanya ditulis
    bila log SEMUA run ada dan seed-nya memang satu."""
    if len(rows) < 2 or any(row.get("G3") is None for row in rows):
        return None
    same_seed = len(seeds) == len(rows) and None not in seeds and len(set(seeds)) == 1
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
    note = ("Operasi ditambahkan berurutan dan tiap run dilatih satu kali" + (" dengan seed yang sama" if same_seed else "")
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


def font_count(root: Path, run: str, recorded: dict[str, int]) -> int | None:
    """Jumlah font latih sebuah run di luar rantai, untuk kalimat bukti "N lawan M font": dari log run atau dari ekspor
    hasil saja. None bila checkpoint-nya tidak ada di mesin ini atau jumlahnya hanya bisa ditebak dari folder font
    sekarang (kalimat bukti tidak mengarang angka dari folder yang bisa sudah berubah)."""
    meta = run_meta(root, run)
    if meta is None:
        return None
    link = {"run": run, "step": meta[0], "args": meta[1], "path": root / "out/checkpoints" / run / "last_snapshot.pt"}
    facts = font_facts(root, link, [], recorded)
    return facts["train"] if facts["source"] != "folder" else None


def data_group(run: str, args: dict, fonts: dict, real: dict, evidence: Evidence, root: Path) -> dict:
    """Butir data latih sintetis. `fonts` = fakta font run resmi (`font_facts`) ditambah "recorded" (jumlah font menurut
    ekspor hasil) dan "chain" (fakta font run-run sebelumnya di rantai, yang bobotnya diwarisi model resmi)."""
    augment = build_augment(args.get("augment") or "none")
    low, high = RENDER_SIZE_RANGE
    earlier = fonts.get("chain") or []
    measured = "" if fonts["current"] else f" yang ada di folder sekarang (run ini dilatih dengan {fonts['train']} font)"
    # Catatan butir render: font bercacat run resmi, font bercacat yang hanya dipakai run sebelumnya di rantai, dan
    # font yang tidak bisa diperiksa karena berkasnya sudah tidak ada. Yang tidak bisa diperiksa disebut, bukan didiamkan.
    notes = []
    if fonts["defects"]:
        notes.append(f"{len(fonts['defects'])} dari {fonts['measured']} font latih{measured} punya cacat yang diketahui "
                     f"({listing(fonts['defects'])}): pada font itu sebagian citra tidak sepadan dengan labelnya, atau "
                     "pasangannya tidak menumpuk di lingkungan training. Rinciannya di halaman Dataset, bagian Font per "
                     "peran.")
    elif not fonts["current"]:
        notes.append(f"Cacat font hanya bisa diperiksa pada {fonts['measured']} dari {fonts['train']} font latih: sisanya "
                     "tidak ada lagi di folder font.")
    inherited = sorted({stem for fact in earlier for stem in fact["defects"] if stem not in fonts["defects"]})
    if inherited:
        runs = [fact["run"] for fact in earlier if any(stem in inherited for stem in fact["defects"])]
        notes.append(f"Run sebelumnya di rantai ({listing(runs)}) dilatih dengan font bercacat yang tidak dipakai run resmi "
                     f"({listing(inherited)}); model resmi mewarisi bobotnya.")
    methods = [
        entry("render", "Render teks dengan shaping OpenType", "data", "official",
              "Teks aksara Jawa dirender dengan font lewat mesin tata huruf HarfBuzz (RAQM), supaya pasangan menumpuk "
              "dan taling pindah ke kiri seperti pada tulisan aksara Jawa yang benar. Tanpa RAQM gambar tetap keluar "
              "tetapi salah, jadi keberadaannya diperiksa di kode.",
              [("Ukuran huruf", f"{low} sampai {high} piksel, diundi per baris"), ("Tinggi citra akhir", f"{H} piksel")],
              files=["src/render.py", "src/dataset.py"], note=" ".join(notes) or None),
    ]

    # Bukti variasi font berasal dari run perlakuan dan kontrolnya sendiri, bukan dari model resmi: itu disebut di
    # kalimatnya. Pasangannya diperiksa dulu (`control_run`), dan jumlah font keduanya dibaca dengan cara yang sama
    # dengan run di rantai.
    font_evidence = None
    if fonts["extra"] > 0:
        for treated, control in CONTROLS["fonts"].items():
            if control_run(root, "fonts", treated) != control:
                continue
            change = evidence.g3_change(pipeline_key(treated) or "", pipeline_key(control) or "")
            counts = [font_count(root, name, fonts.get("recorded") or {}) for name in (control, treated)]
            if change and None not in counts:
                own = "" if treated == run else ", bukan model resmi"
                font_evidence = (f"Pada run {control} lawan {treated} ({counts[0]} lawan {counts[1]} font{own}): "
                                 f"{change[0]}.", change[1])
                break
    if fonts["current"]:
        font_setting = f"{fonts['train']} ({fonts['core']} inti, {fonts['extra']} tambahan)"
    elif fonts.get("source") == "log":
        font_setting = f"{fonts['train']} (tercatat di log run; {fonts['measured']} di antaranya masih ada di folder font)"
    else:
        font_setting = f"{fonts['train']} (menurut ekspor hasil; folder font sekarang berisi {fonts['measured']})"
    methods.append(entry(
        "fonts", "Variasi font", "data", "official",
        f"Tiap baris dirender dengan satu font yang diundi dari {fonts['train']} font. Bentuk huruf cetakan nyata "
        "berbeda dari font mana pun, jadi model perlu melihat banyak ragam bentuk huruf.",
        [("Font latih", font_setting)],
        font_evidence[0] if font_evidence else None, font_evidence[1] if font_evidence else None,
        files=["src/dataset.py", "fonts/extra/SOURCES.md"]))
    if augment:
        ablation = evidence.manifest.get("ablation") or []
        seeds = []  # seed tiap run ablasi menurut log-nya; None = log run itu tidak ada di mesin ini
        for k in range(len(ablation)):
            starts = [row for row in read_log(root / f"out/checkpoints/ablation_{k}/log.jsonl") if row.get("event") == "start"]
            seeds.append(starts[-1].get("args", {}).get("seed") if starts else None)
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
        control = control_run(root, "tracking", run)
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
        # baris tanpa spasi lebih besar dari peluangnya: p + (1 - p) x porsi font berspasi sempit. Aturan font
        # berspasi sempit itu baru ada sesudah run-run di RUNS_BEFORE_NARROW_SPACE_RULE dilatih.
        chance = args["drop_space_prob"]
        ruled = run not in RUNS_BEFORE_NARROW_SPACE_RULE
        narrow = fonts["narrow"] if ruled else 0
        settings = [("Peluang", dec(chance))]
        if narrow:
            settings.append(("Selalu, pada font berspasi sempit", f"{narrow} dari {fonts['measured']} font{measured}"))
            if fonts["current"]:  # porsinya hanya bisa dihitung bila berkas yang diukur memang font run ini
                settings.append(("Porsi baris tanpa spasi", "sekitar " + pct(chance + (1 - chance) * narrow / fonts["measured"], 0)))
        space_notes = []
        if not ruled and fonts["narrow"]:
            space_notes.append(f"Run ini dilatih sebelum aturan font berspasi sempit ada: pada {fonts['narrow']} font, label "
                               "yang masih berspasi memuat spasi yang nyaris tidak tampak di citra.")
        before = [fact["run"] for fact in earlier if fact["run"] in RUNS_BEFORE_NARROW_SPACE_RULE and fact["narrow"]]
        if before:
            space_notes.append(f"Run {listing(before)} di rantai dilatih sebelum aturan font berspasi sempit ada: di font "
                               "itu labelnya masih memuat spasi yang nyaris tidak tampak di citra.")
        methods.append(entry(
            "drop_space", "Membuang spasi", "data", "official",
            "Pada sebagian baris, spasi dihapus dari teks sebelum dirender, sehingga konsonan penutup kata bertemu kata "
            "berikutnya sebagai pasangan. Begitulah buku cetak beraksara Jawa umumnya ditulis."
            + (" Pada font yang spasinya nyaris tak tampak, spasi selalu dibuang supaya label tidak memuat spasi yang "
               "tidak terlihat di citra." if narrow else ""),
            settings,
            f"{num(real['without_space'])} dari {num(real['lines'])} baris cetak nyata "
            f"({pct(real['without_space'] / real['lines'], 1)}) ditulis tanpa spasi.",
            "data/real/nusaaksara/labels.tsv", files=["src/dataset.py"], note=" ".join(space_notes) or None))

    rare_used = bool(args.get("rare_insert_prob") or args.get("rare_opener_prob"))
    # Sisipan aksara langka yang diuji dengan run resmi sebagai kontrolnya (run perlakuan = run resmi + sisipan).
    report, source = None, None
    if not rare_used:
        for treated, control in CONTROLS["rare"].items():
            a, b = pipeline_key(treated) or "", pipeline_key(control) or ""
            if control == run and control_run(root, "rare", treated) == control and (report := evidence.compare(a, b)):
                source = f"out/compare/{a}_vs_{b}.json"
                break
    if rare_used or report:
        measured_rare = None
        if report:
            group = report["rare"]["groups"]["langka"]
            cer = report["cer"]
            low, high = cer["bootstrap_clusters"]["ci95"]
            same = low <= 0 <= high  # selang kepercayaan per halaman memuat nol
            recall, precision = group["recall"], group["precision"]["a"]
            # Kalimatnya mengikuti hasilnya: "tetapi hanya" untuk presisi di bawah separuh, dan "karena itu tidak
            # dipakai" hanya bila G3 tidak lebih baik (alasan yang dicatat proyek: efek samping, bukan G3 terendah).
            measured_rare = (f"Pada {num(report['lines'])} baris nyata, aksara langka yang terbaca benar "
                             f"{'naik' if recall['a'] > recall['b'] else 'berubah'} dari {pct(recall['b'], 1)} ke "
                             f"{pct(recall['a'], 1)}, "
                             + (f"tetapi hanya {pct(precision, 1)} dari keluarannya benar" if precision < 0.5
                                else f"dan {pct(precision, 1)} dari keluarannya benar")
                             + ", dan G3 " + ("tidak berbeda nyata" if same else f"berubah {points(cer['a'] - cer['b'])}")
                             + f" ({pct(cer['a'])} melawan {pct(cer['b'])}). "
                             + ("Run resmi tidak memakainya." if high < 0 else "Karena itu tidak dipakai di run resmi."))
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
    """Apa yang berubah di argumen data tiap run dibanding run sebelumnya, dari argumen di checkpoint-nya (`data_recipe`,
    dibandingkan nilai demi nilai). `counts` = jumlah font latih tiap run. Yang tidak tercatat di argumen (mis. aturan
    font berspasi sempit) tidak terlihat di sini, jadi run tanpa perubahan disebut "argumen data sama", bukan "resep
    sama"; argumen pelatihan (laju belajar, jumlah langkah) tidak dibandingkan di sini."""
    parts = []
    for index, (before, link) in enumerate(zip(links, links[1:])):
        old, new = data_recipe(before["args"]), data_recipe(link["args"])
        changed = []
        if new["augment"] != old["augment"]:
            changed.append("tanpa augmentasi" if new["augment"] == "none" else f"augmentasi {new['augment']}")
        if counts[index + 1] != counts[index]:
            changed.append(f"{counts[index + 1]} font (dari {counts[index]})")
        elif new["extra_fonts"] != old["extra_fonts"]:
            changed.append(f"font tambahan dari {new['extra_fonts']}" if new["extra_fonts"] else "tanpa font tambahan")
        if new["drop_space_prob"] != old["drop_space_prob"]:
            changed.append("tanpa buang spasi" if not new["drop_space_prob"] else "buang spasi" if not old["drop_space_prob"]
                           else f"buang spasi p {dec(new['drop_space_prob'])} (dari {dec(old['drop_space_prob'])})")
        if (new["track_prob"], new["track_max"]) != (old["track_prob"], old["track_max"]):
            changed.append("tanpa jarak antar suku kata" if not new["track_prob"] else "jarak antar suku kata"
                           if not old["track_prob"] else f"jarak antar suku kata p {dec(new['track_prob'])} hingga "
                           f"{dec(new['track_max'])} em (dari p {dec(old['track_prob'])} hingga {dec(old['track_max'])} em)")
        rare_keys = ("rare_insert_prob", "rare_opener_prob", "rare_max_similarity", "rare_attach")
        if any(new[key] != old[key] for key in rare_keys):
            was, now = is_treated("rare", old), is_treated("rare", new)
            changed.append("tanpa sisipan aksara langka" if not now else "sisipan aksara langka" if not was
                           else "sisipan aksara langka dengan pengaturan lain")
        if new["train_lines"] != old["train_lines"]:
            changed.append(f"kumpulan {num(new['train_lines'])} baris (dari {num(old['train_lines'])})")
        if new["seed"] != old["seed"]:
            changed.append(f"seed {new['seed']} (dari {old['seed']})")
        if (new["real_train"], new["real_val"]) != (old["real_train"], old["real_val"]):
            changed.append("data nyata ikut dipakai" if new["real_train"] or new["real_val"] else "tanpa data nyata")
        parts.append(f"{link['run']}: {', '.join(changed) if changed else 'argumen data sama, langkah tambahan'}")
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
    planned = [link.get("planned") or link["args"].get("steps") or link["step"] for link in links]
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


def evaluation_group(run: str, args: dict, real: dict, evidence: Evidence, root: Path, family: dict,
                     test_fonts: list[str]) -> dict:
    """Butir evaluasi. `family` = font latih di rantai dan kemiripannya dengan font uji gerbang: "owners"
    (`chain_fonts`), "dir" (folder font uji), "twins" dan "unknown" (`family_twins` untuk font uji gerbang), dan
    "incomplete" (ada font latih yang berkasnya tidak ada lagi)."""
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
             if control_run(root, kind, treated) == control
             and pipeline_key(treated) in evidence.done and pipeline_key(control) in evidence.done]
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
              note=family_note(family["twins"], "G1 dan G2", run, family["unknown"], family["incomplete"])),
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
        # Font perender evaluasi ini dibaca dari laporannya sendiri (belum tentu font uji gerbang). Angkanya hanya
        # dikutip bila laporan itu memang membaca checkpoint run resmi dan run kontrolnya yang sekarang.
        probe_twins, probe_unknown = family_twins(family["owners"], list(report.get("fonts") or {}), family["dir"])
        control = control_run(root, "tracking", run)
        tied = report_matches(report, root, run) and (control is None or report_matches(report, root, control))
        measured_probe = spacing_note(report, run, control, args.get("track_max") or 0.0) if tied else None
        stale = None if tied else ("Laporan evaluasi yang ada di mesin ini tidak bisa dicocokkan dengan checkpoint run "
                                   "resmi dan run kontrolnya yang sekarang, jadi angkanya tidak dikutip.")
        methods.append(entry(
            "probe", "Evaluasi sintetis tertarget", "evaluation", "used",
            "Citra uji dibuat sengaja untuk satu pertanyaan, misalnya baris tanpa spasi yang direnggangkan sedikit demi "
            "sedikit, lalu dibaca semua model pada citra yang sama. Sebuah gejala jadi bisa diuji tanpa menyentuh data "
            "uji nyata.",
            [("Jarak yang diuji", "; ".join(dec(t) for t in probe["tracking"]) + " em"), ("Baris", probe_lines)],
            measured_probe, "out/compare/spacing_synthetic.json", files=["scripts/eval_spacing.py", "scripts/eval_rare.py"],
            note=" ".join(part for part in (stale, probe_note(probe_twins, probe_unknown)) if part) or None))
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
    for link in links:
        link["planned"] = planned_steps(link["path"])
    fonts = [font_facts(root, link, charset, recorded) for link in links]
    real = nusaaksara_card(root / "data/real/nusaaksara/labels.tsv")
    gates = gate_reports(run, root, real["lines"])
    test_fonts = sorted({Path(name).stem for gate in gates if gate["code"] != "G3" for name in gate["fonts"]})
    # Model resmi mewarisi bobot semua run di rantai, jadi font yang "pernah dilihat model" = gabungan font rantai.
    owners = chain_fonts(fonts)
    twins, unknown = family_twins(owners, [name for gate in gates if gate["code"] != "G3" for name in gate["fonts"]],
                                  test_font_dir)
    family = {"owners": owners, "dir": test_font_dir, "twins": twins, "unknown": unknown,
              "incomplete": any(not fact["current"] for fact in fonts)}
    device = None
    for row in read_log(official["path"].parent / "log.jsonl"):
        if row.get("event") == "start":
            device = row.get("device")
            if row.get("params") not in (None, model["parameters"]["total"]):
                raise SystemExit(f"log run {run} mencatat {row['params']} parameter, model {model['parameters']['total']}")
    groups = [
        ocr_group(model, tokenizer, evidence),
        data_group(run, args, {**fonts[-1], "recorded": recorded, "chain": fonts[:-1]}, real, evidence, root),
        training_group(links, complete, code_facts(), device, [facts["train"] for facts in fonts]),
        decoding_group(key or "", evidence, root),
        evaluation_group(run, args, real, evidence, root, family, test_fonts),
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
