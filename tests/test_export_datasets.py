"""Test scripts/export_datasets.py: kartu data halaman Dataset, pada repo palsu kecil dan (bila ada) data sebenarnya."""

import json
import os
import random
import shutil
import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))  # scripts/ bukan paket

import export_datasets as ed  # noqa: E402
from export_results import (  # noqa: E402
    CHECKPOINT_DIR,
    CHECKPOINTS,
    OFFICIAL_LINES,
    OFFICIAL_RUN,
    OPTIONAL_CHECKPOINTS,
    official_pipeline,
)
from src.dataset import H, TRAIN_FONTS, LengthBucketSampler  # noqa: E402
from src.train import make_loader  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KA, GA, WULU, PANGKON, RARE, NEVER = "ꦏ", "ꦒ", "ꦶ", "꧀", "ꦐ", "ꦑ"
SECRET = "ꦱꦸꦒꦼꦁꦫꦮꦸꦃ"  # teks label data nyata palsu: tidak boleh muncul di keluaran


def test_count_lines_handles_missing_final_newline(tmp_path):
    for name, text, expected in [("kosong.txt", "", 0), ("satu.txt", "a", 1), ("dua.txt", "a\nb\n", 2), ("tiga.txt", "a\nb\nc", 3)]:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8", newline="\n")
        assert ed.count_lines(path) == expected
        assert ed.count_lines(path) == len(text.splitlines())


def start(step):
    return {"event": "start", "step": step, "args": {}}


def test_effective_segments_follow_resumes_and_discard_lost_steps():
    # Satu proses dari awal sampai akhir.
    assert ed.effective_segments([start(0), {"step": 100}, {"step": 1500, "event": "end"}], 1500) == [(0, 1500)]
    # fase6_ctrl: proses pertama mati sebelum checkpoint, lalu berhenti dan dilanjutkan di 500 dan 884.
    rows = [start(0), start(0), {"step": 500, "val_cer": 0.1}, {"step": 500, "event": "end"}, start(500),
            {"step": 884, "event": "end"}, start(884), {"step": 1000}, {"step": 1500, "event": "end"}]
    assert ed.effective_segments(rows, 1500) == [(0, 500), (500, 884), (884, 1500)]
    # Proses mati di langkah 1130, dilanjutkan dari checkpoint langkah 1000: langkah 1000-1130 hilang.
    rows = [start(0), {"step": 1000, "val_cer": 0.4}, {"step": 1130}, start(1000), {"step": 1500, "event": "end"}]
    assert ed.effective_segments(rows, 1500) == [(0, 1000), (1000, 1500)]
    # Mundur ke checkpoint yang lebih awal membuang segmen sesudahnya.
    rows = [start(0), {"step": 500}, start(500), {"step": 700}, start(500), {"step": 900}]
    assert ed.effective_segments(rows, 900) == [(0, 500), (500, 900)]
    # Checkpoint yang dipakai anaknya lebih awal dari langkah terakhir di log (fase5_quick: 500 dari ~950).
    assert ed.effective_segments([start(0), {"step": 500, "val_cer": 0.1}, {"step": 900}], 500) == [(0, 500)]
    # Baris tanpa langkah dan log kosong.
    assert ed.effective_segments([{"event": "note"}, start(0), {"loss": 1.0}, {"step": 10}], 10) == [(0, 10)]
    assert ed.effective_segments([], 10) == []


class IndexLines(torch.utils.data.Dataset):
    """Dataset palsu berbentuk SyntheticLines: target tiap sampel = indeksnya, supaya batch DataLoader bisa dibaca."""

    def __init__(self, lines):
        self.lines = list(lines)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return torch.zeros(1, H, 8), torch.tensor([idx], dtype=torch.long)


def loader_batches(lines, batch_size, workers, seed, passes=1):
    loader = make_loader(IndexLines(lines), batch_size, True, workers, seed)
    return [[y.tolist() for _x, y, _in, _tgt in loader] for _ in range(passes)]


def test_schedule_reproduces_the_real_dataloader_order():
    rng = random.Random(7)
    lines = [KA * rng.randint(1, 30) + str(i) for i in range(500)]
    schedule = ed.Schedule(lines)

    # src.train: random.Random(seed).shuffle(train_lines); train_lines[:n]
    shuffled = list(lines)
    random.Random(3).shuffle(shuffled)
    pool = schedule.pool(120, 3)
    assert [lines[i] for i in pool] == shuffled[:120]
    assert schedule.pool(60, 3) == pool[:60]  # kumpulan kecil = awalan kumpulan besar

    # 120 baris, batch 8 = 15 batch per epoch. Dua epoch dari DataLoader sungguhan, tanpa dan dengan worker.
    for workers in (0, 1, 2):
        first, second = loader_batches(shuffled[:120], 8, workers, 3, passes=2)
        replay = schedule.batches(120, 3, 8, 30, workers)
        assert [[pool.index(i) for i in batch] for batch in replay] == first + second, workers
        assert schedule.batches(120, 3, 8, 5, workers) == replay[:5]  # proses lanjutan mengulang dari awal
        assert schedule.batches(120, 3, 8, 0, workers) == []
    # Dengan worker, DataLoader memanggil iter(sampler) dua kali sebelum batch pertama: urutannya BUKAN panggilan
    # pertama sampler. Menyamakannya dengan panggilan pertama memilih baris yang lain (temuan tinjauan 2026-10-05).
    naive = list(iter(LengthBucketSampler(shuffled[:120], 8, shuffle=True, seed=3)))
    assert [[pool.index(i) for i in batch] for batch in schedule.batches(120, 3, 8, 15, 0)] == naive
    assert [[pool.index(i) for i in batch] for batch in schedule.batches(120, 3, 8, 15, 2)] != naive
    assert schedule.batches(120, 3, 8, 15, 1) == schedule.batches(120, 3, 8, 15, 2)  # satu worker sudah cukup


def test_short_batches_are_counted_as_they_are():
    # 100 baris, batch 8: kelompok panjang berisi 100 baris = 12 batch penuh + satu batch berisi 4.
    lines = [KA * (i % 17 + 1) for i in range(100)]
    batches = ed.Schedule(lines).batches(100, 0, 8, 13, 0)
    assert sorted(len(b) for b in batches) == [4] + [8] * 12
    assert sum(len(b) for b in batches) == 100 == len({i for b in batches for i in b})


def test_pipeline_key_follows_the_results_export():
    checkpoints = {**CHECKPOINTS, **{spec["key"]: CHECKPOINT_DIR / spec["run"] / "last_snapshot.pt"
                                     for spec in OPTIONAL_CHECKPOINTS}}
    for key, path in checkpoints.items():
        run = path.parent.name
        assert ed.pipeline_key(run) == official_pipeline(checkpoints, run) == key
    assert ed.pipeline_key(OFFICIAL_RUN) is not None
    assert ed.pipeline_key("run_tak_dikenal") is None


def test_font_rules_match_training():
    assert ed.font_files(ROOT / "fonts") == TRAIN_FONTS
    assert set(ed.CORE_FONTS) == {p.name for p in TRAIN_FONTS}
    assert len(ed.SPACING) == 73 and KA in ed.SPACING and "꧐" in ed.SPACING and "꧈" in ed.SPACING
    assert WULU not in ed.SPACING and PANGKON not in ed.SPACING
    for path in TRAIN_FONTS:
        own = ed.advances(path)
        assert set(own) == set(ed.SPACING) and own[KA] and own[KA] > 0
    noto, tuladha = (ed.advances(p) for p in TRAIN_FONTS)
    assert sum(1 for ch in ed.SPACING if noto[ch] is not None and noto[ch] == tuladha[ch]) < 0.5 * len(ed.SPACING)
    assert ed.missing_glyphs(TRAIN_FONTS[0], [" ", KA, "꧟"]) == []


def test_font_sources_table_is_parsed(tmp_path):
    path = tmp_path / "SOURCES.md"
    path.write_text("# Font\n\n| File | Folder | Sumber | Lisensi | Status visual |\n|---|---|---|---|---|\n"
                    "| `A.ttf` | `extra` | https://contoh/a | tidak tercantum | tampak benar |\n"
                    "| `B.otf` | `extra_rejected` | https://contoh/b | CC BY-NC-ND | DITOLAK: wulu tergeser |\n", encoding="utf-8")
    table = ed.font_sources(path)
    assert table == {
        "A.ttf": {"folder": "extra", "source": "https://contoh/a", "license": "tidak tercantum", "status": "tampak benar"},
        "B.otf": {"folder": "extra_rejected", "source": "https://contoh/b", "license": "CC BY-NC-ND",
                  "status": "DITOLAK: wulu tergeser"},
    }
    assert ed.font_sources(tmp_path / "tidak-ada.md") == {}


@pytest.mark.skipif(not (ROOT / "fonts/extra").is_dir(), reason="font tambahan tidak ikut repo")
def test_every_extra_font_has_a_source_row():
    table = ed.font_sources(ROOT / "fonts/extra/SOURCES.md")
    for _, folder in ed.FONT_GROUPS[1:]:
        for path in ed.font_files(ROOT / folder):
            assert table[path.name]["folder"] == Path(folder).name, path


def save_ckpt(path: Path, step: int, **args) -> Path:
    base = {"run": path.parent.name, "init": "", "train_lines": 400, "val_lines": 10, "batch_size": 8, "seed": 0,
            "steps": step, "augment": "none", "extra_fonts": ""}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "args": {**base, **args}}, path)
    return path


def write_log(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fake_repo(root: Path, run: str = "runB") -> Path:
    """Repo OCR palsu: 2.000 baris latih, rantai runA -> runB, laporan gerbang, data nyata, font."""
    rng = random.Random(1)
    train = [KA * rng.randint(20, 60) + GA for _ in range(2000)]
    train[5] += RARE  # satu baris = 0,05% < 0,1%: langka
    val, test = [GA * 25 for _ in range(100)], [KA * 25 for _ in range(100)]
    (root / "data/splits").mkdir(parents=True)
    for name, lines in (("train", train), ("val", val), ("test", test)):
        (root / "data/splits" / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (root / "data/corpus_stats.json").write_text(json.dumps({
        "articles": 40, "candidates": 2400, "roundtrip_passed": 2300, "roundtrip_pass_rate": 2300 / 2400, "duplicates": 110,
        "injected": {"train": 8, "val": 1, "test": 1}, "injected_total": 10,
        "split_lines": {"train": 2000, "val": 100, "test": 100}, "total_lines": 2200, "leaked_lines": 0,
        "length_histogram": {"20-29": 900, "30-39": 1300}}), encoding="utf-8")
    (root / "data/tokenizer.json").write_text(json.dumps(
        {"blank": 0, "charset": [f"U+{ord(ch):04X}" for ch in (" ", KA, GA, RARE, NEVER)]}), encoding="utf-8")

    (root / "fonts/extra").mkdir(parents=True)
    for font in TRAIN_FONTS:
        shutil.copy(font, root / "fonts" / font.name)
    shutil.copy(TRAIN_FONTS[0], root / "fonts/extra/Salinan.ttf")
    (root / "fonts/extra/SOURCES.md").write_text(
        "| File | Folder | Sumber | Lisensi | Status visual |\n|---|---|---|---|---|\n"
        "| `Salinan.ttf` | `extra` | https://contoh/salinan | tidak tercantum | tampak benar |\n", encoding="utf-8")
    (root / "ujifont").mkdir()
    shutil.copy(TRAIN_FONTS[0], root / "ujifont/javatext.ttf")  # font uji palsu = salinan Noto

    ckpt = root / "out/checkpoints"
    first = save_ckpt(ckpt / "runA" / "last_snapshot.pt", 6, train_lines=200)
    write_log(first.parent / "log.jsonl", [start(0), {"step": 4, "val_cer": 0.5, "val_lines": 10},
                                           start(4), {"step": 6, "val_cer": 0.2, "val_lines": 10},
                                           {"step": 8, "val_cer": 0.3, "val_lines": 10}])  # jalan terus sesudah snapshot
    last = save_ckpt(ckpt / run / "last_snapshot.pt", 10, init="out/checkpoints/runA/last_snapshot.pt", workers=2,
                     extra_fonts="fonts/extra", augment="fase5", drop_space_prob=0.5, track_prob=0.5, track_max=0.3)
    write_log(last.parent / "log.jsonl", [start(0), {"step": 5, "val_cer": 0.1, "val_lines": 10},
                                          {"step": 10, "val_cer": 0.05, "val_lines": 10}, {"step": 10, "event": "end"}])

    (root / "data/real/nusaaksara").mkdir(parents=True)
    (root / "data/real/nusaaksara/labels.tsv").write_text(
        "image_path\ttext\tsource_id\tcondition\n"
        f"images/a.png\t{SECRET}\thal_0\tscan_buku_cetak\n"
        f"images/b.png\t{SECRET} {KA}\thal_0\tscan_buku_cetak\n"
        f"images/c.png\t{KA}{GA}\thal_1\tscan_buku_cetak\n", encoding="utf-8", newline="\n")
    (root / "data/real/commons").mkdir(parents=True)
    (root / "data/real/commons/drafts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        {"id": "x1", "source_file": "Papan A.jpg", "license": "CC BY-SA 4.0", "draft": SECRET},
        {"id": "x2", "source_file": "Papan A.jpg", "license": "CC BY-SA 4.0", "draft": SECRET},
        {"id": "x3", "source_file": "Papan B.jpg", "license": "CC0", "draft": SECRET}]), encoding="utf-8")

    (root / "out/eval").mkdir(parents=True)
    checkpoint = f"out/checkpoints/{run}/last_snapshot.pt"
    for code, augment in (("G1", "none"), ("G2", "heavy")):
        (root / "out/eval" / f"{run}_{code}_10k.json").write_text(json.dumps({
            "goal": code, "checkpoint": checkpoint, "split": "test", "augment": augment, "cer": 0.01,
            "results": {"javatext.ttf": {"lines": OFFICIAL_LINES - 2, "rejected_by_min_frames": 2},
                        "semua": {"lines": OFFICIAL_LINES - 2}}}), encoding="utf-8")
    # Laporan lain yang juga membaca data uji: dua tes cepat, satu evaluasi penuh run lain, satu G3 tes cepat.
    for name, report in (("q100_G1", {"goal": "G1", "split": "test", "results": {"semua": {"lines": 100}}}),
                         ("q100_x_trainfonts", {"goal": "G1", "split": "test", "results": {
                             "a.ttf": {"lines": 100}, "b.ttf": {"lines": 100}, "semua": {"lines": 200}}}),
                         ("lain_G1_10k", {"goal": "G1", "split": "test", "results": {
                             "javatext.ttf": {"lines": OFFICIAL_LINES - 3, "rejected_by_min_frames": 3},
                             "semua": {"lines": OFFICIAL_LINES - 3}}}),
                         ("q100_val", {"goal": "G1", "split": "val", "results": {"semua": {"lines": 100}}}),
                         ("q100_G3", {"goal": "G3", "results": {"semua": {"lines": 3}}}),
                         ("bukan_laporan", {"items": [1, 2]})):
        (root / "out/eval" / f"{name}.json").write_text(json.dumps(report), encoding="utf-8")
    (root / "out/eval" / f"{run}_G3_full.json").write_text(json.dumps({
        "goal": "G3", "checkpoint": checkpoint, "pad_ratio": 0.0, "cer": 0.2, "results": {"semua": {"lines": 3}}}),
        encoding="utf-8")
    return root


def test_build_describes_a_fake_repo(tmp_path):
    root = fake_repo(tmp_path)
    card = ed.build(root, "runB", root / "ujifont")
    dump = json.dumps(card, ensure_ascii=False)
    assert SECRET not in dump  # teks label data nyata tidak pernah ikut
    assert card["schema"] == ed.SCHEMA and card["warnings"] == []
    assert card["official"] == {"run": "runB", "pipeline": None, "checkpoint": "out/checkpoints/runB/last_snapshot.pt",
                                "step": 10}

    assert [s["lines"] for s in card["splits"]] == [2000, 100, 100]
    assert sum(s["share"] for s in card["splits"]) == pytest.approx(1.0)
    assert card["corpus"]["lines"] == 2200 and card["corpus"]["other"] == 2200 - (2300 - 110 + 10)
    assert card["corpus"]["length_histogram"] == [{"range": "20-29", "lines": 900}, {"range": "30-39", "lines": 1300}]
    assert card["corpus"]["split_buckets"] == {"train": 90, "val": 5, "test": 5}
    assert card["corpus"]["max_roundtrip_cer"] == 0.05 and card["corpus"]["min_count"]["train"] == 100

    first, last = card["lineage"]["runs"]
    assert (first["run"], first["init"], first["pool"], first["steps"], first["samples"]) == ("runA", None, 200, 6, 48)
    assert first["segments"] == [[0, 4], [4, 6]] and first["distinct_lines"] == 4 * 8  # proses kedua mengulang awal
    assert first["val_steps"] == [4, 6]  # pemeriksaan di langkah 8 terjadi sesudah checkpoint yang dipakai runB
    assert (first["fonts"], first["extra_fonts"], first["augment"]) == (2, 0, "none")
    assert (last["run"], last["init"], last["pool"], last["steps"], last["samples"]) == ("runB", "runA", 400, 10, 80)
    assert last["distinct_lines"] == 80 and last["processes"] == 1 and last["val_steps"] == [5, 10]
    assert (last["fonts"], last["extra_fonts"], last["track_prob"], last["drop_space_prob"]) == (3, 1, 0.5, 0.5)
    lineage = card["lineage"]
    assert lineage["complete"] and lineage["steps"] == 16 and lineage["samples"] == 128 and lineage["pool"] == 400
    # Gabungan rantai dihitung ulang di sini tanpa Schedule: runA tanpa worker (panggilan pertama sampler), runB dengan
    # worker (panggilan kedua). Gabungannya lebih besar daripada jadwal run resmi saja.
    train = (root / "data/splits/train.txt").read_text(encoding="utf-8").splitlines()
    order = list(range(len(train)))
    random.Random(0).shuffle(order)
    sampler_a = LengthBucketSampler([train[i] for i in order[:200]], 8, shuffle=True, seed=0)
    seen_a = {order[i] for batch in list(iter(sampler_a))[:4] for i in batch}
    sampler_b = LengthBucketSampler([train[i] for i in order[:400]], 8, shuffle=True, seed=0)
    iter(sampler_b)
    seen_b = {order[i] for batch in list(iter(sampler_b))[:10] for i in batch}
    assert (len(seen_a), len(seen_b)) == (32, 80)
    assert lineage["distinct_lines"] == len(seen_a | seen_b) > 80

    usage = card["usage"]
    assert usage["train"] == last
    assert usage["val"] == {"lines": 10, "steps": [5, 10], "fonts": 2}
    assert usage["test"]["lines"] == OFFICIAL_LINES and [g["code"] for g in usage["test"]["gates"]] == ["G1", "G2"]
    # Dua tes cepat membaca 100 baris pertama (yang berfont dua: baris yang sama dua kali, bukan 200 baris).
    assert (usage["test"]["quick"], usage["test"]["quick_max_lines"], usage["test"]["full"]) == (2, 100, 1)
    assert usage["test"]["others"] == []
    assert (first["fonts_source"], last["fonts_source"]) == ("folder", "folder")
    assert usage["test"]["gates"][1]["augment"] == "heavy" and usage["test"]["gates"][0]["fonts"] == ["javatext.ttf"]
    assert usage["real"] == {"lines": 3, "reports": 2, "others": [],
                             "gates": [{"code": "G3", "lines": 3, "fonts": [], "split": None, "augment": None,
                                        "source": "out/eval/runB_G3_full.json"}]}
    assert first["font_names"] is None and last["font_names"] is None  # log lama tidak mencatat nama font

    rare = card["rare"]
    assert (rare["codepoints"], rare["javanese"], rare["charset"]) == (2, 4, 5)  # RARE (1 baris) dan NEVER (0 baris)
    assert rare["train_lines"] == {"min": 0, "max": 1, "mean": 0.5}
    assert rare["scheduled_lines"]["max"] <= rare["pool_lines"]["max"] <= 1
    assert 1 <= rare["never_scheduled"] <= 2

    corpus, real, commons = card["datasets"]
    assert corpus["roles"] == {"train": 2000, "val": 100, "test": 100} and corpus["gates"] == ["G1", "G2"]
    assert (real["count"], real["pages"], real["without_space"], real["roles"]) == (3, 2, 2, {"test": 3})
    assert real["shareable"] is False and real["license"] == "non-komersial"
    assert (commons["count"], commons["files"], commons["verified"], commons["status"]) == (3, 2, 0, "pending")
    assert commons["pending"] == 3
    assert commons["license"] == "CC BY-SA 4.0, CC0" and commons["roles"] == {} and commons["planned"] == ["train", "val"]
    assert real["share_note"].startswith("Lisensi non-komersial. Tidak disebarkan repo ini")
    assert (last["real_train"], last["real_val"]) == (False, False)

    fonts = {f["file"]: f for f in card["fonts"]}
    assert [f["group"] for f in card["fonts"]] == ["core", "core", "extra", "test"]
    assert fonts[TRAIN_FONTS[0].name]["roles"] == ["train", "val"] and fonts[TRAIN_FONTS[0].name]["in_repo"]
    assert fonts["Salinan.ttf"]["roles"] == ["train"] and fonts["Salinan.ttf"]["review"] == "tampak benar"
    assert fonts["javatext.ttf"]["roles"] == ["test"] and fonts["javatext.ttf"]["available"]
    same = fonts["Salinan.ttf"]["shared_with_test"]
    assert same["family"] and same["font"] == "javatext.ttf" and same["of"] == 73 and same["same"] >= 70
    assert fonts[TRAIN_FONTS[0].name]["shared_with_test"]["family"]  # font uji palsu = salinan font inti pertama
    assert not fonts[TRAIN_FONTS[1].name]["shared_with_test"]["family"]
    assert fonts["Salinan.ttf"]["drops_space"] is False and fonts["Salinan.ttf"]["missing"] == []

    assert card["support"][0] == {"key": "charset", "file": "data/tokenizer.json", "characters": 5, "classes": 6}


def test_build_reports_what_it_cannot_verify(tmp_path):
    root = fake_repo(tmp_path)
    (root / "out/checkpoints/runA/last_snapshot.pt").unlink()  # leluhur dihapus: rantai berhenti di run resmi
    (root / "out/checkpoints/runB/log.jsonl").unlink()          # tanpa log: dihitung seolah satu proses
    (root / "data/real/commons/drafts.jsonl").unlink()
    card = ed.build(root, "runB", root / "tidak-ada")
    assert [r["run"] for r in card["lineage"]["runs"]] == ["runB"] and card["lineage"]["complete"] is False
    assert card["usage"]["train"]["distinct_lines"] == 80 and card["usage"]["train"]["segments"] == [[0, 10]]
    assert [d["key"] for d in card["datasets"]] == ["corpus", "nusaaksara"]
    test_font = next(f for f in card["fonts"] if f["group"] == "test")
    assert test_font["available"] is False
    assert all("shared_with_test" not in f for f in card["fonts"])
    assert len(card["warnings"]) == 2 and "Log run runB" in card["warnings"][0] and "javatext.ttf" in card["warnings"][1]


def test_build_stops_on_inconsistent_inputs(tmp_path):
    root = fake_repo(tmp_path)
    train = root / "data/splits/train.txt"
    original = train.read_text(encoding="utf-8")
    train.write_text(original + KA * 30 + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(SystemExit, match="corpus_stats.json mencatat 2000"):
        ed.build(root, "runB", root / "ujifont")
    train.write_text(original, encoding="utf-8", newline="\n")

    report = root / "out/eval/runB_G2_10k.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    report.write_text(json.dumps({**data, "augment": "none"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="laporan G2 .* tidak sah"):
        ed.build(root, "runB", root / "ujifont")
    report.unlink()
    with pytest.raises(SystemExit, match="laporan G2 .* tidak ada"):
        ed.build(root, "runB", root / "ujifont")
    report.write_text(json.dumps(data), encoding="utf-8")

    shutil.copy(TRAIN_FONTS[0], root / "fonts/extra/TanpaCatatan.ttf")
    with pytest.raises(SystemExit, match="TanpaCatatan.ttf tidak ada di tabel"):
        ed.build(root, "runB", root / "ujifont")
    (root / "fonts/extra/TanpaCatatan.ttf").unlink()

    with pytest.raises(SystemExit, match="tidak ada"):
        ed.build(root, "runC", root / "ujifont")  # run resmi tanpa checkpoint


def test_main_writes_the_card(tmp_path, monkeypatch, capsys):
    root = fake_repo(tmp_path / "repo")
    card = ed.build(root, "runB", root / "ujifont")
    monkeypatch.setattr(ed, "build", lambda: card)
    # Run yang tidak dikenal ekspor hasil tidak punya kunci pipeline: web tidak bisa mencocokkannya, jadi tidak ditulis.
    assert card["official"]["pipeline"] is None
    with pytest.raises(SystemExit, match="runB tidak dikenal scripts/export_results.py"):
        ed.main(["--out", str(tmp_path / "hasil")])
    assert not (tmp_path / "hasil/datasets.json").exists()
    card["official"]["pipeline"] = "crnn_runB"
    ed.main(["--out", str(tmp_path / "hasil")])
    written = json.loads((tmp_path / "hasil/datasets.json").read_text(encoding="utf-8"))
    assert written == json.loads(json.dumps(card))
    out = capsys.readouterr().out
    assert "run resmi runB" in out and "datasets.json" in out


REAL = [ROOT / "data/splits/train.txt", ROOT / "data/real/nusaaksara/labels.tsv",
        ROOT / "out/checkpoints" / OFFICIAL_RUN / "last_snapshot.pt"]


@pytest.mark.skipif(not all(p.exists() for p in REAL), reason="data, checkpoint, atau laporan resmi tidak ada di mesin ini")
def test_real_card_is_internally_consistent():
    card = ed.build()
    assert card["official"]["run"] == OFFICIAL_RUN
    corpus, splits = card["corpus"], card["splits"]
    assert sum(s["lines"] for s in splits) == corpus["lines"] and corpus["other"] == 0 and corpus["leaked_lines"] == 0
    assert sum(h["lines"] for h in corpus["length_histogram"]) == corpus["lines"]
    train = card["usage"]["train"]
    assert 0 < train["distinct_lines"] <= min(train["samples"], train["pool"]) <= splits[0]["lines"]
    lineage = card["lineage"]
    assert lineage["runs"][-1] == train and lineage["complete"]
    assert train["distinct_lines"] <= lineage["distinct_lines"] <= lineage["pool"] <= splits[0]["lines"]
    assert card["usage"]["val"]["lines"] <= splits[1]["lines"] and card["usage"]["test"]["lines"] <= splits[2]["lines"]
    roles = [set(f["roles"]) for f in card["fonts"]]
    assert sum("train" in r for r in roles) == train["fonts"] and sum("val" in r for r in roles) == len(TRAIN_FONTS)
    assert all(f["roles"] == [] for f in card["fonts"] if f["group"] in ("review", "rejected"))
    fonts = {f["file"]: f for f in card["fonts"]}
    if "CarakanJawa.otf" in fonts and "javatext.ttf" in fonts and fonts["javatext.ttf"]["available"]:
        # Terukur 2026-10-05 dan dicatat di CLAUDE.md: 72 dari 73 lebar maju sama; tiga font berspasi sempit.
        shared = fonts["CarakanJawa.otf"]["shared_with_test"]
        assert shared["family"] and shared["same"] < shared["of"] == len(ed.SPACING)
        assert fonts["CarakanJawa.otf"]["missing"] == ["U+A98E"]
        assert sum(f.get("shared_with_test", {}).get("family", False) for f in card["fonts"]) == 1
        assert {name for name, f in fonts.items() if f.get("drops_space")} == {
            "GumregahNew.ttf", "abmAksaJawa-Regular.ttf", "abmAksaJawa-Bold.ttf"}
    if len(lineage["runs"]) > 1:
        assert lineage["distinct_lines"] > train["distinct_lines"]
    assert lineage["samples"] == sum(r["samples"] for r in lineage["runs"]) <= sum(r["steps"] * r["batch_size"] for r in lineage["runs"])
    rare = card["rare"]
    assert rare["codepoints"] > 0 and rare["scheduled_lines"]["mean"] <= rare["pool_lines"]["mean"] <= rare["train_lines"]["mean"]

    # Teks label NusaAksara tidak boleh bocor ke kartu yang akan diimpor web dan ditampilkan.
    dump = json.dumps(card, ensure_ascii=False)
    with (ROOT / "data/real/nusaaksara/labels.tsv").open(encoding="utf-8", newline="") as f:
        labels = [line.split("\t")[1] for line in f.read().splitlines()[1:]]
    assert len(labels) == card["usage"]["real"]["lines"]
    assert not any(label in dump for label in labels if len(label) >= 4)


def test_rare_characters_are_counted_in_pool_and_schedule_separately(tmp_path):
    root = fake_repo(tmp_path)
    path = root / "data/splits/train.txt"
    train = path.read_text(encoding="utf-8").splitlines()
    order = list(range(len(train)))
    random.Random(0).shuffle(order)
    pool = order[:400]
    sampler = LengthBucketSampler([train[i] for i in pool], 8, shuffle=True, seed=0)
    iter(sampler)  # runB dilatih dengan worker
    scheduled = {pool[i] for batch in list(iter(sampler))[:10] for i in batch}
    in_schedule = min(scheduled)
    in_pool_only = min(set(pool) - scheduled)
    outside = min(set(order[400:]))
    # Ganti karakter terakhir (panjang baris tetap, jadi urutan batch tidak berubah).
    train = [line.replace(RARE, "") + (GA if RARE in line else "") for line in train]
    for index, ch in ((in_schedule, "\ua990"), (in_pool_only, "\ua991"), (outside, "\ua993")):
        train[index] = train[index][:-1] + ch
    path.write_text("\n".join(train) + "\n", encoding="utf-8", newline="\n")
    tokenizer = {"blank": 0, "charset": [f"U+{ord(ch):04X}" for ch in (" ", KA, GA, "\ua990", "\ua991", "\ua993", "\ua9ce")]}
    (root / "data/tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")

    card = ed.build(root, "runB", root / "ujifont")
    rare = card["rare"]
    assert (rare["codepoints"], rare["javanese"], rare["charset"]) == (4, 6, 7)
    assert rare["train_lines"] == {"min": 0, "max": 1, "mean": 0.75}
    assert rare["pool_lines"] == {"min": 0, "max": 1, "mean": 0.5}       # dua dari empat ada di kumpulan run resmi
    assert rare["scheduled_lines"] == {"min": 0, "max": 1, "mean": 0.25}  # hanya satu yang dijadwalkan
    assert rare["never_scheduled"] == 3
    # U+A9CE belum dipakai Unicode: tidak ada font yang punya glyph-nya.
    for font in card["fonts"]:
        if font["roles"]:
            assert font["missing"] == ["U+A9CE"], font["file"]


def test_chain_follows_every_ancestor_and_reports_odd_logs(tmp_path):
    root = fake_repo(tmp_path)
    ckpt = root / "out/checkpoints"
    zero = save_ckpt(ckpt / "run0" / "last_snapshot.pt", 3, train_lines=100)
    write_log(zero.parent / "log.jsonl", [start(0), {"step": 3, "event": "end"}])
    save_ckpt(ckpt / "runA" / "last_snapshot.pt", 6, train_lines=200, init="out/checkpoints/run0/last_snapshot.pt")
    card = ed.build(root, "runB", root / "ujifont")
    assert [(r["run"], r["init"]) for r in card["lineage"]["runs"]] == [("run0", None), ("runA", "run0"), ("runB", "runA")]
    assert card["lineage"]["steps"] == 19 and card["lineage"]["complete"] and card["warnings"] == []

    # Run yang dimundurkan lalu melewati langkah checkpoint-nya lagi, dan proses lanjutan dengan --train-lines lain:
    # keduanya diberi tahu, karena hitungan dari log belum tentu benar.
    write_log(ckpt / "runA" / "log.jsonl", [start(0), {"step": 6, "val_cer": 0.3}, start(2), {"step": 6, "val_cer": 0.2},
                                            {"step": 6, "event": "end"}])
    write_log(ckpt / "runB" / "log.jsonl", [start(0), {"step": 5}, {"event": "start", "step": 5, "args": {"train_lines": 123}},
                                            {"step": 10, "event": "end"}])
    (root / "fonts/extra/Salinan.ttf").unlink()
    warnings = ed.build(root, "runB", root / "ujifont")["warnings"]
    assert len(warnings) == 3
    assert "Log run runA: sebuah proses mulai lagi di langkah 2 sesudah proses sebelumnya melewati langkah checkpoint 6" in warnings[0]
    assert "Run runB dilanjutkan dengan argumen data yang berbeda" in warnings[1]
    assert "font tambahan dari fonts/extra, tetapi folder itu kosong" in warnings[2]

    # Pemunduran yang sama, tetapi proses kedua memeriksa di langkah lain (--eval-every lain): tetap diperingatkan.
    shutil.copy(TRAIN_FONTS[0], root / "fonts/extra/Salinan.ttf")
    write_log(ckpt / "runB" / "log.jsonl", [start(0), {"step": 10, "val_cer": 0.05}, {"step": 10, "event": "end"}])
    write_log(ckpt / "runA" / "log.jsonl", [start(0), {"step": 6, "val_cer": 0.3}, start(2), {"step": 5, "val_cer": 0.2},
                                            {"step": 7, "val_cer": 0.2}])
    warnings = ed.build(root, "runB", root / "ujifont")["warnings"]
    assert len(warnings) == 1 and "Log run runA: sebuah proses mulai lagi di langkah 2" in warnings[0]
    # Dimundurkan dua kali: yang disebut pemunduran TERAKHIR (proses-proses sesudahnya yang dihitung).
    write_log(ckpt / "runA" / "log.jsonl", [start(0), {"step": 6, "val_cer": 0.3}, start(2), {"step": 7, "val_cer": 0.2},
                                            start(4), {"step": 6, "val_cer": 0.2}])
    warnings = ed.build(root, "runB", root / "ujifont")["warnings"]
    assert len(warnings) == 1 and "Log run runA: sebuah proses mulai lagi di langkah 4 sesudah" in warnings[0]
    # Bukan pemunduran: proses pertama mati di langkah 5 (sebelum langkah checkpoint 6) lalu dilanjutkan dari 4, dan
    # proses yang dilanjutkan tepat di langkah checkpoint run itu.
    write_log(ckpt / "runA" / "log.jsonl", [start(0), {"step": 5, "val_cer": 0.3}, start(4), {"step": 6, "val_cer": 0.2},
                                            start(6), {"step": 9, "val_cer": 0.2}])
    assert ed.build(root, "runB", root / "ujifont")["warnings"] == []
    # y03: proses lanjutan dengan --workers lain mengubah urutan batch (0 lawan > 0), jadi ikut diperingatkan.
    write_log(ckpt / "runB" / "log.jsonl", [start(0), {"step": 5}, {"event": "start", "step": 5, "args": {"workers": 0}},
                                            {"step": 10, "event": "end"}])
    warnings = ed.build(root, "runB", root / "ujifont")["warnings"]
    assert len(warnings) == 1 and "Run runB dilanjutkan dengan argumen data yang berbeda" in warnings[0]


def test_paths_outside_the_repo_and_the_last_histogram_bin(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    assert ed.relative(root / "out/checkpoints/x/last_snapshot.pt", root) == "out/checkpoints/x/last_snapshot.pt"
    outside = ed.relative(tmp_path / "pengguna rahasia" / "model" / "last_snapshot.pt", root)
    assert outside == "(luar repo)/model/last_snapshot.pt" and "rahasia" not in outside

    # src.corpus.length_histogram memasukkan baris sepanjang MAX_LEN (80) ke keranjang "70-79".
    assert ed.histogram({"20-29": 5, "70-79": 7}) == [{"range": "20-29", "lines": 5}, {"range": "70-80", "lines": 7}]
    assert ed.histogram({"20-29": 5, "30-39": 7})[-1]["range"] == "30-39"
    assert ed.histogram({}) == []


def test_main_refuses_values_that_are_not_strict_json(tmp_path, monkeypatch):
    card = ed.build(fake_repo(tmp_path / "repo"), "runB", tmp_path / "repo" / "ujifont")
    card["official"]["pipeline"] = "crnn_runB"
    card["usage"]["train"]["track_max"] = float("nan")
    monkeypatch.setattr(ed, "build", lambda: card)
    with pytest.raises(ValueError):
        ed.main(["--out", str(tmp_path / "hasil")])
    assert not (tmp_path / "hasil/datasets.json").exists()


def test_real_training_data_changes_roles_and_leaves_the_schedule_unknown(tmp_path):
    root = fake_repo(tmp_path)
    commons = root / "data/real/commons"
    header = "image_path\ttext\tsource_id\tcondition\n"
    (commons / "labels.tsv").write_text(header + f"a.png\t{KA}\tx1\tpapan\n" + f"b.png\t{GA}\tx2\tpapan\n",
                                        encoding="utf-8", newline="\n")
    # Dua dari tiga draf sudah diverifikasi tetapi belum dipakai melatih: satu masih menunggu.
    card = ed.build(root, "runB", root / "ujifont")
    waiting = card["datasets"][2]
    assert (waiting["verified"], waiting["pending"], waiting["status"], waiting["roles"]) == (2, 1, "pending", {})

    # Semua draf sudah diverifikasi tetapi belum dipakai run resmi: terverifikasi, bukan "dipakai".
    (commons / "labels.tsv").write_text(header + "".join(f"{n}.png\t{KA}\tx{n}\tpapan\n" for n in range(3)),
                                        encoding="utf-8", newline="\n")
    ready = ed.build(root, "runB", root / "ujifont")["datasets"][2]
    assert (ready["verified"], ready["pending"], ready["status"], ready["roles"], ready["planned"]) == (
        3, 0, "verified", {}, ["train", "val"])

    # Run resmi dilatih dan divalidasi dengan data nyata (Fase 6): jadwal baris sintetis tidak dihitung, ditulis null.
    (commons / "labels_train.tsv").write_text(header + f"a.png\t{KA}\tx1\tpapan\n", encoding="utf-8", newline="\n")
    (commons / "labels_val.tsv").write_text(header + f"b.png\t{GA}\tx2\tpapan\n", encoding="utf-8", newline="\n")
    (commons / "labels.tsv").write_text(header + "".join(f"{n}.png\t{KA}\tx{n}\tpapan\n" for n in range(3)),
                                        encoding="utf-8", newline="\n")
    save_ckpt(root / "out/checkpoints/runB/last_snapshot.pt", 10, init="out/checkpoints/runA/last_snapshot.pt",
              real_train="data/real/commons/labels_train.tsv", real_val="data/real/commons/labels_val.tsv")
    card = ed.build(root, "runB", root / "ujifont")
    train = card["usage"]["train"]
    assert train["distinct_lines"] is None and train["real_train"] and train["real_val"] and train["samples"] == 80
    assert card["lineage"]["distinct_lines"] is None and card["lineage"]["runs"][0]["distinct_lines"] == 32
    assert card["rare"]["scheduled_lines"] == {} and card["rare"]["never_scheduled"] is None
    assert card["rare"]["pool_lines"]["max"] <= 1 and card["rare"]["codepoints"] == 2
    assert any("--overfit atau --real-train" in warning for warning in card["warnings"])
    used = card["datasets"][2]
    assert (used["status"], used["pending"], used["roles"], used["planned"]) == ("used", 0, {"train": 1, "val": 1}, [])


def test_font_roles_and_remarks_follow_folders_and_measurements(tmp_path, monkeypatch):
    """Aturan font teruji tanpa font tambahan (yang tidak ikut repo): ukurannya dibuat-buat, aturannya yang asli."""
    root = fake_repo(tmp_path)
    (root / "fonts/extra_review").mkdir()
    shutil.copy(TRAIN_FONTS[1], root / "fonts/extra_review/Ragu.ttf")
    with (root / "fonts/extra/SOURCES.md").open("a", encoding="utf-8") as f:
        f.write("| `Ragu.ttf` | `extra_review` | https://contoh/ragu | tidak tercantum | MERAGUKAN: contoh |\n")
    # Salinan: spasi sempit, dan lebar majunya sama dengan font uji di 72 dari 73 aksara (seperti CarakanJawa dan
    # javatext sungguhan). Font inti berbeda di semua aksara.
    reference = {ch: 0.5 for ch in ed.SPACING}
    close = {**reference, ed.SPACING[0]: None}

    def advances(path):
        name = Path(path).name
        return reference if name == "javatext.ttf" else close if name == "Salinan.ttf" else {ch: 0.7 for ch in ed.SPACING}

    monkeypatch.setattr(ed, "advances", advances)
    monkeypatch.setattr(ed, "space_ratio", lambda path: 0.05 if Path(path).name == "Salinan.ttf" else 0.2)

    card = ed.build(root, "runB", root / "ujifont")
    fonts = {f["file"]: f for f in card["fonts"]}
    assert [f["group"] for f in card["fonts"]] == ["core", "core", "extra", "test", "review"]
    salinan = fonts["Salinan.ttf"]
    assert salinan["drops_space"] is True
    assert salinan["shared_with_test"] == {"font": "javatext.ttf", "same": 72, "of": 73, "family": True}
    for core in TRAIN_FONTS:
        assert fonts[core.name]["drops_space"] is False
        assert fonts[core.name]["shared_with_test"] == {"font": "javatext.ttf", "same": 0, "of": 73, "family": False}
    # Font di folder yang bukan --extra-fonts run resmi tidak punya peran dan tidak diukur.
    ragu = fonts["Ragu.ttf"]
    assert ragu["roles"] == [] and ragu["review"] == "MERAGUKAN: contoh" and ragu["in_repo"] is False
    assert not {"drops_space", "missing", "shared_with_test"} & set(ragu)
    assert card["usage"]["train"]["fonts"] == 3 and card["warnings"] == []


def test_samples_validation_lines_and_corpus_leftovers_are_counted_as_they_are(tmp_path):
    """Empat hal yang lolos uji mutasi 2026-10-06: test lama tidak membedakan hitungan yang benar dari yang salah."""
    root = fake_repo(tmp_path)
    ckpt = root / "out/checkpoints/runB"
    # Kumpulan 100 baris, batch 8: 12 batch penuh + satu batch berisi 4. 13 langkah = satu epoch = 100 sampel, bukan
    # 13 x 8 = 104. --val-lines 500 lebih besar dari bagian validasi (100 baris): yang dibaca seluruh bagian itu.
    save_ckpt(ckpt / "last_snapshot.pt", 13, init="out/checkpoints/runA/last_snapshot.pt", train_lines=100, workers=2,
              val_lines=500)
    write_log(ckpt / "log.jsonl", [start(0), {"step": 13, "val_cer": 0.05, "val_lines": 100}, {"step": 13, "event": "end"}])
    # Statistik korpus yang jumlahnya tidak dijelaskan langkah pembangunannya: selisihnya ditulis, bukan dianggap nol.
    stats_path = root / "data/corpus_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats_path.write_text(json.dumps({**stats, "duplicates": 105}), encoding="utf-8")

    card = ed.build(root, "runB", root / "ujifont")
    train = card["usage"]["train"]
    assert (train["steps"], train["batch_size"], train["pool"]) == (13, 8, 100)
    assert (train["samples"], train["distinct_lines"]) == (100, 100)
    assert card["lineage"]["samples"] == 48 + 100 and card["lineage"]["steps"] == 6 + 13
    assert card["usage"]["val"] == {"lines": 100, "steps": [13], "fonts": 2}
    assert card["corpus"]["other"] == 2200 - (2300 - 105 + 10) == -5
    assert card["warnings"] == []

    # Berkas bagian latih ditulis ulang sesudah training: pemakaian dihitung dari berkas yang sekarang, dan itu disebut.
    # Snapshot disalin ulang tiap evaluasi (waktunya bisa lebih baru dari korpus), jadi log training ikut dilihat:
    # di sini korpus lebih tua dari semua snapshot tetapi lebih baru dari log run pertama.
    train_path = root / "data/splits/train.txt"
    now = train_path.stat().st_mtime
    for run in ("runA", "runB"):
        os.utime(root / "out/checkpoints" / run / "last_snapshot.pt", (now + 7200, now + 7200))
        os.utime(root / "out/checkpoints" / run / "log.jsonl", (now + 3600, now + 3600))
    assert ed.build(root, "runB", root / "ujifont")["warnings"] == []
    os.utime(root / "out/checkpoints/runA/log.jsonl", (now - 3600, now - 3600))
    warnings = ed.build(root, "runB", root / "ujifont")["warnings"]
    assert len(warnings) == 1 and warnings[0].startswith("data/splits/train.txt lebih baru daripada checkpoint")


def test_other_readers_of_the_test_split_and_font_counts_from_logs(tmp_path):
    root = fake_repo(tmp_path)
    # Evaluasi di luar out/eval yang membaca bagian uji secara acak: aksara langka sintetis (dua laporan) dan beam.
    settings = {"split": "data/splits/test.txt", "lines": 2000, "sampling": "acak"}
    (root / "out/compare/fase7").mkdir(parents=True)
    (root / "out/compare/rare_synthetic.json").write_text(json.dumps(
        {"settings": settings, "checkpoints": {"a": {}, "b": {}, "c": {}}}), encoding="utf-8")
    (root / "out/compare/fase7/rare_synthetic.json").write_text(json.dumps(
        {"settings": {**settings, "lines": 50}, "checkpoints": {"a": {}, "b": {}}}), encoding="utf-8")
    (root / "out/compare/val_synthetic.json").write_text(json.dumps({"settings": {"split": "data/splits/val.txt", "lines": 9}}),
                                                         encoding="utf-8")
    (root / "out/compare/lain/").mkdir()
    (root / "out/compare/lain/rare_synthetic.json").write_text(json.dumps(
        {"settings": {"split": "data/splits/val.txt", "lines": 77}, "checkpoints": {"a": {}}}), encoding="utf-8")
    (root / "out/beam").mkdir()
    (root / "out/beam/x.json").write_text(json.dumps({"clean_heldout_font": {"lines": 300}, "real": {"lines": 3}}), encoding="utf-8")
    (root / "out/beam/rusak.json").write_text("{bukan json", encoding="utf-8")
    (root / "out/beam/tanpa.json").write_text(json.dumps({"dev": {"lines": 300}}), encoding="utf-8")
    # d01: laporan tanpa hasil per font (hanya "semua") tetap dihitung barisnya: ini evaluasi penuh, bukan tes cepat.
    (root / "out/eval/lain2_G1_10k.json").write_text(json.dumps(
        {"goal": "G1", "split": "test", "results": {"semua": {"lines": OFFICIAL_LINES}}}), encoding="utf-8")

    card = ed.build(root, "runB", root / "ujifont")
    assert (card["usage"]["test"]["quick"], card["usage"]["test"]["quick_max_lines"], card["usage"]["test"]["full"]) == (2, 100, 2)
    # N11: evaluasi di luar out/eval yang membaca SEMUA baris data nyata ikut tercatat.
    assert card["usage"]["real"]["others"] == [{"kind": "beam", "file": "out/beam/x.json", "lines": 3}]
    assert card["usage"]["real"]["reports"] == 2
    assert card["usage"]["test"]["others"] == [
        {"kind": "rare_synthetic", "file": "out/compare/fase7/rare_synthetic.json", "lines": 50, "checkpoints": 2},
        {"kind": "rare_synthetic", "file": "out/compare/rare_synthetic.json", "lines": 2000, "checkpoints": 3},
        {"kind": "beam", "file": "out/beam/x.json", "lines": 300, "checkpoints": 1}]

    # F08 / N02: log run yang lebih baru mencatat NAMA fontnya. Jumlahnya dipakai apa adanya walau folder sudah berubah,
    # peran latih di tabel font hanya untuk nama yang tercatat, dan selisihnya dengan folder diperingatkan.
    ckpt = root / "out/checkpoints/runB"
    core = [font.name for font in TRAIN_FONTS]
    names = core + ["Salinan.ttf", "SudahDihapus.ttf"]
    ending = [{"step": 10, "val_cer": 0.05}, {"step": 10, "event": "end"}]
    write_log(ckpt / "log.jsonl", [{**start(0), "fonts": names}, *ending])
    card = ed.build(root, "runB", root / "ujifont")
    train = card["usage"]["train"]
    assert (train["fonts"], train["extra_fonts"], train["fonts_source"], train["font_names"]) == (4, 2, "log", names)
    assert card["warnings"] == ["Run runB: log mencatat 4 font latih, tetapi 1 di antaranya tidak ada lagi di folder font "
                                "(SudahDihapus.ttf); font itu tidak ada di tabel font kartu ini."]
    assert sorted(f["file"] for f in card["fonts"] if "train" in f["roles"]) == sorted(core + ["Salinan.ttf"])
    # Font yang ditambahkan ke folder SESUDAH run dilatih tidak diberi peran latih, dan itu diperingatkan.
    write_log(ckpt / "log.jsonl", [{**start(0), "fonts": core}, *ending])
    card = ed.build(root, "runB", root / "ujifont")
    assert (card["usage"]["train"]["fonts"], card["usage"]["train"]["extra_fonts"]) == (2, 0)
    assert card["warnings"] == ["Run runB: folder font sekarang memuat 1 font yang tidak tercatat di log run itu (Salinan.ttf); "
                                "font itu tidak diberi peran latih."]
    roles = {f["file"]: f["roles"] for f in card["fonts"]}
    assert roles["Salinan.ttf"] == [] and all(roles[name] == ["train", "val"] for name in core)
    assert "shared_with_test" not in next(f for f in card["fonts"] if f["file"] == "Salinan.ttf")
    # Run yang dilanjutkan: daftar proses TERAKHIR sebelum langkah checkpoint yang dipakai (proses itulah yang menulis
    # checkpoint-nya). Proses yang mulai tepat di langkah checkpoint baru melanjutkannya: daftarnya bukan milik
    # checkpoint ini.
    write_log(ckpt / "log.jsonl", [{**start(0), "fonts": names}, {"step": 5, "val_cer": 0.1}, {**start(5), "fonts": names[:3]},
                                   *ending, {**start(10), "fonts": core[:1]}])
    card = ed.build(root, "runB", root / "ujifont")
    train = card["usage"]["train"]
    assert (train["fonts"], train["extra_fonts"], train["fonts_source"]) == (3, 1, "log") and card["warnings"] == []
    assert ed.logged_fonts([{**start(10), "fonts": core}], 10) is None and ed.logged_fonts([start(0)], 10) is None
    assert ed.logged_fonts([{**start(9), "fonts": core}, {**start(10), "fonts": []}], 10) == core
    # Proses tanpa baris sintetis mencatat daftar kosong: tidak ada yang dibandingkan dengan folder.
    write_log(ckpt / "log.jsonl", [{**start(0), "fonts": []}, *ending])
    card = ed.build(root, "runB", root / "ujifont")
    assert (card["usage"]["train"]["fonts"], card["usage"]["train"]["font_names"]) == (0, []) and card["warnings"] == []
    assert not any("train" in f["roles"] for f in card["fonts"]) and {f["file"]: f["roles"] for f in card["fonts"]}[core[0]] == ["val"]

    (root / "out/results").mkdir()
    manifest = {"pipelines": [{"key": "crnn_b", "config": "crnn:runB@10 (lanjutan runA@6) · 5 font · aug fase5 · greedy"},
                              {"key": "crnn_a", "config": "crnn:runA@6 · 2 font · greedy"},
                              {"key": "vlm", "config": "vlm:frontier · zero-shot"}]}

    def record(config_b: str, config_a: str = "crnn:runA@6 · 2 font · greedy") -> list[str]:
        manifest["pipelines"][0]["config"], manifest["pipelines"][1]["config"] = config_b, config_a
        (root / "out/results/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return ed.build(root, "runB", root / "ujifont")["warnings"]

    (root / "out/results/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert ed.recorded_fonts(root) == {"runB": 5, "runA": 2}
    # Jumlah dari log tidak dibandingkan dengan hitungan ekspor hasil: nama-namanya sudah dibandingkan sendiri di atas.
    write_log(ckpt / "log.jsonl", [{**start(0), "fonts": core + ["Salinan.ttf"]}, *ending])
    card = ed.build(root, "runB", root / "ujifont")
    assert card["usage"]["train"]["fonts"] == 3 and card["warnings"] == []

    # Tanpa daftar di log, jumlahnya dari folder sekarang. Bila ekspor hasil mencatat jumlah lain, itu diperingatkan:
    # untuk run resmi maupun run sebelumnya di rantai (d05), dan juga bila folder kini berisi LEBIH banyak font (d06).
    write_log(ckpt / "log.jsonl", [start(0), *ending])
    card = ed.build(root, "runB", root / "ujifont")
    train = card["usage"]["train"]
    assert (train["fonts"], train["fonts_source"], train["font_names"]) == (3, "folder", None)
    assert card["warnings"] == ["Run runB: ekspor hasil mencatat 5 font latih, tetapi folder font sekarang berisi 3; jumlah dan "
                                "daftar font di kartu ini mengikuti folder sekarang, bukan font yang dipakai saat run itu dilatih."]
    warnings = record("crnn:runB@10 · 2 font · greedy", "crnn:runA@6 · 7 font · greedy")
    assert [w.split(";")[0] for w in warnings] == [
        "Run runA: ekspor hasil mencatat 7 font latih, tetapi folder font sekarang berisi 2",
        "Run runB: ekspor hasil mencatat 2 font latih, tetapi folder font sekarang berisi 3"]
    assert record("crnn:runB@10 · 3 font · greedy") == []
    # d07: daftar font bercacat (CLAUDE.md, tiga cacat data training) dipakai kartu metode; isinya dijaga di sini.
    assert ed.FONT_DEFECTS == ("BasaJan.ttf", "NewKramawirya.ttf") and set(ed.FONT_DEFECTS) <= set(ed.FONT_NOTES)


def test_shared_advances_count_only_glyphs_both_fonts_have():
    count = len(ed.SPACING)
    full = {ch: 600.0 for ch in ed.SPACING}
    assert ed.shared_advances(full, full) == {"same": count, "of": count, "family": True}
    # Glyph yang tidak ada di kedua font bukan "lebar yang sama": dua font kosong tidak sekeluarga.
    empty = {ch: None for ch in ed.SPACING}
    assert ed.shared_advances(empty, empty) == {"same": 0, "of": count, "family": False}
    assert ed.shared_advances(empty, full)["same"] == 0 and ed.shared_advances(full, empty)["same"] == 0
    # Ambang keluarga: paling sedikit FAMILY_SHARE dari semua codepoint berlebar sama.
    need = -(-count * 9 // 10)  # pembulatan ke atas dari 90%
    assert ed.FAMILY_SHARE == 0.9

    def differing(n):
        return {ch: (600.0 if i < n else 601.0 + i) for i, ch in enumerate(ed.SPACING)}

    assert ed.shared_advances(differing(need), full) == {"same": need, "of": count, "family": True}
    assert ed.shared_advances(differing(need - 1), full) == {"same": need - 1, "of": count, "family": False}

