"""Test scripts/export_datasets.py: kartu data halaman Dataset, pada repo palsu kecil dan (bila ada) data sebenarnya."""

import json
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
from src.dataset import TRAIN_FONTS, LengthBucketSampler  # noqa: E402

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


def test_schedule_reproduces_src_train_order():
    rng = random.Random(7)
    lines = [KA * rng.randint(1, 30) + str(i) for i in range(500)]
    schedule = ed.Schedule(lines)

    # src.train: random.Random(seed).shuffle(train_lines); train_lines[:n]
    shuffled = list(lines)
    random.Random(3).shuffle(shuffled)
    assert [lines[i] for i in schedule.pool(120, 3)] == shuffled[:120]
    assert schedule.pool(60, 3) == schedule.pool(120, 3)[:60]  # kumpulan kecil = awalan kumpulan besar

    batches = list(iter(LengthBucketSampler(shuffled[:120], 8, shuffle=True, seed=3)))
    expected = {shuffled[i] for batch in batches[:5] for i in batch}
    got = schedule.scheduled(120, 3, 8, 5)
    assert {lines[i] for i in got} == expected and len(got) == 40
    assert schedule.scheduled(120, 3, 8, 0) == set()
    assert schedule.scheduled(120, 3, 8, len(batches) + 4) == set(schedule.pool(120, 3))  # lebih dari satu epoch
    assert schedule.scheduled(120, 3, 8, 3) < schedule.scheduled(120, 3, 8, 5)  # proses lanjutan mengulang awal


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
                                           start(4), {"step": 6, "val_cer": 0.2, "val_lines": 10}, {"step": 6, "event": "end"}])
    last = save_ckpt(ckpt / run / "last_snapshot.pt", 10, init="out/checkpoints/runA/last_snapshot.pt",
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
    assert card["corpus"]["length_histogram"][0] == {"range": "20-29", "lines": 900}
    assert card["corpus"]["split_buckets"] == {"train": 90, "val": 5, "test": 5}
    assert card["corpus"]["max_roundtrip_cer"] == 0.05 and card["corpus"]["min_count"]["train"] == 100

    first, last = card["lineage"]["runs"]
    assert (first["run"], first["init"], first["pool"], first["steps"], first["samples"]) == ("runA", None, 200, 6, 48)
    assert first["segments"] == [[0, 4], [4, 6]] and first["distinct_lines"] == 4 * 8  # proses kedua mengulang awal
    assert (first["fonts"], first["extra_fonts"], first["augment"]) == (2, 0, "none")
    assert (last["run"], last["init"], last["pool"], last["steps"], last["samples"]) == ("runB", "runA", 400, 10, 80)
    assert last["distinct_lines"] == 80 and last["processes"] == 1 and last["val_steps"] == [5, 10]
    assert (last["fonts"], last["extra_fonts"], last["track_prob"], last["drop_space_prob"]) == (3, 1, 0.5, 0.5)
    lineage = card["lineage"]
    assert lineage["complete"] and lineage["steps"] == 16 and lineage["samples"] == 128 and lineage["pool"] == 400
    assert max(first["distinct_lines"], last["distinct_lines"]) <= lineage["distinct_lines"] <= 32 + 80

    usage = card["usage"]
    assert usage["train"] == last
    assert usage["val"] == {"lines": 10, "steps": [5, 10], "fonts": 2}
    assert usage["test"]["lines"] == OFFICIAL_LINES and [g["code"] for g in usage["test"]["gates"]] == ["G1", "G2"]
    assert usage["test"]["gates"][1]["augment"] == "heavy" and usage["test"]["gates"][0]["fonts"] == ["javatext.ttf"]
    assert usage["real"] == {"lines": 3, "gates": [{"code": "G3", "lines": 3, "fonts": [], "split": None, "augment": None,
                                                    "source": "out/eval/runB_G3_full.json"}]}

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
    assert commons["license"] == "CC BY-SA 4.0, CC0" and commons["roles"] == {} and commons["planned"] == ["train", "val"]

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
    assert lineage["samples"] == sum(r["steps"] * r["batch_size"] for r in lineage["runs"])
    assert card["usage"]["val"]["lines"] <= splits[1]["lines"] and card["usage"]["test"]["lines"] <= splits[2]["lines"]
    roles = [set(f["roles"]) for f in card["fonts"]]
    assert sum("train" in r for r in roles) == train["fonts"] and sum("val" in r for r in roles) == len(TRAIN_FONTS)
    assert all(f["roles"] == [] for f in card["fonts"] if f["group"] in ("review", "rejected"))
    rare = card["rare"]
    assert rare["codepoints"] > 0 and rare["scheduled_lines"]["mean"] <= rare["pool_lines"]["mean"] <= rare["train_lines"]["mean"]

    # Teks label NusaAksara tidak boleh bocor ke kartu yang akan diimpor web dan ditampilkan.
    dump = json.dumps(card, ensure_ascii=False)
    with (ROOT / "data/real/nusaaksara/labels.tsv").open(encoding="utf-8", newline="") as f:
        labels = [line.split("\t")[1] for line in f.read().splitlines()[1:]]
    assert len(labels) == card["usage"]["real"]["lines"]
    assert not any(label in dump for label in labels if len(label) >= 4)
