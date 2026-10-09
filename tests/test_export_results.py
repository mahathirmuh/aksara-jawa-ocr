"""Test scripts/export_results.py: run lanjutan opsional (fase6, fase7), run resmi, dan cek angka resmi, tanpa model."""

import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))  # scripts/ bukan paket

from export_results import (  # noqa: E402
    CHECKPOINTS,
    MAX_CONFIG,
    OFFICIAL_LINES,
    OFFICIAL_RUN,
    OPTIONAL_CHECKPOINTS,
    PIPELINES,
    TARGETS,
    add_optional,
    check_official,
    check_optional_run,
    gate_problem,
    insert_after,
    official_gates,
    official_pipeline,
    optional_official,
    optional_pipeline,
)

RARE, CTRL, TRACK, TRACK_RARE, CTRL7 = OPTIONAL_CHECKPOINTS  # urutan deklarasi = urutan baris di web
OPTIONAL_KEYS = ["crnn_fase6_rare", "crnn_fase6_ctrl", "crnn_fase7_track", "crnn_fase7_track_rare", "crnn_fase7_ctrl"]
RARE_CONFIG = ("crnn:fase6_rare@1500 (lanjutan fase5_fonts@1298) · 10 font · aug fase5 · "
               "+ aksara langka (sisip 0,3 · adeg-adeg 0,15) · greedy")
CTRL_CONFIG = ("crnn:fase6_ctrl@1500 (lanjutan fase5_fonts@1298) · 10 font · aug fase5 · "
               "tanpa aksara langka (pembanding) · greedy")
TRACK_CONFIG = ("crnn:fase7_track@1500 (lanjutan fase6_ctrl@1500) · 10 font · aug fase5 · "
                "jarak antar-aksara p 0,5 hingga 0,3 em · greedy")
TRACK_RARE_CONFIG = ("crnn:fase7_track_rare@1500 (lanjutan fase6_ctrl@1500) · 10 font · aug fase5 · "
                     "jarak antar-aksara p 0,5 hingga 0,3 em · + aksara langka tersaring "
                     "(sisip 0,3 · adeg-adeg 0,15 · mirip >= 0,9 · tempel) · greedy")
CTRL7_CONFIG = ("crnn:fase7_ctrl@1500 (lanjutan fase6_ctrl@1500) · 10 font · aug fase5 · "
                "tanpa jarak tambahan (pembanding) · greedy")
# Opsi ketiga run fase7 seperti di scripts/run_fase7.sh (TRACK, RARE_FIXED) dan scripts/run_fase7_ctrl.sh.
FASE7 = {
    "fase7_track": {"track": 0.5},
    "fase7_track_rare": {"track": 0.5, "insert": 0.3, "opener": 0.15, "similarity": 0.9, "attach": True},
    "fase7_ctrl": {},
}


def run_args(run: str, insert: float = 0.0, opener: float = 0.0, steps: int = 1500, init: str = "") -> dict:
    """Bagian vars(args) src.train yang dibaca ekspor, seperti di checkpoint fase6: belum ada opsi fase7."""
    return {"run": run, "steps": steps, "stop_step": 0, "augment": "fase5", "rare_insert_prob": insert,
            "rare_opener_prob": opener, "init": init}


def fase7_args(run: str, track: float = 0.0, insert: float = 0.0, opener: float = 0.0, similarity: float = 0.0,
               attach: bool = False, track_max: float = 0.3, steps: int = 1500, init: str = "") -> dict:
    """`run_args` + opsi yang ada sejak fase7, dengan default src.train (track_max 0.3 juga saat track_prob 0)."""
    return {**run_args(run, insert, opener, steps, init), "rare_max_similarity": similarity, "rare_attach": attach,
            "track_prob": track, "track_max": track_max}


def save_ckpt(path: Path, step: int, args: dict) -> Path:
    """Checkpoint palsu: hanya kunci yang dibaca `checkpoint_meta`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "args": args}, path)
    return path


def fase6_dir(tmp_path: Path, runs: list[str]) -> Path:
    """out/checkpoints palsu: fase5_fonts@1298 sebagai bobot awal, lalu last_snapshot.pt selesai untuk `runs`."""
    init = save_ckpt(tmp_path / "fase5_fonts" / "last_snapshot.pt", 1298, run_args("fase5_fonts"))
    for run in runs:
        insert, opener = (0.3, 0.15) if run == "fase6_rare" else (0.0, 0.0)
        save_ckpt(tmp_path / run / "last_snapshot.pt", 1500, run_args(run, insert, opener, init=str(init)))
    return tmp_path


def fase7_dir(tmp_path: Path, runs: list[str]) -> Path:
    """`fase6_dir` dengan kedua run fase6 selesai, lalu last_snapshot.pt selesai untuk `runs` fase7.

    Bobot awal fase7 = fase6_ctrl@1500, jadi kedua run fase6 selalu ikut, seperti di out/checkpoints sebenarnya.
    """
    fase6_dir(tmp_path, ["fase6_rare", "fase6_ctrl"])
    init = tmp_path / "fase6_ctrl" / "last_snapshot.pt"
    for run in runs:
        save_ckpt(tmp_path / run / "last_snapshot.pt", 1500, fase7_args(run, init=str(init), **FASE7[run]))
    return tmp_path


def keys(pipelines: list[dict]) -> list[str]:
    return [p["key"] for p in pipelines]


def metric(pipeline: str, value: float, scope: str = "nusaaksara_745") -> dict:
    return {"scope": scope, "pipeline": pipeline, "lines": 745, "cer": value}


# --- urutan & keikutsertaan ----------------------------------------------------------------------------------


def test_import_does_not_add_optional_runs():
    # Dulu run lanjutan disisipkan saat impor modul; sekarang hanya lewat add_optional di main().
    assert not set(OPTIONAL_KEYS) & set(keys(PIPELINES))
    assert not set(OPTIONAL_KEYS) & set(CHECKPOINTS)


def test_optional_runs_are_declared_in_web_order_and_named_after_their_run():
    assert keys(OPTIONAL_CHECKPOINTS) == OPTIONAL_KEYS
    for spec in OPTIONAL_CHECKPOINTS:
        # scripts/compare_runs.py mencari out/eval/<run>_G3_full.json dengan membuang "crnn_" dari kunci.
        assert spec["key"] == f"crnn_{spec['run']}" and spec["label"] == f"CRNN {spec['run']}"


def test_insert_after_keeps_declaration_order_right_after_beam():
    out = insert_after(PIPELINES, [{"key": "crnn_fase6_rare"}, {"key": "crnn_fase6_ctrl"}])
    at = keys(out).index("crnn_fonts_beam")
    assert keys(out)[at + 1: at + 4] == ["crnn_fase6_rare", "crnn_fase6_ctrl", "vlm_zeroshot"]
    assert keys(out)[:at + 1] == keys(PIPELINES)[:at + 1]
    assert len(out) == len(PIPELINES) + 2


def test_add_optional_only_existing_snapshot_with_step_in_config(tmp_path):
    pipelines, checkpoints, included = add_optional(PIPELINES, CHECKPOINTS, fase6_dir(tmp_path, ["fase6_rare"]))
    assert included == [RARE]
    k = keys(pipelines)
    assert k[k.index("crnn_fonts_beam") + 1] == "crnn_fase6_rare"
    assert "crnn_fase6_ctrl" not in k
    rare = pipelines[k.index("crnn_fase6_rare")]
    assert rare == {"key": "crnn_fase6_rare", "label": "CRNN fase6_rare", "kind": "crnn", "status": "done",
                    "config": RARE_CONFIG}
    assert checkpoints == {**CHECKPOINTS, "crnn_fase6_rare": tmp_path / "fase6_rare" / "last_snapshot.pt"}
    # Salinan: daftar global tetap.
    assert "crnn_fase6_rare" not in keys(PIPELINES) and "crnn_fase6_rare" not in CHECKPOINTS


def test_add_optional_both_runs_rare_before_ctrl(tmp_path):
    pipelines, checkpoints, included = add_optional(PIPELINES, CHECKPOINTS,
                                                    fase6_dir(tmp_path, ["fase6_ctrl", "fase6_rare"]))
    assert included == [RARE, CTRL]
    k = keys(pipelines)
    at = k.index("crnn_fonts_beam")
    assert k[at + 1: at + 3] == ["crnn_fase6_rare", "crnn_fase6_ctrl"]
    assert pipelines[at + 2]["config"] == CTRL_CONFIG
    assert list(checkpoints) == [*CHECKPOINTS, "crnn_fase6_rare", "crnn_fase6_ctrl"]


def test_add_optional_all_runs_fase6_then_fase7_with_config_from_checkpoint(tmp_path):
    root = fase7_dir(tmp_path, ["fase7_ctrl", "fase7_track_rare", "fase7_track"])
    pipelines, checkpoints, included = add_optional(PIPELINES, CHECKPOINTS, root)
    assert included == [RARE, CTRL, TRACK, TRACK_RARE, CTRL7]
    k = keys(pipelines)
    at = k.index("crnn_fonts_beam")
    assert k[at + 1: at + 7] == [*OPTIONAL_KEYS, "vlm_zeroshot"]
    assert k[:at + 1] == keys(PIPELINES)[:at + 1] and len(pipelines) == len(PIPELINES) + 5
    assert [p["config"] for p in pipelines[at + 1: at + 6]] == [RARE_CONFIG, CTRL_CONFIG, TRACK_CONFIG,
                                                                TRACK_RARE_CONFIG, CTRL7_CONFIG]
    assert pipelines[at + 3] == {"key": "crnn_fase7_track", "label": "CRNN fase7_track", "kind": "crnn",
                                 "status": "done", "config": TRACK_CONFIG}
    assert [p["label"] for p in pipelines[at + 4: at + 6]] == ["CRNN fase7_track_rare", "CRNN fase7_ctrl"]
    assert list(checkpoints) == [*CHECKPOINTS, *OPTIONAL_KEYS]
    assert checkpoints["crnn_fase7_track_rare"] == tmp_path / "fase7_track_rare" / "last_snapshot.pt"
    assert not set(OPTIONAL_KEYS) & set(keys(PIPELINES)) and not set(OPTIONAL_KEYS) & set(CHECKPOINTS)


@pytest.mark.parametrize("finished, expected", [
    ([], []),  # rantai fase7 belum menyelesaikan satu run pun: ekspor sama dengan sebelum fase7
    (["fase7_track"], [TRACK]),
    (["fase7_track", "fase7_track_rare"], [TRACK, TRACK_RARE]),
    (["fase7_track", "fase7_ctrl"], [TRACK, CTRL7]),  # fase7_track_rare gagal, kontrol tetap dijalankan
])
def test_add_optional_skips_fase7_runs_that_have_no_snapshot_yet(tmp_path, finished, expected):
    root = fase7_dir(tmp_path, finished)
    init = str(root / "fase6_ctrl" / "last_snapshot.pt")
    for run in sorted(set(FASE7) - set(finished)):
        # Run yang masih dilatih punya last.pt (di sini langkah 700), tetapi last_snapshot.pt baru disalin
        # scripts/fase6_common.sh::evaluate setelah langkah terakhir.
        save_ckpt(root / run / "last.pt", 700, fase7_args(run, init=init, **FASE7[run]))
    pipelines, checkpoints, included = add_optional(PIPELINES, CHECKPOINTS, root)
    assert included == [RARE, CTRL, *expected]
    exported = ["crnn_fase6_rare", "crnn_fase6_ctrl", *keys(expected)]
    k = keys(pipelines)
    at = k.index("crnn_fonts_beam")
    assert k[at + 1: at + 2 + len(exported)] == [*exported, "vlm_zeroshot"]
    assert list(checkpoints) == [*CHECKPOINTS, *exported]


def test_add_optional_without_snapshots_changes_nothing(tmp_path):
    pipelines, checkpoints, included = add_optional(PIPELINES, CHECKPOINTS, tmp_path)
    assert pipelines == PIPELINES and pipelines is not PIPELINES
    assert checkpoints == CHECKPOINTS and included == []


def test_add_optional_stops_on_unfinished_snapshot(tmp_path):
    save_ckpt(tmp_path / "fase6_ctrl" / "last_snapshot.pt", 884, run_args("fase6_ctrl"))
    with pytest.raises(SystemExit, match="crnn_fase6_ctrl: last_snapshot.pt di langkah 884.*1500.*belum selesai"):
        add_optional(PIPELINES, CHECKPOINTS, tmp_path)


def test_add_optional_stops_on_unfinished_fase7_snapshot(tmp_path):
    root = fase7_dir(tmp_path, ["fase7_track"])
    init = str(root / "fase6_ctrl" / "last_snapshot.pt")
    save_ckpt(root / "fase7_track_rare" / "last_snapshot.pt", 500,
              fase7_args("fase7_track_rare", init=init, **FASE7["fase7_track_rare"]))
    with pytest.raises(SystemExit, match="crnn_fase7_track_rare: last_snapshot.pt di langkah 500.*1500.*belum selesai"):
        add_optional(PIPELINES, CHECKPOINTS, root)


# --- langkah & flag aksara langka ----------------------------------------------------------------------------


def test_check_optional_run_accepts_finished_runs_with_matching_flags():
    check_optional_run(RARE, 1500, run_args("fase6_rare", 0.3, 0.15))
    check_optional_run(RARE, 1500, run_args("fase6_rare", 0.3, 0.0))  # adeg-adeg pembuka tidak wajib
    check_optional_run(CTRL, 1500, run_args("fase6_ctrl"))
    check_optional_run(CTRL, 1500, {"run": "fase6_ctrl", "steps": 1500})  # checkpoint lama tanpa opsi aksara langka


@pytest.mark.parametrize("spec, step, args, message", [
    (RARE, 884, run_args("fase6_rare", 0.3, 0.15), "langkah 884, padahal run dijadwalkan 1500"),
    (CTRL, 1500, run_args("fase6_ctrl", steps=0), "belum selesai"),  # --steps 0: selesai tidak bisa dipastikan
    (RARE, 1500, run_args("fase6_rare", 0.0, 0.15), "run aksara langka, tetapi rare_insert_prob = 0"),
    (CTRL, 1500, run_args("fase6_ctrl", 0.3, 0.0), "run pembanding harus tanpa aksara langka"),
    (CTRL, 1500, run_args("fase6_ctrl", 0.0, 0.15), "rare_opener_prob = 0.15"),
    (RARE, 1500, run_args("fase6_ctrl", 0.3, 0.15), "milik run 'fase6_ctrl', bukan 'fase6_rare'"),
])
def test_check_optional_run_rejects(spec, step, args, message):
    with pytest.raises(SystemExit, match=message):
        check_optional_run(spec, step, args)


# --- jarak antar-aksara & saringan aksara langka (fase7) -----------------------------------------------------


def test_check_optional_run_treats_options_missing_from_old_checkpoints_as_off():
    # Checkpoint fase6 dibuat sebelum opsi fase7 ada: args-nya tidak punya keempat kunci ini.
    old_rare, old_ctrl = run_args("fase6_rare", 0.3, 0.15), run_args("fase6_ctrl")
    assert not {"track_prob", "track_max", "rare_max_similarity", "rare_attach"} & (set(old_rare) | set(old_ctrl))
    check_optional_run(RARE, 1500, old_rare)
    check_optional_run(CTRL, 1500, old_ctrl)
    # Run fase6 yang diulang dengan kode sekarang menyimpan opsi itu dengan nilai mati.
    check_optional_run(RARE, 1500, fase7_args("fase6_rare", insert=0.3, opener=0.15))
    check_optional_run(CTRL, 1500, fase7_args("fase6_ctrl"))
    check_optional_run(CTRL, 1500, {**run_args("fase6_ctrl"), "track_prob": None, "rare_attach": None})


def test_check_optional_run_accepts_fase7_runs_with_matching_arguments():
    for spec in (TRACK, TRACK_RARE, CTRL7):
        check_optional_run(spec, 1500, fase7_args(spec["run"], **FASE7[spec["run"]]))
    check_optional_run(TRACK_RARE, 1500, fase7_args("fase7_track_rare", 0.5, 0.3, 0.0, 0.9, True))  # tanpa pembuka
    # src.train hanya membuat RareText bila ada sisipan/pembuka: saringan pada run tanpa aksara langka tidak dipakai.
    check_optional_run(TRACK, 1500, fase7_args("fase7_track", 0.5, similarity=0.9, attach=True))
    check_optional_run(CTRL7, 1500, fase7_args("fase7_ctrl", similarity=0.9, attach=True))


@pytest.mark.parametrize("spec, args, message", [
    (TRACK, fase7_args("fase7_track"), "crnn_fase7_track: run jarak antar-aksara, tetapi track_prob = 0"),
    (TRACK, fase7_args("fase7_track", 0.5, track_max=0.0), "tetapi track_prob = 0.5, track_max = 0"),
    (TRACK, run_args("fase7_track"), "run jarak antar-aksara, tetapi track_prob = 0, track_max = 0"),  # args gaya fase6
    (TRACK, fase7_args("fase7_track", 0.5, insert=0.3), "crnn_fase7_track: run jarak antar-aksara harus tanpa aksara"),
    (TRACK, fase7_args("fase7_track", 0.5, opener=0.15), "tanpa aksara langka, tetapi .*rare_opener_prob = 0.15"),
    (TRACK_RARE, fase7_args("fase7_track_rare", 0.0, 0.3, 0.15, 0.9, True),
     "crnn_fase7_track_rare: run jarak antar-aksara, tetapi track_prob = 0, track_max = 0.3"),
    (TRACK_RARE, fase7_args("fase7_track_rare", 0.5, 0.0, 0.15, 0.9, True),
     "crnn_fase7_track_rare: run aksara langka, tetapi rare_insert_prob = 0"),
    (TRACK_RARE, fase7_args("fase7_track_rare", 0.5, 0.3, 0.15, 0.0, True),
     "crnn_fase7_track_rare: run aksara langka tersaring, tetapi rare_max_similarity = 0, rare_attach = True"),
    (TRACK_RARE, fase7_args("fase7_track_rare", 0.5, 0.3, 0.15, 0.9, False),
     "tersaring, tetapi rare_max_similarity = 0.9, rare_attach = False"),
    (TRACK_RARE, {**run_args("fase7_track_rare", 0.3, 0.15), "track_prob": 0.5, "track_max": 0.3},
     "tersaring, tetapi rare_max_similarity = 0, rare_attach = False"),  # saringan tidak tercatat = mati
    (CTRL7, fase7_args("fase7_ctrl", 0.5), "crnn_fase7_ctrl: run tanpa jarak tambahan, tetapi track_prob = 0.5"),
    (CTRL7, fase7_args("fase7_ctrl", insert=0.3), "crnn_fase7_ctrl: run pembanding harus tanpa aksara langka"),
    (CTRL7, fase7_args("fase7_ctrl", opener=0.15), "run pembanding harus tanpa aksara langka.*rare_opener_prob = 0.15"),
    # Run fase6 dengan opsi fase7 menyala: config-nya tidak menyebut jarak maupun saringan, jadi akan menyesatkan.
    (RARE, fase7_args("fase6_rare", 0.5, 0.3, 0.15), "crnn_fase6_rare: run tanpa jarak tambahan, tetapi track_prob"),
    (CTRL, fase7_args("fase6_ctrl", 0.5), "crnn_fase6_ctrl: run tanpa jarak tambahan, tetapi track_prob = 0.5"),
    (RARE, fase7_args("fase6_rare", 0.0, 0.3, 0.15, 0.9),
     "crnn_fase6_rare: run aksara langka tanpa saringan, tetapi rare_max_similarity = 0.9, rare_attach = False"),
    (RARE, fase7_args("fase6_rare", 0.0, 0.3, 0.15, 0.0, True),
     "tanpa saringan, tetapi rare_max_similarity = 0, rare_attach = True"),
    (TRACK, fase7_args("fase7_ctrl", 0.5), "milik run 'fase7_ctrl', bukan 'fase7_track'"),
    (TRACK_RARE, fase7_args("fase7_track", 0.5), "milik run 'fase7_track', bukan 'fase7_track_rare'"),
    (CTRL7, fase7_args("fase7_ctrl", steps=3000), "langkah 1500, padahal run dijadwalkan 3000"),
])
def test_check_optional_run_rejects_arguments_that_do_not_match_the_run(spec, args, message):
    with pytest.raises(SystemExit, match=message):
        check_optional_run(spec, 1500, args)


def test_optional_pipeline_builds_config_from_checkpoint_arguments():
    args = fase7_args("fase7_track_rare", 0.25, 0.2, 0.0, 0.95, True, track_max=0.6)
    assert optional_pipeline(TRACK_RARE, 1200, args, "")["config"] == (
        "crnn:fase7_track_rare@1200 · 10 font · aug fase5 · jarak antar-aksara p 0,25 hingga 0,6 em · "
        "+ aksara langka tersaring (sisip 0,2 · adeg-adeg 0 · mirip >= 0,95 · tempel) · greedy")


def test_configs_fit_the_web_column_and_a_redirected_console():
    optional = [RARE_CONFIG, CTRL_CONFIG, TRACK_CONFIG, TRACK_RARE_CONFIG, CTRL7_CONFIG]
    assert MAX_CONFIG == 255  # kolom string Laravel = varchar(255) di PostgreSQL
    assert max(len(config) for config in [p["config"] for p in PIPELINES] + optional) <= MAX_CONFIG
    for config in optional:
        # main() mencetak config run lanjutan. stdout yang dialihkan ke berkas/pipa di Windows memakai codepage ANSI
        # (cp1252 di mesin ini) kecuali PYTHONIOENCODING diset; karakter di luar itu menghentikan ekspor.
        config.encode("cp1252")


def test_optional_pipeline_stops_when_config_exceeds_the_web_column():
    args = {**fase7_args("fase7_track", 0.5), "augment": "+".join(["binarize"] * 30)}  # daftar op yang panjang
    with pytest.raises(SystemExit, match=r"crnn_fase7_track: config \d+ karakter, batas kolom di web 255"):
        optional_pipeline(TRACK, 1500, args, "fase6_ctrl@1500")


# --- laporan resmi -------------------------------------------------------------------------------------------


def test_optional_official_requires_report_for_each_included_run(tmp_path):
    assert optional_official([], tmp_path) == {}
    with pytest.raises(SystemExit, match="crnn_fase6_rare: laporan resmi .*fase6_rare_G3_full.json tidak ada"):
        optional_official([RARE], tmp_path)
    report = tmp_path / "fase6_rare_G3_full.json"
    report.write_text(json.dumps({"cer": 0.34057793259088454}), encoding="utf-8")
    assert optional_official([RARE], tmp_path) == {"crnn_fase6_rare": (0.34057793259088454, str(report))}
    with pytest.raises(SystemExit, match="fase6_ctrl_G3_full.json tidak ada"):
        optional_official([RARE, CTRL], tmp_path)


def test_optional_official_requires_report_for_each_included_fase7_run(tmp_path):
    report = tmp_path / "fase7_track_G3_full.json"
    report.write_text(json.dumps({"cer": 0.25}), encoding="utf-8")
    assert optional_official([TRACK], tmp_path) == {"crnn_fase7_track": (0.25, str(report))}
    # Snapshot sudah disalin tetapi evaluasi 745 barisnya belum selesai: ekspor penuh berhenti sebelum inferensi.
    with pytest.raises(SystemExit, match=r"crnn_fase7_track_rare: laporan resmi .*fase7_track_rare_G3_full\.json tidak "
                                         r"ada \(buat dengan src\.evaluate out/checkpoints/fase7_track_rare/"
                                         r"last_snapshot\.pt .* --name fase7_track_rare_G3_full\)"):
        optional_official([TRACK, TRACK_RARE, CTRL7], tmp_path)


def test_check_official_passes_within_tolerance_and_ignores_other_scopes():
    official = {"crnn_fonts": (0.37246725212186627, "fonts"), "crnn_fase6_rare": (0.34057793259088454, "rare")}
    check_official([metric("crnn_fonts", 0.37246725212186627), metric("crnn_fase6_rare", 0.34057793259088454 + 1e-12),
                    metric("crnn_fase6_rare", 0.9, scope="blind_50")], official)


def test_check_official_stops_on_difference():
    official = {"crnn_fase6_rare": (0.34057793259088454, "out/eval/fase6_rare_G3_full.json")}
    with pytest.raises(SystemExit, match="crnn_fase6_rare: CER .* berbeda dari laporan out/eval/fase6_rare_G3_full"):
        check_official([metric("crnn_fase6_rare", 0.34057793259088454 + 2e-9)], official)


def test_check_official_stops_when_pipeline_has_no_full_metric():
    official = {"crnn_fase6_ctrl": (0.35, "out/eval/fase6_ctrl_G3_full.json")}
    with pytest.raises(SystemExit, match="crnn_fase6_ctrl: tidak ada CER 745 baris"):
        check_official([metric("crnn_fase6_ctrl", 0.35, scope="blind_50")], official)


# --- run resmi: gerbang G1-G3 dan kunci "official" -------------------------------------------------------


def gate_report(code: str, run: str, cer: float = 0.01, lines: int | None = None, **changes) -> dict:
    """Laporan src.evaluate yang sah untuk gerbang `code` milik `run`; `changes` menimpa kunci tingkat atas."""
    checkpoint = f"out/checkpoints/{run}/last_snapshot.pt"
    if code == "G3":
        report = {"checkpoint": checkpoint, "results": {"semua": {"lines": 745 if lines is None else lines, "cer": cer}},
                  "pad_ratio": 0.0}
    else:
        n = OFFICIAL_LINES if lines is None else lines
        block = {"lines": n, "cer": cer, "rejected_by_min_frames": 0}
        report = {"checkpoint": checkpoint, "split": "test", "augment": "none" if code == "G1" else "heavy",
                  "results": {"javatext.ttf": block, "semua": {"lines": n, "cer": cer}}}
    return {**report, "goal": code, "target": TARGETS[code], "cer": cer, "passed": cer < TARGETS[code], **changes}


def write_gate_reports(eval_dir: Path, run: str, prefix: str | None = None, cers=(0.003, 0.012, 0.21)) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    for (code, suffix), value in zip([("G1", "G1_10k"), ("G2", "G2_10k"), ("G3", "G3_full")], cers):
        (eval_dir / f"{prefix or run}_{suffix}.json").write_text(json.dumps(gate_report(code, run, value)), encoding="utf-8")


def test_official_pipeline_is_the_pipeline_that_reads_the_official_checkpoint(tmp_path):
    root = fase7_dir(tmp_path, ["fase7_track"])
    _, checkpoints, _ = add_optional(PIPELINES, CHECKPOINTS, root)
    assert official_pipeline(checkpoints, "fase7_track") == "crnn_fase7_track"
    assert official_pipeline(checkpoints, "fase5_fonts") == "crnn_fonts"  # run lama: kunci pendek
    with pytest.raises(SystemExit, match="fase7_ctrl.*tidak ikut diekspor"):  # snapshot belum ada
        official_pipeline(checkpoints, "fase7_ctrl")


def test_official_run_is_declared_as_a_followup_run_or_base_checkpoint():
    runs = [spec["run"] for spec in OPTIONAL_CHECKPOINTS] + [path.parent.name for path in CHECKPOINTS.values()]
    assert OFFICIAL_RUN in runs


def test_demo_service_defaults_to_the_official_checkpoint():
    # Demo (src/serve.py) dan angka resmi harus membaca checkpoint yang sama.
    from src import serve

    assert serve.DEFAULT_CHECKPOINT == f"out/checkpoints/{OFFICIAL_RUN}/last_snapshot.pt"


def test_official_gates_come_from_the_official_runs_reports(tmp_path):
    snapshot = save_ckpt(tmp_path / "ckpt" / "fase7_track" / "last_snapshot.pt", 1500, fase7_args("fase7_track", 0.5))
    write_gate_reports(tmp_path / "eval", "fase7_track")
    gates = official_gates("fase7_track", "CRNN fase7_track", snapshot, 745, tmp_path / "eval")

    assert [g["code"] for g in gates] == ["G1", "G2", "G3", "G4"]
    assert [g["value"] for g in gates[:3]] == [0.003, 0.012, 0.21]
    assert [g["passed"] for g in gates] == [True, True, False, True]
    assert [g["target"] for g in gates[:3]] == [0.02, 0.05, 0.08]
    assert [g["source"] for g in gates[:3]] == ["out/eval/fase7_track_G1_10k.json", "out/eval/fase7_track_G2_10k.json",
                                                "out/eval/fase7_track_G3_full.json"]
    assert all("CRNN fase7_track" in g["basis"] for g in gates[:3])
    assert gates[0]["basis"].startswith("10.000 baris · javatext") and "CarakanJawa" in gates[0]["basis"]
    assert gates[2]["basis"].startswith("745 baris NusaAksara")
    assert all(len(g["basis"]) <= MAX_CONFIG and len(g["name"]) <= MAX_CONFIG for g in gates)  # kolom string di web


def test_official_gates_use_short_report_names_for_old_runs(tmp_path):
    snapshot = save_ckpt(tmp_path / "ckpt" / "fase5_fonts" / "last_snapshot.pt", 1298, run_args("fase5_fonts"))
    write_gate_reports(tmp_path / "eval", "fase5_fonts", prefix="fonts")
    gates = official_gates("fase5_fonts", "CRNN fase5_fonts", snapshot, 745, tmp_path / "eval")
    assert gates[0]["source"] == "out/eval/fonts_G1_10k.json" and gates[2]["source"] == "out/eval/fonts_G3_full.json"


def test_official_gates_stop_when_a_report_is_missing_or_stale(tmp_path):
    eval_dir = tmp_path / "eval"
    snapshot = save_ckpt(tmp_path / "ckpt" / "fase7_track" / "last_snapshot.pt", 1500, fase7_args("fase7_track", 0.5))
    write_gate_reports(eval_dir, "fase7_track")
    (eval_dir / "fase7_track_G2_10k.json").unlink()
    with pytest.raises(SystemExit, match=r"laporan G2 .*tidak ada \(G1/G2: scripts/make_official.sh fase7_track\)"):
        official_gates("fase7_track", "CRNN fase7_track", snapshot, 745, eval_dir)

    # Run dilatih ulang sesudah dievaluasi: laporan lama tidak boleh menjadi angka resmi checkpoint baru.
    write_gate_reports(eval_dir, "fase7_track")
    newer = (eval_dir / "fase7_track_G1_10k.json").stat().st_mtime + 60
    os.utime(snapshot, (newer, newer))
    with pytest.raises(SystemExit, match="laporan G1 .*lebih tua dari last_snapshot.pt"):
        official_gates("fase7_track", "CRNN fase7_track", snapshot, 745, eval_dir)


@pytest.mark.parametrize("code, changes, lines, message", [
    ("G1", {"goal": "G2"}, None, "goal 'G2', bukan G1"),
    ("G1", {"checkpoint": "out/checkpoints/fase6_ctrl/last_snapshot.pt"}, None, "bukan milik run fase7_track"),
    ("G1", {}, 100, "100 baris, bukan 10000"),  # tes cepat q100 bukan angka resmi
    ("G1", {"augment": "heavy"}, None, "augmentasi 'heavy', bukan 'test' dan 'none'"),
    ("G2", {"augment": "fase5"}, None, "augmentasi 'fase5', bukan 'test' dan 'heavy'"),
    ("G2", {"split": "val"}, None, "split 'val'"),
    ("G3", {"pad_ratio": 0.06}, None, "pad_ratio 0.06"),
    ("G3", {}, 100, "100 baris, bukan seluruh 745 baris nyata"),
])
def test_gate_problem_rejects_reports_that_are_not_the_promised_evaluation(code, changes, lines, message):
    report = gate_report(code, "fase7_track", lines=lines, **changes)
    assert message in gate_problem(code, report, "fase7_track", 745)


def test_gate_problem_accepts_valid_reports_and_counts_rejected_lines():
    for code in ("G1", "G2", "G3"):
        assert gate_problem(code, gate_report(code, "fase7_track"), "fase7_track", 745) == ""
    # Baris yang ditolak T >= 1,5L tidak dinilai tetapi tetap bagian dari 10.000 baris yang dievaluasi.
    report = gate_report("G1", "fase7_track", lines=OFFICIAL_LINES - 3)
    report["results"]["javatext.ttf"]["rejected_by_min_frames"] = 3
    assert gate_problem("G1", report, "fase7_track", 745) == ""
    # Font lain ikut dievaluasi: bukan gerbang G1 (javatext saja).
    report = gate_report("G1", "fase7_track")
    report["results"]["NotoSansJavanese-Regular.ttf"] = dict(report["results"]["javatext.ttf"])
    assert "bukan javatext.ttf saja" in gate_problem("G1", report, "fase7_track", 745)


# --- metrik karakter (recall, presisi, F1) untuk tabel perbandingan web ------------------------------------


def test_aggregate_adds_character_metrics_consistent_with_cer():
    from export_results import aggregate

    ka, na, ta = "ꦏ", "ꦤ", "ꦠ"
    refs = {"b1": ka + na + ta, "b2": ka + na}
    hyps = {"b1": ka + na + ta, "b2": ka}  # b2: na hilang -> M 4, D 1
    m = aggregate(refs, hyps, ["b1", "b2"])
    assert m["lines"] == 2 and m["cer"] == pytest.approx(1 / 5) and m["exact"] == 0.5
    assert (m["precision"], m["recall"], m["char_accuracy"]) == (1.0, pytest.approx(0.8), pytest.approx(0.8))
    assert m["f1"] == pytest.approx(2 * 0.8 / 1.8)
    # CER dan 1 - recall sama-sama memakai pembilang hapus + tertukar bila tidak ada sisipan.
    assert m["cer"] == pytest.approx(1 - m["recall"])


def test_class_metrics_block_only_covers_pipelines_with_every_line():
    from export_results import class_metrics_block

    ka, na, ta = "ꦏ", "ꦤ", "ꦠ"
    refs = {"b1": ka + na + ta, "b2": ka + na}
    hyps = {"crnn_a": {"b1": ka + na + ta, "b2": ka}, "vlm": {"b1": ka + na + ta}}  # vlm hanya 1 baris: dilewati
    block = class_metrics_block(refs, hyps, ["b1", "b2"], "nusaaksara_745")
    assert block["scope"] == "nusaaksara_745"
    assert {it["pipeline"] for it in block["items"]} == {"crnn_a"}
    rows = {it["char"]: it for it in block["items"]}
    assert rows[na] == {"pipeline": "crnn_a", "char": na, "code": "U+A9A4", "name": "JAVANESE LETTER NA",
                        "ref": 2, "hyp": 1, "tp": 1, "fn": 1, "fp": 0, "precision": 1.0, "recall": 0.5,
                        "f1": pytest.approx(2 * 1.0 * 0.5 / 1.5)}
    assert [it["char"] for it in block["items"]] == [ka, na, ta]  # paling sering di referensi dulu
