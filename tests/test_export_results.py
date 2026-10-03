"""Test scripts/export_results.py: run lanjutan opsional (fase6) dan cek angka resmi, tanpa model."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))  # scripts/ bukan paket

from export_results import (  # noqa: E402
    CHECKPOINTS,
    OPTIONAL_CHECKPOINTS,
    PIPELINES,
    add_optional,
    check_official,
    check_optional_run,
    insert_after,
    optional_official,
)

RARE, CTRL = OPTIONAL_CHECKPOINTS  # urutan deklarasi: rare lalu ctrl
RARE_CONFIG = ("crnn:fase6_rare@1500 (lanjutan fase5_fonts@1298) · 10 font · aug fase5 · "
               "+ aksara langka (sisip 0,3 · adeg-adeg 0,15) · greedy")
CTRL_CONFIG = ("crnn:fase6_ctrl@1500 (lanjutan fase5_fonts@1298) · 10 font · aug fase5 · "
               "tanpa aksara langka (pembanding) · greedy")


def run_args(run: str, insert: float = 0.0, opener: float = 0.0, steps: int = 1500, init: str = "") -> dict:
    """Bagian vars(args) src.train yang dibaca ekspor."""
    return {"run": run, "steps": steps, "stop_step": 0, "augment": "fase5", "rare_insert_prob": insert,
            "rare_opener_prob": opener, "init": init}


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


def keys(pipelines: list[dict]) -> list[str]:
    return [p["key"] for p in pipelines]


def metric(pipeline: str, value: float, scope: str = "nusaaksara_745") -> dict:
    return {"scope": scope, "pipeline": pipeline, "lines": 745, "cer": value}


# --- urutan & keikutsertaan ----------------------------------------------------------------------------------


def test_import_does_not_add_optional_runs():
    # Dulu run lanjutan disisipkan saat impor modul; sekarang hanya lewat add_optional di main().
    assert not {"crnn_fase6_rare", "crnn_fase6_ctrl"} & set(keys(PIPELINES))
    assert not {"crnn_fase6_rare", "crnn_fase6_ctrl"} & set(CHECKPOINTS)


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


def test_add_optional_without_snapshots_changes_nothing(tmp_path):
    pipelines, checkpoints, included = add_optional(PIPELINES, CHECKPOINTS, tmp_path)
    assert pipelines == PIPELINES and pipelines is not PIPELINES
    assert checkpoints == CHECKPOINTS and included == []


def test_add_optional_stops_on_unfinished_snapshot(tmp_path):
    save_ckpt(tmp_path / "fase6_ctrl" / "last_snapshot.pt", 884, run_args("fase6_ctrl"))
    with pytest.raises(SystemExit, match="crnn_fase6_ctrl: last_snapshot.pt di langkah 884.*1500.*belum selesai"):
        add_optional(PIPELINES, CHECKPOINTS, tmp_path)


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
