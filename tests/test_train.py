"""Test pemecahan batch lebar (OOM fase5_quick) — Fase 5."""

import pytest
import torch
from torch import nn

from src.dataset import DOWNSAMPLE, H
from src.train import ctc_loss, split_batch

# in_lens semuanya berbeda supaya sampel asal bisa dikenali dari in_len-nya.
WIDTHS = [803, 2401, 1210, 3999, 402, 2000, 1601, 7]
LENGTHS = [5, 30, 12, 60, 3, 25, 20, 1]
BUDGET = 8000


def make_batch():
    g = torch.Generator().manual_seed(0)
    X = torch.zeros(len(WIDTHS), 1, H, max(WIDTHS))
    for i, w in enumerate(WIDTHS):
        X[i, :, :, :w] = torch.rand(1, H, w, generator=g)
    Y = torch.randint(1, 10, (sum(LENGTHS),), generator=g)
    in_lens = torch.tensor([max(1, w // DOWNSAMPLE) for w in WIDTHS])
    return X, Y, in_lens, torch.tensor(LENGTHS)


def test_batch_within_budget_is_untouched():
    batch = make_batch()
    chunks = split_batch(*batch, max_columns=len(WIDTHS) * max(WIDTHS))
    assert len(chunks) == 1 and chunks[0][0] is batch[0]
    assert len(split_batch(*batch, max_columns=0)) == 1


def test_split_keeps_every_sample_intact_within_budget():
    X, Y, in_lens, tgt_lens = make_batch()
    offsets = [0, *torch.cumsum(tgt_lens, 0).tolist()]
    index = {int(n): i for i, n in enumerate(in_lens)}
    chunks = split_batch(X, Y, in_lens, tgt_lens, BUDGET)
    assert len(chunks) > 1
    seen = []
    for cx, cy, c_in, c_tgt in chunks:
        assert cx.shape[0] * cx.shape[-1] <= BUDGET
        c_offsets = [0, *torch.cumsum(c_tgt, 0).tolist()]
        for j, n in enumerate(c_in.tolist()):
            i = index[n]
            seen.append(i)
            w = WIDTHS[i]
            assert cx.shape[-1] >= w, "potongan tidak boleh memangkas tinta"
            assert torch.equal(cx[j, :, :, :w], X[i, :, :, :w])
            assert torch.equal(cy[c_offsets[j] : c_offsets[j + 1]], Y[offsets[i] : offsets[i + 1]])
            assert int(c_tgt[j]) == LENGTHS[i]
    assert sorted(seen) == list(range(len(WIDTHS)))


def test_weighted_chunk_losses_equal_full_batch_loss():
    X, Y, in_lens, tgt_lens = make_batch()
    g = torch.Generator().manual_seed(1)
    per_sample = torch.randn(len(WIDTHS), int(in_lens.max()), 10, generator=g)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    cpu = torch.device("cpu")
    full = ctc_loss(ctc, per_sample, Y, in_lens, tgt_lens, cpu)

    index = {int(n): i for i, n in enumerate(in_lens)}
    total = torch.tensor(0.0)
    for cx, cy, c_in, c_tgt in split_batch(X, Y, in_lens, tgt_lens, BUDGET):
        rows = [index[int(n)] for n in c_in]
        logits = per_sample[rows, : int(c_in.max())]
        total = total + ctc_loss(ctc, logits, cy, c_in, c_tgt, cpu) * (cx.shape[0] / X.shape[0])
    assert torch.allclose(total, full, atol=1e-5)


def test_start_record_lists_the_fonts_the_process_renders_with(tmp_path):
    """Event start log mencatat nama font perender data sintetis proses itu; kartu data dan kartu metode membacanya."""
    import json
    from pathlib import Path

    from src.dataset import TRAIN_FONTS
    from src.train import parse_args, start_record

    extra = tmp_path / "extra"
    extra.mkdir()
    for name in ("B.otf", "A.ttf", "catatan.md"):
        (extra / name).write_bytes(b"")
    core = [Path(font).name for font in TRAIN_FONTS]
    args = parse_args(["--run", "uji", "--extra-fonts", str(extra), "--train-lines", "100"])
    record = start_record(args, 5, 123, "cpu")
    assert record == {"event": "start", "step": 5, "args": vars(args), "params": 123, "device": "cpu",
                      "fonts": core + ["A.ttf", "B.otf"]}
    assert json.loads(json.dumps(record)) == record  # bisa ditulis ke log.jsonl apa adanya
    assert start_record(parse_args(["--run", "uji", "--overfit", "32"]), 0, 1, "cpu")["fonts"] == core
    # --overfit merender baris sintetis berapa pun --train-lines (build_datasets mengambil N baris pertama).
    assert start_record(parse_args(["--run", "uji", "--overfit", "32", "--train-lines", "0"]), 0, 1, "cpu")["fonts"] == core
    # Proses tanpa baris sintetis (--train-lines 0 dengan --real-train) tidak merender dengan font apa pun: daftarnya
    # kosong, bukan daftar font inti.
    real = parse_args(["--run", "uji", "--extra-fonts", str(extra), "--train-lines", "0", "--real-train", "labels.tsv"])
    assert start_record(real, 0, 1, "cpu")["fonts"] == []
    # main() tidak bisa dijalankan di test (ia melatih model), jadi tempat catatan itu ditulis dijaga lewat sumbernya,
    # dan font yang dicatat harus daftar yang sama dengan yang dipakai merender (training_fonts di build_datasets).
    import inspect

    from src import train

    assert "log(start_record(args, step, n_params, device))" in inspect.getsource(train.main)
    assert "train_fonts = training_fonts(args)" in inspect.getsource(train.build_datasets)


def test_training_fonts_are_core_fonts_plus_the_extra_folder(tmp_path):
    # Daftar ini dicatat di event start log; kartu data dan kartu metode membacanya dari sana.
    import shutil

    from src.dataset import TRAIN_FONTS
    from src.train import parse_args, training_fonts

    assert training_fonts(parse_args(["--run", "x"])) == list(TRAIN_FONTS)
    shutil.copy(TRAIN_FONTS[0], tmp_path / "Tambahan.TTF")
    (tmp_path / "catatan.md").write_text("bukan font", encoding="utf-8")
    fonts = training_fonts(parse_args(["--run", "x", "--extra-fonts", str(tmp_path)]))
    assert [font.name for font in fonts] == [font.name for font in TRAIN_FONTS] + ["Tambahan.TTF"]
    empty = tmp_path / "kosong"
    empty.mkdir()
    with pytest.raises(ValueError, match="Tidak ada font"):
        training_fonts(parse_args(["--run", "x", "--extra-fonts", str(empty)]))


def test_new_data_options_default_to_off():
    # Opsi fase7 mati secara bawaan, supaya perintah run lama (fase5, fase6) menghasilkan data yang sama.
    from src.train import parse_args

    args = parse_args(["--run", "x"])
    assert (args.track_prob, args.rare_max_similarity, args.rare_attach) == (0.0, 0.0, False)
    args = parse_args(["--run", "x", "--track-prob", "0.5", "--track-max", "0.25", "--rare-max-similarity", "0.9",
                       "--rare-attach"])
    assert (args.track_prob, args.track_max, args.rare_max_similarity, args.rare_attach) == (0.5, 0.25, 0.9, True)
