"""Fase 4-6 — training CRNN + CTC.

  python -m src.train --run overfit32 --overfit 32 --epochs 1000 --eval-every 25   # 4a
  python -m src.train --run base --train-lines 50000                                # 4b
  python -m src.train --run aug --init out/checkpoints/base/best.pt --augment train # 5
  python -m src.train --run real --init out/checkpoints/aug/best.pt --lr 1e-4 \\
      --real-train data/real/labels_train.tsv --real-val data/real/labels_val.tsv   # 6

Checkpoint ke out/checkpoints/<run>/ (last.pt, best.pt, log.jsonl). Run yang
terhenti (mis. --time-budget-hours habis) dilanjutkan dengan --resume.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.augment import build_augment
from src.dataset import DOWNSAMPLE, H, TRAIN_FONTS, LengthBucketSampler, SyntheticLines, collate
from src.decode import cer, greedy_decode
from src.model import CRNN
from src.real import ConcatLines, RealLines, read_labels
from src.text_augment import RareText
from src.tokenizer import BLANK, TOKENIZER_PATH, Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits"
CHECKPOINT_DIR = ROOT / "out" / "checkpoints"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="crnn")
    p.add_argument("--overfit", type=int, default=0, help="latih dan evaluasi pada N baris pertama saja (Fase 4a)")
    p.add_argument("--train-lines", type=int, default=50_000, help="baris sintetis; 0 = tanpa sintetis")
    p.add_argument("--val-lines", type=int, default=1_000)
    p.add_argument("--epochs", type=float, default=20)
    p.add_argument("--steps", type=int, default=0, help="total langkah; menimpa --epochs (siklus LR selesai tepat)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-batch-columns", type=int, default=64_000,
                   help="batas B x lebar per forward; batch lebih lebar dipecah dengan akumulasi gradien "
                        "(~6,7 GiB di XPU). 0 = tanpa batas")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--channels", default="32,64,128,256")
    p.add_argument("--hidden", type=int, default=256)
    # Tanpa packing, frame padding tidak pernah diawasi CTC dan model belajar
    # memancarkan simbol di sana; inference satu baris (tanpa padding) lalu
    # menghasilkan "pada lingsa" palsu di tepi kanan. Packing: ~20% lebih lambat di XPU.
    p.add_argument("--no-pack", dest="packed", action="store_false",
                   help="matikan packing LSTM (lebih cepat, tapi training != inference)")
    p.add_argument("--augment", default="none", help="preset src/augment.py untuk data sintetis (Fase 5)")
    p.add_argument("--extra-fonts", default="", help="direktori font tambahan untuk data training; val tetap font inti")
    p.add_argument("--drop-space-prob", type=float, default=0.0,
                   help="peluang spasi dihapus dari baris sintetis (teks nyata umumnya tanpa spasi)")
    p.add_argument("--rare-insert-prob", type=float, default=0.0,
                   help="peluang menyisipkan satu aksara langka (murda, swara, pa cerek, ...) ke baris sintetis")
    p.add_argument("--rare-opener-prob", type=float, default=0.0,
                   help="peluang membuka baris sintetis dengan pada adeg-adeg (pembuka paragraf cetakan)")
    p.add_argument("--rare-max-similarity", type=float, default=0.0,
                   help="lewati aksara langka yang glyph-nya di font terpilih mirip glyph lain (IoU >= nilai ini); "
                        "0 = sisipkan semua, seperti fase6_rare")
    p.add_argument("--rare-attach", action="store_true",
                   help="tempelkan pada/pangrangkep sisipan ke kata sebelumnya, tanpa menambah spasi")
    p.add_argument("--track-prob", type=float, default=0.0,
                   help="peluang baris sintetis dirender dengan jarak tambahan antar suku kata")
    p.add_argument("--track-max", type=float, default=0.3,
                   help="jarak tambahan terbesar (em), diundi seragam dari 0 per baris")
    p.add_argument("--real-train", default="", help="labels_train.tsv citra nyata (Fase 6)")
    p.add_argument("--real-val", default="", help="labels_val.tsv untuk memilih checkpoint (Fase 6)")
    p.add_argument("--real-repeat", type=int, default=10, help="berapa kali data nyata diulang per epoch")
    p.add_argument("--real-augment", default="rotate+blur+noise+jpeg")
    p.add_argument("--real-pad-ratio", type=float, default=0.0, help="margin putih pada citra nyata (train & val)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "xpu"])
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=0, help="thread torch; 0 = default")
    p.add_argument("--eval-every", type=int, default=0, help="langkah; 0 = sekali per epoch")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--time-budget-hours", type=float, default=0, help="berhenti rapi setelah sekian jam; 0 = tanpa batas")
    p.add_argument("--target-cer", type=float, default=0, help="berhenti begitu val CER di bawah nilai ini")
    p.add_argument("--stop-step", type=int, default=0,
                   help="berhenti rapi di langkah ini tanpa mengubah jadwal LR dari --steps (run pembanding)")
    p.add_argument("--init", default="", help="checkpoint awal (bobot saja), untuk Fase 5/6")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def prevent_sleep() -> None:
    """Cegah Windows tidur selama proses training hidup.

    Laptop dengan Modern Standby membekukan proses saat layar mati karena idle;
    training 4b pernah berhenti hampir 3 jam karenanya. SYSTEM + DISPLAY required
    diperlukan karena di Modern Standby layar mati berarti standby. Tidak mengubah
    pengaturan daya; efeknya hilang saat proses selesai. Menutup lid tetap menidurkan.
    """
    import sys

    if sys.platform != "win32":
        return
    import ctypes

    es_continuous, es_system_required, es_display_required = 0x80000000, 0x00000001, 0x00000002
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required | es_display_required)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            name = "xpu"
        else:
            name = "cpu"
    return torch.device(name)


def read_split(name: str) -> list[str]:
    return (SPLITS_DIR / f"{name}.txt").read_text(encoding="utf-8").splitlines()


def assert_downsample(model: CRNN, device: torch.device) -> None:
    """Invariant: DOWNSAMPLE harus sama dengan stride lebar model SEBENARNYA."""
    width = 257
    was_training = model.training
    model.eval()
    with torch.no_grad():
        frames = model(torch.zeros(1, 1, H, width, device=device)).shape[1]
    model.train(was_training)
    if frames != width // DOWNSAMPLE:
        raise RuntimeError(f"Model menghasilkan {frames} frame untuk lebar {width}; DOWNSAMPLE={DOWNSAMPLE} salah")


def ctc_loss(ctc: nn.CTCLoss, logits, Y, in_lens, tgt_lens, device: torch.device) -> torch.Tensor:
    log_probs = logits.float().log_softmax(-1).permute(1, 0, 2)  # (T, B, C)
    if device.type != "cuda":
        log_probs = log_probs.cpu()  # CTCLoss hanya punya kernel akselerasi untuk CUDA
    return ctc(log_probs, Y, in_lens, tgt_lens)


def split_batch(X, Y, in_lens, tgt_lens, max_columns: int) -> list[tuple]:
    """Pecah batch yang terlalu lebar menjadi potongan berisi <= max_columns kolom (B x W).

    Lebar batch mengikuti citra terlebar, dan memori aktivasi CNN sebanding dengan B x W
    (~105 KiB per kolom saat training di XPU). fase5_quick OOM di langkah ~950 pada batch
    32 baris panjang. Sampel diurutkan dari yang terlebar supaya padding tiap potongan minim;
    tiap potongan dipotong ke lebar sampel terlebarnya. in_lens tidak berubah.
    """
    batch, width = X.shape[0], X.shape[-1]
    if max_columns <= 0 or batch * width <= max_columns:
        return [(X, Y, in_lens, tgt_lens)]
    offsets = [0, *torch.cumsum(tgt_lens, 0).tolist()]
    order = sorted(range(batch), key=lambda i: -int(in_lens[i]))
    chunks, start = [], 0
    while start < batch:
        # Lebar asli w memenuhi w // DOWNSAMPLE == in_len, jadi w < (in_len + 1) * DOWNSAMPLE.
        chunk_width = min(width, (int(in_lens[order[start]]) + 1) * DOWNSAMPLE)
        idx = order[start : start + max(1, max_columns // chunk_width)]
        chunks.append((
            X[idx, :, :, :chunk_width],
            torch.cat([Y[offsets[i] : offsets[i + 1]] for i in idx]),
            in_lens[idx],
            tgt_lens[idx],
        ))
        start += len(idx)
    return chunks


def _worker_init(_worker_id: int) -> None:
    torch.set_num_threads(1)  # worker hanya merender


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=LengthBucketSampler(dataset.lines, batch_size, shuffle=shuffle, seed=seed),
        collate_fn=collate,
        num_workers=workers,
        persistent_workers=workers > 0,
        worker_init_fn=_worker_init,
    )


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, packed: bool, max_columns: int = 0) -> tuple[float, list[tuple[str, str]]]:
    """CER pada urutan logis. Referensi = target yang di-decode balik (round-trip terjamin Fase 1)."""
    model.eval()
    refs, hyps = [], []
    for batch in loader:
        if batch is None:
            continue
        for X, Y, in_lens, tgt_lens in split_batch(*batch, max_columns):
            logits = model(X.to(device), in_lens if packed else None)
            hyps += greedy_decode(logits, in_lens, tokenizer)
            refs += [tokenizer.to_logical(tokenizer.decode(y.tolist())) for y in torch.split(Y, tgt_lens.tolist())]
    model.train()
    return cer(refs, hyps), list(zip(refs, hyps))


def checkpoint_payload(model, opt, sched, step, best_cer, args, channels, tokenizer) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": step,
        "best_cer": best_cer,
        "args": vars(args),
        "model_config": {"n_classes": tokenizer.n_classes, "channels": list(channels), "hidden": args.hidden},
        # Checkpoint membawa tokenizer supaya inference berjalan tanpa data/.
        "tokenizer": {
            "charset": [f"U+{ord(c):04X}" for c in tokenizer.charset],
            "reorder": {k: list(v) for k, v in tokenizer.reorder.items()},
        },
    }


def save_atomic(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    # Windows: os.replace gagal kalau file tujuan sedang dibuka proses lain (mis. notebook
    # yang membaca checkpoint). Coba ulang sebentar, jangan hentikan training berjam-jam.
    for _ in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(1)
    os.replace(tmp, path)


def training_fonts(args) -> list[Path]:
    """Font perender data latih: font inti, ditambah isi --extra-fonts. Namanya dicatat di event start log, supaya
    kartu data dan kartu metode tidak perlu menebak font sebuah run dari folder font yang bisa berubah sesudahnya."""
    fonts = list(TRAIN_FONTS)
    if args.extra_fonts:
        extra = sorted(p for p in Path(args.extra_fonts).glob("*") if p.suffix.lower() in {".ttf", ".otf"})
        if not extra:
            raise ValueError(f"Tidak ada font .ttf/.otf di {args.extra_fonts}")
        fonts += extra
    return fonts


def build_datasets(args, tokenizer: Tokenizer):
    train_lines, val_lines = read_split("train"), read_split("val")
    if args.overfit:
        train_lines = train_lines[: args.overfit]
        val_lines = train_lines
    else:
        random.Random(args.seed).shuffle(train_lines)
        train_lines = train_lines[: args.train_lines]
        val_lines = val_lines[: args.val_lines]

    train_fonts = training_fonts(args)
    if args.extra_fonts:
        print(f"font training: {len(train_fonts)} ({len(train_fonts) - len(TRAIN_FONTS)} tambahan dari {args.extra_fonts})")

    parts = []
    if train_lines:
        rare_text = None
        if args.rare_insert_prob or args.rare_opener_prob:
            rare_text = RareText.from_lines(train_lines, tokenizer.charset, args.rare_insert_prob, args.rare_opener_prob,
                                            max_similarity=args.rare_max_similarity, attach=args.rare_attach)
            print(f"aksara langka: {len(rare_text.rare)} codepoint, sisip p={args.rare_insert_prob}, "
                  f"pembuka p={args.rare_opener_prob}"
                  + (f", lewati glyph mirip (IoU >= {args.rare_max_similarity})" if args.rare_max_similarity else "")
                  + (", tanpa spasi tambahan" if args.rare_attach else ""))
        if args.track_prob:
            print(f"jarak antar suku kata: p={args.track_prob}, 0-{args.track_max} em")
        synthetic = SyntheticLines(
            train_lines, tokenizer, train_fonts, augment=build_augment(args.augment),
            deterministic=bool(args.overfit), seed=args.seed, drop_space_prob=args.drop_space_prob,
            rare_text=rare_text, track_prob=args.track_prob, track_max=args.track_max,
        )
        parts.append((synthetic, 1))
    if args.real_train:
        # Data nyata diulang supaya tidak tenggelam di antara baris sintetis.
        real = RealLines(
            read_labels(args.real_train), tokenizer, augment=build_augment(args.real_augment),
            seed=args.seed, pad_ratio=args.real_pad_ratio,
        )
        parts.append((real, args.real_repeat))
    if not parts:
        raise ValueError("Tidak ada data training: --train-lines 0 dan tanpa --real-train")
    train_ds = parts[0][0] if len(parts) == 1 and parts[0][1] == 1 else ConcatLines(parts)

    if args.real_val:
        val_ds = RealLines(read_labels(args.real_val), tokenizer, deterministic=True, pad_ratio=args.real_pad_ratio)
    else:
        # Overfit (4a) menguji hafalan sampel yang SAMA: citra val harus identik
        # dengan citra training, bukan teks yang sama dengan font/ukuran lain.
        val_seed = args.seed if args.overfit else args.seed + 1
        val_ds = SyntheticLines(val_lines, tokenizer, TRAIN_FONTS, deterministic=True, seed=val_seed)
    return train_ds, val_ds


def main(argv=None) -> None:
    args = parse_args(argv)
    prevent_sleep()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    run_dir = CHECKPOINT_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    tokenizer = Tokenizer.load(TOKENIZER_PATH)
    train_ds, val_ds = build_datasets(args, tokenizer)
    train_loader = make_loader(train_ds, args.batch_size, True, args.workers, args.seed)
    val_loader = make_loader(val_ds, args.batch_size, False, args.workers, args.seed)

    channels = tuple(int(c) for c in args.channels.split(","))
    model = CRNN(tokenizer.n_classes, channels=channels, hidden=args.hidden).to(device)
    assert_downsample(model, device)

    steps_per_epoch = len(train_loader)
    total_steps = args.steps or max(1, round(args.epochs * steps_per_epoch))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    step, best_cer = 0, float("inf")
    if args.resume and (run_dir / "last.pt").exists():
        ckpt = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        step, best_cer = ckpt["step"], ckpt["best_cer"]
        print(f"lanjut dari langkah {step}, best CER {best_cer:.4f}")
    elif args.init:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=False)["model"])
        print(f"bobot awal dari {args.init}")

    def log(record: dict) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"device={device} params={n_params / 1e6:.2f}M train={len(train_ds):,} val={len(val_ds):,} "
        f"steps/epoch={steps_per_epoch} total_steps={total_steps} threads={torch.get_num_threads()}"
    )
    log({"event": "start", "step": step, "args": vars(args), "params": n_params, "device": str(device),
         "fonts": [Path(font).name for font in training_fonts(args)]})

    eval_every = args.eval_every or steps_per_epoch
    started = time.time()
    window_start, window_samples = time.time(), 0
    split_batches = 0
    model.train()
    done = step >= total_steps
    while not done:
        for batch in train_loader:
            if batch is None:
                continue
            X, Y, in_lens, tgt_lens = batch
            opt.zero_grad(set_to_none=True)
            chunks = split_batch(X, Y, in_lens, tgt_lens, args.max_batch_columns)
            split_batches += len(chunks) > 1
            loss = 0.0
            for cx, cy, c_in, c_tgt in chunks:
                logits = model(cx.to(device, non_blocking=True), c_in if args.packed else None)
                # CTCLoss 'mean' merata-ratakan per sampel; bobot n/B membuat jumlah potongan = batch utuh.
                part = ctc_loss(ctc, logits, cy, c_in, c_tgt, device) * (cx.shape[0] / X.shape[0])
                part.backward()
                loss += float(part.detach())
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), args.clip))
            opt.step()
            sched.step()
            step += 1
            window_samples += X.shape[0]

            if step % args.log_every == 0:
                rate = window_samples / (time.time() - window_start)
                window_start, window_samples = time.time(), 0
                record = {"step": step, "loss": float(loss), "grad_norm": grad_norm,
                          "lr": sched.get_last_lr()[0], "samples_per_s": round(rate, 2),
                          "split_batches": split_batches}
                log(record)
                print(f"step {step}/{total_steps} loss {float(loss):.4f} grad {grad_norm:.2f} "
                      f"lr {record['lr']:.2e} {rate:.1f} sampel/s")

            out_of_time = args.time_budget_hours > 0 and time.time() - started > args.time_budget_hours * 3600
            finished = step >= total_steps or 0 < args.stop_step <= step
            if step % eval_every == 0 or finished or out_of_time:
                val_cer, pairs = evaluate(model, val_loader, tokenizer, device, args.packed, args.max_batch_columns)
                improved = val_cer < best_cer
                best_cer = min(best_cer, val_cer)
                payload = checkpoint_payload(model, opt, sched, step, best_cer, args, channels, tokenizer)
                save_atomic(payload, run_dir / "last.pt")
                if improved:
                    save_atomic(payload, run_dir / "best.pt")
                errors = [(r, h) for r, h in pairs if r != h][:5]
                log({"step": step, "epoch": round(step / steps_per_epoch, 3), "val_cer": val_cer,
                     "val_lines": len(pairs), "best_cer": best_cer, "hours": (time.time() - started) / 3600,
                     "errors": errors})
                print(f"== step {step} epoch {step / steps_per_epoch:.2f}: val CER {val_cer:.4%} "
                      f"(best {best_cer:.4%}, {len(pairs)} baris)")
                target = 0.01 if args.overfit else args.target_cer
                if target and val_cer < target:
                    print(f"target tercapai: val CER < {target:.2%}")
                    done = True
            if finished or out_of_time:
                if out_of_time and not finished:
                    print("anggaran waktu habis; lanjutkan dengan --resume")
                done = True
            if done:
                break
    log({"event": "end", "step": step, "best_cer": best_cer, "hours": (time.time() - started) / 3600})


if __name__ == "__main__":
    main()
