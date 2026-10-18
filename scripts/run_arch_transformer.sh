#!/usr/bin/env bash
# Pembanding arsitektur: CNN + Transformer + CTC (src.model.TransformerCTC) dilatih dengan resep persis run `base`
# (CRNN 4b): dari nol, 50.000 baris, 2 font inti, tanpa augmentasi, lr 1e-3, OneCycle untuk 15.000 langkah, val tiap
# 500 langkah. `base` berhenti di langkah 1.500 (target val CER 0,5% tercapai) dan dilanjutkan sekali di langkah 1.000,
# jadi lengan ini berhenti di 1.000, dilanjutkan, dan berhenti di 1.500 (urutan batch sama; render tetap acak per run).
# Bedanya hanya pemodel urutan (BiLSTM -> Transformer) dan worker render 4 (base: 2; render memang acak per worker).
# Log: out/arch_chain.log. Evaluasi sesudahnya: scripts/eval_arch.sh.
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
LOG=out/arch_chain.log
PY=.venv-xpu/Scripts/python.exe
RUN=arch_transformer
say() { echo "$* $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"; }
ARGS=(--run "$RUN" --arch transformer --train-lines 50000 --val-lines 1000 --steps 15000 --batch-size 32 --lr 1e-3
      --eval-every 500 --log-every 100 --device xpu --workers 4 --seed 0)
say "mulai $RUN"
if ! [ -f "out/checkpoints/$RUN/last.pt" ]; then
  "$PY" -m src.train "${ARGS[@]}" --stop-step 1000 >> "$LOG" 2>&1 || { say "segmen 1 gagal"; exit 1; }
fi
"$PY" -m src.train "${ARGS[@]}" --resume --stop-step 1500 >> "$LOG" 2>&1 || { say "segmen 2 gagal"; exit 1; }
say "training selesai"
