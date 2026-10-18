#!/usr/bin/env bash
# Pembanding arsitektur dengan resep PENUH: CNN + Transformer + CTC dilatih ulang lewat rantai yang sama dengan model resmi
# CRNN (base -> fase5_quick -> fase5_fonts -> fase6_ctrl -> fase7_track -> fase8_std -> fase9_ctrl), dimulai dari
# arch_transformer (= tahap base). Argumen tiap tahap disalin dari event `start` log run CRNN-nya: langkah terencana 1.500,
# titik berhenti yang sama (fase5_quick 500 karena run CRNN-nya crash OOM sesudah checkpoint 500; fase5_fonts 1.298 karena
# batas waktu 3 jam), dan titik lanjut fase6_ctrl di 500 dan 884. Perbedaan yang diketahui: kode render sekarang (aturan
# spasi font sempit sejak 2026-09-14 berlaku juga untuk tahap fase5_fonts), dan render acak per run.
# Sesudah tiap tahap: bacaan greedy 745 baris nyata (scripts/method_matrix.py crnn, group arsitektur), supaya tiap tahap
# bisa dibandingkan dengan CRNN di tahap yang sama walau rantainya belum selesai. Sesudah tahap terakhir: G1/G2 2.000 baris
# javatext untuk Transformer dan CRNN resmi, lalu skor, ekspor, dan impor web.
# Log: out/archT_chain.log. Dijalankan ulang = melanjutkan (penanda DONE_<langkah> per segmen).
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
LOG=out/archT_chain.log
PY=.venv-xpu/Scripts/python.exe
CPU=.venv/Scripts/python.exe
say() { echo "$* $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"; }
# Worker render 2, bukan 4 seperti run CRNN: 2026-10-18 19:18 training macet (semua thread menunggu, 0 CPU) saat commit
# memori sistem 74,2/75,5 GB; 2 worker cukup (render fase5 ~99 ms/sampel/inti, training ~4 sampel/dtk) dan menghemat ~5 GB.
# Urutan batch sama (aturan seed sampler sama untuk semua worker > 0); render memang acak per worker.
COMMON=(--arch transformer --augment fase5 --drop-space-prob 0.5 --train-lines 100000 --val-lines 500 --eval-every 500
        --workers 2 --seed 0 --steps 1500 --device xpu --resume)

# seg RUN STOP ARGS... : satu proses training sampai langkah STOP (0 = sampai 1.500), dilewati bila penandanya ada. Proses
# yang gagal atau dihentikan penjaga (scripts/arch_watchdog.sh, bila macet) dicoba ulang dari checkpoint terakhir, 3 kali.
seg() {
  local run=$1 stop=$2; shift 2
  local mark="out/checkpoints/$run/DONE_${stop}"
  if [ -f "$mark" ]; then say "lewati $run sampai $stop (sudah)"; return 0; fi
  local stoparg=(); [ "$stop" != 0 ] && stoparg=(--stop-step "$stop")
  local attempt
  for attempt in 1 2 3 4; do
    say "mulai $run sampai ${stop/#0/1500} (percobaan $attempt)"
    if "$PY" -m src.train --run "$run" "${COMMON[@]}" "$@" "${stoparg[@]}" >> "$LOG" 2>&1; then
      mkdir -p "out/checkpoints/$run" && touch "$mark"
      say "selesai $run sampai ${stop/#0/1500}"
      return 0
    fi
    say "GAGAL-COBA $run percobaan $attempt; lanjut dari checkpoint terakhir"
    sleep 30
  done
  say "GAGAL $run"; exit 1
}

# bacaan 745 baris nyata untuk tahap yang selesai (CPU, beberapa menit; di latar supaya training tahap berikutnya tidak
# menunggu; tidak menghentikan rantai bila gagal). real RUN LABEL [CHECKPOINT]
real() {
  local run=$1 label=$2 ck=${3:-out/checkpoints/$1/last.pt}
  [ -f "out/methods/$run.jsonl" ] && return 0
  "$CPU" scripts/method_matrix.py crnn "$ck" --name "$run" --label "$label" --group arsitektur \
    --threads 2 >> "out/archT_real_$run.log" 2>&1 && say "bacaan nyata $run" || say "bacaan nyata $run GAGAL (rantai lanjut)"
}

say "=== rantai Transformer resep penuh"
seg archT_fase5_quick 500 --init out/checkpoints/arch_transformer/last.pt --lr 5e-4
seg archT_fase5_fonts 1298 --init out/checkpoints/archT_fase5_quick/last.pt --lr 3e-4 --extra-fonts fonts/extra
real archT_fase5_fonts "Transformer, tahap fase5_fonts" &
F6=(--init out/checkpoints/archT_fase5_fonts/last.pt --lr 3e-4 --extra-fonts fonts/extra)
seg archT_fase6_ctrl 500 "${F6[@]}"
seg archT_fase6_ctrl 884 "${F6[@]}"
seg archT_fase6_ctrl 0 "${F6[@]}"
real archT_fase6_ctrl "Transformer, tahap fase6_ctrl" &
TR=(--lr 3e-4 --extra-fonts fonts/extra --track-prob 0.5 --track-max 0.3)
seg archT_fase7_track 0 --init out/checkpoints/archT_fase6_ctrl/last.pt "${TR[@]}"
real archT_fase7_track "Transformer, tahap fase7_track" &
seg archT_fase8_std 0 --init out/checkpoints/archT_fase7_track/last.pt "${TR[@]}" --standard-prob 0.5
real archT_fase8_std "Transformer, tahap fase8_std" &
seg archT_fase9_ctrl 0 --init out/checkpoints/archT_fase8_std/last.pt "${TR[@]}" --standard-prob 0.5 --font-fixes
wait
real archT_fase9_ctrl "CNN + Transformer + CTC, resep penuh"

say "evaluasi sintetis"
for pair in "archT_fase9_ctrl:out/checkpoints/archT_fase9_ctrl/last.pt" "arch_crnn_full:out/checkpoints/fase9_ctrl/last_snapshot.pt"; do
  name=${pair%%:*}; ck=${pair#*:}
  [ -f "out/eval/${name}_G1_2k.json" ] || "$CPU" -m src.evaluate "$ck" --lines 2000 --fonts C:/Windows/Fonts/javatext.ttf --workers 2 --name "${name}_G1_2k" >> "$LOG" 2>&1
  [ -f "out/eval/${name}_G2_2k.json" ] || "$CPU" -m src.evaluate "$ck" --lines 2000 --fonts C:/Windows/Fonts/javatext.ttf --augment heavy --workers 2 --name "${name}_G2_2k" >> "$LOG" 2>&1
done
real arch_crnn_full "CNN + BiLSTM + CTC, resep penuh (model resmi)" out/checkpoints/fase9_ctrl/last_snapshot.pt
"$CPU" scripts/method_matrix.py score >> "$LOG" 2>&1 && "$CPU" scripts/method_matrix.py export >> "$LOG" 2>&1 \
  && (cd web && php artisan aksara:matrix --no-ansi >> "../$LOG" 2>&1)
say "=== rantai selesai"
