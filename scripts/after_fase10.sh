#!/usr/bin/env bash
# Sesudah rantai fase 10 (scripts/run_fase10.sh): titik akhir yang ditetapkan sebelum ada angka (CLAUDE.md, "Aturan
# analisis fase 10"), lalu kartu dan web. Angka resmi TIDAK diganti (keputusan user).
#   1. ekspor hasil (kedua run fase 10 ikut), pembanding G3 untuk semua baris DAN hanya baris cetak (titik akhir utama:
#      308 baris cetak, fase10_rare - fase10_ctrl), CER per kelompok kemiringan dan jenis tulisan, bentuk ejaan baku;
#   2. kartu data dan kartu metode, lalu impor ke web;
#   3. evaluasi sintetis aksara langka (javatext, 2.000 baris split test, kedua run fase 10 dan run dasarnya).
# Argumen pertama = run dasar (sama dengan run_fase10.sh). Menunggu baris akhir rantai yang muncul SESUDAH skrip ini
# mulai. Proses mandiri dari root repo:
#   Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList "scripts/after_fase10.sh fase9_slant" -WindowStyle Hidden
# Log: out/after_fase10.log.
set -u
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
BASE=${1:-}
case "$BASE" in fase9_slant | fase9_ctrl | fase8_std) ;; *) echo "pakai: scripts/after_fase10.sh RUN_DASAR" >&2; exit 2 ;; esac
LOG="$PWD/out/after_fase10.log"  # mutlak: impor web berjalan dari folder web
CHAIN=out/fase10_chain.log
PY=.venv/Scripts/python.exe

say() { echo "$* $(date '+%H:%M:%S')" >> "$LOG"; }

run() {  # $1 = nama langkah; sisanya = perintah. Keluaran ke log; mengembalikan exit perintah.
    local name=$1
    shift
    "$@" 2>&1 | grep -v -E "Warning|warnings.warn" >> "$LOG"
    local code=${PIPESTATUS[0]}
    say "$name exit=$code"
    return "$code"
}

start=$(wc -l < "$CHAIN" 2>/dev/null || echo 0)
echo "menunggu rantai fase10 (dasar $BASE, baris log ke-$start) $(date '+%Y-%m-%d %H:%M:%S')" > "$LOG"
until tail -n +"$((start + 1))" "$CHAIN" 2>/dev/null | grep -qE "^== selesai|^== .*(berhenti|gagal)"; do sleep 60; done
say "rantai: $(tail -n +"$((start + 1))" "$CHAIN" | grep -E '^== selesai|^== .*(berhenti|gagal)' | tail -1)"
for r in fase10_rare fase10_ctrl; do
    if ! [ "out/eval/${r}_G3_full.json" -nt "out/checkpoints/$r/last.pt" ] || ! [ -f "out/checkpoints/$r/last_snapshot.pt" ]; then
        say "$r belum selesai dievaluasi; berhenti"
        exit 1
    fi
done

run "ekspor hasil" $PY scripts/export_results.py || { say "ekspor gagal; berhenti"; exit 1; }
for pair in "crnn_fase10_rare crnn_fase10_ctrl" "crnn_fase10_rare crnn_$BASE" "crnn_fase10_ctrl crnn_$BASE"; do
    run "compare_runs $pair" $PY scripts/compare_runs.py $pair
    run "compare_runs $pair --kind cetak" $PY scripts/compare_runs.py $pair --kind cetak
    run "eval_slant real $pair" $PY scripts/eval_slant.py real $pair
done
run "eval_spelling real crnn_fase10_rare crnn_fase10_ctrl" $PY scripts/eval_spelling.py real crnn_fase10_rare crnn_fase10_ctrl
run "kartu data" $PY scripts/export_datasets.py
run "kartu metode" $PY scripts/export_methods.py
(cd web && run "impor web" php artisan aksara:import --no-ansi)
run "eval_rare synthetic" $PY scripts/eval_rare.py --checkpoints fase10_rare fase10_ctrl "$BASE" --workers 2 \
    --out out/compare/fase10
say "selesai"
