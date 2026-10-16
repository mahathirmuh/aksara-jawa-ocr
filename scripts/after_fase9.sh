#!/usr/bin/env bash
# Sesudah rantai fase 9 (scripts/run_fase9.sh): titik akhir yang ditetapkan sebelum ada angka (CLAUDE.md, "Aturan
# analisis fase 9"), lalu kartu dan web. Angka resmi TIDAK diganti (keputusan user).
#   1. ekspor hasil (kedua run fase 9 ikut), pembanding G3 dengan SK per halaman, CER per kelompok kemiringan baris
#      (titik akhir utama: baris miring, fase9_slant - fase9_ctrl), dan bentuk ejaan baku (efek samping);
#   2. kartu data dan kartu metode, lalu impor ke web;
#   3. dosis-respons kemiringan sintetis (javatext, 300 baris val, miring 0 sampai 0,7).
# Menunggu baris akhir rantai yang muncul SESUDAH skrip ini mulai. Proses mandiri dari root repo:
#   Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList scripts/after_fase9.sh -WindowStyle Hidden
# Log: out/after_fase9.log.
set -u
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
LOG="$PWD/out/after_fase9.log"  # mutlak: impor web berjalan dari folder web
CHAIN=out/fase9_chain.log
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
echo "menunggu rantai fase9 (baris log ke-$start) $(date '+%Y-%m-%d %H:%M:%S')" > "$LOG"
until tail -n +"$((start + 1))" "$CHAIN" 2>/dev/null | grep -qE "^== selesai|^== .*(berhenti|gagal)"; do sleep 60; done
say "rantai: $(tail -n +"$((start + 1))" "$CHAIN" | grep -E '^== selesai|^== .*(berhenti|gagal)' | tail -1)"
for r in fase9_slant fase9_ctrl; do
    if ! [ "out/eval/${r}_G3_full.json" -nt "out/checkpoints/$r/last.pt" ] || ! [ -f "out/checkpoints/$r/last_snapshot.pt" ]; then
        say "$r belum selesai dievaluasi; berhenti"
        exit 1
    fi
done

run "ekspor hasil" $PY scripts/export_results.py || { say "ekspor gagal; berhenti"; exit 1; }
for pair in "crnn_fase9_slant crnn_fase9_ctrl" "crnn_fase9_slant crnn_fase8_std" "crnn_fase9_ctrl crnn_fase8_std"; do
    run "compare_runs $pair" $PY scripts/compare_runs.py $pair
    run "eval_slant real $pair" $PY scripts/eval_slant.py real $pair
done
run "eval_spelling real crnn_fase9_slant crnn_fase9_ctrl" $PY scripts/eval_spelling.py real crnn_fase9_slant crnn_fase9_ctrl
run "kartu data" $PY scripts/export_datasets.py
run "kartu metode" $PY scripts/export_methods.py
(cd web && run "impor web" php artisan aksara:import --no-ansi)
run "eval_slant synthetic" $PY scripts/eval_slant.py synthetic fase9_slant fase9_ctrl fase8_std --lines 300
say "selesai"
