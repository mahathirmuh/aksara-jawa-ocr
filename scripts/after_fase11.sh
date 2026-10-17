#!/usr/bin/env bash
# Sesudah rantai fase 11 (scripts/run_fase11.sh): titik akhir yang ditetapkan sebelum ada angka (CLAUDE.md, "Aturan
# analisis fase 11"), lalu kartu dan web. Angka resmi TIDAK diganti (keputusan user).
#   1. ekspor hasil (kedua run fase 11 ikut), pembanding G3 untuk semua baris DAN hanya baris cetak (titik akhir utama:
#      308 baris cetak, fase11_neighbor - fase11_ctrl), CER per jenis tulisan, tanda sisipan dan baris berpecahan
#      (eval_neighbor real), bentuk ejaan baku;
#   2. kartu data dan kartu metode, lalu impor ke web;
#   3. evaluasi sintetis pecahan baris tetangga (javatext, 300 baris val, kedua run fase 11 dan run dasarnya).
# Dengan dasar fase9_ctrl, fase11_ctrl mengulang resep fase10_ctrl persis (run_fase11.sh): pasangan fase11_ctrl lawan
# fase10_ctrl ikut dibandingkan sebagai ukuran derau antar-run.
# Argumen pertama = run dasar (sama dengan run_fase11.sh). Menunggu baris akhir rantai yang muncul SESUDAH skrip ini
# mulai. Proses mandiri dari root repo:
#   Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList "scripts/after_fase11.sh fase9_ctrl" -WindowStyle Hidden
# Log: out/after_fase11.log.
set -u
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
BASE=${1:-}
case "$BASE" in fase9_ctrl | fase10_ctrl | fase10_rare) ;; *) echo "pakai: scripts/after_fase11.sh RUN_DASAR" >&2; exit 2 ;; esac
PAIRS=("crnn_fase11_neighbor crnn_fase11_ctrl" "crnn_fase11_neighbor crnn_$BASE" "crnn_fase11_ctrl crnn_$BASE")
[ "$BASE" = fase9_ctrl ] && PAIRS+=("crnn_fase11_ctrl crnn_fase10_ctrl")
LOG="$PWD/out/after_fase11.log"  # mutlak: impor web berjalan dari folder web
CHAIN=out/fase11_chain.log
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
echo "menunggu rantai fase11 (dasar $BASE, baris log ke-$start) $(date '+%Y-%m-%d %H:%M:%S')" > "$LOG"
until tail -n +"$((start + 1))" "$CHAIN" 2>/dev/null | grep -qE "^== selesai|^== .*(berhenti|gagal)"; do sleep 60; done
say "rantai: $(tail -n +"$((start + 1))" "$CHAIN" | grep -E '^== selesai|^== .*(berhenti|gagal)' | tail -1)"
for r in fase11_neighbor fase11_ctrl; do
    if ! [ "out/eval/${r}_G3_full.json" -nt "out/checkpoints/$r/last.pt" ] || ! [ -f "out/checkpoints/$r/last_snapshot.pt" ]; then
        say "$r belum selesai dievaluasi; berhenti"
        exit 1
    fi
done

run "ekspor hasil" $PY scripts/export_results.py || { say "ekspor gagal; berhenti"; exit 1; }
for pair in "${PAIRS[@]}"; do
    run "compare_runs $pair" $PY scripts/compare_runs.py $pair
    run "compare_runs $pair --kind cetak" $PY scripts/compare_runs.py $pair --kind cetak
    run "eval_slant real $pair" $PY scripts/eval_slant.py real $pair
    run "eval_neighbor real $pair" $PY scripts/eval_neighbor.py real $pair
done
run "eval_spelling real crnn_fase11_neighbor crnn_fase11_ctrl" $PY scripts/eval_spelling.py real crnn_fase11_neighbor crnn_fase11_ctrl
run "kartu data" $PY scripts/export_datasets.py
run "kartu metode" $PY scripts/export_methods.py
(cd web && run "impor web" php artisan aksara:import --no-ansi)
run "eval_neighbor synthetic" $PY scripts/eval_neighbor.py synthetic fase11_neighbor fase11_ctrl "$BASE" --lines 300
say "selesai"
