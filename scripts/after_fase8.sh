#!/usr/bin/env bash
# Sesudah rantai fase 8 (scripts/run_fase8.sh): titik akhir yang ditetapkan sebelum ada angka (CLAUDE.md, "Aturan
# analisis fase 8"), lalu kartu dan web, lalu beam + LM untuk fase8_std.
#   1. ekspor hasil (kedua run fase 8 ikut), pembanding G3 dengan SK per halaman, dan bentuk ejaan baku di baris nyata:
#      fase8_std - fase8_ctrl (titik akhir utama), dan keduanya lawan fase7_track;
#   2. kartu data dan kartu metode, lalu impor ke web (angka resmi TIDAK diganti: itu keputusan user);
#   3. evaluasi sintetis ejaan baku (javatext, ejaan korpus dan ejaan baku, 300 baris val);
#   4. model bahasa berejaan baku (separuh baris), lalu beam + LM untuk fase8_std dengan model bahasa lama dan yang
#      baru, keduanya disetel di dev yang dirender seperti training fase8_std (ejaan baku p 0,5).
# Menunggu baris akhir rantai yang muncul SESUDAH skrip ini mulai. Proses mandiri dari root repo:
#   Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList scripts/after_fase8.sh -WindowStyle Hidden
# Log: out/after_fase8.log.
set -u
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
LOG="$PWD/out/after_fase8.log"  # mutlak: impor web berjalan dari folder web
CHAIN=out/fase8_chain.log
PY=.venv/Scripts/python.exe
LM_OLD=data/charlm/o5_n300000_ds0.5.pkl
LM_STD=data/charlm/o5_n300000_ds0.5_std0.5.pkl

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
echo "menunggu rantai fase8 (baris log ke-$start) $(date '+%Y-%m-%d %H:%M:%S')" > "$LOG"
until tail -n +"$((start + 1))" "$CHAIN" 2>/dev/null | grep -qE "^== selesai|^== .*(berhenti|gagal)"; do sleep 60; done
say "rantai: $(tail -n +"$((start + 1))" "$CHAIN" | grep -E '^== selesai|^== .*(berhenti|gagal)' | tail -1)"
for r in fase8_std fase8_ctrl; do
    if ! [ "out/eval/${r}_G3_full.json" -nt "out/checkpoints/$r/last.pt" ] || ! [ -f "out/checkpoints/$r/last_snapshot.pt" ]; then
        say "$r belum selesai dievaluasi; berhenti"
        exit 1
    fi
done

run "ekspor hasil" $PY scripts/export_results.py || { say "ekspor gagal; berhenti"; exit 1; }
for pair in "crnn_fase8_std crnn_fase8_ctrl" "crnn_fase8_std crnn_fase7_track" "crnn_fase8_ctrl crnn_fase7_track"; do
    run "compare_runs $pair" $PY scripts/compare_runs.py $pair
    run "eval_spelling real $pair" $PY scripts/eval_spelling.py real $pair
done
run "kartu data" $PY scripts/export_datasets.py
run "kartu metode" $PY scripts/export_methods.py
(cd web && run "impor web" php artisan aksara:import --no-ansi)

run "eval_spelling synthetic" $PY scripts/eval_spelling.py synthetic fase8_std fase8_ctrl fase7_track --lines 300
[ -f "$LM_STD" ] || run "model bahasa ejaan baku" $PY -m src.charlm --order 5 --lines 300000 --drop-space-prob 0.5 --standard-prob 0.5
for spec in "fase8_std_o5|$LM_OLD" "fase8_std_o5std|$LM_STD"; do
    name=${spec%%|*}
    lm=${spec#*|}
    run "beam_eval $name" $PY scripts/beam_eval.py out/checkpoints/fase8_std/last_snapshot.pt --lm "$lm" --name "$name" \
        --dev-standard-prob 0.5
done
say "selesai"
