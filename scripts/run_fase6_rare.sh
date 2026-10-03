#!/usr/bin/env bash
# Perbandingan terkendali: fase5_fonts + aksara langka vs fase5_fonts + langkah yang sama tanpa aksara langka.
# Log ringkas: out/fase6_rare_chain.log. Dijalankan dari root repo.
set -u
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
LOG=out/fase6_rare_chain.log
INIT=out/checkpoints/fase5_fonts/last_snapshot.pt
JT=C:/Windows/Fonts/javatext.ttf
REAL=data/real/nusaaksara/labels.tsv
COMMON=(--init "$INIT" --augment fase5 --drop-space-prob 0.5 --extra-fonts fonts/extra --train-lines 100000
        --val-lines 500 --batch-size 32 --lr 3e-4 --steps 1500 --eval-every 500 --log-every 100
        --time-budget-hours 4.5 --device xpu --workers 4 --resume)  # --resume: lanjut dari last.pt bila ada

evaluate() {  # $1 = nama run
    local ck="out/checkpoints/$1/last_snapshot.pt"
    cp "out/checkpoints/$1/last.pt" "$ck"
    echo "== evaluasi $1 $(date '+%H:%M:%S')" >> "$LOG"
    for spec in "$1_G3_full|--real $REAL --lines 0" "$1_G3_full_pad006|--real $REAL --lines 0 --pad-ratio 0.06" \
                "q100_$1_G1|--split test --lines 100 --fonts $JT" "q100_$1_G2|--split test --lines 100 --fonts $JT --augment heavy"; do
        name=${spec%%|*}; args=${spec#*|}
        .venv/Scripts/python.exe -m src.evaluate "$ck" $args --workers 0 --name "$name" 2>&1 \
            | grep -E "^G[123]:|info, bukan gerbang|Error|Traceback" | sed "s/; detail.*//; s/^/  $name: /" >> "$LOG"
    done
}

for run in fase6_rare fase6_ctrl; do
    extra=()
    [ "$run" = fase6_rare ] && extra=(--rare-insert-prob 0.3 --rare-opener-prob 0.15)
    echo "== mulai $run $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
    .venv-xpu/Scripts/python.exe -m src.train --run "$run" "${COMMON[@]}" "${extra[@]}" > "out/$run.stdout" 2>&1
    echo "   train exit=$? $(date '+%H:%M:%S')" >> "$LOG"
    grep -E "aksara langka|== step|anggaran waktu" "out/$run.stdout" | tail -5 | sed 's/^/   /' >> "$LOG"
    [ -f "out/checkpoints/$run/last.pt" ] && evaluate "$run"
done
echo "== selesai $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
