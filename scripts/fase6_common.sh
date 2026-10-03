# Bagian bersama rantai fase6; disertakan dengan `. scripts/fase6_common.sh` dari root repo, bukan dijalankan.
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
LOG=out/fase6_rare_chain.log
INIT=out/checkpoints/fase5_fonts/last_snapshot.pt
JT=C:/Windows/Fonts/javatext.ttf
REAL=data/real/nusaaksara/labels.tsv
TOTAL=1500
# Sama dengan scripts/run_fase6_rare.sh kecuali --time-budget-hours 0. Anggaran itu memakai jam dinding, jadi ikut
# menghitung waktu laptop tidur dan menghentikan run di langkah sembarang (2026-10-02: tidur 17:40-20:58, fase6_rare
# terhenti jauh sebelum langkah 1500). Tanpa anggaran, tidur hanya menjeda training.
COMMON=(--init "$INIT" --augment fase5 --drop-space-prob 0.5 --extra-fonts fonts/extra --train-lines 100000
        --val-lines 500 --batch-size 32 --lr 3e-4 --steps "$TOTAL" --eval-every 500 --log-every 100
        --time-budget-hours 0 --device xpu --workers 4 --resume)  # --resume: lanjut dari last.pt bila ada
RARE=(--rare-insert-prob 0.3 --rare-opener-prob 0.15)

now() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "$*" >> "$LOG"; }
is_number() { case "$1" in '' | *[!0-9]*) return 1 ;; esac; }

training_count() {  # jumlah proses src.train yang hidup; keluaran kosong (PowerShell gagal) dianggap "masih ada"
    powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'src\.train' }).Count" 2>/dev/null | tr -d '\r\n '
}

wait_no_training() {  # dua training sekaligus berebut GPU
    while [ "$(training_count)" != "0" ]; do sleep 15; done
}

saved_step() {  # $1 = run -> langkah di last.pt; 0 bila belum ada; kosong bila gagal dibaca
    local ck="out/checkpoints/$1/last.pt"
    [ -f "$ck" ] || { echo 0; return; }
    .venv/Scripts/python.exe -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['step'])" "$ck" 2>/dev/null | tr -d '\r\n '
}

run_train() {  # $1 = run; sisanya = argumen tambahan untuk src.train
    local run=$1
    shift
    .venv-xpu/Scripts/python.exe -m src.train --run "$run" "${COMMON[@]}" "$@" >> "out/$run.stdout" 2>&1
    note "   train exit=$? $(date '+%H:%M:%S')"
    grep -E "== step|Error|Traceback" "out/$run.stdout" | tail -3 | sed 's/^/   /' >> "$LOG"
}

train_to_end() {  # $1 = run; sisanya = argumen tambahan. Ulangi --resume sampai last.pt mencapai $TOTAL.
    local run=$1 tries=0 step
    shift
    while :; do
        step=$(saved_step "$run")
        is_number "$step" || { note "   $run: last.pt tidak terbaca"; return 1; }
        [ "$step" -ge "$TOTAL" ] && return 0
        tries=$((tries + 1))
        [ "$tries" -gt 4 ] && { note "   $run: berhenti setelah 4 percobaan, masih di langkah $step"; return 1; }
        note "== lanjut $run dari langkah $step $(now)"
        run_train "$run" "$@"
    done
}

evaluate() {  # $1 = run. Hasil yang sudah lebih baru dari last.pt dilewati (skrip dijalankan ulang).
    local ck="out/checkpoints/$1/last_snapshot.pt" spec name args
    cp "out/checkpoints/$1/last.pt" "$ck"
    note "== evaluasi $1 $(date '+%H:%M:%S')"
    for spec in "$1_G3_full|--real $REAL --lines 0" "$1_G3_full_pad006|--real $REAL --lines 0 --pad-ratio 0.06" \
                "q100_$1_G1|--split test --lines 100 --fonts $JT" "q100_$1_G2|--split test --lines 100 --fonts $JT --augment heavy"; do
        name=${spec%%|*}
        args=${spec#*|}
        [ "out/eval/$name.json" -nt "out/checkpoints/$1/last.pt" ] && { note "  $name: sudah ada"; continue; }
        .venv/Scripts/python.exe -m src.evaluate "$ck" $args --workers 0 --name "$name" 2>&1 \
            | grep -E "^G[123]:|info, bukan gerbang|Error|Traceback" | sed "s/; detail.*//; s/^/  $name: /" >> "$LOG"
    done
}
