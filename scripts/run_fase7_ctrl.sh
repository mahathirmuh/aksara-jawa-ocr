#!/usr/bin/env bash
# Run kontrol fase7_ctrl: sama dengan fase7_track TANPA jarak antar suku kata (1.500 langkah dari fase6_ctrl), dan
# berhenti/lanjut di langkah yang sama dengan fase7_track. Memisahkan efek tracking dari efek +1.500 langkah:
# fase5_fonts -> fase6_ctrl (tanpa perubahan data) saja sudah menurunkan G3 5,1 poin.
# Menunggu rantai scripts/run_fase7.sh selesai atau berhenti (baris baru di out/fase7_chain.log), lalu berjalan.
# Dari root repo, sebagai proses mandiri:
#   Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList scripts/run_fase7_ctrl.sh -WindowStyle Hidden
set -u
. scripts/fase6_common.sh
LOG=out/fase7_chain.log
INIT=out/checkpoints/fase6_ctrl/last_snapshot.pt
# HARUS sama dengan COMMON di scripts/run_fase7.sh.
COMMON=(--init "$INIT" --augment fase5 --drop-space-prob 0.5 --extra-fonts fonts/extra --train-lines 100000
        --val-lines 500 --batch-size 32 --lr 3e-4 --steps "$TOTAL" --eval-every 500 --log-every 100
        --time-budget-hours 0 --device xpu --workers 4 --resume)

segment() {  # sama dengan scripts/run_fase7.sh: $1 = run, $2 = langkah awal, $3 = langkah akhir, sisanya argumen
    local run=$1 from=$2 to=$3 tries=0 step stop dir
    shift 3
    dir=out/checkpoints/$run
    while :; do
        step=$(saved_step "$run")
        is_number "$step" || { note "   $run: last.pt tidak terbaca"; return 1; }
        [ "$step" -ge "$to" ] && return 0
        if [ "$step" -ne "$from" ]; then  # sisa percobaan yang terhenti di tengah segmen
            note "   $run: last.pt di langkah $step, kembali ke awal segmen ($from)"
            if [ "$from" -eq 0 ]; then rm -f "$dir/last.pt"; else cp "$dir/seg_$from.pt" "$dir/last.pt" || return 1; fi
        elif [ "$from" -gt 0 ]; then
            cp "$dir/last.pt" "$dir/seg_$from.pt" || return 1
        fi
        tries=$((tries + 1))
        [ "$tries" -gt 3 ] && { note "   $run: segmen $from->$to gagal 3 kali"; return 1; }
        note "== $run segmen $from->$to (percobaan $tries) $(now)"
        stop=()
        [ "$to" -lt "$TOTAL" ] && stop=(--stop-step "$to")
        run_train "$run" "$@" "${stop[@]}"
    done
}

start=$(wc -l < "$LOG" 2>/dev/null || echo 0)
until tail -n +"$((start + 1))" "$LOG" 2>/dev/null | grep -qE "^== selesai|^== .*berhenti"; do sleep 120; done
wait_no_training

step=$(saved_step fase7_track)
if ! is_number "$step" || [ "$step" -lt "$TOTAL" ]; then
    note "== fase7_ctrl dilewati: fase7_track belum mencapai langkah $TOTAL (${step:-?}) $(now)"
    exit 1
fi
bounds=$(.venv/Scripts/python.exe -c "
import json, sys
steps = {r['step'] for r in map(json.loads, open(sys.argv[1], encoding='utf-8')) if r.get('event') == 'start'}
print(' '.join(str(s) for s in sorted(steps) if 0 < s < int(sys.argv[2])))
" out/checkpoints/fase7_track/log.jsonl "$TOTAL") || { note "== fase7_ctrl: log fase7_track tidak terbaca $(now)"; exit 1; }
bounds=${bounds//$'\r'/}
note "== fase7_ctrl: titik lanjut mengikuti fase7_track: ${bounds:-tidak ada} $(now)"
prev=0
for to in $bounds "$TOTAL"; do
    segment fase7_ctrl "$prev" "$to" || { note "== fase7_ctrl: segmen $prev->$to gagal $(now)"; exit 1; }
    prev=$to
done
evaluate fase7_ctrl
note "== fase7_ctrl selesai $(now)"
