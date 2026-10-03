#!/usr/bin/env bash
# Fase 7: dua run 1.500 langkah dari fase6_ctrl.
#   fase7_track       jarak antar suku kata acak (--track-prob 0.5 --track-max 0.3): sasaran spasi berlebih di G3
#   fase7_track_rare  sama + sisipan aksara langka yang diperbaiki (glyph mirip dilewati, tanpa spasi tambahan)
# Run kedua berhenti dan lanjut di langkah yang sama dengan run pertama (lihat scripts/run_fase6_ctrl.sh: urutan
# batch diulang dari awal di tiap proses), jadi keduanya hanya berbeda di aksara langka. Boleh dijalankan ulang.
# Dari root repo, sebagai proses mandiri:
#   Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList scripts/run_fase7.sh -WindowStyle Hidden
# Log ringkas: out/fase7_chain.log.
set -u
. scripts/fase6_common.sh
LOG=out/fase7_chain.log
INIT=out/checkpoints/fase6_ctrl/last_snapshot.pt
COMMON=(--init "$INIT" --augment fase5 --drop-space-prob 0.5 --extra-fonts fonts/extra --train-lines 100000
        --val-lines 500 --batch-size 32 --lr 3e-4 --steps "$TOTAL" --eval-every 500 --log-every 100
        --time-budget-hours 0 --device xpu --workers 4 --resume)
TRACK=(--track-prob 0.5 --track-max 0.3)
RARE_FIXED=(--rare-insert-prob 0.3 --rare-opener-prob 0.15 --rare-max-similarity 0.9 --rare-attach)

segment() {  # $1 = run, $2 = langkah awal, $3 = langkah akhir, sisanya = argumen tambahan untuk src.train
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

[ -f "$INIT" ] || { note "== $INIT tidak ada; rantai fase7 berhenti $(now)"; exit 1; }
note "== rantai fase7 $(now)"
wait_no_training

train_to_end fase7_track "${TRACK[@]}" || { note "== fase7_track tidak selesai; rantai berhenti $(now)"; exit 1; }
evaluate fase7_track

# Langkah tempat fase7_track dimulai ulang dari checkpoint = event "start" dengan langkah > 0 (kosong bila tidak ada).
bounds=$(.venv/Scripts/python.exe -c "
import json, sys
steps = {r['step'] for r in map(json.loads, open(sys.argv[1], encoding='utf-8')) if r.get('event') == 'start'}
print(' '.join(str(s) for s in sorted(steps) if 0 < s < int(sys.argv[2])))
" out/checkpoints/fase7_track/log.jsonl "$TOTAL") || { note "== log fase7_track tidak terbaca; rantai berhenti $(now)"; exit 1; }
bounds=${bounds//$'\r'/}
note "== fase7_track_rare: titik lanjut mengikuti fase7_track: ${bounds:-tidak ada} $(now)"
prev=0
for to in $bounds "$TOTAL"; do
    segment fase7_track_rare "$prev" "$to" "${TRACK[@]}" "${RARE_FIXED[@]}" \
        || { note "== fase7_track_rare: segmen $prev->$to gagal; rantai berhenti $(now)"; exit 1; }
    prev=$to
done
evaluate fase7_track_rare
note "== selesai $(now)"
