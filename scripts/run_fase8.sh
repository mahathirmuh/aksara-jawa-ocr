#!/usr/bin/env bash
# Fase 8: dua run 1.500 langkah dari fase7_track (resep fase7_track: jarak antar suku kata acak).
#   fase8_std   + ejaan baku pada separuh baris (--standard-prob 0.5, src/spelling.py): nya di depan ca/ja, pa cerek,
#               nga lelet, keret, angka diapit pada pangkat. Korpus memakai konvensi alat transliterasi, cetakan ejaan baku.
#   fase8_ctrl  sama TANPA ejaan baku, berhenti dan lanjut di langkah yang sama dengan fase8_std: memisahkan efek ejaan
#               dari efek +1.500 langkah.
# Boleh dijalankan ulang (lanjut dari last.pt). Dari root repo, sebagai proses mandiri:
#   Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList scripts/run_fase8.sh -WindowStyle Hidden
# Log ringkas: out/fase8_chain.log. Selama berjalan: charger terpasang, jangan Hibernate/Sleep/Shut down, jangan
# menjalankan evaluasi besar atau test suite penuh, dan jangan mengedit src/ (lihat CLAUDE.md, Diagnostik cepat).
set -u
. scripts/fase6_common.sh
LOG=out/fase8_chain.log
INIT=out/checkpoints/fase7_track/last_snapshot.pt
COMMON=(--init "$INIT" --augment fase5 --drop-space-prob 0.5 --extra-fonts fonts/extra --train-lines 100000
        --val-lines 500 --batch-size 32 --lr 3e-4 --steps "$TOTAL" --eval-every 500 --log-every 100
        --time-budget-hours 0 --device xpu --workers 4 --resume --track-prob 0.5 --track-max 0.3)
STANDARD=(--standard-prob 0.5)

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

[ -f "$INIT" ] || { note "== $INIT tidak ada; rantai fase8 berhenti $(now)"; exit 1; }
note "== rantai fase8 $(now)"
wait_no_training

train_to_end fase8_std "${STANDARD[@]}" || { note "== fase8_std tidak selesai; rantai berhenti $(now)"; exit 1; }
# Rantai yang dijalankan ulang sesudah terputus di run kontrol (2026-10-13 18:50, laptop mati): fase8_std sudah
# dievaluasi, dan menyalin ulang last_snapshot.pt akan membuatnya lebih baru dari laporan-laporannya.
cmp -s out/checkpoints/fase8_std/last.pt out/checkpoints/fase8_std/last_snapshot.pt || evaluate fase8_std

# Langkah tempat fase8_std dimulai ulang dari checkpoint = event "start" dengan langkah > 0 (kosong bila tidak ada).
bounds=$(.venv/Scripts/python.exe -c "
import json, sys
steps = {r['step'] for r in map(json.loads, open(sys.argv[1], encoding='utf-8')) if r.get('event') == 'start'}
print(' '.join(str(s) for s in sorted(steps) if 0 < s < int(sys.argv[2])))
" out/checkpoints/fase8_std/log.jsonl "$TOTAL") || { note "== log fase8_std tidak terbaca; rantai berhenti $(now)"; exit 1; }
bounds=${bounds//$'\r'/}
note "== fase8_ctrl: titik lanjut mengikuti fase8_std: ${bounds:-tidak ada} $(now)"
prev=0
for to in $bounds "$TOTAL"; do
    segment fase8_ctrl "$prev" "$to" || { note "== fase8_ctrl: segmen $prev->$to gagal; rantai berhenti $(now)"; exit 1; }
    prev=$to
done
evaluate fase8_ctrl
note "== selesai $(now)"
