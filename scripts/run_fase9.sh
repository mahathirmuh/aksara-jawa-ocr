#!/usr/bin/env bash
# Fase 9: dua run 1.500 langkah dari fase8_std (resep fase8_std: jarak antar suku kata acak + ejaan baku p 0,5).
#   fase9_slant  + render miring seperti huruf italic cetakan (--slant-prob 0.5 --slant-max 0.6). Analisis galat
#                fase8_std: 270 dari 745 baris nyata dicetak miring (CER 28,3% lawan 15,7% baris tegak, 45% suntingan),
#                dan render bersih yang dimiringkan 0,5 dibaca 15,4% (tegak 0,16%).
#   fase9_ctrl   sama TANPA render miring, berhenti dan lanjut di langkah yang sama dengan fase9_slant.
# Kedua run memakai --font-fixes: BasaJan (pasangan tidak menumpuk di .venv-xpu) dikeluarkan, dan NewKramawirya tidak
# merender baris berpangkon + lungsi (digambar seperti lingsa). Jadi selisih keduanya hanya render miring.
# Boleh dijalankan ulang (lanjut dari last.pt). Dari root repo, sebagai proses mandiri:
#   Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList scripts/run_fase9.sh -WindowStyle Hidden
# Log ringkas: out/fase9_chain.log. Selama berjalan: charger terpasang, jangan Sleep/Shut down (Hibernate hanya menjeda),
# jangan menjalankan evaluasi besar atau test suite penuh, dan jangan mengedit src/ (lihat CLAUDE.md, Diagnostik cepat).
set -u
. scripts/fase6_common.sh
LOG=out/fase9_chain.log
INIT=out/checkpoints/fase8_std/last_snapshot.pt
COMMON=(--init "$INIT" --augment fase5 --drop-space-prob 0.5 --extra-fonts fonts/extra --train-lines 100000
        --val-lines 500 --batch-size 32 --lr 3e-4 --steps "$TOTAL" --eval-every 500 --log-every 100
        --time-budget-hours 0 --device xpu --workers 4 --resume --track-prob 0.5 --track-max 0.3
        --standard-prob 0.5 --font-fixes)
SLANT=(--slant-prob 0.5 --slant-max 0.6)

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

[ -f "$INIT" ] || { note "== $INIT tidak ada; rantai fase9 berhenti $(now)"; exit 1; }
note "== rantai fase9 $(now)"
wait_no_training

train_to_end fase9_slant "${SLANT[@]}" || { note "== fase9_slant tidak selesai; rantai berhenti $(now)"; exit 1; }
# Bila rantai dijalankan ulang sesudah terputus di run kontrol, fase9_slant sudah dievaluasi: menyalin ulang
# last_snapshot.pt akan membuatnya lebih baru dari laporan-laporannya.
cmp -s out/checkpoints/fase9_slant/last.pt out/checkpoints/fase9_slant/last_snapshot.pt || evaluate fase9_slant

# Langkah tempat fase9_slant dimulai ulang dari checkpoint = event "start" dengan langkah > 0 (kosong bila tidak ada).
bounds=$(.venv/Scripts/python.exe -c "
import json, sys
steps = {r['step'] for r in map(json.loads, open(sys.argv[1], encoding='utf-8')) if r.get('event') == 'start'}
print(' '.join(str(s) for s in sorted(steps) if 0 < s < int(sys.argv[2])))
" out/checkpoints/fase9_slant/log.jsonl "$TOTAL") || { note "== log fase9_slant tidak terbaca; rantai berhenti $(now)"; exit 1; }
bounds=${bounds//$'\r'/}
note "== fase9_ctrl: titik lanjut mengikuti fase9_slant: ${bounds:-tidak ada} $(now)"
prev=0
for to in $bounds "$TOTAL"; do
    segment fase9_ctrl "$prev" "$to" || { note "== fase9_ctrl: segmen $prev->$to gagal; rantai berhenti $(now)"; exit 1; }
    prev=$to
done
evaluate fase9_ctrl
note "== selesai $(now)"
