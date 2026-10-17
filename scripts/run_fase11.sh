#!/usr/bin/env bash
# Fase 11: pecahan baris tetangga di tepi atas dan bawah citra latih (src/neighbors.py, src.train --neighbor-prob),
# dinilai pada baris CETAK (CLAUDE.md, "Aturan analisis fase 11").
#   fase11_neighbor  dasar + 1.500 langkah, resep dasar + --neighbor-prob 0.5 (pita bawah baris lain di atas citra dan
#                    pita atas baris lain di bawahnya, tiap sisi kosong dengan peluang 0,3; label tetap baris utama).
#   fase11_ctrl      sama TANPA pecahan, berhenti dan lanjut di langkah yang sama dengan fase11_neighbor.
# fase11_neighbor dihentikan dan dilanjutkan di langkah 1.000 (seperti fase10_rare, yang terhenti di sana karena baterai),
# jadi dengan dasar fase9_ctrl, fase11_ctrl mengulang fase10_ctrl persis (bobot awal, argumen, seed, urutan batch, titik
# lanjut); bedanya hanya acak render dan perangkat: selisih keduanya = derau antar-run. Gangguan lain (charger tercabut)
# menambah titik lanjut di kedua lengan fase 11.
# Argumen pertama = run dasar; resep dasarnya mengikuti run itu. Boleh dijalankan ulang (lanjut dari last.pt). Dari root
# repo, sebagai proses mandiri:
#   Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList "scripts/run_fase11.sh fase9_ctrl" -WindowStyle Hidden
# Log ringkas: out/fase11_chain.log. Selama berjalan: charger terpasang, jangan Sleep/Shut down, jangan menjalankan
# evaluasi besar atau test suite penuh, dan jangan mengedit src/ (lihat CLAUDE.md, Diagnostik cepat).
set -u
. scripts/fase6_common.sh
BASE=${1:-}
case "$BASE" in
    fase9_ctrl | fase10_ctrl) BASE_ARGS=(--standard-prob 0.5 --font-fixes) ;;
    fase10_rare) BASE_ARGS=(--standard-prob 0.5 --font-fixes --rare-insert-prob 0.3 --rare-opener-prob 0.15
                            --rare-max-similarity 0.9 --rare-attach) ;;
    *) echo "pakai: scripts/run_fase11.sh fase9_ctrl|fase10_ctrl|fase10_rare" >&2; exit 2 ;;
esac
FIRST_STOP=1000
LOG=out/fase11_chain.log
INIT=out/checkpoints/$BASE/last_snapshot.pt
COMMON=(--init "$INIT" --augment fase5 --drop-space-prob 0.5 --extra-fonts fonts/extra --train-lines 100000
        --val-lines 500 --batch-size 32 --lr 3e-4 --steps "$TOTAL" --eval-every 500 --log-every 100
        --time-budget-hours 0 --device xpu --workers 4 --resume --track-prob 0.5 --track-max 0.3 "${BASE_ARGS[@]}")
NEIGHBOR=(--neighbor-prob 0.5)

segment() {  # sama dengan scripts/run_fase10.sh: $1 = run, $2 = langkah awal, $3 = langkah akhir, sisanya argumen
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

[ -f "$INIT" ] || { note "== $INIT tidak ada; rantai fase11 berhenti $(now)"; exit 1; }
note "== rantai fase11 (dasar $BASE) $(now)"
wait_no_training

leader_to() {  # $1 = langkah akhir; ulangi --resume sampai last.pt fase11_neighbor mencapainya (berhenti di sana bila < TOTAL)
    local to=$1 tries=0 step stop
    while :; do
        step=$(saved_step fase11_neighbor)
        is_number "$step" || { note "   fase11_neighbor: last.pt tidak terbaca"; return 1; }
        [ "$step" -ge "$to" ] && return 0
        tries=$((tries + 1))
        [ "$tries" -gt 4 ] && { note "   fase11_neighbor: berhenti setelah 4 percobaan, masih di langkah $step"; return 1; }
        note "== lanjut fase11_neighbor dari langkah $step sampai $to $(now)"
        stop=()
        [ "$to" -lt "$TOTAL" ] && stop=(--stop-step "$to")
        run_train fase11_neighbor "${NEIGHBOR[@]}" "${stop[@]}"
    done
}
{ leader_to "$FIRST_STOP" && leader_to "$TOTAL"; } || { note "== fase11_neighbor tidak selesai; rantai berhenti $(now)"; exit 1; }
# Bila rantai dijalankan ulang sesudah terputus di run kontrol, fase11_neighbor sudah dievaluasi: menyalin ulang
# last_snapshot.pt akan membuatnya lebih baru dari laporan-laporannya.
cmp -s out/checkpoints/fase11_neighbor/last.pt out/checkpoints/fase11_neighbor/last_snapshot.pt || evaluate fase11_neighbor

# Langkah tempat fase11_neighbor dimulai ulang dari checkpoint = event "start" dengan langkah > 0 (kosong bila tidak ada).
bounds=$(.venv/Scripts/python.exe -c "
import json, sys
steps = {r['step'] for r in map(json.loads, open(sys.argv[1], encoding='utf-8')) if r.get('event') == 'start'}
print(' '.join(str(s) for s in sorted(steps) if 0 < s < int(sys.argv[2])))
" out/checkpoints/fase11_neighbor/log.jsonl "$TOTAL") || { note "== log fase11_neighbor tidak terbaca; rantai berhenti $(now)"; exit 1; }
bounds=${bounds//$'\r'/}
note "== fase11_ctrl: titik lanjut mengikuti fase11_neighbor: ${bounds:-tidak ada} $(now)"
prev=0
for to in $bounds "$TOTAL"; do
    segment fase11_ctrl "$prev" "$to" || { note "== fase11_ctrl: segmen $prev->$to gagal; rantai berhenti $(now)"; exit 1; }
    prev=$to
done
evaluate fase11_ctrl
note "== selesai $(now)"
