#!/usr/bin/env bash
# Fase 10: sisipan aksara langka versi perbaikan (resep fase7_track_rare) di atas dasar yang dipilih dari hasil fase 9,
# dinilai pada baris CETAK (CLAUDE.md, adendum fase 9: 437 dari 745 baris uji ternyata tulisan tangan, di luar scope).
#   fase10_rare  dasar + 1.500 langkah, resep dasar + --rare-insert-prob 0.3 --rare-opener-prob 0.15
#                --rare-max-similarity 0.9 --rare-attach. Analisis ulang 2026-10-16: di fase 7, sisipan ini menurunkan
#                CER 308 baris cetak 0,80 poin [-1,33; -0,28] (SK halaman), tertutup tulisan tangan di G3 keseluruhan.
#   fase10_ctrl  sama TANPA sisipan, berhenti dan lanjut di langkah yang sama dengan fase10_rare.
# Kedua lengan memakai --font-fixes (perbaikan data fase 9). Argumen pertama = run dasar; resep dasarnya mengikuti run
# itu. Boleh dijalankan ulang (lanjut dari last.pt). Dari root repo, sebagai proses mandiri:
#   Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList "scripts/run_fase10.sh fase9_slant" -WindowStyle Hidden
# Log ringkas: out/fase10_chain.log. Selama berjalan: charger terpasang, jangan Sleep/Shut down, jangan menjalankan
# evaluasi besar atau test suite penuh, dan jangan mengedit src/ (lihat CLAUDE.md, Diagnostik cepat).
set -u
. scripts/fase6_common.sh
BASE=${1:-}
case "$BASE" in
    fase9_slant) BASE_ARGS=(--standard-prob 0.5 --font-fixes --slant-prob 0.5 --slant-max 0.6) ;;
    fase9_ctrl) BASE_ARGS=(--standard-prob 0.5 --font-fixes) ;;
    fase8_std) BASE_ARGS=(--standard-prob 0.5 --font-fixes) ;;
    *) echo "pakai: scripts/run_fase10.sh fase9_slant|fase9_ctrl|fase8_std" >&2; exit 2 ;;
esac
LOG=out/fase10_chain.log
INIT=out/checkpoints/$BASE/last_snapshot.pt
COMMON=(--init "$INIT" --augment fase5 --drop-space-prob 0.5 --extra-fonts fonts/extra --train-lines 100000
        --val-lines 500 --batch-size 32 --lr 3e-4 --steps "$TOTAL" --eval-every 500 --log-every 100
        --time-budget-hours 0 --device xpu --workers 4 --resume --track-prob 0.5 --track-max 0.3 "${BASE_ARGS[@]}")
RARE=(--rare-insert-prob 0.3 --rare-opener-prob 0.15 --rare-max-similarity 0.9 --rare-attach)

segment() {  # sama dengan scripts/run_fase9.sh: $1 = run, $2 = langkah awal, $3 = langkah akhir, sisanya argumen
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

[ -f "$INIT" ] || { note "== $INIT tidak ada; rantai fase10 berhenti $(now)"; exit 1; }
note "== rantai fase10 (dasar $BASE) $(now)"
wait_no_training

train_to_end fase10_rare "${RARE[@]}" || { note "== fase10_rare tidak selesai; rantai berhenti $(now)"; exit 1; }
# Bila rantai dijalankan ulang sesudah terputus di run kontrol, fase10_rare sudah dievaluasi: menyalin ulang
# last_snapshot.pt akan membuatnya lebih baru dari laporan-laporannya.
cmp -s out/checkpoints/fase10_rare/last.pt out/checkpoints/fase10_rare/last_snapshot.pt || evaluate fase10_rare

# Langkah tempat fase10_rare dimulai ulang dari checkpoint = event "start" dengan langkah > 0 (kosong bila tidak ada).
bounds=$(.venv/Scripts/python.exe -c "
import json, sys
steps = {r['step'] for r in map(json.loads, open(sys.argv[1], encoding='utf-8')) if r.get('event') == 'start'}
print(' '.join(str(s) for s in sorted(steps) if 0 < s < int(sys.argv[2])))
" out/checkpoints/fase10_rare/log.jsonl "$TOTAL") || { note "== log fase10_rare tidak terbaca; rantai berhenti $(now)"; exit 1; }
bounds=${bounds//$'\r'/}
note "== fase10_ctrl: titik lanjut mengikuti fase10_rare: ${bounds:-tidak ada} $(now)"
prev=0
for to in $bounds "$TOTAL"; do
    segment fase10_ctrl "$prev" "$to" || { note "== fase10_ctrl: segmen $prev->$to gagal; rantai berhenti $(now)"; exit 1; }
    prev=$to
done
evaluate fase10_ctrl
note "== selesai $(now)"
