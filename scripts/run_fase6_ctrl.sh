#!/usr/bin/env bash
# Run kontrol fase6_ctrl: sama dengan fase6_rare tanpa aksara langka, dan dengan TITIK LANJUT yang sama.
#
# LengthBucketSampler mengulang urutan batch dari awal setiap kali proses training dimulai, jadi run yang
# terhenti lalu dilanjutkan melihat ulang baris-baris awal dan tidak pernah sampai ke baris yang lebih jauh.
# fase6_rare terhenti beberapa kali (sesi berakhir, laptop tidur). Supaya pembandingnya hanya berbeda di aksara
# langka, fase6_ctrl dihentikan dan dilanjutkan di langkah yang sama (--stop-step), dibaca dari log fase6_rare.
# Segmen yang gagal di tengah diulang dari checkpoint awal segmennya (seg_<langkah>.pt), bukan dari last.pt.
# Dipanggil oleh scripts/run_fase6_resume.sh; boleh dijalankan ulang. Dari root repo.
set -u
. scripts/fase6_common.sh
run=fase6_ctrl
dir=out/checkpoints/$run

# Langkah tempat fase6_rare dimulai ulang dari checkpoint = semua event "start" dengan langkah > 0.
bounds=$(.venv/Scripts/python.exe -c "
import json, sys
steps = {r['step'] for r in map(json.loads, open(sys.argv[1], encoding='utf-8')) if r.get('event') == 'start'}
print(' '.join(str(s) for s in sorted(steps) if 0 < s < int(sys.argv[2])))
" out/checkpoints/fase6_rare/log.jsonl "$TOTAL") || { note "== $run: log fase6_rare tidak terbaca; berhenti $(now)"; exit 1; }
bounds=${bounds//$'\r'/}
[ -n "$bounds" ] || { note "== $run: titik lanjut fase6_rare kosong (seharusnya minimal 500); berhenti $(now)"; exit 1; }
note "== $run: titik lanjut mengikuti fase6_rare: $bounds $(now)"

segment() {  # $1 = langkah awal, $2 = langkah akhir
    local from=$1 to=$2 tries=0 step stop
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
        run_train "$run" "${stop[@]}"
    done
}

wait_no_training
prev=0
for to in $bounds "$TOTAL"; do
    segment "$prev" "$to" || { note "== $run: segmen $prev->$to gagal; rantai berhenti TANPA hasil kontrol $(now)"; exit 1; }
    prev=$to
done
evaluate "$run"
note "== selesai $(now)"
