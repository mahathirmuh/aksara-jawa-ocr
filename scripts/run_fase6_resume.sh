#!/usr/bin/env bash
# Lanjutan rantai fase6 setelah laptop tidur (2026-10-02): tunggu trainer yang sedang berjalan keluar, hentikan
# rantai lama, lanjutkan fase6_rare sampai langkah 1500, evaluasi, lalu serahkan ke scripts/run_fase6_ctrl.sh.
# Boleh dijalankan ulang kapan saja (lanjut dari last.pt). Dari root repo, sebagai proses mandiri:
#   Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList scripts/run_fase6_resume.sh -WindowStyle Hidden
# $1 (opsional) = PID rantai lama scripts/run_fase6_rare.sh. Log ringkas: out/fase6_rare_chain.log.
set -u
. scripts/fase6_common.sh

note "== rantai lanjutan $(now): menunggu trainer yang berjalan keluar"
wait_no_training
if [ -n "${1:-}" ]; then
    # Tanpa ini rantai lama mengevaluasi run yang belum selesai lalu memulai fase6_ctrl bersamaan dengan run ini.
    taskkill //F //T //PID "$1" > /dev/null 2>&1
    sleep 3
    rm -f out/checkpoints/fase6_rare/last_snapshot.pt
    note "   rantai lama (PID $1) dihentikan $(now)"
    wait_no_training
fi

train_to_end fase6_rare "${RARE[@]}" || { note "== fase6_rare tidak selesai; rantai berhenti $(now)"; exit 1; }
evaluate fase6_rare
exec bash scripts/run_fase6_ctrl.sh
