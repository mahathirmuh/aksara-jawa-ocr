#!/usr/bin/env bash
# Setelah rantai fase6 selesai: ekspor hasil (termasuk fase6_rare & fase6_ctrl), impor ke web, lalu analisis rare vs
# ctrl (scripts/compare_runs.py pada 745 baris nyata, scripts/eval_rare.py sintetis tertarget font javatext).
# Hanya baris log rantai yang ditulis SETELAH skrip ini mulai yang dihitung, jadi aman dijalankan ulang. Bila rantai
# berhenti tanpa hasil kontrol, run yang sudah lengkap tetap diekspor dan diimpor, tanpa analisis.
# Proses mandiri dari root repo; log out/after_fase6.log.
set -u
export PYTHONIOENCODING=utf-8
LOG=out/after_fase6.log
CHAIN=out/fase6_rare_chain.log
start=$(wc -l < "$CHAIN" 2>/dev/null || echo 0)
echo "menunggu rantai fase6 selesai $(date '+%Y-%m-%d %H:%M:%S') (log rantai sejak baris $((start + 1)))" > "$LOG"
new_lines() { tail -n +"$((start + 1))" "$CHAIN" 2>/dev/null; }
until new_lines | grep -qE "^== selesai|^== .*(berhenti|tidak selesai)"; do sleep 120; done
if new_lines | grep -q "^== selesai"; then status=lengkap; else status=gagal; fi
echo "rantai $status $(date '+%H:%M:%S'); ekspor hasil" >> "$LOG"

.venv/Scripts/python.exe scripts/export_results.py 2>&1 | grep -v -E "Warning|warnings.warn" >> "$LOG"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "ekspor gagal; impor & analisis dilewati $(date '+%H:%M:%S')" >> "$LOG"
    exit 1
fi
(cd web && php artisan aksara:import --no-ansi) >> "$LOG" 2>&1
echo "impor exit=$? $(date '+%H:%M:%S')" >> "$LOG"

if [ "$status" = lengkap ]; then
    .venv/Scripts/python.exe scripts/compare_runs.py crnn_fase6_rare crnn_fase6_ctrl >> "$LOG" 2>&1
    echo "compare_runs exit=$? $(date '+%H:%M:%S')" >> "$LOG"
    .venv/Scripts/python.exe scripts/eval_rare.py --workers 2 >> "$LOG" 2>&1
    echo "eval_rare exit=$? $(date '+%H:%M:%S')" >> "$LOG"
fi
echo "selesai $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
