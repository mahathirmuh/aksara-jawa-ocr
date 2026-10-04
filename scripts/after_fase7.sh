#!/usr/bin/env bash
# Sesudah rantai fase7: ekspor -> impor ke web -> pembanding -> evaluasi sintetis (jarak antar-aksara, aksara langka).
#   Tahap A: sesudah "== selesai" di out/fase7_chain.log (fase7_track & fase7_track_rare selesai dievaluasi).
#   Tahap B: sesudah "== fase7_ctrl selesai" (run kontrol); mengulang ekspor dan pembanding terhadap kontrol.
# Baris yang sudah ada di log saat skrip mulai ikut dihitung, jadi boleh dijalankan sesudah rantai lewat.
# Evaluasi sintetis memakai cache, jadi tahap B hanya membayar bacaan checkpoint kontrol.
# Proses mandiri dari root repo; log out/after_fase7.log.
set -u
export PYTHONIOENCODING=utf-8
LOG=out/after_fase7.log
CHAIN=out/fase7_chain.log
PY=.venv/Scripts/python.exe

say() { echo "$* $(date '+%H:%M:%S')" >> "$LOG"; }

ready() {  # ekspor berhenti total bila sebuah run punya snapshot tetapi laporan G3-nya belum ditulis
    local r
    for r in fase7_track fase7_track_rare fase7_ctrl; do
        [ -f "out/checkpoints/$r/last_snapshot.pt" ] || continue
        [ "out/eval/${r}_G3_full.json" -nt "out/checkpoints/$r/last_snapshot.pt" ] || return 1
    done
}

publish() {  # ekspor + impor; gagal = 1 (out/results dan web tidak berubah)
    local tries=0
    until ready; do
        tries=$((tries + 1))
        [ "$tries" -gt 20 ] && { say "laporan G3 belum lengkap setelah 20 menit; ekspor dibatalkan"; return 1; }
        sleep 60
    done
    say "ekspor mulai"
    $PY scripts/export_results.py 2>&1 | grep -v -E "Warning|warnings.warn" >> "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { say "ekspor gagal; impor dilewati"; return 1; }
    (cd web && php artisan aksara:import --no-ansi) >> "$LOG" 2>&1
    say "impor exit=$?"
}

compare() {
    $PY scripts/compare_runs.py "$1" "$2" >> "$LOG" 2>&1
    say "compare_runs $1 vs $2 exit=$?"
}

spacing() {  # >= 300 baris: aturan analisis fase 7 di CLAUDE.md
    $PY scripts/eval_spacing.py --lines 300 --threads 4 >> "$LOG" 2>&1
    say "eval_spacing exit=$?"
}

echo "menunggu rantai fase7 $(date '+%Y-%m-%d %H:%M:%S')" > "$LOG"
until grep -qE "^== selesai|^== .*berhenti" "$CHAIN" 2>/dev/null; do sleep 60; done
say "tahap A: $(grep -E '^== selesai|^== .*berhenti' "$CHAIN" | tail -1)"
if publish; then
    compare crnn_fase7_track crnn_fase6_ctrl
    [ -f out/checkpoints/fase7_track_rare/last_snapshot.pt ] && compare crnn_fase7_track_rare crnn_fase7_track
fi
spacing
if [ -f out/checkpoints/fase7_track_rare/last_snapshot.pt ]; then
    # Direktori sendiri: out/compare/rare_synthetic.* berisi hasil fase6 dan tidak boleh tertimpa.
    $PY scripts/eval_rare.py --checkpoints fase7_track_rare fase7_track fase6_rare fase6_ctrl --workers 2 --threads 4 \
        --out out/compare/fase7 >> "$LOG" 2>&1
    say "eval_rare exit=$?"
fi
say "tahap A selesai"

until grep -qE "^== fase7_ctrl (selesai|dilewati)|^== fase7_ctrl: .*(gagal|tidak terbaca)" "$CHAIN" 2>/dev/null; do sleep 60; done
say "tahap B: $(grep -E '^== fase7_ctrl' "$CHAIN" | tail -1)"
if [ -f out/checkpoints/fase7_ctrl/last_snapshot.pt ] && publish; then
    compare crnn_fase7_track crnn_fase7_ctrl
    compare crnn_fase7_ctrl crnn_fase6_ctrl
    spacing
fi
say "selesai"
