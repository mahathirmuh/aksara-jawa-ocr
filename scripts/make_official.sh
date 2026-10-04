#!/usr/bin/env bash
# Menjadikan sebuah run angka resmi (gerbang G1-G3, kesalahan aksara, titik "resmi" di web):
#   1. evaluasi resmi G1 dan G2 pada 10.000 baris javatext untuk checkpoint-nya (CPU, ~30 menit per gerbang bila mesin
#      tidak dibebani). Menunggu training selesai dulu: training fase7 memakai ~27 GB memori commit, dan evaluasi ini
#      menambah ~6 GB (2026-10-04: commit 75,3 dari 76,4 GB saat keduanya berjalan, evaluasi dihentikan);
#   2. menunggu scripts/after_fase7.sh selesai, supaya tidak ada dua ekspor atau dua eval_spacing sekaligus;
#   3. ekspor (scripts/export_results.py harus sudah menetapkan run itu lewat OFFICIAL_RUN; kalau belum, berhenti
#      tanpa mengekspor), impor ke web, pembanding terhadap run kontrol, evaluasi sintetis jarak (>= 300 baris).
# Boleh dijalankan ulang: evaluasi yang laporannya lebih baru dari snapshot dilewati, evaluasi sintetis memakai cache.
# Proses mandiri dari root repo; log out/make_official.log:
#   Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList scripts/make_official.sh -WindowStyle Hidden
set -u
export PYTHONIOENCODING=utf-8
RUN=${1:-fase7_track}
LOG=out/make_official.log
PY=.venv/Scripts/python.exe
CK=out/checkpoints/$RUN/last_snapshot.pt
JT=C:/Windows/Fonts/javatext.ttf

say() { echo "$* $(date '+%H:%M:%S')" >> "$LOG"; }

after_fase7_running() {  # keluaran kosong (PowerShell gagal) dianggap "masih berjalan"
    powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -match 'after_fase7' }).Count" 2>/dev/null | tr -d '\r\n '
}

training_count() {  # jumlah proses src.train yang hidup; keluaran kosong (PowerShell gagal) dianggap "masih ada"
    powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'src\.train' }).Count" 2>/dev/null | tr -d '\r\n '
}

echo "angka resmi -> $RUN $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
[ -f "$CK" ] || { say "$CK tidak ada; berhenti"; exit 1; }
while [ "$(training_count)" != "0" ]; do sleep 30; done
say "tidak ada training yang berjalan"

for spec in "G1|" "G2|--augment heavy"; do
    goal=${spec%%|*}
    extra=${spec#*|}
    name=${RUN}_${goal}_10k
    if [ "out/eval/$name.json" -nt "$CK" ]; then
        say "$name: sudah ada"
        continue
    fi
    say "evaluasi $name mulai"
    $PY -m src.evaluate "$CK" --lines 10000 --fonts "$JT" $extra --name "$name" 2>&1 \
        | grep -E "CER|Error|Traceback" | sed "s/; detail.*//; s/^/  /" >> "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { say "evaluasi $name gagal; berhenti"; exit 1; }
done

until grep -q "^selesai " out/after_fase7.log 2>/dev/null || [ "$(after_fase7_running)" = "0" ]; do sleep 60; done

official=$($PY -c "import sys; sys.path.insert(0, 'scripts'); import export_results as e; print(getattr(e, 'OFFICIAL_RUN', ''))" 2>/dev/null | tr -d '\r\n ')
if [ "$official" != "$RUN" ]; then
    say "scripts/export_results.py menetapkan '${official:-fase5_fonts}' sebagai resmi, bukan $RUN; ekspor dibatalkan"
    exit 1
fi

say "ekspor mulai"
$PY scripts/export_results.py 2>&1 | grep -v -E "Warning|warnings.warn" >> "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "ekspor gagal; impor dilewati"; exit 1; }
(cd web && php artisan aksara:import --no-ansi) >> "$LOG" 2>&1
say "impor exit=$?"

if [ -f out/checkpoints/fase7_ctrl/last_snapshot.pt ]; then
    for pair in "crnn_fase7_track crnn_fase7_ctrl" "crnn_fase7_ctrl crnn_fase6_ctrl"; do
        $PY scripts/compare_runs.py $pair >> "$LOG" 2>&1
        say "compare_runs $pair exit=$?"
    done
fi
$PY scripts/eval_spacing.py --lines 301 --threads 4 >> "$LOG" 2>&1
say "eval_spacing exit=$?"
say "selesai"
