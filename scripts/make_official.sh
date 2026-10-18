#!/usr/bin/env bash
# Menjadikan sebuah run angka resmi (gerbang G1-G3, kesalahan aksara, titik "resmi" di web):
#   1. evaluasi resmi G1 dan G2 pada 10.000 baris javatext untuk checkpoint-nya (CPU, ~30 menit per gerbang bila mesin
#      tidak dibebani). Menunggu training selesai dulu: training fase7 memakai ~27 GB memori commit, dan evaluasi ini
#      menambah ~6 GB (2026-10-04: commit 75,3 dari 76,4 GB saat keduanya berjalan, evaluasi dihentikan);
#   2. menunggu skrip lanjutan rantai (scripts/after_fase*.sh) selesai, supaya tidak ada dua ekspor atau dua
#      eval_spacing sekaligus;
#   3. ekspor (scripts/export_results.py harus sudah menetapkan run itu lewat OFFICIAL_RUN; kalau belum, berhenti
#      tanpa mengekspor), kartu data halaman Dataset (scripts/export_datasets.py), impor ke web, pembanding terhadap
#      run kontrol (per run, lihat `pairs` di bawah), evaluasi sintetis jarak (>= 300 baris, run resmi ikut dibaca),
#      lalu kartu metode halaman Metode
#      (scripts/export_methods.py) paling akhir, karena buktinya dikutip dari semua keluaran di atas.
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

after_running() {  # skrip lanjutan rantai mana pun; keluaran kosong (PowerShell gagal) dianggap "masih berjalan"
    powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='bash.exe'\" | Where-Object { \$_.CommandLine -match 'after_fase[0-9]' }).Count" 2>/dev/null | tr -d '\r\n '
}

pairs() {  # pembanding (scripts/compare_runs.py) yang buktinya dikutip kartu untuk run resmi $1
    case "$1" in
        fase7_track) echo "crnn_fase7_track:crnn_fase7_ctrl crnn_fase7_ctrl:crnn_fase6_ctrl" ;;
        fase8_std) echo "crnn_fase8_std:crnn_fase8_ctrl crnn_fase8_std:crnn_fase7_track crnn_fase8_ctrl:crnn_fase7_track" ;;
        fase9_slant) echo "crnn_fase9_slant:crnn_fase9_ctrl crnn_fase9_slant:crnn_fase8_std crnn_fase9_ctrl:crnn_fase8_std" ;;
        # fase9_ctrl: pasangannya sendiri, lalu dua lanjutan model resmi yang diuji (fase 10 dan 11) dan pasangan di
        # rantainya yang dikutip kartu (aksara langka fase 7; jarak antar suku kata).
        fase9_ctrl) echo "crnn_fase9_slant:crnn_fase9_ctrl crnn_fase9_ctrl:crnn_fase8_std crnn_fase10_rare:crnn_fase10_ctrl crnn_fase11_neighbor:crnn_fase11_ctrl crnn_fase7_track_rare:crnn_fase7_track crnn_fase7_track:crnn_fase7_ctrl" ;;
    esac
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

until [ "$(after_running)" = "0" ]; do sleep 60; done

official=$($PY -c "import sys; sys.path.insert(0, 'scripts'); import export_results as e; print(getattr(e, 'OFFICIAL_RUN', ''))" 2>/dev/null | tr -d '\r\n ')
if [ "$official" != "$RUN" ]; then
    say "scripts/export_results.py menetapkan '${official:-fase5_fonts}' sebagai resmi, bukan $RUN; ekspor dibatalkan"
    exit 1
fi

say "ekspor mulai"
$PY scripts/export_results.py 2>&1 | grep -v -E "Warning|warnings.warn" >> "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || { say "ekspor gagal; impor dilewati"; exit 1; }
# Kartu data halaman Dataset mengikuti run resmi yang sama (pemakaian data latih dibaca dari checkpoint-nya);
# aksara:import di bawah ikut mengimpornya. Bila gagal, hasil OCR tetap diimpor dan halaman Dataset menandai
# kartu lamanya tertinggal.
$PY scripts/export_datasets.py 2>&1 | grep -v -E "Warning|warnings.warn" >> "$LOG"
say "ekspor kartu data exit=${PIPESTATUS[0]}"
(cd web && php artisan aksara:import --no-ansi) >> "$LOG" 2>&1
say "impor exit=$?"

# Berkas pembanding (dan, untuk run berejaan baku, laporan bentuk ejaan baku; untuk run fase 9, CER per kelompok
# kemiringan baris) dihitung ulang dari ekspor ini: kartu metode hanya mengutip berkas yang CER-nya sama dengan manifest
# yang baru.
for pair in $(pairs "$RUN"); do
    a=${pair%%:*}
    b=${pair#*:}
    $PY scripts/compare_runs.py "$a" "$b" >> "$LOG" 2>&1
    say "compare_runs $a $b exit=$?"
    case "$RUN" in
        fase8_*)
            $PY scripts/eval_spelling.py real "$a" "$b" >> "$LOG" 2>&1
            say "eval_spelling real $a $b exit=$?" ;;
        fase9_*)
            $PY scripts/eval_slant.py real "$a" "$b" >> "$LOG" 2>&1
            say "eval_slant real $a $b exit=$?" ;;
    esac
done
# Run resmi ikut dibaca evaluasi jarak (kartu hanya mengutip laporan yang membaca checkpoint resmi sekarang); bacaan
# run lain diambil dari cache.
spacing=(fase7_track fase7_ctrl fase7_track_rare fase6_ctrl fase5_fonts)
case " ${spacing[*]} " in *" $RUN "*) ;; *) spacing=("$RUN" "${spacing[@]}") ;; esac
$PY scripts/eval_spacing.py --lines 301 --threads 4 --checkpoints "${spacing[@]}" >> "$LOG" 2>&1
say "eval_spacing exit=$?"
# Kartu metode halaman Metode mengutip manifest hasil, berkas pembanding, dan evaluasi sintetis di atas, jadi
# diekspor dan diimpor paling akhir. Sampai di sini halaman Metode masih memuat kartu lama dan menandainya tidak
# sejalan dengan hasil yang baru diimpor. Bila gagal, kartu lama dipertahankan.
$PY scripts/export_methods.py 2>&1 | grep -v -E "Warning|warnings.warn" >> "$LOG"
say "ekspor kartu metode exit=${PIPESTATUS[0]}"
(cd web && php artisan aksara:methods --no-ansi) >> "$LOG" 2>&1
say "impor kartu metode exit=$?"
say "selesai"
