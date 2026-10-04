# Aksara OCR Lab (web)

Tampilan hasil OCR aksara Jawa dan alur lengkapnya: citra baris → aksara → Latin → arti → tingkat tutur.
Laravel 12 + starter kit Livewire (Flux, Volt), PostgreSQL, Tailwind 4, Chart.js.

Semua angka OCR dihitung di repo Python (folder induk). Web hanya **menampilkan** hasil yang diekspor dan
memanggil layanan model untuk halaman Demo. Aturan proyek ada di `../CLAUDE.md`.

## Menjalankan

```bash
# 1. Di folder repo OCR (induk web/): ekspor hasil (~9 menit di CPU) dan jalankan layanan model
.venv/Scripts/python scripts/export_results.py
.venv/Scripts/python -m uvicorn src.serve:app --host 127.0.0.1 --port 8011

# 2. Di folder web/
composer install && npm install && npm run build   # sekali
php artisan migrate
php artisan aksara:import                           # out/results/ -> database
php artisan aksara:annotations                      # transliterasi & arti manusia NusaAksara
php artisan serve --port=8010                       # http://127.0.0.1:8010, daftar akun di /register
```

Cara termudah menyalakan ketiga layanan sekaligus (Laravel 8010, model OCR 8011, terjemahan 8012) sebagai
proses mandiri yang tetap hidup setelah terminal ditutup:

```powershell
powershell -ExecutionPolicy Bypass -File web\services.ps1 start    # juga: status, stop
```

`php artisan aksara:annotations --fetch` mengunduh ulang anotasi dari HuggingFace (beberapa menit).
Setelah model baru atau eksperimen baru: jalankan ulang `export_results.py` lalu `php artisan aksara:import`.

### Tahap 3 (arti) dan tahap 4 (tingkat tutur)

```bash
# Terjemahan batch Jawa -> Indonesia dengan NLLB-200-distilled-600M lokal (unduhan pertama ~2,5 GB, CPU)
php artisan aksara:translate                        # sumber: transliterasi manusia + keluaran OCR beam+LM
# Layanan terjemahan untuk Demo (port 8012)
../.venv/Scripts/python -m uvicorn translate_service:app --app-dir tools --host 127.0.0.1 --port 8012
```

- Terjemahan dinilai dengan chrF & BLEU (sacrebleu) terhadap terjemahan manusia NusaAksara. NLLB berlisensi
  CC-BY-NC 4.0 (non-komersial). Tidak ada teks yang dikirim ke layanan luar.
- Tingkat tutur memakai leksikon penanda ngoko/madya/krama (`app/Support/SpeechLevel.php`) dan menampilkan
  kata buktinya. Label manusia diisi di Penjelajah baris (tombol ngoko/madya/krama/campur; saringan
  "tutur belum dilabel"); Ringkasan menghitung akurasi leksikon terhadap label itu.

## Database (PostgreSQL)

PostgreSQL 18 (layanan Windows, port 5432). `.env`: `DB_CONNECTION=pgsql`, `DB_HOST=127.0.0.1`,
`DB_DATABASE=its_aksara`, `DB_USERNAME=postgres`, `DB_PASSWORD=...` (hanya di `.env`, tidak ikut git).

```bash
php artisan migrate
php artisan aksara:import && php artisan aksara:annotations && php artisan aksara:translate --import-only
```

`--import-only` membaca `storage/app/mt/output.jsonl` dan `summary.json` (hasil NLLB, ~1 jam CPU). Bila berkas itu
hilang tetapi database masih memuat terjemahannya, tulis ulang dengan `php artisan aksara:translate --dump`
(hanya membaca database; menolak menimpa `output.jsonl` yang lebih lengkap dari database).

Test memakai SQLite in-memory (`phpunit.xml` memaksanya, jadi variabel `DB_*` di shell diabaikan). Untuk menguji
di PostgreSQL pakai database uji terpisah, JANGAN `its_aksara` (test mengosongkan database yang dipakainya):
`php artisan test -c phpunit.pgsql.xml` (selalu `its_aksara_test`). `tests/TestCase.php` menolak database bernama
`its_aksara` sebelum migrasi.

Port: Laravel 8010, layanan model 8011, layanan terjemahan 8012 (8000/8001 dipakai proyek lain di laptop ini).

Hanya akun dan label tingkat tutur yang tidak bisa dibangun ulang dari file. Cadangkan labelnya:
`php artisan aksara:labels export` (-> `database/labels/speech_levels.json`), pulihkan dengan `aksara:labels import`.

## Halaman

| Menu | Isi |
|---|---|
| Ringkasan | Gerbang G1–G4, jarak G3 ke target, status empat tahap alur, fase |
| Perbandingan | CER per pipeline (745 baris, 50 baris uji buta), pipeline yang direncanakan, hasil sintetis |
| Ablasi | 12 run augmentasi Fase 5 |
| Penjelajah baris | Citra, label, transliterasi, arti, tingkat tutur, dan keluaran tiap pipeline dengan beda per suku kata; arahkan kursor ke suku kata CRNN greedy untuk melihat kolom citra yang dibacanya |
| Kesalahan aksara | Aksara tertukar, hilang, tambahan |
| Demo | Unggah potongan satu baris → FastAPI `/predict`, dengan skor CTC dan LM setiap kandidat |

## Kontrak data (skema 1)

`../out/results/` dari `scripts/export_results.py`: `manifest.json` (pipeline, pipeline resmi, gerbang, metrik,
ablasi, kesalahan aksara), `lines.jsonl`, `predictions.jsonl`. `aksara:import` menolak skema lain.

**Pipeline resmi** ditetapkan ekspor, bukan web: kunci `official` di manifest (`OFFICIAL_RUN` di
`scripts/export_results.py`; manifest lama tanpa kunci itu berarti `crnn_fonts`). Impor menandainya di kolom
`pipelines.official`, dan semua halaman membacanya lewat `Pipeline::official()`: tanda "angka G3 resmi", titik penuh
dan kartu tahap 1 di Ringkasan, judul Kesalahan aksara (tabelnya dihitung ekspor dari pipeline itu), serta urutan
dan angka CER di daftar Penjelajah. Gerbang G1–G3 di manifest berasal dari laporan resmi run yang sama. Demo
menampilkan nama checkpoint yang dimuat layanan model (`/health`); bawaannya di `src/serve.py` = checkpoint resmi.

## Lisensi data

Citra dan anotasi NusaAksara berlisensi non-komersial dan hanya dipakai sebagai data uji. Web ini untuk
pemakaian lokal dengan login; citra dibaca langsung dari repo OCR, tidak disalin. Hanya Noto Sans Javanese
(OFL) yang disajikan sebagai font; font training lain tidak boleh disebar.

## Test

```bash
php artisan test
```

Test tidak menyentuh `storage/app/mt`: `tests/TestCase.php` mengalihkan `aksara.mt_dir` ke folder sementara per test.
