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
# Layanan terjemahan untuk Demo dan halaman Terjemahan (port 8012); satu model, dua arah
../.venv/Scripts/python -m uvicorn translate_service:app --app-dir tools --host 127.0.0.1 --port 8012
```

- Terjemahan dinilai dengan chrF & BLEU (sacrebleu) terhadap terjemahan manusia NusaAksara. NLLB berlisensi
  CC-BY-NC 4.0 (non-komersial). Tidak ada teks yang dikirim ke layanan luar.
- API layanan: `POST /translate` dengan `{"texts": [...], "source": "id", "target": "jv"}` (atau sebaliknya; bahasa
  Jawa selalu beraksara Latin) menjawab `{"translations": [...]}`; bentuk lama `{"text": "..."}` (Jawa → Indonesia)
  tetap dipakai Demo. Layanan harus dinyalakan ulang sesudah `tools/nllb.py` atau `translate_service.py` berubah.
- Model tidak mengenal pepet bertanda "ê" (draf transliterasi menulisnya begitu): panggilan langsung membuang tanda
  itu lewat `TranslationService::forModel()`. Batch `aksara:translate` belum, supaya angka yang sudah dilaporkan
  tetap bisa diulang.
- Tingkat tutur memakai leksikon penanda ngoko/madya/krama (`app/Support/SpeechLevel.php`) dan menampilkan
  kata buktinya. Label manusia diisi di Penjelajah baris (tombol ngoko/madya/krama/campur; saringan
  "tutur belum dilabel"); Ringkasan menghitung akurasi leksikon terhadap label itu.

### Alih aksara dan pemulihan tanda é (halaman Terjemahan)

```bash
../.venv/Scripts/python tools/taling_lexicon.py     # opsional: bangun ulang database/dictionaries/jv-taling.json
```

- `App\Support\AksaraWriter::fromLatin()` menulis bahasa Jawa Latin dalam aksara Jawa menurut aturan: vokal tanpa
  konsonan pembuka memakai ha, ng/r/h penutup suku kata menjadi cecak/layar/wignyan, konsonan mati diberi pangkon,
  rê/lê ditulis pa cerek/nga lelet, konsonan + rê memakai keret, n di depan c/j ditulis nya, angka diapit pada
  pangkat. Tidak memakai aksara murda/swara. Ejaannya sama dengan 2.086 dari 2.134 lema kamus bahasa Jawa (97,8%;
  `tests/Unit/AksaraWriterTest.php` menjaga angka itu).
- `App\Support\TalingRestorer` memulihkan é/è pada kata tanpa tanda memakai `database/dictionaries/jv-taling.json`
  (8.709 kata dari Wikipedia bahasa Jawa; pada artikel yang ditahan, kata ber-e yang benar naik dari 53% ke 90%).
  Membangun ulang leksikon butuh snapshot Wikipedia proyek (`../data/raw/jvwiki-20231101.parquet`).
- Arah balik (`Transliterator::toLatin`) tetap draf: ꦲ di tengah kata selalu dibaca "h" dan teks tanpa spasi tidak
  punya batas kata.

### Kamus kata (halaman Kamus)

```bash
../.venv/Scripts/python tools/dictionaries.py       # opsional: unduh ulang sumber dan bangun database/dictionaries/
php artisan aksara:dictionary                       # impor kamus bahasa Jawa dan bahasa Indonesia (--only=jv|id)
```

- Data pihak ketiga, bukan hasil hitung proyek. Berkas jadinya ikut git (`database/dictionaries/jv.jsonl.gz`,
  `id.jsonl.gz`, `sources.json`), jadi `aksara:dictionary` cukup sesudah `migrate`. Sumber dan lisensi tiap berkas:
  `database/dictionaries/SUMBER.md`; juga ditampilkan di kaki tiap kamus, dan tiap entri bertaut ke halaman asalnya.
- Sumber sekarang: entri bahasa Jawa dan bahasa Indonesia di Wiktionary bahasa Inggris (CC BY-SA 4.0, lewat
  ekstraksi kaikki.org), jadi artinya berbahasa Inggris. Kamus Jawa membawa ejaan aksara, ragam, dan padanan
  ngoko/krama dari kepala entrinya.
- **KBBI tidak disalin**: isinya hak cipta Badan Bahasa dan tidak dirilis dengan lisensi terbuka, begitu juga kamus
  yang isinya salinan KBBI (Kateglo). Untuk definisi resmi, hasil pencarian satu kata menautkan ke KBBI Daring.
- Format berkas (satu entri per baris): `word`, `glosses[]`, `gloss_lang` (`id`/`en`/`jv`), `source`, lalu bila ada
  `pos`, `aksara`, `register`, `examples[]`, `note`, `url`. Pengimpor mengganti satu kamus seutuhnya dalam satu
  transaksi dan menolak berkas yang punya baris rusak.

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

Kamus kata diisi ulang dengan `php artisan aksara:dictionary`. Hanya akun dan label tingkat tutur yang tidak bisa
dibangun ulang dari file. Cadangkan labelnya:
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
| Terjemahan | Indonesia → Jawa beraksara Jawa dan sebaliknya. Model NLLB lokal menerjemahkan (per kalimat, paling banyak 12); web memulihkan tanda é (`TalingRestorer`), mengalihaksarakan (`AksaraWriter`, `Transliterator`), dan menampilkan teks Jawa Latin yang boleh disunting sehingga aksaranya mengikuti. Alih aksara tetap jalan tanpa layanan model. Tautan: `?arah=jv-id`, `?q=`, `?jawa=` |
| Kamus | Satu kotak pencarian untuk lima rujukan alur. **Kamus bahasa Jawa** dan **kamus bahasa Indonesia** (data pihak ketiga, `aksara:dictionary`): kata dicari tanpa peduli diakritik dan huruf besar, hasil diurutkan persis sama → berawalan → memuat, bisa dicari balik dari artinya, beberapa kata sekaligus diartikan per kata, dan aksara yang ditempel dicari lewat bacaan Latin drafnya. Lalu **kamus aksara** (91 codepoint charset tokenizer: nama, kode, bacaan Latin draf, jumlah kemunculan di label uji), **kamus koreksi OCR** (model bahasa karakter untuk beam search: greedy vs beam + LM per data, dari hasil yang diimpor), **leksikon tingkat tutur** (kata penanda ngoko/madya/krama tahap 4). Aksara yang ditempel ke kotak cari diurai per codepoint |

## Tampilan

Gaya mengikuti konsep aplikasi admin berbasis template Tabler (permintaan user 2026-10-05): sidebar terang yang
bisa diciutkan dengan item aktif biru, bilah atas putih berisi judul halaman, latar abu kebiruan dengan kartu putih
bergaris tipis, tabel rapat 12 px, huruf Inter 13 px. Stack tetap Tailwind 4 + Flux; tidak ada Bootstrap.

- **Satu sumber gaya:** `resources/css/app.css`. Isinya token warna dan permukaan (`--surface`, `--border-soft`,
  `--brand`, `--good|bad|warn`, dan seterusnya; semuanya punya nilai untuk tema gelap) serta kelas komponen:
  `.card` / `.card-body` / `.section-title`, `.stat-card`, `.table-wrap` + `.data-table`, `.status status-*`,
  `.badge`, `.chip`, `.alert`, `.nav-tabs`, `.btn-pagination`. Halaman baru memakai kelas itu, bukan gugus utilitas
  warna sendiri.
- **Kerangka:** `components/layouts/app/sidebar.blade.php` (sidebar Flux `collapsible`, bilah atas, footer); isi
  halaman dibungkus `.container-xl` (maks. 1440 px) di `layouts/app.blade.php`. Halaman masuk dan daftar:
  `layouts/auth/split.blade.php`.
- **Huruf dan ikon:** Inter 4.1 dari paket npm `inter-ui` (OFL, dibundel Vite, tanpa CDN); aksara Jawa tetap Noto
  Sans Javanese. Ikon menu = Tabler Icons (MIT) sebagai komponen `flux:icon.ti-*` di `resources/views/flux/icon/`.
- **Lambang:** gambar pilihan user (2026-10-05), sumbernya `resources/brand/logo.png`: ubin putih bersudut bulat
  berisi keris di atas atap joglo, diapit bulir padi di atas ombak, dengan sebaris tulisan bergaya aksara Jawa.
  `tools/logo.py` menurunkan semua berkasnya: `public/img/logo/logo.png` (ubin utuh, di atas judul halaman masuk),
  `public/img/logo/mark.png` (lambangnya saja di ubin putih, untuk sidebar lewat `x-app-logo`; tulisannya tidak
  terbaca di bawah ~100 px), `favicon.ico`, `favicon-192.png`, dan `apple-touch-icon.png`. Untuk ukuran kecil tinta
  lambang ditebalkan sedikit supaya tidak memudar. Dibuat ulang dari akar repo:
  `.venv/Scripts/python web/tools/logo.py web`. Tulisan aksara di gambar itu belum diperiksa pembaca aksara.
- **Halaman masuk:** panel kiri berisi korsel tiga foto (naskah beraksara Jawa di Museum Sonobudoyo, halaman
  Serat Damar Wulan, gerbang Kraton Yogyakarta), latarnya foto naskah yang diburamkan. Semua dari Wikimedia Commons
  berlisensi bebas (CC0, domain publik, CC BY 4.0); sumber dan pembuatnya di `public/img/login/SUMBER.md` dan di
  keterangan tiap foto. Dibuat ulang dengan `../.venv/Scripts/python tools/login_images.py public/img/login`.
- Setelah mengubah CSS/JS atau menambah kelas utilitas di view: `npm run build` (`public/build` tidak ikut git).
- Yang diikuti hanya konsep visualnya: logo, nama, dan foto aplikasi rujukan tidak dipakai.

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
(OFL) yang disajikan sebagai font aksara; font training lain tidak boleh disebar. Huruf antarmuka Inter (OFL) dan
Tabler Icons (MIT) boleh disebar ulang.
Kamus kata (`database/dictionaries/`) berasal dari Wiktionary dan berlisensi CC BY-SA 4.0: atribusi dan tautan ke
halaman asal harus tetap ada; rinciannya di `database/dictionaries/SUMBER.md`.

## Test

```bash
php artisan test
```

Test tidak menyentuh `storage/app/mt`: `tests/TestCase.php` mengalihkan `aksara.mt_dir` ke folder sementara per test.
