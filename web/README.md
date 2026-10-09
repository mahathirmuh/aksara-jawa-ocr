# Aksara OCR Lab (web)

Tampilan hasil OCR aksara Jawa dan alur lengkapnya: citra baris → aksara → Latin → arti → tingkat tutur.
Laravel 12 + starter kit Livewire (Flux, Volt), PostgreSQL, Tailwind 4, Chart.js.

Semua angka OCR dihitung di repo Python (folder induk). Web hanya **menampilkan** hasil yang diekspor dan
memanggil layanan model untuk halaman Demo. Aturan proyek ada di `../CLAUDE.md`.

## Menjalankan

```bash
# 1. Di folder repo OCR (induk web/): ekspor hasil (~14 menit di CPU saat diukur dengan run fase6; bertambah
#    dengan jumlah run yang diekspor) dan jalankan layanan model
.venv/Scripts/python scripts/export_results.py
.venv/Scripts/python scripts/export_datasets.py     # kartu data halaman Dataset (~20 detik)
.venv/Scripts/python scripts/export_methods.py      # kartu metode halaman Metode (beberapa detik; sesudah export_results.py)
.venv/Scripts/python -m uvicorn src.serve:app --host 127.0.0.1 --port 8011

# 2. Di folder web/
composer install && npm install && npm run build   # sekali
php artisan migrate
php artisan aksara:import                           # out/results/ -> database (termasuk datasets.json dan methods.json bila ada)
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
Kartu halaman Dataset dan Metode bisa diperbarui sendiri: `export_datasets.py` lalu `php artisan aksara:datasets`,
`export_methods.py` lalu `php artisan aksara:methods`. Kartu metode mengutip angka dari `out/results/manifest.json`,
jadi diekspor sesudah `export_results.py`; selang kepercayaan selisih antar-run dibacanya dari `out/compare/`
(`scripts/compare_runs.py`) bila CER di berkas itu sama dengan manifest, dan evaluasi tertarget
(`scripts/eval_spacing.py`) bila laporannya membaca checkpoint yang sekarang.
Ketiga perintah impor langsung menulis "PERINGATAN" bila kartu yang baru diimpor tidak sejalan dengan hasil OCR yang
diimpor (kartu untuk run lain, atau buktinya dari ekspor hasil yang lain), dengan skrip dan perintah yang perlu
diulang.

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
  pangkat. Tidak memakai aksara murda/swara. Ejaannya sama dengan 2.173 dari 2.231 lema berejaan baku di kamus bahasa Jawa (97,4%;
  `tests/Unit/AksaraWriterTest.php` menjaga angka itu).
- `App\Support\TalingRestorer` memulihkan é/è pada kata tanpa tanda memakai `database/dictionaries/jv-taling.json`
  (8.714 kata: 8.615 dari Wikipedia bahasa Jawa dan 99 dari lema kamus Wiktionary bahasa Inggris; pada artikel
  Wikipedia yang ditahan, kata ber-e yang benar naik dari 53% ke 90%).
  Membangun ulang leksikon butuh snapshot Wikipedia proyek (`../data/raw/jvwiki-20231101.parquet`).
- Arah balik (`Transliterator::toLatin`) tetap draf: ꦲ di tengah kata selalu dibaca "h" dan teks tanpa spasi tidak
  punya batas kata.

### Kamus kata (halaman Kamus)

```bash
../.venv/Scripts/python tools/dictionaries.py       # opsional: bangun ulang database/dictionaries/ (--refresh = unduh ulang sumber)
php artisan aksara:dictionary                       # impor kamus bahasa Jawa dan bahasa Indonesia (--only=jv|id)
```

- Data pihak ketiga, bukan hasil hitung proyek. Berkas jadinya ikut git (`database/dictionaries/jv.jsonl.gz`,
  `id.jsonl.gz`, `sources.json`), jadi `aksara:dictionary` cukup sesudah `migrate`. Sumber dan lisensi tiap berkas:
  `database/dictionaries/SUMBER.md`; juga ditampilkan di kaki tiap kamus, dan tiap entri bertaut ke halaman asalnya.
- Sumber (semuanya CC BY-SA 4.0):
  - Wiktionary bahasa Inggris lewat ekstraksi kaikki.org: bahasa Jawa 3.966 entri (arti berbahasa Inggris, ejaan
    aksara di 3.148 entri, ragam di 654, padanan ngoko/krama dari kepala entrinya) dan bahasa Indonesia 39.966 entri
    (arti berbahasa Inggris).
  - Wikikamus (Wiktionary bahasa Indonesia), bagian bahasa Jawa, dump 2026-10-01: 1.922 entri dengan arti berbahasa
    Indonesia dan 942 contoh kalimat berterjemahan. Bagian bertanda impor KBBI dibuang. Ragam dan ejaan aksara tidak
    diambil dari sini (tidak andal); label ragam dan dialeknya tampil sebagai bagian teks arti.
  Kamus Jawa seluruhnya 5.888 entri (4.415 lema). Arti berbahasa Indonesia didahulukan, dan kata Indonesia menemukan
  padanan Jawanya lewat pencarian balik (ketik "makan" untuk mangan, dhahar, madhang).
- Pencarian (`App\Support\DictionarySearch`): diakritik dan huruf besar diabaikan; satu kata dicari sebagai katanya
  (imbuhan dengan tanda hubungnya: "-an"; kata ulang yang diketik berspasi, "anak anak", menampilkan anak-anak lebih
  dulu); beberapa kata dicari sebagai frasa utuh ("mau tak mau", "ke- -an") lalu diartikan per kata; tanda kutip yang
  membungkus kata dibuang; kueri dipotong di 500 karakter. Urutan: persis sama, berawalan, memuat; lalu kata lain yang
  artinya memuat kata itu. Urutan dan hasilnya sama di PostgreSQL dan SQLite.
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
php artisan aksara:datasets     # hanya bila datasets.json diekspor sesudah aksara:import
php artisan aksara:methods      # hanya bila methods.json diekspor sesudah aksara:import
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
| Perbandingan | CER, CER tanpa spasi, baris persis, presisi, recall, F1 per pipeline (745 baris, 50 baris uji buta) dengan definisinya, pipeline yang direncanakan, hasil sintetis |
| Ablasi | 12 run augmentasi Fase 5 |
| Dataset | Kartu data: besar korpus, pembagian latih / validasi / uji dengan arti tiap bagian dan berapa yang dipakai run resmi, asal korpus (dari artikel ke baris, sebaran panjang), rantai checkpoint run resmi (langkah, sampel, baris berbeda), daftar dataset (peran, sumber, lisensi, boleh disebar atau tidak), font per peran dengan catatannya, batasan data, dan data pendukung |
| Metode | Daftar metode dan machine learning yang dipakai, per tahap: pembaca aksara (CNN, BiLSTM, lapisan keluaran, CTC, decoding greedy, tokenizer urutan visual, normalisasi kontras), data latih sintetis (render, variasi font, augmentasi, jarak antar suku kata, buang spasi, sisipan aksara langka), pelatihan (AdamW, OneCycle, pemotongan gradien, batch per panjang, pelatihan bertahap), koreksi sesudah baca (beam search + model bahasa karakter), evaluasi dan statistik (CER, gerbang, bootstrap berpasangan, run kontrol, ablasi, evaluasi sintetis tertarget, uji buta VLM), lalu tahap lanjutan alur milik web (alih aksara, NLLB-200, chrF/BLEU, tingkat tutur, pemulih tanda é). Tiap butir punya jenis, status (dipakai model resmi, dipakai, tersedia, diuji lalu tidak dipakai, pembanding), penjelasan, berkas kodenya, dan bila ada: pengaturan, bukti terukur dengan sumbernya, serta catatan batasan (mis. font uji yang sekeluarga dengan font latih, font latih bercacat, ablasi satu seed, angka terjemahan yang dihitung dengan ejaan yang tidak dikenal modelnya). Di atasnya: empat angka pokok dan alur dari citra ke teks Latin, yang lalu bercabang ke arti dan tingkat tutur; di bawahnya: metode yang direncanakan |
| Penjelajah baris | Citra, label, transliterasi, arti, tingkat tutur, dan keluaran tiap pipeline dengan beda per suku kata; arahkan kursor ke suku kata CRNN greedy untuk melihat kolom citra yang dibacanya |
| Kesalahan aksara | Aksara tertukar, hilang, tambahan; recall, presisi, F1 per aksara (macro-F1, mikro, 20 aksara F1 terendah) untuk pipeline resmi |
| Demo | Unggah potongan satu baris → FastAPI `/predict`, dengan skor CTC dan LM setiap kandidat |
| Terjemahan | Indonesia → Jawa beraksara Jawa dan sebaliknya. Model NLLB lokal menerjemahkan (per kalimat, paling banyak 12); web memulihkan tanda é (`TalingRestorer`), mengalihaksarakan (`AksaraWriter`, `Transliterator`), dan menampilkan teks Jawa Latin yang boleh disunting sehingga aksaranya mengikuti. Alih aksara tetap jalan tanpa layanan model. Tautan: `?arah=jv-id`, `?q=`, `?jawa=` |
| Kamus | Satu kotak pencarian untuk lima rujukan alur. **Kamus bahasa Jawa** dan **kamus bahasa Indonesia** (data pihak ketiga, `aksara:dictionary`): kata dicari tanpa peduli diakritik dan huruf besar, hasil diurutkan persis sama → berawalan → memuat, bisa dicari balik dari artinya (arti kamus Jawa sebagian berbahasa Indonesia, jadi "makan" menemukan mangan dan dhahar), frasa dan imbuhan ("mau tak mau", "-an") dicari seperti diketik, beberapa kata sekaligus diartikan per kata, dan aksara yang ditempel dicari lewat bacaan Latin drafnya dan ejaan aksaranya. Lalu **kamus aksara** (91 codepoint charset tokenizer: nama, kode, bacaan Latin draf, jumlah kemunculan di label uji), **kamus koreksi OCR** (model bahasa karakter untuk beam search: greedy vs beam + LM per data, dari hasil yang diimpor), **leksikon tingkat tutur** (kata penanda ngoko/madya/krama tahap 4). Aksara yang ditempel ke kotak cari diurai per codepoint |

## Tampilan

Gaya mengikuti konsep aplikasi admin berbasis template Tabler (permintaan user 2026-10-05): sidebar terang yang
bisa diciutkan dengan item aktif biru, bilah atas putih berisi judul halaman, latar abu kebiruan dengan kartu putih
bergaris tipis, tabel rapat 12 px, huruf Inter 13 px. Stack tetap Tailwind 4 + Flux; tidak ada Bootstrap.

- **Satu sumber gaya:** `resources/css/app.css`. Isinya token warna dan permukaan (`--surface`, `--border-soft`,
  `--brand`, `--good|bad|warn`, dan seterusnya; semuanya punya nilai untuk tema gelap) serta kelas komponen:
  `.card` / `.card-body` / `.section-title`, `.stat-card`, `.table-wrap` + `.data-table`, `.status status-*`,
  `.badge`, `.chip`, `.alert`, `.nav-tabs`, `.btn-pagination`. Halaman baru memakai kelas itu, bukan gugus utilitas
  warna sendiri. Halaman Dataset menambah `.split-bar` (batang pembagian), `.meter` (batang kecil), `.split-dot` dan
  `.role-chip` (penanda peran); warna bagiannya `.split-train|val|test` = `--series-1|2|3`. Seri ketiga abu
  kebiruan. Warna bukan satu-satunya pembeda: tiap bagian selalu disertai namanya (terang-gelap biru dan abu itu hanya
  berbeda 1,8 kali). Alur `.meter` berwarna permukaan dengan garis tepi dari warna teks redup, supaya batasnya
  terlihat dan isi jingga tetap berkontras 3:1.
  Halaman Metode menambah `.flow` + `.flow-step` (alur langkah; `.is-learned` = langkah yang memakai model hasil
  belajar, ditandai warna DAN tulisan "ML" lewat `.flow-tag`; `.flow-branches` = dua langkah terakhir yang sama-sama
  berangkat dari teks Latin) dan `.method-list` / `.method-row` / `.method-evidence` / `.method-note` /
  `.method-settings` (satu butir: nama dan status, penjelasan dengan kotak "Terukur" dan kotak "Catatan", pengaturan;
  tiga kolom di layar lebar, dua di layar sedang, satu di ponsel).
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
ablasi, kesalahan aksara, metrik per aksara), `lines.jsonl`, `predictions.jsonl`. `aksara:import` menolak skema lain.

**Metrik karakter (sejak 2026-10-09, kunci opsional di tiap butir `metrics`; ekspor lama tanpa kunci ini tetap
diimpor dengan nilai null dan halaman menampilkan "–"):** `precision`, `recall`, `f1`, `char_accuracy`, dihitung
`src/metrics.py` dari penjajaran karakter yang sama dengan recall aksara langka `scripts/compare_runs.py` (biaya
Levenshtein minimum, lalu kecocokan terbanyak). Dari jumlah cocok M, tertukar S, hilang D, tambahan I: CER = (S + D +
I)/|label|, recall = M/|label|, presisi = M/|keluaran|, F1 harmonik keduanya, akurasi karakter = M/(M + S + D + I).
`class_metrics` (`scope`, `items[]` dengan `pipeline`, `char`, `code`, `name`, `ref`, `hyp`, `tp`, `fn`, `fp`,
`precision`, `recall`, `f1`) = angka yang sama per karakter, untuk tiap pipeline yang punya prediksi di semua baris;
tabel `class_metrics` diganti utuh saat impor. Perbandingan menampilkan presisi/recall/F1 di samping CER;
Kesalahan aksara menampilkan macro-F1, mikro, dan aksara dengan F1 terendah (yang muncul ≥ 10 kali di label).

**Kartu data (skema 1, kontrak sendiri):** `../out/results/datasets.json` dari `scripts/export_datasets.py`, diimpor
`php artisan aksara:datasets` (atau ikut `aksara:import`) ke tabel `dataset_reports`: satu baris berisi dokumennya
utuh. Semua angka data OCR di halaman Dataset berasal dari kartu ini; yang dihitung web hanya data pendukung miliknya
sendiri (kamus kata dan anotasi dari tabel web, leksikon é dari berkasnya). Kalimat yang bergantung pada isi kartu
(jumlah run di rantai, seed, batasan data) dipilih dari nilai kartu, jadi tetap benar untuk kartu run lain; nilai
yang tidak dihitung ekspor (`null`, misalnya jadwal baris run `--real-train`) ditampilkan "tidak dihitung". Kolom
`payload` bertipe `json`, bukan `jsonb`: jsonb PostgreSQL mengurutkan ulang kunci objek, dan halaman tidak boleh
bergantung pada urutan kunci (peran selalu diurutkan latih, validasi, uji). Kartu memuat kunci pipeline run yang
dihitungnya; bila berbeda dari `Pipeline::official()`, halaman menampilkan peringatan "Kartu data dan hasil OCR
tidak sejalan", tanpa menebak sisi mana yang tertinggal. Kunci yang ditambahkan sesudah tinjauan 2026-10-06:
`usage.test.others` (evaluasi di luar `out/eval` yang mengambil barisnya secara acak dari seluruh bagian uji:
aksara langka sintetis dan beam + LM; karena itu halaman tidak lagi menyiratkan hanya N baris pertama yang pernah
dibaca), `usage.real.others` (evaluasi di luar `out/eval` yang membaca seluruh data nyata: laporan beam + LM; halaman
menjumlahkannya dengan laporan `out/eval` dan menulis "paling sedikit N kali", karena pembacaan yang tidak
meninggalkan laporan tidak terhitung), serta `fonts_source` dan `font_names` per run (`log` bila log training mencatat
nama font perender data latihnya, dan nama itulah yang ditulis; `folder` bila jumlahnya dihitung dari folder font
sekarang, yang diperingatkan bila ekspor hasil mencatat jumlah lain). Bila log run resmi mencatat nama font, peran
"latih" di tabel font hanya diberikan kepada nama yang tercatat, dan kartu memperingatkan font tercatat yang sudah
tidak ada di folder serta font di folder yang tidak tercatat. Kalimat pembuka pembagian menulis "teks uji tidak pernah
dilihat saat latih" hanya bila kartu mencatat nol baris berteks sama di lebih dari satu bagian; kalau tidak,
jumlahnya yang disebut.

**Kartu metode (skema 1, kontrak sendiri):** `../out/results/methods.json` dari `scripts/export_methods.py`, diimpor
`php artisan aksara:methods` (atau ikut `aksara:import`) ke tabel `method_reports`. Isinya `model` (arsitektur dan
jumlah parameter, dihitung dari model yang dibangun dari checkpoint run resmi), `groups[]` (tiap kelompok: `key`,
`title`, `intro`, `methods[]` dengan `key`, `name`, `kind`, `status`, `summary`, `settings` berupa pasangan
`[label, nilai]`, `evidence`, `evidence_source`, `note`, `files[]`), `planned[]`, dan `chain_complete` (rantai
checkpoint run resmi terbaca utuh atau tidak). `evidence` = hasil ukur, `note` = batasan yang harus dibaca bersama
butir itu; keduanya boleh kosong. Penjelasan, pengaturan, bukti, dan catatan butir milik repo OCR disusun Python:
angka dan pengaturannya dibaca dari checkpoint, kode, log, dan berkas hasil, sedangkan kalimat penjelasannya ditulis
tangan di skrip. Aturan kutipnya: selisih antar-run selalu dihitung dari manifest; selang kepercayaannya hanya diambil
dari berkas pembanding yang CER-nya sama dengan manifest; sebuah run hanya disebut "run kontrol" bila kedua
checkpoint-nya ada dan memang berbeda di satu perlakuan itu saja (langkah, argumen pelatihan, dan resep data sama);
evaluasi sintetis hanya dikutip bila laporannya membaca checkpoint yang sekarang (langkahnya sama, dan sidik
berkasnya sama bila laporan mencatatnya), dan bila tidak, catatan butirnya menyebut bahwa angkanya tidak dikutip.
Bukti selalu membawa sumbernya: skrip menolak butir berbukti tanpa sumber, dan impor menolak kartu yang begitu.
Web menampilkannya apa adanya dan hanya menambah kelompok "Tahap lanjutan alur" dari tabel dan konstanta web sendiri
(`Metode::webGroup()`); kunci butir kelompok itu (`Metode::WEB_KEYS`) tidak boleh dipakai butir kartu, dan impor
menolak kartu yang memakainya. Kartu memuat `results_generated` (waktu ekspor hasil yang dikutip buktinya): bila
berbeda dari hasil yang diimpor, atau kartu dibuat untuk run lain, halaman menampilkan peringatan "Kartu metode dan
hasil tidak sejalan". Angka terjemahan di butir NLLB-200 disertai catatan selama batch terjemahan masih mengirim
pepet bertanda "ê" ke model (`TranslationService::BATCH_KEEPS_PEPET_MARK`): ejaan itu tidak dikenal model, jadi chrF
dan BLEU-nya kemungkinan terlalu rendah sampai batchnya dijalankan ulang. Angka tingkat tutur dihitung pada label
yang sama dengan Ringkasan (`LineAnnotation::scorableSpeechLabels()`), dan label yang belum bisa dinilai disebut
jumlahnya.

**Dua pengaman impor kartu** (`App\Services\CardImporter`, dasar `DatasetImporter` dan `MethodImporter`): (1) bentuk:
tiap kunci di `shape()` wajib ada dan bertipe benar (notasi titik, `*` = setiap butir; tipe `number`, `count` =
bilangan tidak negatif yang boleh pecahan (rata-rata, jarak), `int` = bilangan bulat tidak negatif (jumlah baris,
langkah, lapis, parameter), `share` = bilangan dari 0 sampai 1 (porsi dan peluang), `string`, `text` = teks tidak
kosong, `bool`, `list`, `map`; akhiran `?` = boleh tidak ada atau `null`, tetapi harus bertipe benar bila ada), dengan
pesan yang menyebut kuncinya. Isi daftar yang dicetak apa adanya (kode karakter yang tidak ada di sebuah font, catatan
font, calon peran dataset) dan angka data pendukung ikut bertipe. Ditambah pemeriksaan khusus tiap kartu (`check()`:
urutan bagian, pasangan pengaturan, bukti yang harus punya sumber, kunci kelompok dan butir yang tidak boleh berulang
atau memakai kunci butir milik web);
(2) uji tampil: sebelum kartu lama diganti, halaman dirender sekali dengan kartu calon di dalam transaksi, dan impor
dibatalkan bila render gagal karena apa pun. Kartu yang ditolak tidak pernah mengganti kartu yang sudah ada.
Pengaman ini menangkap kartu yang RUSAK; kartu yang bentuknya sah tetapi angkanya salah (disunting tangan) tetap
diterima, jadi kebenaran angka adalah tanggung jawab skrip ekspor dan test-nya.
Kode keluar: `aksara:datasets` dan `aksara:methods` gagal bila kartunya ditolak. `aksara:import` yang berhasil
mengimpor hasil OCR tetapi menolak sebuah kartu menulis "PERINGATAN" dengan alasannya dan tetap keluar dengan kode
sukses, karena hasilnya sudah tersimpan. Lain halnya galat database (`CardStorageException`, mis. tabel kartu belum
dimigrasi): itu bukan kartu yang ditolak, pesannya menunjuk ke `php artisan migrate`, dan `aksara:import` pun keluar
dengan kode gagal.

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
