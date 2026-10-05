# Aksara Jawa OCR — CRNN + CTC

## Baca dulu

Baca `PLAN.md` sebelum mengerjakan apa pun. Kerjakan **SATU fase per sesi**.
Jangan lanjut ke fase berikutnya sebelum kriteria lulus fase sekarang terpenuhi.

---

## Environment mesin ini

Terverifikasi 2026-09-13. Berbeda dari asumsi PLAN.md §4.1 — perhatikan.

| Item | Status | Konsekuensi |
|---|---|---|
| OS | **Windows 11 native** (bukan WSL2) | PLAN.md §4.1 menyarankan WSL2 — tidak perlu |
| CPU | Intel Core Ultra 7 165U, 14 thread, 32 GB RAM | training CPU ~300 ms/sampel (model 4.6M, lebar 1200) |
| GPU | **tanpa NVIDIA**; Intel iGPU dipakai lewat PyTorch **XPU** | ~140 ms/sampel pada batch 32 — ~2x CPU |
| `.venv` | Python 3.12, `--system-site-packages` (torch 2.10 **cpu**), Pillow 11.3 RAQM=True, uharfbuzz | korpus, tokenizer, test |
| `.venv-xpu` | torch 2.14 **xpu** + dependensi proyek, RAQM=True; **Pillow 12.3.0 / HarfBuzz 14.2.1, beda dari `.venv` (11.3.0 / 11.2.1)** | training. Render BasaJan berbeda antar lingkungan (lihat catatan font); cek shaping harus dijalankan dengan interpreter ini juga |
| Node.js 24 | `scripts/translit/` (honocoroko) | hanya untuk membangun korpus |
| `fonts/` | `NotoSansJavanese-Regular.ttf`, `TuladhaJejegOT-Regular.ttf` | font training |
| `C:/Windows/Fonts/javatext.ttf` | Javanese Text (Microsoft) | font uji G1/G2; ikut konsensus urutan visual; tidak dipakai merender data training, tetapi **sekeluarga dengan font training CarakanJawa** (lihat catatan font), jadi bukan lagi font yang belum pernah dilihat; lisensinya tidak mengizinkan redistribusi |

> **JANGAN jalankan `pip install --no-binary :all: Pillow`** yang ada di PLAN.md §4.4.
> Instruksi itu untuk kasus RAQM mati di Linux. Di mesin ini RAQM sudah aktif lewat
> wheel resmi. Rebuild dari source di Windows butuh toolchain C, akan gagal, dan
> merusak Pillow yang sudah benar.

Tetap **assert** RAQM di dalam kode. Statusnya benar hari ini, tapi assert-nya
melindungi dari environment lain dan dari regresi saat upgrade.

### Font — koreksi terhadap PLAN.md §4.5

- **Tuladha Jejeg: pakai versi OT, BUKAN versi asli.** Tuladha Jejeg asli
  (`TuladhaJejeg_gr.ttf` dari aksaradinusantara.com) hanya punya tabel **Graphite**
  (`Silf`/`Glat`/`Gloc`), **tanpa GSUB/GPOS**. RAQM/HarfBuzz tidak membaca Graphite:
  font itu dirender tanpa error tapi pasangan tidak terbentuk — kegagalan senyap.
  Yang dipakai: konversi OpenType `github.com/akufadhl/Tuladha-Jejeg-OT` (OFL 1.1).
- Bentuk wulu/suku/layar Tuladha OT yang tampak janggal (lengkung tinggi, sapuan
  diagonal) **sama** dengan Javanese Text Microsoft — itu gaya tipografis, bukan bug.
- Link PLAN.md `bennylin.github.io/jawa-fonts.html` sudah **404**.
- Font baru wajib punya tabel GSUB + GPOS. `scripts/verify_shaping.py` memeriksanya
  (dengan kontrol negatif: font Graphite asli harus ditandai MASALAH).
- **Cek otomatis Fase 0 (4 cek): GSUB/GPOS, glyph tersedia, pasangan menumpuk, wulu menempel di
  atas ka.** Cek wulu ditambahkan setelah Istaka & Nawatura lolos 3 cek pertama padahal wulu/layar
  tergeser ke kiri; kontrol negatifnya kini tertangkap (lebar 1,50x & 1,56x ka, batas 1,3x).
  Cek otomatis tetap TIDAK menangkap: artefak kotak (Budyamarsudi), susun-3 bertabrakan,
  sapuan terlalu lebar. Pemeriksaan mata tetap wajib.
- **Font tambahan (keputusan user 2026-09-13: boleh dipakai untuk merender data training riset
  non-komersial; file tidak disebarkan).** Sumber & status di `fonts/extra/SOURCES.md`:
  `fonts/extra/` 8 font tampak benar (diperiksa Claude; user mendelegasikan keputusan, **belum diperiksa mata manusia**; dipakai run `fase5_fonts`),
  `fonts/extra_review/` 4 meragukan, `fonts/extra_rejected/` 2 ditolak. Lembar:
  `out/shaping_extra/contact_sheet_*.png`, `out/shaping_review/contact_sheet.png`.
  Dipakai lewat `src.train --extra-fonts fonts/extra` (hanya data training; val tetap font inti).
  `verify_shaping.py --fonts-dir DIR --out DIR`.
- **Tiga cacat data training yang ditemukan 2026-10-03 (tinjauan independen, diverifikasi ulang), berlaku untuk
  SEMUA run `--extra-fonts` sejak `fase5_fonts`; belum diperbaiki supaya run fase7 tetap sebanding:**
  1. **BasaJan dirender TANPA shaping di lingkungan training.** `.venv-xpu` memakai Pillow 12.3.0 / HarfBuzz
     14.2.1, `.venv` Pillow 11.3.0 / HarfBuzz 11.2.1. Di `.venv-xpu` GSUB BasaJan tidak diterapkan: `ꦏ꧀ꦏꦏ`
     @64 px = 163x69 (pangkon terlihat, aksara berjajar) vs 100x71 di `.venv` (pasangan menumpuk). Sembilan font
     training lain dan javatext identik di kedua lingkungan. `verify_shaping.py`, lembar kontak, dan pytest
     berjalan di `.venv`, jadi kegagalan senyap ini tidak pernah terlihat. ~10% sampel sintetis menampilkan
     "C + pangkon + C" berjajar; label tetap cocok dengan citra (cacat tipografi, bukan derau label).
     Perbaikan nanti: keluarkan BasaJan atau samakan versi Pillow, dan jalankan cek shaping tiap font training
     dengan interpreter training di awal `src.train` (raise bila gagal).
  2. **javatext bukan lagi font di luar data training.** CarakanJawa (di `fonts/extra`) dan javatext punya
     advance yang sama pada 72 dari 73 aksara/angka/pada dan bentuk yang nyaris sama (IoU median 0,86): satu
     keluarga huruf. G1/G2 sejak `fase5_fonts` mengukur generalisasi di dalam keluarga itu, bukan ke font yang
     belum pernah dilihat. Sebut batasan ini saat melaporkan G1/G2; metrik sintetis baru sebaiknya memakai juga
     satu font di luar keluarga ini.
  3. **NewKramawirya:** pada lungsi sesudah pangkon digambar dengan glyph pada lingsa (substitusi kontekstual),
     jadi dua label untuk gambar yang sama di font itu (pangkon+lungsi ada di 12% baris train).
- **Lebar spasi font tambahan tidak seragam (terukur 2026-09-14, spasi/lebar ka @64px):** Noto 0,21,
  Tuladha OT 0,18, javatext 0,21; **GumregahNew 0,00** (celah kata hasil shaping 0,09),
  **abmAksaJawa Regular & Bold 0,10**, ARDemak 0,14, BasaJan 0,35. Pada font berspasi sempit, label
  memuat spasi yang tidak tampak di citra = derau label. Val bersih (font inti) `fase5_fonts` langkah
  500 = 0,82% vs 0,22% `fase5_quick`; 4 dari 5 contoh kesalahan = spasi hilang. Pulih sendiri: val
  0,27% di langkah 1000, 0,24% di 1298; G1 100 baris membaik 0,79% -> 0,41%. **Diperbaiki
  2026-09-14** (sesudah `fase5_fonts`/`fase5_core`; berlaku untuk ablasi): `SyntheticLines` selalu membuang
  spasi dari teks + label bila `space_ratio(font)` (celah kata hasil shaping / lebar ka) < `MIN_SPACE_RATIO`
  0,12 → hanya GumregahNew & abmAksaJawa Regular/Bold. Font inti & javatext tidak terpengaruh, jadi
  G1/G2 tetap sebanding; urutan angka acak dipertahankan.

---

## Cara menjalankan

```bash
python scripts/verify_shaping.py                 # Fase 0 -> out/shaping/contact_sheet.png (periksa MATA)
python -m src.corpus                             # Fase 2 -> data/splits, out/corpus_report.md (~10 menit)
python -m src.tokenizer                          # Fase 1 -> data/tokenizer.json, out/charset.md
python -m pytest                                 # semua test, termasuk round-trip 1 juta baris (~6 menit)
.venv-xpu/Scripts/python -m src.train --run overfit32 --overfit 32 --epochs 1000 --eval-every 25   # 4a
.venv-xpu/Scripts/python -m src.train --run base --train-lines 50000                              # 4b
.venv-xpu/Scripts/python scripts/ablation.py --init out/checkpoints/base/best_4b_step1500.pt --steps 600 --train-lines 100000 --extra-fonts fonts/extra --drop-space-prob 0.5 --real data/real/nusaaksara/labels.tsv                   # 5
.venv-xpu/Scripts/python -m src.evaluate CKPT --fonts C:/Windows/Fonts/javatext.ttf               # G1
python -m src.real split data/real/labels.tsv                                                     # 6
python -m src.infer CKPT baris.png
python -m src.gpt check                          # wadah API GPT (ablasi koreksi pasca-OCR); kunci di .env (template .env.example)
python -m src.charlm --order 5 --lines 300000 --drop-space-prob 0.5              # LM karakter dari split train (kondisi "kamus")
python scripts/beam_eval.py CKPT --lm data/charlm/o5_n300000_ds0.5.pkl --name N  # beam+LM vs greedy: disetel di dev sintetis, NusaAksara sekali
python scripts/export_results.py                 # kontrak data web -> out/results/ (~14 menit CPU dengan run fase6)
python scripts/compare_runs.py A B               # dua pipeline di out/results: CER + SK bootstrap baris/halaman, recall aksara langka & adeg-adeg -> out/compare/
python scripts/eval_rare.py --workers 2          # sintetis tertarget aksara langka (split test, font javatext, 3 kondisi x 3 checkpoint; ~1-2 jam CPU)
python scripts/eval_spacing.py --lines 301 --threads 4   # dosis-respons jarak antar suku kata (split val, javatext): spasi palsu per 100 batas suku kata; memakai cache
bash scripts/make_official.sh RUN                # evaluasi resmi G1/G2 10.000 baris untuk RUN, lalu ekspor + impor (OFFICIAL_RUN harus sudah RUN)
.venv/Scripts/python -m uvicorn src.serve:app --host 127.0.0.1 --port 8011   # layanan model untuk Demo
cd web && php artisan aksara:import && php artisan serve --port=8010                  # web UI (lihat web/README)
```

---

## Invariant yang TIDAK BOLEH diubah tanpa diskusi

Ini keputusan desain yang sudah diargumentasikan di PLAN.md §3, bukan preferensi gaya.

1. **`H = 96`.** BUKAN 32, bukan 64. Tumpukan pasangan butuh ruang vertikal.
   Semua tutorial CRNN memakai 32 — abaikan, tutorial itu untuk aksara Latin.
2. **Rendering WAJIB `ImageFont.Layout.RAQM`.** Assert-nya, jangan asumsikan.
   Tanpa RAQM shaping OpenType gagal **SECARA SENYAP** — gambar tetap keluar
   tanpa error, tapi pasangan tidak menumpuk dan taling di posisi salah.
3. **`blank = indeks 0`.** Charset 1-indexed.
4. **`input_lengths` dari lebar citra ASLI**, bukan lebar setelah padding.
   Ini bug CTC paling umum yang beredar di internet.
5. **`DOWNSAMPLE` harus konsisten dengan stride lebar CNN sebenarnya.**
   Verifikasi dengan assert pada output model, jangan asumsikan.
6. **Semua string NFC** sebelum encode dan sebelum hitung CER.
7. **Training pada urutan visual, decode mengembalikan ke urutan logis.**
   Jangan campur. CER selalu dihitung pada urutan logis.
8. **Aturan reorder diturunkan dari uharfbuzz**, jangan tulis manual dari ingatan.

---

## Keputusan implementasi (tambahan / penyimpangan dari PLAN.md)

- **Urutan visual (tokenizer).** HarfBuzz dijalankan di 3 font per suku kata.
  Pre-base = tanda yang pada **mayoritas** kemunculannya diletakkan sebelum aksara
  dasar: hasilnya **taling (U+A9BA) dan dirga mure (U+A9BB)**, masing-masing 100%;
  cakra hanya 2,6%, suku 0,01%. Urutan visual kanonis = pre-base di awal suku kata,
  sisanya logis. Jangan kembali ke "mayoritas per pola": cakra pada pasangan
  ditaruh font sebelum aksara dasar secara tidak konsisten antar konsonan, dan
  pendekatan itu membuat 144 ribu baris gagal round-trip.
- **Korpus**: Wikipedia Jawa snapshot 20231101 versi parquet (wikimedia/wikipedia
  di HuggingFace), bukan dump XML. sastra.org belum dipakai (lisensi per dokumen
  harus dicek manusia).
- **honocoroko punya cacat sistematis** yang ditambal di `scripts/translit/translit.js`:
  konsonan penutup kata tanpa pangkon (`taun` -> ꦠꦈꦤ), digraf th/dh/ng/ny terpecah
  bila tidak diikuti "a" (`kanthi` -> ꦠ꧀ꦲꦶ), dan `useSwara` default yang tidak baku.
  Filter round-trip TIDAK bisa menangkap cacat digraf karena arah balik ikut salah.
- **Test pasangan Fase 0**: PLAN.md memakai U+A9A5 berlabel "ta"; itu **PA**, dan
  pasangan ha/sa/pa ditulis di samping. Dipakai U+A9A0 (TA).
- **LSTM WAJIB packing** (`model(X, in_lens)`) saat training DAN evaluasi batch.
  Tanpa packing, frame padding tidak pernah diawasi CTC; model belajar memancarkan
  simbol di sana, lalu inference satu baris (tanpa padding) mengeluarkan "pada
  lingsa" palsu di frame terakhir. Terukur di overfit 32: tanpa packing CER 1,12%
  dengan 14 lingsa palsu; model yang sama dengan kondisi training ditiru 0,07%.
  Biaya packing ~20% di XPU (138 vs 117 ms/sampel). `--no-pack` hanya untuk eksperimen.
- **Ukur gerbang dengan inference satu baris**, bukan val batch saat training:
  hanya itu yang sama dengan pemakaian sebenarnya. Mode overfit memakai seed render
  yang sama untuk val (sebelumnya berbeda font -> CER mentok palsu).
- **Augmentasi** ditulis sendiri dengan PIL/numpy (`src/augment.py`), tanpa
  albumentations.
- **Fase 6**: `src/real.py` memvalidasi anotasi dan membagi split per `source_id`.
- **Celah domain data nyata (terukur di notebook):** 87% baris NusaAksara tanpa spasi,
  94% potongan dipotong rapat sampai tinta menyentuh tepi. Opsi yang tersedia, semuanya
  mati secara default supaya run lama & definisi G2 tidak berubah:
  `--drop-space-prob` (train), augmentasi `tight` (tidak masuk preset `heavy`/`train`;
  pakai eksplisit, mis. `--augment tight+blur`), `--pad-ratio` (evaluate/infer) dan
  `--real-pad-ratio` (train). Bandingkan `tight` vs `--pad-ratio` pada checkpoint yang layak
  sebelum memilih.
- **Normalisasi kontras di `to_tensor`:** latar (median) -> 0, tinta (persentil 1) -> 1, per
  citra. Tanpa ini, model 4b (latar selalu persis 0 saat training) jatuh dari CER 0,9% ke 74,5%
  hanya karena kertas abu-abu. Render bersih tidak berubah (G1 100 baris 0,81% -> 0,79%).
- **G3 & NusaAksara (keputusan user 2026-09-13): NusaAksara tetap HANYA test.** Halaman 0–86
  berasal dari satu majalah (tipografi sama), jadi split per halaman akan membocorkan font ke test.
  Analisis checkpoint fase5 langkah 500 pada 745 baris: CER literal 70,3%; bila spasi dibuang dari
  keluaran 54,0% (label umumnya tanpa spasi; `evaluate.py` mencatatnya sebagai `cer_hyp_no_space`,
  bukan angka gerbang); potongan bersih 45,8%, bertekstur raster 72,8%; ejaan klasik (murda,
  pa cerek, dll.) hanya 0,3 poin; filter median saat inferensi 1–4 poin. Celah utama = bentuk huruf
  cetak, bukan derau -> lever: font tambahan, lalu data nyata non-NusaAksara (Commons terverifikasi).
  Terbukti 2026-09-14: `fase5_core` vs `fase5_fonts` (beda hanya 8 font tambahan) G3 745 baris 66,6% vs 37,2%.
- **VLM vs CRNN (keputusan user 2026-09-14): CRNN+CTC tetap sistem utama; VLM hanya alat bantu**
  (kandidat pengoreksi/guru; VLM zero-shot TIDAK disarankan sebagai pembuat label draf setelah uji buta
  di bawah, karena pembaca aksara bisa terjangkar ke draf yang 91% salah). Alasan: metrik
  CER karakter menuntut transkripsi setia piksel (VLM bisa mengarang/"membetulkan" teks), aksara Jawa
  langka di data VLM, fine-tune VLM tidak layak di laptop ini, dan celah G3 sudah terbukti bisa
  diturunkan lewat font (66,6% -> 37,2%). **Tinjau ulang** bila VLM tanpa fine-tune mencapai CER
  < ~20% pada baris nyata yang sama tempat CRNN 37% tanpa halusinasi, dan hasilnya bertahan pada data
  yang pasti belum dilihat model (mis. papan Commons terverifikasi), karena NusaAksara mungkin sudah
  masuk data pretraining. **Uji buta 2026-09-14** (50 baris NusaAksara acak seed 0, VLM frontier
  zero-shot tanpa melihat label): VLM CER 91,0% (0/50 persis, 17 baris CER ≥100%, keluaran runtuh ke
  sedikit aksara, mis. ka 204x vs 63 di referensi) vs CRNN `fase5_fonts` 38,4% (tanpa spasi 28,0%;
  margin 0,06: 34,6%); CRNN lebih baik di 47 baris. Literatur terverifikasi sejalan: NusaAksara (ACL
  2025) CER >1 untuk GPT-4o dkk. pada OCR baris Jawa; GlotOCR Bench (2026) 0% baris CER≤5% untuk
  VLM open. Pola kesalahan CRNN yang terlihat (untuk Fase 5/6): adeg-adeg di tepi kiri potongan
  hilang (0/10, terbaca taling/suku), aksara murda/nga lelet tertukar angka Jawa, cecak↔layar,
  deretan suku kata bertaling runtuh jadi taling bertumpuk, potongan bertekstur raster kehilangan
  aksara dasar.
- **Ablasi koreksi pasca-OCR (keputusan user 2026-09-14; BELUM dikerjakan, setelah ablasi Fase 5).**
  Beam search + n-gram LM ada di PLAN.md §3; korektor LLM = perluasan yang disetujui user. Semua
  kondisi memakai keluaran checkpoint CRNN yang SAMA; angka greedy selalu dilaporkan di sampingnya.
  - Kondisi: greedy; kamus = beam + LM **karakter** dari split train (87% label nyata tanpa spasi,
    jadi bukan kamus kata); kontrol transliterasi bolak-balik tanpa LLM; lalu tiap LLM tanpa & dengan
    kamus. "Hybrid" = kandidat LLM dinilai ulang dengan skor CTC dari logit CRNN + LM karakter, supaya
    LLM tidak "membetulkan" teks yang benar.
  - LLM lokal, **semuanya dibandingkan** (efek CPT dan base vs instruct), **dijalankan di laptop**
    (cloud nanti, setelah semua hasil ablasi ada): Gemma-2-9B base, Gemma-2-9B-it,
    `GoToCompany/gemma2-9b-cpt-sahabatai-v1-base`, `...-v1-instruct`. Model card (dicek 2026-09-14):
    gemma-2-9b → Gemma-SEA-LION-v3-9B → Sahabat-AI base (CPT 50B token, Jawa 1,5B) → instruct;
    aksara Jawa tidak disebut (data Jawa kemungkinan Latin); Gemma Community License; konteks 8192.
    Kuantisasi harus sama untuk keempatnya. Base butuh prompt few-shot, instruct memakai format chat:
    format prompt ikut berbeda, catat sebagai pengganggu perbandingan. Cara menjalankannya di laptop
    belum dibuat.
    Kecepatan 9B di laptop ini belum diukur.
  - GPT: model dipilih belakangan (opsi per 2026-09-14: `gpt-5.6-sol`, `gpt-5.6-luna`,
    `gpt-6-astra`; tanpa snapshot bertanggal, dukungan `temperature` tidak terdokumentasi).
  - Wadah API **khusus GPT** (keputusan user): `src/gpt.py`, kunci `OPENAI_API_KEY` dari environment
    variable atau `.env` (tidak pernah ditulis ke log/cache; `.env` ada di `.gitignore`), Responses
    API, cache respons `out/gpt_cache/` berisi nama model dari server + waktu. `python -m src.gpt check`.
  - Protokol: prompt/bobot disetel di dev set bukan NusaAksara (val sintetis augmentasi `fase5` yang
    dibaca CRNN + Commons terverifikasi); NusaAksara dipakai sekali di akhir. Metrik: CER, CER tak
    peka spasi, **tingkat koreksi berlebih** (karakter benar diubah jadi salah). Cek dulu lisensi
    NusaAksara sebelum mengirim teks/citranya ke API.
- **Beam search + LM karakter ("kamus") — hasil pertama 2026-09-14** (`out/beam/fonts_o5.json`,
  checkpoint `fase5_fonts`, LM order 5 Witten-Bell dari 300.000 baris train, drop-space 0,5, beam 16).
  alpha/beta dipilih dari 24 kombinasi di dev (300 baris val, font training + `fase5`): **alpha 0,25,
  beta 1,0**; alpha ≥ 1 merugikan. Dev 3,25% → 2,34% (beam tanpa LM saja 3,07%). Render bersih font
  held-out (300 baris test): 0,38% → 0,24% (14 baris membaik, 2 memburuk). **NusaAksara 745 baris, sekali:
  greedy 37,25% → beam 33,83%** (tanpa spasi 25,22% → 22,40%); 413 baris membaik, **157 memburuk**
  (koreksi berlebih nyata, belum dianalisis). Greedy di skrip = angka resmi persis (cek kewajaran).
  Bukan angka gerbang resmi: jalur resmi `src/evaluate.py` tetap greedy sampai diputuskan.
- **OCR + VLM dan web UI (keputusan user 2026-09-24).**
  - **Dua pembaca, satu hakim:** CRNN tetap pembaca utama (offline, cepat) dan "hakim piksel"; VLM
    fine-tune = pembaca kedua yang menyumbang kandidat. Semua kandidat (greedy, N-best beam, VLM, LLM)
    dinilai ulang dengan log P_CTC(teks | citra) dari CRNN + alpha * LM karakter, lalu dipilih yang
    tertinggi, supaya VLM tidak bisa mengarang teks yang tidak didukung piksel. VLM zero-shot tidak
    dipakai (uji buta 91%). Jalur resmi tetap CRNN sampai gabungan terbukti lebih baik di dev.
  - **Pola pipeline ala Docling, tanpa dependensi Docling:** praproses -> pembaca (CRNN greedy | CRNN
    beam+LM | VLM | GPT, bisa diganti lewat konfigurasi) -> penggabung (tidak ada | CTC+LM rescoring) ->
    pascaproses (NFC, urutan logis). Setiap hasil mencatat nama pipeline & versi model. Layout/reading
    order/tabel Docling TIDAK dipakai (di luar scope; nanti proyek terpisah: halaman -> layout -> potongan
    baris -> engine aksara ini). Kandidat VLM: GraniteDocling-258M / SmolDocling-256M sebagai baseline
    zero-shot di CPU dan kandidat fine-tune di iGPU (kecepatan belum diukur); VLM besar di cloud/API.
    **Ditinjau ulang 2026-10-02 (user bertanya "baiknya pakai Docling?", lalu setuju): tetap TANPA Docling.**
    Alasan (dari pengetahuan umum, belum diuji di mesin ini): mesin OCR bawaannya (EasyOCR, Tesseract, RapidOCR)
    tidak membaca aksara Jawa; model tata letaknya memberi blok, bukan potongan baris, jadi pendeteksi baris
    tetap harus dibuat sendiri; dan dengan G3 37% keluaran halaman utuh belum berguna. Docling tidak mengubah
    angka CER. Tinjau lagi bila G3 mendekati target ATAU user memutuskan produk akhir menerima halaman utuh
    (itu perluasan scope); langkah pertamanya uji tata letak Docling pada 3–5 halaman cetak beraksara Jawa,
    dengan CRNN dipasang sebagai plugin OCR.
  - **Web UI di folder `web/`** (keputusan user 2026-09-24: satu folder dengan repo ini, bukan repo
    terpisah). Laravel 12 + starter kit Livewire (Flux, Volt), PHP 8.4, SQLite; Tailwind 4, Chart.js;
    webfont aksara hanya Noto Sans Javanese. Navigasi = **sidebar** (Hasil: Ringkasan, Perbandingan,
    Ablasi; Analisis: Penjelajah baris, Kesalahan aksara; Alat: Demo, Kamus). Mockup disetujui user 2026-09-24:
    <https://claude.ai/artifact/9NY5nf3RwJfJQpT479Sibp>. Lokal dengan login (citra NusaAksara
    non-komersial). Python menghitung semua angka OCR; Laravel hanya menampilkan (`php artisan
    aksara:import` dari `out/results/`) dan memanggil `src/serve.py` (FastAPI di `.venv`, 127.0.0.1) untuk demo.
    **Gaya & desain (permintaan user 2026-10-05): mengikuti konsep proyek rujukan
    `C:\Users\itsupport\Documents\Apps\Merdeka\Base-Apps-Merdeka`** (template admin Tabler + palet biru/abu:
    sidebar terang dengan item aktif biru `#3b82f6`, bilah atas putih, kartu putih radius 4 px di latar `#f6f8fb`,
    Inter 13 px). Mockup 2026-09-24 hanya menggambarkan isi dan navigasi, bukan lagi gayanya. Stack tetap Tailwind
    4 + Flux: token dan kelas komponen di `web/resources/css/app.css` (rincian di `web/README.md` bagian
    Tampilan); halaman baru memakai kelas itu. Hanya konsep visual yang diikuti: logo, nama, dan foto perusahaan
    rujukan tidak disalin (repo ini publik), dan proyek rujukan hanya dibaca. Setelah mengubah CSS/JS/kelas di
    view wajib `npm run build`. Halaman dipotret tanpa login lewat salinan statis (tinker merender HTML, Edge
    tanpa kepala memotret); dua jebakannya: `app('livewire')->flushState()` harus dipanggil di antara dua render
    dalam satu proses (kalau tidak, hanya halaman pertama yang mendapat skrip Livewire sehingga grafik kosong), dan
    jendela Edge tidak bisa lebih sempit dari ~500 px (tampilan ponsel dipotret lewat iframe 390 px).
    **Menu Kamus (permintaan user 2026-10-05, "isinya mengenai kamus/dictionary"; isinya keputusan Claude, belum
    dikonfirmasi):** `web/app/Livewire/Pages/Kamus.php` menampilkan tiga rujukan yang memang ada di alur: kamus
    aksara (charset `data/tokenizer.json`, lewat `App\Support\AksaraCatalog`; bacaan Latin = draf Transliterator),
    kamus koreksi OCR (kondisi "kamus" = beam + LM karakter; angka greedy vs beam diambil dari metrik yang diimpor,
    dengan peringatan bila checkpoint-nya bukan pipeline resmi), dan leksikon tingkat tutur (`SpeechLevel::lexicon()`).
    Tidak ada kamus kata Jawa-Indonesia: datanya tidak ada di proyek ini, jadi tidak dibuat-buat.
    **Foto halaman masuk (user 2026-10-05: "kamu saja yang cari dan pilihkan gambarnya", konsep sama dengan
    rujukan):** korsel tiga foto di panel kiri + latar foto gelap, semuanya dari Wikimedia Commons berlisensi bebas
    dan dipilih Claude: naskah beraksara Jawa di Museum Sonobudoyo (CC0), halaman Serat Damar Wulan British Library
    MSS Jav 89 (domain publik), gerbang Donopratono Kraton Yogyakarta (Chainwit., CC BY 4.0: atribusi WAJIB tetap
    ada di keterangan foto dan `web/public/img/login/SUMBER.md`). Dibuat ulang lewat `web/tools/login_images.py`.
    **Lambang aplikasi = gambar pilihan user (2026-10-05: "ubah logonya jadi ini saja"; lambang SVG buatan Claude
    ditolak):** sumber `web/resources/brand/logo.png` (ikon 1254 px: keris di atas atap joglo, bulir padi, ombak,
    sebaris tulisan bergaya aksara Jawa). `web/tools/logo.py` menurunkan `public/img/logo/logo.png` (ubin utuh,
    halaman masuk), `public/img/logo/mark.png` (lambang saja, sidebar), `favicon.ico`, `favicon-192.png`,
    `apple-touch-icon.png`; jangan diganti tanpa permintaan user. Tulisan di gambar itu buatan pembuat gambar,
    bentuknya tidak sama dengan `ꦲꦏ꧀ꦱꦫ ꦗꦮ` hasil font, dan belum diperiksa pembaca aksara (sudah disampaikan ke user).
  - **Database web: PostgreSQL 18** (keputusan user 2026-09-25; layanan Windows `postgresql-x64-18` sudah
    terpasang, port 5432). Database **`its_aksara`** (dibuat user 2026-10-02; UTF8), pengguna `postgres`;
    kredensial HANYA di `web/.env` (diabaikan git), jangan ditulis di file lain. Database uji
    `its_aksara_test` (dibuat Claude) untuk menjalankan test di PostgreSQL: `php artisan test -c phpunit.pgsql.xml`.
    JANGAN arahkan test ke `its_aksara` (RefreshDatabase mengosongkan database). Pengaman (2026-10-03):
    `phpunit.xml` memaksa SQLite in-memory (`force="true"`, jadi `DB_*` yang tertinggal di shell diabaikan) dan
    `tests/TestCase.php::createApplication` menolak database bernama `its_aksara` sebelum migrasi. Test memakai folder
    sementara untuk `storage/app/mt` (dulu `TranslationTest` menimpa hasil NLLB; dipulihkan dari DB lewat
    `php artisan aksara:translate --dump`). 77 test lolos di SQLite (2026-10-05, sesudah restyle dan menu Kamus).
    **Port:** Laravel 8010, layanan model 8011, layanan terjemahan 8012 — 8000/8001 dipakai proyek lain
    milik user di laptop yang sama (jangan dihentikan). Test tetap bisa di SQLite in-memory
    (`phpunit.xml`); kode dijaga netral: `whereJsonContains` untuk tag, tabel turunan untuk ORDER BY berekspresi
    (PostgreSQL menolak alias kolom di dalam ekspresi). Isi DB kecuali akun & label manusia bisa dibangun ulang
    (`aksara:import`, `aksara:annotations`, `aksara:translate --import-only`); label tingkat tutur dicadangkan
    ke `web/database/labels/speech_levels.json` lewat `php artisan aksara:labels export|import`.
  - **Kontrak data (skema 1):** `scripts/export_results.py` -> `out/results/manifest.json` (pipeline,
    gerbang, metrik, ablasi, kesalahan aksara), `lines.jsonl`, `predictions.jsonl` (teks, CER, segmen beda
    per suku kata dari `src/align.py`, kolom citra per suku kata dari alignment CTC). Skrip berhenti kalau
    angka 745 baris beda dari laporan resmi. Uji buta VLM tersimpan di `out/eval/vlm_blind_50.json`.
  - **Angka resmi = satu run, ditetapkan di satu tempat (2026-10-04).** `OFFICIAL_RUN` di
    `scripts/export_results.py` menentukan pipeline resmi: gerbang G1–G3 di manifest dibaca dari laporan
    `out/eval/<run>_G1_10k.json`, `_G2_10k.json`, `_G3_full.json` (run lama `fase5_fonts` memakai awalan
    `fonts_`), tabel kesalahan aksara dihitung dari pipeline itu, dan manifest membawa kunci `"official"`.
    Ekspor berhenti bila sebuah laporan tidak ada, bukan milik checkpoint run itu, lebih tua dari
    `last_snapshot.pt`-nya, atau bukan evaluasi yang dijanjikan (G1/G2: 10.000 baris split test, javatext saja,
    bersih / `heavy`; G3: semua baris nyata tanpa margin; tes cepat q100 ditolak). Web menyimpan tandanya di
    `pipelines.official` dan membacanya lewat `Pipeline::official()` (manifest lama tanpa kunci itu =
    `crnn_fonts`); tidak ada lagi nama pipeline resmi yang tertanam di halaman. Demo: `DEFAULT_CHECKPOINT` di
    `src/serve.py` harus run yang sama (dijaga `tests/test_export_results.py`); bawaan Demo = greedy, karena
    bobot LM beam disetel pada `fase5_fonts` dan belum diuji ulang. **Mengganti run resmi:**
    `bash scripts/make_official.sh RUN` (evaluasi G1/G2 10.000 baris, ~30 menit per gerbang di CPU; menunggu
    training selesai dulu), ubah `OFFICIAL_RUN` + `DEFAULT_CHECKPOINT`, lalu ekspor dan impor.
  - **Seluruh alur = scope proyek (keputusan user 2026-09-24):** OCR -> transliterasi -> arti (Indonesia)
    -> tingkat tutur (ngoko / madya / krama / campur). Transliterasi & arti **manusia** untuk 745 label
    tersedia dari NusaAksara (config `Image Transliteration` / `Image Translation`, cocok 745/745, lisensi
    non-komersial sama). Transliterasi draf = aturan deterministik aksara -> Latin di `web/`, berlabel
    "bantu baca, bukan hasil OCR"; **CER draf vs transliterasi manusia 7,4% pada 745 label** (huruf saja,
    2026-09-24; kelemahan diketahui: ꦲ tengah baris selalu "h", geminasi ditulis ganda, tanpa spasi kata).
    **Tahap 3:** NLLB-200-distilled-600M lokal (`web/tools/`, CC-BY-NC), `php artisan aksara:translate`,
    dinilai chrF/BLEU (sacrebleu) terhadap terjemahan manusia, dari transliterasi manusia DAN dari keluaran
    OCR; layanan Demo port 8012. Hasil 2026-09-25 (745 baris): dari transliterasi manusia **chrF 36,2 / BLEU
    10,5**; dari keluaran OCR beam+LM -> Latin draf chrF 19,4 / BLEU 1,0 (NLLB sering menyalin teks rusak apa
    adanya). ~1 jam CPU untuk 1.490 kalimat; `--import-only` mengimpor ulang tanpa menerjemahkan. **Tahap 4:** leksikon penanda ngoko/madya/krama + afiks krama
    (`web/app/Support/SpeechLevel.php`) dengan kata bukti; label manusia diisi di Penjelajah (data uji),
    akurasi di Ringkasan. Pada 745 transliterasi manusia (2026-09-24): tak tentu 328, ngoko 255, krama 155,
    madya 6, campur 1; rata-rata 1 penanda per baris, jadi per baris lemah (potongan baris, bukan kalimat). Setiap tahap
    ditampilkan dari label DAN dari keluaran OCR, supaya error OCR tidak terbaca sebagai error tahap lain.
- **Karakter langka hampir tak terlihat saat training (terukur di split train 2026-09-14, tanpa
  melihat label test).** 44 codepoint (semua murda, mahaprana, swara, pa cerek, nga lelet, rerenggan,
  pada adeg/adeg-adeg, dll.) masing-masing hanya di **100 baris** (0,011%; hasil `inject_rare`), jadi
  ~11 kali per run `--train-lines 100000`; adeg-adeg **tidak pernah** di awal baris. Cocok dengan uji
  buta (adeg-adeg tepi kiri 0/10, murda tertukar angka). `src/text_augment.py` `RareText` menyisipkan
  ≤1 codepoint langka (`insert_prob`) dan adeg-adeg pembuka baris (`opener_prob`) SEBELUM render; syarat:
  well-formed, round-trip tokenizer, dan glyph ada di cmap font (dibaca uharfbuzz; fontTools tidak ada di
  `.venv-xpu`). `insert_codepoint` pindah dari `src/corpus.py` ke sini. **Disambungkan 2026-09-24**
  (`src.train --rare-insert-prob --rare-opener-prob`; probabilitas 0 tidak mengonsumsi angka acak, sisipan
  sebelum `drop_space`). Terukur di 745 baris NusaAksara (`scripts/compare_runs.py`): semua checkpoint
  sampai `fase5_fonts` (greedy maupun beam+LM) membaca benar **0 dari 495** karakter langka (221 murda) dan
  **0 dari 112** adeg-adeg pembuka baris, dan tidak pernah mengeluarkan satu pun dari 44 codepoint itu;
  21 dari 44 codepoint tidak muncul di referensi. Efek langsung maksimal aksara langka pada G3 = 2,4 poin
  (adeg-adeg saja 0,55 poin).
  - **Eksperimen `fase6_rare` vs `fase6_ctrl`** (`scripts/run_fase6_resume.sh`, `scripts/run_fase6_ctrl.sh`):
    `fase5_fonts` + 1.500 langkah (lr 3e-4, augmentasi `fase5`, drop-space 0,5, 10 font) dengan dan tanpa
    `--rare-insert-prob 0.3 --rare-opener-prob 0.15`. `LengthBucketSampler` mengulang urutan batch dari awal
    di tiap proses (epoch tidak disimpan di checkpoint), jadi run yang dilanjutkan melihat ulang baris awal:
    `fase6_rare` dilanjutkan di langkah 500 dan 884, dan `fase6_ctrl` dihentikan/dilanjutkan di langkah yang
    sama lewat `--stop-step` (segmen gagal diulang dari `seg_<langkah>.pt`). Kedua lengan melihat 19.712
    baris unik dari 48.000 sampel. Render (font, ukuran, augmentasi, drop-space) tetap acak per run, jadi
    citra kedua lengan TIDAK berpasangan dan derau antar-run belum terukur: selisih G3 keseluruhan < ~2–3
    poin tidak bisa dibedakan dari derau. **Titik akhir utama = recall aksara langka & adeg-adeg pembuka**
    (dan adeg-adeg palsu), lalu CER baris tanpa aksara langka sebagai efek samping. Asal nilai pembuka 0,15
    tidak terdokumentasi dan kebetulan sama dengan proporsi baris berawalan adeg-adeg di test (112/745), yang
    sudah terlihat di ekspor web saat itu: anggap kemungkinan penyetelan pada test set; setel run berikutnya
    di evaluasi sintetis. Efek samping yang diketahui: pada windu (U+A9C6) disisipkan sebagai token lepas
    dan glyph-nya identik dengan angka nol di Noto & ARDemak, sehingga `fase6_rare` membaca nol tunggal
    sebagai pada windu di val; homoglif lain: E ↔ angka enam, pa murda ↔ angka delapan, nga lelet ↔ angka dua.
    **`fase6_rare` selesai 2026-10-03 05:24** (val bersih 0,197%): G3 745 baris **34,06%** (pad 0,06:
    32,17%; spasi dibuang 22,70%), q100 G1 0,32%, G2 1,22%; dasar `fase5_fonts` 37,25%. **`fase6_ctrl` selesai
    2026-10-03 13:31** (val bersih 0,216%): G3 **32,17%** (pad 0,06: 30,43%; spasi dibuang 22,89%), q100 G1 0,27%,
    G2 0,95%.
  - **Hasil `fase6_rare` − `fase6_ctrl` (2026-10-03; `out/compare/crnn_fase6_rare_vs_crnn_fase6_ctrl.md`,
    `out/compare/rare_synthetic.md`).** Sisipan aksara langka **berhasil untuk tujuannya**: recall 44 codepoint di
    745 baris nyata 42,8% vs 0% (SK halaman [+36,3; +50,3]), murda 20,8% vs 0%, adeg-adeg pembuka 103/112 (92,0%)
    vs 0; sintetis javatext 2.000 baris: recall 73,7% (bersih) dan 59,3% (augmentasi `fase5`) vs 0%, pembuka
    100% vs 0%, teks biasa tanpa efek samping (CER 0,33% vs 0,32%; `fase5_fonts` 0,42%). **Efek samping pada
    cetakan nyata:** presisi aksara langka hanya 39,6% (536 dikeluarkan, 212 benar; 154 di baris yang tidak punya
    aksara langka), adeg-adeg palsu 24/633 baris (3,8%), dan aksara umum ikut tertukar: dda → da mahaprana 70x
    (recall dda 38,1% → 21,8%), ra → pada isen-isen 28x, taling/tarung → tolong 17x (recall tarung 86,6% → 75,6%),
    sa murda → ba murda 11x. Lengan rare juga mengeluarkan lebih banyak spasi (2.636 vs 2.168; referensi hanya
    110). Akibatnya G3 literal rare **lebih buruk 1,89 poin** dari kontrol (SK halaman [+1,06; +2,89]; 409 baris
    tanpa aksara langka +2,79 poin), sedangkan tanpa spasi setara (−0,19 [−0,65; +0,35]). Satu run per lengan:
    derau antar-run belum terukur. **Kesimpulan menurut aturan di bawah:** (1) terpenuhi, (2) ada efek samping
    nyata di cetakan → sisipan dipertahankan untuk run berikutnya HANYA setelah diperbaiki: lewati codepoint yang
    glyph-nya di font terpilih identik/nyaris identik dengan codepoint lain (cek bitmap), jangan menambah spasi di
    sekitar sisipan/pembuka, dan setel peluangnya di evaluasi sintetis. Dasar run berikutnya = `fase6_ctrl`
    (lebih baik dari `fase5_fonts` juga di sintetis berpasangan: teks biasa −0,10 poin [−0,14; −0,07]).
  - **Spasi = galat terbesar yang tersisa di G3 (terukur 2026-10-03):** `fase6_ctrl` mengeluarkan 2.168 spasi,
    referensi hanya 110; 91% di antara aksara (bukan di samping pada), jadi model membaca renggang antar-aksara
    cetakan sebagai spasi kata. Itu ~9,3 poin dari 32,17%. Kandidat: augmentasi jarak antar-aksara (tracking)
    dan lebar spasi yang lebih bervariasi; `--drop-space-prob 1.0` akan menggagalkan G1/G2 (teks sintetis berspasi).
  - **Fase 7 (dijalankan 2026-10-03 22:12, `scripts/run_fase7.sh`, log `out/fase7_chain.log`):** dua run 1.500
    langkah dari `fase6_ctrl` (lr 3e-4, augmentasi `fase5`, drop-space 0,5, 10 font).
    `fase7_track` = `--track-prob 0.5 --track-max 0.3`: baris dirender per suku kata dengan jarak tambahan
    seragam per baris (0–0,3 em; `src/render.py::syllable_origins` mengambil posisi dari tata letak baris utuh,
    jadi tanpa jarak tambahan citranya identik dengan render biasa di 9 dari 11 font, beda < 0,2% tinta di
    GumregahNew & NewKramawirya). Spasi kata menjadi lebar spasi + 2x jarak, antar-aksara = jarak.
    `fase7_track_rare` = sama + `--rare-insert-prob 0.3 --rare-opener-prob 0.15 --rare-max-similarity 0.9
    --rare-attach`: aksara langka yang glyph-nya di font terpilih punya IoU ≥ 0,9 dengan glyph codepoint
    SEJENIS dilewati, dan pada/pangrangkep sisipan ditempel ke kata sebelumnya tanpa spasi tambahan (attach
    hanya mengubah ~2,6% baris). Run kedua meniru titik lanjut run pertama. Saringan diperbaiki 2026-10-03
    23:20, sebelum run kedua mulai, setelah tinjauan independen (`src/text_augment.py::glyph_similarity`):
    (i) codepoint dipilih dulu baru disaring, jadi jatah yang dilewati tidak pindah ke codepoint lain
    (versi pertama menaikkan 21 codepoint lain 22%); (ii) sandhangan dibandingkan tanpa tinta ka (versi pertama
    menganggap cecak mirip "tanpa tanda"); (iii) pasangan aksara-angka tidak disaring (E = angka enam, pa murda
    = angka delapan, ga/la/ya = angka 1/7/9 kembar di banyak font termasuk javatext, dan model membedakannya
    dari konteks; pada windu = angka nol tetap disaring karena keduanya token lepas). Hasil di lingkungan
    training: 38 dari 439 pasangan (font, codepoint langka) tersaring, 0–12 per font, tidak ada codepoint yang
    tersisa di ≤ 3 font, adeg-adeg selalu ikut (IoU terbesar 0,50). Da mahaprana ~ dda hanya tersaring di 3 font
    (abmAksaJawa Regular/Bold, GumregahNew: 0,94; CarakanJawa 0,897, Tuladha 0,87, nyk 0,86 lolos), jadi
    jangan berharap kebingungan dda → da mahaprana hilang. Batasan: ambang 0,9 dipilih SETELAH melihat
    kebingungan di test set dan tidak punya celah alami (dataran baru di ≥ 0,97 = kembar persis); keputusan di
    tepi ambang berubah menurut ukuran render (18 dari 439 pada 56–72 px); saringan dan attach tidak terpisah.
    G3 run ini bersifat eksploratif; titik akhir utamanya `scripts/eval_rare.py`.
    **Run ketiga `fase7_ctrl`** (`scripts/run_fase7_ctrl.sh`, menunggu rantai selesai): sama dengan
    `fase7_track` tanpa tracking, titik lanjut ditiru, supaya efek tracking terpisah dari efek +1.500 langkah.
  - **Aturan analisis fase 7, ditetapkan 2026-10-03 SEBELUM ada angka:** (a) *Tracking*: titik akhir utama =
    spasi berlebih di 745 baris (jumlah spasi keluaran; `fase6_ctrl` 2.168 vs 110 di referensi) dan CER
    literal `fase7_track` − `fase6_ctrl` dengan SK halaman; "membantu" hanya bila SK tidak memuat 0, CER tanpa
    spasi tidak memburuk, dan evaluasi sintetis teks berjarak (javatext, belum dibuat) searah; G1/G2 dan teks
    biasa sintetis tidak boleh memburuk. Perbedaan `fase7_track` vs `fase6_ctrl` memuat tracking, +1.500
    langkah, siklus LR baru, dan teks baru sekaligus, jadi efek tracking pada G3 dibaca dari `fase7_track` −
    `fase7_ctrl` (SK halaman). Titik akhir utama tracking = **dosis-respons sintetis**: spasi palsu per 100
    batas suku kata pada baris val tanpa spasi yang dirender berjarak. Dasar terukur 2026-10-03 (8 baris,
    javatext; model tanpa tracking): 0 / 54–64 / 91–93 pada jarak 0 / 0,15 / 0,30 em untuk `fase5_fonts`,
    `fase6_ctrl`, `fase6_rare` (+1.500 langkah tanpa tracking tidak mengubahnya); pada 150 citra bersih
    `fase6_ctrl`: CER 0,29% → 15,2% → 37,1% sedangkan CER tanpa spasi datar 0,32%. Gejala G3 tereproduksi di
    sintetis. Skrip evaluasinya (≥ 300 baris, berspasi & tanpa spasi, 0–0,6 em, + recall spasi asli) dibuat
    sebelum membaca hasil. (b) *Aksara langka diperbaiki*:
    `fase7_track_rare` − `fase7_track` pada recall 44 codepoint & adeg-adeg pembuka (harus tetap > 0, SK
    halaman), presisi keluaran aksara langka (fase6_rare 39,6%), recall dda (fase6: 38,1% → 21,8%), pembuka
    palsu (3,8%), dan CER 409 baris tanpa aksara langka. (c) G3 keseluruhan dilaporkan dengan SK sebagai arah;
    checkpoint dasar berikutnya dipilih dari (a)–(b) dan sintetis, bukan dari G3.
  - **Run kontrol `fase7_ctrl` selesai 2026-10-04 14:13** (val bersih 0,183%; terbaik 0,164% di langkah 1000):
    G3 **30,75%** (pad 0,06: 29,32%; spasi dibuang 21,28%), q100 G1 0,27%, G2 0,81%. Jadi +1.500 langkah tanpa
    tracking hanya menurunkan G3 1,42 poin dari `fase6_ctrl` (32,17%), sedangkan `fase7_track` 21,46%: **efek
    tracking ≈ −9,3 poin** (`fase7_track` − `fase7_ctrl`), hampir seluruhnya dari spasi (CER spasi dibuang 20,99%
    vs 21,28%). `out/compare/crnn_fase7_track_vs_crnn_fase7_ctrl.md`: CER **−9,29 poin, SK halaman [−11,92;
    −6,83]** (baris [−10,20; −8,40]); CER spasi dibuang −0,28 [−0,81; +0,21] (tidak berbeda); 484 baris membaik,
    123 memburuk, 138 sama; halaman 73 / 14 / 5; spasi keluaran 145 vs 2.201 (referensi 110).
    `crnn_fase7_ctrl_vs_crnn_fase6_ctrl.md`: −1,42 [−2,36; −0,40], CER spasi dibuang −1,61 [−2,27; −0,92].
    Aturan (a) terpenuhi: SK tidak memuat 0, CER tanpa spasi tidak memburuk, dosis-respons sintetis searah,
    G1/G2 tidak memburuk. Satu run per lengan; derau antar-run tetap belum terukur.
  - **Angka resmi = `fase7_track` (keputusan user 2026-10-04, menggantikan `fase5_fonts`).** Evaluasi resmi
    10.000 baris javatext (`out/eval/fase7_track_G1_10k.json`, `_G2_10k.json`): **G1 0,27%** (0,2686%; sebelumnya
    0,34%), **G2 1,20%** (1,1965%; sebelumnya 2,01%), **G3 21,46%** (sebelumnya 37,25%; target < 8% belum
    tercapai). Diimpor ke web 2026-10-04 15:47 (14 pipeline, 6.755 prediksi); kesalahan aksara `fase7_track`:
    2.165 tertukar, 1.302 hilang, 711 tambahan (`fase5_fonts`: 2.266 / 2.143 / 626). Dipilih di antara
    `fase7_track` (21,46%) dan `fase7_track_rare` (21,18%, tidak berbeda nyata) berdasarkan efek samping sisipan
    aksara langka (aturan (b)), bukan G3 terendah; tetap satu run yang dibandingkan pada test set, jadi angkanya
    sedikit optimistis. Beam + LM (33,83%) dan terjemahan "dari keluaran OCR" (chrF 19,4) masih milik checkpoint
    `fase5_fonts`: beam + LM belum dievaluasi di atas `fase7_track`.
  - **Hasil fase 7 di 745 baris nyata (2026-10-04; `out/compare/crnn_fase7_track_vs_crnn_fase6_ctrl.md`,
    `crnn_fase7_track_rare_vs_crnn_fase7_track.md`).**
    `fase7_track`: G3 **21,46%** (pad 0,06: 19,99%; spasi dibuang 20,99%; baris persis 5,6%), q100 G1 0,29%,
    G2 1,17%, val bersih 0,234%. Terhadap `fase6_ctrl`: −10,71 poin (SK halaman [−13,55; −8,16]), CER tanpa
    spasi −1,89 [−2,63; −1,12], spasi keluaran 2.168 → **145** (referensi 110), 542 baris membaik dan 81
    memburuk. Aturan (a): spasi berlebih hilang dan CER tanpa spasi tidak memburuk; berapa bagian dari −10,71
    yang milik tracking menunggu `fase7_ctrl`.
    `fase7_track_rare`: G3 **21,18%** (pad 0,06: 19,78%; spasi dibuang 20,78%), q100 G1 0,32%, G2 0,97%.
    Terhadap `fase7_track`: CER −0,27 [−0,88; +0,38] (tidak berbeda); recall 44 codepoint langka 49,7% vs 0
    (SK halaman [+43,5; +57,0]), murda 33,9%, adeg-adeg pembuka 101/112, salah baca aksara → angka turun (pa
    murda → 8: 9 → 1; nga lelet → 2: 20 → 8). TETAPI efek samping tidak berkurang: presisi keluaran aksara
    langka 40,6% (606 dikeluarkan, 246 benar; `fase6_rare` 39,6%), da mahaprana 84 keluaran dengan presisi
    14%, recall dda 39,6% → 25,7%, ca murda 35 dan pada isen-isen 36 keluaran palsu, pembuka palsu 22/633
    (3,5%), CER 409 baris tanpa aksara langka +0,94 [+0,10; +1,86]. Aturan (b): saringan glyph mirip TIDAK
    menekan keluaran palsu (penyebabnya bukan glyph kembar di font training); manfaat dan biaya sisipan saling
    menghapus di G3. **Checkpoint dasar berikutnya = `fase7_track`**; sisipan aksara langka perlu cara lain
    (dosis lebih rendah, font yang bentuk aksara langkanya dekat dengan cetakan, atau data nyata).
  - **Hasil sintetis fase 7 (2026-10-04; `out/compare/spacing_synthetic.md`, `out/compare/fase7/rare_synthetic.md`;
    citra yang sama untuk semua checkpoint, font javatext, tanpa NusaAksara).** *Dosis-respons jarak* (300 baris
    val, bersih, 64 px; jalan akhir 2026-10-04 16:10 dengan run kontrol): spasi palsu per 100 batas suku kata pada
    baris tanpa spasi, jarak 0 / 0,1 / 0,2 / 0,3 / 0,45 / 0,6 em (dua terakhir di luar rentang latih 0–0,3):
    `fase7_track` **0 / 0 / 0 / 0 / 4,1 / 67,5**; `fase7_track_rare` 0 / 0 / 0 / 0 / 4,6 / 72,2; kontrol
    `fase7_ctrl` **0 / 15,3 / 84,3 / 92,5 / 94,2 / 87,6**; `fase6_ctrl` 0 / 16,5 / 85,0 / 92,3 / 94,1 / 92,4;
    `fase5_fonts` 0 / 10,3 / 81,1 / 91,8 / 93,4 / 89,3. Selisih `fase7_track` − `fase7_ctrl` pada 0,3 em −92,5
    [−93,2; −91,8]: gejalanya hilang karena tracking, bukan karena tambahan langkah. Recall spasi asli (baris
    berspasi) `fase7_track` 99,65–100% di semua jarak; `fase7_ctrl` 99,8% → 63,0% pada 0,3 em dan 25,4% pada
    0,6 em (celah kata yang sangat lebar tidak lagi dibaca spasi). Biaya pada jarak 0 terhadap kontrol tidak
    nyata: recall spasi −0,18 poin [−0,46; +0,06]. CER tak peka spasi datar (0,63–0,74% tanpa spasi; 0,28–0,40%
    berspasi), jadi bentuk aksara tetap terbaca. Batas: di luar rentang latih gejalanya kembali (0,6 em: 67,5),
    jadi cetakan yang lebih renggang dari 0,3–0,45 em butuh `--track-max` lebih besar. *Aksara langka*
    (2.000 baris test): teks biasa `fase7_track` 0,33% vs `fase6_ctrl` 0,32% (+0,02 [−0,01; +0,05]: tracking
    tanpa efek samping pada teks bersih; dengan augmentasi `fase5` −0,21 [−0,29; −0,12]); `fase7_track_rare`
    recall 44 codepoint 73,5% bersih / 62,2% teraugmentasi (`fase6_rare` 73,7% / 59,3%), presisi 86,1% / 84,1%
    (90,0% / 86,4%), murda 74,4% / 52,0% (66,3% / 41,7%), pembuka 100%, pembuka palsu 0% / 0,2%. Di sintetis
    presisi aksara langka tinggi (84–90%) padahal di cetakan nyata 40%: keluaran palsu itu masalah celah domain
    bentuk huruf, bukan sesuatu yang bisa disetel di javatext.
  - **Aturan analisis, ditetapkan 2026-10-03 SEBELUM angka `fase6_ctrl` ada** (`compare_runs.py crnn_fase6_rare
    crnn_fase6_ctrl` + evaluasi sintetis tertarget font javatext): (1) "sisipan membantu aksara langka" hanya bila
    SK 95% bootstrap per halaman untuk selisih recall 44 codepoint (n=495) atau recall adeg-adeg pembuka (n=112)
    tidak memuat 0, searah di evaluasi sintetis; (2) efek samping = CER 409 baris tanpa aksara langka, adeg-adeg
    palsu, dan CER kondisi "biasa" sintetis; (3) selisih G3 keseluruhan dilaporkan dengan SK per halaman dan
    dibaca sebagai arah (satu run per lengan); (4) memilih lengan berdasarkan G3 = memilih di test set, jadi
    lengan yang dipakai untuk run berikutnya diputuskan dari (1)–(2) dan val sintetis, bukan dari G3.
  **CarakanJawa tidak punya aksara O (U+A98E):** baris korpus ber-O yang dirender dengan font itu =
  kotak .notdef berlabel O (derau kecil di `fase5_fonts` & ablasi).
- **Tes cepat 100 baris** (`--lines 100`, juga untuk `--real`, sampel acak seed 0) dipakai untuk
  memutuskan arah; ketidakpastian ~±1 poin. Angka laporan tetap 10.000 baris (G1/G2) dan seluruh
  745 baris nyata (G3). Hasil di `out/eval/q100_*.json`.
- **Augmentasi `heavy` tidak mirip pindaian nyata.** `heavy` menghasilkan citra abu-abu buram;
  potongan NusaAksara justru hitam-putih tajam (seperti hasil binarisasi), goresan tebal, berbintik,
  rapat, tanpa spasi (lihat `out/heavy_vs_real.png`). Lulus G2 tidak menjamin G3. Kandidat augmentasi
  Fase 5 untuk G3: binarisasi + speckle, tebal-tipis goresan (dilate/erode), variasi skala teks.
- **Preset augmentasi:** `heavy` (G2, dibekukan, 8 op p=1), `train` (op yang sama, p=0.5),
  `fase5` (12 op p=0.5, termasuk `tight`, `stroke`, `binarize`, `speckle` yang meniru pindaian).
  Biaya `fase5` ~99 ms/sampel per core -> pakai `--workers 4`. Run cepat Fase 5:
  `.venv-xpu/Scripts/python -m src.train --run fase5_quick --init out/checkpoints/base/best_4b_step1500.pt --augment fase5 --drop-space-prob 0.5 --lr 5e-4 --steps 1500 --workers 4`.
  `best.pt` dipilih dari val **bersih**, jadi untuk G2/G3 evaluasi `last.pt`.
- **Korpus:** `valid_aksara` menolak rantai pasangan > 3 (`MAX_PASANGAN_CHAIN`); berlaku saat
  korpus dibangun ulang. Korpus saat ini masih memuat 0,11% baris seperti itu.
- **Mengedit `src/` saat training berjalan:** di Windows, worker validasi di-spawn ulang pada
  evaluasi pertama run dan mengimpor ulang `src.train` beserta semua modul yang diimpornya.
  Galat sintaks/impor di modul mana pun menghentikan training. Selalu `py_compile` + impor di
  `.venv-xpu` + pytest sebelum evaluasi berikutnya.

---

## Scope

Repo ini berisi dua bagian (keputusan user 2026-09-24, menggantikan "satu repo = OCR saja"):

- **`src/`, `scripts/`, `tests/` (Python) HANYA mengerjakan citra baris → Unicode aksara Jawa.**
  Semua metrik OCR (CER, gerbang G1–G4) dihitung di sini.
- **`web/` (Laravel)** menampilkan hasil dan menampung tahap lanjutan alur: transliterasi,
  arti (terjemahan Indonesia), dan tingkat tutur. Logika tahap-tahap itu TIDAK boleh masuk `src/`
  dan tidak boleh memengaruhi angka OCR.

JANGAN bangun di repo ini:

- layout analysis / deteksi baris dari foto halaman
- tulisan tangan, naskah lontar

Setiap tahap alur punya metrik dan data ujinya sendiri. Jangan menilai tahap 2–4 hanya dari
keluaran OCR: tampilkan juga hasilnya dari label, supaya sumber error tetap terpisah.

> Pengecualian yang sah: transliterator Latin→Aksara **dipakai sebagai alat**
> untuk membuat korpus di Fase 2. Itu bukan produk repo ini, itu tooling.

---

## Aturan kerja

- Tulis test **dulu** untuk tokenizer dan collate, sebelum menulis model.
- Sebelum training penuh: **overfit 32 sampel** sampai CER mendekati 0.
  Kalau tidak bisa overfit 32 sampel, ada bug struktural — **laporkan dan
  berhenti**, jangan lanjut ke dataset penuh, jangan tambah data,
  jangan ganti arsitektur.
- Citra training **tidak pernah disimpan ke disk** — render on-the-fly di
  DataLoader worker.
- Kalau sebuah kriteria lulus tidak tercapai, **laporkan apa adanya** dengan
  angkanya. Jangan longgarkan kriteria supaya kelihatan lulus.
- Pipe ke `tail`/`grep` menelan exit code pytest. Tangkap `$?` sebelum pipe.

---

## Gerbang per fase

Detail lengkap di PLAN.md §7.

| Fase | Deliverable | Gerbang | Status |
|---|---|---|---|
| 0 | `src/render.py`, `scripts/verify_shaping.py` | 13 kasus × 2 font, **diperiksa mata manusia** | cek otomatis lolos; **pemeriksaan mata user belum** |
| 1 | `src/tokenizer.py` + test | round-trip 100% pada ≥100k baris | **lolos**: 1.034.357/1.034.357, 93 kelas |
| 2 | `src/corpus.py`, `data/corpus_jv.txt` | ≥200k baris, ≥95% lolos round-trip | **lolos**: 1.034.357 baris, 95,82%, bocor 0 |
| 3 | `src/dataset.py` + test shape | assert `model(dummy).shape[1] == W // DOWNSAMPLE` | **lolos**: 0/300 melanggar T≥1.5L |
| 4 | `src/model.py`, `train.py`, `decode.py`, `infer.py` | 4a: CER<1% @32 → 4b: **CER<2% synthetic bersih** | 4a **lolos**: CER inference satu baris 0,07% (31/32 persis) dalam 375 langkah; 4b: berhenti di langkah 1500 karena val CER 0,376% < `--target-cer 0.005` (val = teks baru, font training); checkpoint tetap `out/checkpoints/base/best_4b_step1500.pt`; evaluasi test/G1–G3: `out/eval/base_*.json` |
| 5 | `src/augment.py`, `scripts/ablation.py` | CER<5% synthetic augmentasi berat | tes cepat 100 baris (bukan angka resmi) `fase5_quick` langkah 500: G1 0,79%, G2 2,88%, G3 76,0% (spasi dibuang 57,5%); run 1500 langkah crash OOM di ~950; `fase5_fonts` (10 font: 2 inti + 8 tambahan) dilanjutkan dari langkah 500, berhenti di 1298 (batas 3 jam): G1 0,41%, G2 1,44%, **G3 39,1%** (margin 0,06: 35,2%; spasi dibuang 22,9%); **G3 resmi 745 baris: 37,2%** (margin 0,06: 34,9%; spasi dibuang 25,2% / 23,1%); pembanding font inti `fase5_core` (init, augmentasi, jadwal LR & langkah sama lewat `--steps 1500 --stop-step 1298`): G1 1,64%, G2 1,82%, G3 745 baris 66,6% (margin 0,06: 63,3%) → **font tambahan menurunkan G3 ~29 poin**, langkah tambahan saja ~8 poin (G3 q100 76,0% → 67,7%); **evaluasi resmi `fase5_fonts` 10.000 baris javatext: G1 0,34%, G2 2,01% → gerbang Fase 5 (G2 < 5%) LOLOS** (`out/eval/fonts_G1_10k.json`, `fonts_G2_10k.json`); **ablasi selesai 2026-09-15** (`out/ablation.md`; 12 run kumulatif dari 4b, 600 langkah, 10 font, drop-space 0,5, satu seed): val berat 65,1% → 12,0% lewat 7 op preset (blur −21, noise −10, rotate −7,5), lalu naik ke 19,3% dengan op pindaian; **G3 745 baris 75,1% → 44,1%** (tanpa spasi 69,8% → 30,0%): op preset hampir datar kecuali contrast −14,8; op pindaian **tight −14,6, stroke −10,5**, binarize −2,4, speckle −1,2; val bersih 0,34% → 0,43%. Batasan: G3 = test set (bias seleksi), satu seed (rotate +12,8 menunjukkan fluktuasi antar-run bisa besar), urutan kumulatif mencampur interaksi |
| 6 | `src/real.py`, `--real-train` | CER<8% pada ≥200 baris nyata | test: `data/real/nusaaksara/labels.tsv` 745 baris (lisensi **non-komersial**, HANYA test); fine-tune: 43 baris papan Commons menunggu verifikasi pembaca aksara (`out/verifikasi_aksara.html` → `scripts/import_review.py`); **belum lolos**: terbaik sejauh ini `fase7_track` (masih sintetis saja; `fase6_ctrl` + 1.500 langkah dengan jarak antar suku kata acak) G3 **21,46%** (pad 0,06: 19,99%; spasi dibuang 20,99%) dan `fase7_track_rare` 21,18% (tidak berbeda nyata); kontrol `fase7_ctrl` (langkah sama tanpa jarak) 30,75% → efek jarak −9,29 poin (SK halaman [−11,92; −6,83]); sebelumnya `fase6_ctrl` 32,17%, `fase6_rare` 34,06%, beam+LM atas `fase5_fonts` 33,83%. **Angka resmi sejak 2026-10-04 = `fase7_track`** (keputusan user): G1 0,27%, G2 1,20% (10.000 baris javatext), G3 21,46%; sebelumnya `fase5_fonts` 0,34% / 2,01% / 37,25% |

**Fase 0 tidak bisa diotomatiskan.** Render PNG-nya, lalu **minta user membuka dan
memeriksa sendiri**. Jangan menyatakan Fase 0 lulus berdasarkan "tidak ada error".

---

## Konteks aksara

Aksara Jawa = **abugida**, blok Unicode **U+A980–U+A9DF**.

- **Pasangan**: `C1 + U+A9C0 (pangkon/virama) + C2` → C2 menumpuk **di bawah** C1.
  Bisa bertingkat tiga. **Kecuali pasangan ha, sa, pa** (U+A9B2, U+A9B1, U+A9A5):
  ditulis di **samping** C1, tidak menumpuk — terukur sama di Noto dan Tuladha OT.
- **Taling U+A9BA**: **pre-base** — digambar di **KIRI** aksara dasar tapi disimpan
  **SESUDAHNYA** di Unicode. Inilah yang melanggar monotonisitas CTC.
- **Taling-tarung** (`U+A9BA` + `U+A9B4`): **circumfix**, mengapit aksara dasar
  dari kiri dan kanan.
- **Cakra U+A9BF** membungkus konsonannya dari kiri-bawah, tapi di kolom yang sama —
  bukan pre-base.
- **Sandhangan** mengelilingi aksara dasar dari **4 sisi** — atas (wulu U+A9B6),
  bawah (suku U+A9B8), atas-kanan (layar U+A982), bawah-kanan (cakra U+A9BF).

Konsekuensi: **satu kolom piksel bisa memuat 3 codepoint berbeda.** Karena itu
tidak ada segmentasi karakter — model membaca seluruh baris sekaligus.

**Tidak perlu bisa membaca aksara Jawa** untuk mengerjakan repo ini. Semua
verifikasi lewat round-trip test dan perbandingan visual dengan gambar referensi.

---

## Diagnostik cepat

Kalau training bermasalah, cek ini dulu sebelum menyalahkan arsitektur:

| Gejala | Penyebab paling mungkin |
|---|---|
| Loss turun tapi CER mentok ~90% (model keluar blank terus) | `in_lens` dari lebar padded, atau `DOWNSAMPLE` ≠ stride CNN |
| CER punya lantai ~5–10% yang tidak bisa ditembus | Tokenizer tidak round-trip, atau NFC tidak konsisten |
| Tidak bisa overfit 32 sampel | Bug struktural — cek grad norm, LR, `zero_infinity=True` |
| Loss = inf / NaN | Target lebih panjang dari input — filter T ≥ 1.5L |
| Semua taling salah posisi | RAQM tidak aktif |
| CER tinggi hanya pada teks padat | Pelanggaran T ≥ L |
| Bagus di synthetic, hancur di nyata | Augmentasi tidak menutup gap domain |
| Loss & val CER melonjak saat LR masih naik (OneCycle warmup) | Terjadi di 4b langkah 1000 (lr 7,6e-4, grad 6,8 > clip 5, CER 11% → 41%); pulih sendiri di 1100–1200. Aturan: lanjut kalau 2 log berikutnya loss < 0,1, kalau tidak restart dari `best.pt` dengan `--lr 5e-4` |
| Log berhenti berjam-jam, proses masih hidup | Laptop di-Hibernate/Sleep lewat menu, atau masuk Modern Standby saat memakai baterai. Cek `Get-WinEvent` id 187 (`ApiCallerName` = siapa yang meminta), 42, 506/507, 1. **Terukur 2026-10-02/03:** jeda 4b (13 Sep) dan jeda `fase6_rare` 17:39–20:58 = Hibernate dari menu Start (kebiasaan harian di laptop ini); malam berikutnya charger dicabut 23:07 dan laptop diam di Modern Standby 23:11–04:47 (training membeku). `prevent_sleep()` hanya menahan tidur karena idle, tidak bisa menolak permintaan user. Setelan daya sudah benar (idle sleep Never; lid & tombol daya Do nothing), jadi layar padam atau lid tertutup tidak menghentikan training. Proses dibekukan lalu lanjut sendiri saat bangun, TETAPI **`--time-budget-hours` memakai jam dinding** sehingga waktu tidur ikut terhitung dan run berhenti di langkah sembarang. Rantai `scripts/fase6_common.sh` memakai `--time-budget-hours 0` |
| Laptop mati mendadak, Kernel-Power 41 dengan `BugcheckCode=0` | Baterai habis. **Hibernate gagal selama Docker Desktop/WSL2 hidup:** driver Hyper-V `vpcivsp` menolak transisi daya (Kernel-Power 40), termasuk Hibernate saat baterai kritis (2026-10-03 06:28 → mati 06:31, `fase6_ctrl` hilang di langkah ~300). Training menguras baterai ~21 W (penuh ≈ 1 jam 50 menit). Selama training: charger tetap terpasang, jangan pilih Hibernate/Sleep/Shut down. `BugcheckCode≠0` = crash Windows: sesi 1 `fase6_rare` (25 Sep ~02:00) mati karena bugcheck 0x19C WIN32K_POWER_WATCHDOG_TIMEOUT, bukan karena sesi Claude |
| Training crash `PermissionError` saat simpan checkpoint | Proses lain (notebook) sedang membaca `best.pt`; `save_atomic` kini mencoba ulang 10x |
| Training crash `XPU out of memory` | Lebar batch = citra terlebar; memori training ~105 KiB per kolom B×W (batch 32 × 2.900 px ≈ 9,6 GiB), dan iGPU berbagi memori dengan sistem (~6,6 GiB dipakai proses lain). `fase5_quick` crash di langkah ~950, checkpoint terakhir langkah 500. `train.py --max-batch-columns` (default 64.000 ≈ 6,7 GiB) memecah batch lebar dengan akumulasi gradien berbobot n/B (sama dengan batch utuh kecuali statistik BatchNorm); jumlahnya tercatat sebagai `split_batches` di log |
| Memori sistem habis saat training (commit mendekati batas, RAM tersedia < 1 GB) | Proses training fase7 memakai ~17 GB memori privat + 8 worker (4 render, 4 val) à 0,8–1,6 GB ≈ 27 GB commit, dan naik selama run. Terukur 2026-10-04 13:53: `src.evaluate` 10.000 baris (2 worker, ~6,4 GB) dijalankan bersamaan → commit 75,3 dari 76,4 GB, RAM tersedia 0,2 GB, laju training turun ke 1,8 sampel/dtk; evaluasi dihentikan sebelum ada yang gagal. Jangan menjalankan evaluasi besar, ekspor, atau test suite penuh selama training: `last.pt` hanya disimpan tiap 500 langkah, jadi satu kegagalan alokasi membuang sampai ~2 jam. Cek: `Get-CimInstance Win32_OperatingSystem` (`TotalVirtualMemorySize − FreeVirtualMemory` vs `TotalVirtualMemorySize`). Layanan terjemahan (port 8012) memegang ~3,6 GB dan boleh dihentikan sementara |
| Training/server berhenti sendiri ~30 menit setelah dijalankan Claude | Tugas background Claude Code punya batas waktu (~30 menit, terukur 2026-10-02: training, `artisan serve`, uvicorn semuanya dihentikan) dan ikut mati saat sesi berakhir. Proses panjang WAJIB dijalankan sebagai proses Windows mandiri: training `Start-Process "C:\Program Files\Git\bin\bash.exe" -ArgumentList scripts/run_fase6_ctrl.sh -WorkingDirectory <repo> -WindowStyle Hidden` (lanjut otomatis lewat `--resume`), server `web\services.ps1 start\|stop\|status`. Setelah restart/mati listrik TIDAK ada yang menyala sendiri: jalankan ulang rantai, `scripts/after_fase6.sh`, dan `services.ps1 start`. Sebelum menjalankan ulang, pastikan tidak ada proses `src.train` lama yang masih hidup (dua training berebut GPU). `taskkill /T` dari Git Bash tidak mengenai program MSYS (`sleep`, `grep`) yang di-exec, karena induk Windows-nya sudah keluar; program native (python) tetap kena. Jangan mengedit skrip bash yang sedang berjalan: bash membaca berkas sedikit demi sedikit |
| Worker DataLoader crash "Glyph terpotong tepi kanvas" | Rantai pasangan panjang dari kata serapan (4b crash di langkah ~1300). `render.py` kini memperbesar kanvas 3→6→12 em; `SyntheticLines` melewati `RenderClipped`/`EmptyRender` saja (RAQM mati tetap dilempar). Korpus: rantai pasangan ≥4 = 0,11% baris train, semuanya sampah transliterasi — **filter di `src/corpus.py` saat korpus dibangun ulang** |
