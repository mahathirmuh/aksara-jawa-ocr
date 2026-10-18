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
python -m src.charlm --order 5 --lines 300000 --drop-space-prob 0.5 --standard-prob 0.5   # LM yang sama dengan ejaan baku di separuh baris
python scripts/beam_eval.py CKPT --lm data/charlm/o5_n300000_ds0.5.pkl --name N  # beam+LM vs greedy: disetel di dev sintetis, NusaAksara sekali
python scripts/beam_eval.py CKPT --lm data/charlm/o5_n300000_ds0.5_std0.5.pkl --name RUN_o5std --dev-standard-prob 0.5   # run berejaan baku: dev dirender seperti training-nya
python scripts/beam_eval.py CKPT --lm LM --name RUN_o5std --dev-from-checkpoint   # dev dirender dengan resep training checkpoint itu sendiri (argumen + font dari log; fase 9 ke atas)
python scripts/export_results.py                 # kontrak data web -> out/results/ (~14 menit CPU dengan run fase6)
python scripts/export_datasets.py                # kartu data halaman Dataset -> out/results/datasets.json (~20 dtk CPU, tanpa model)
python scripts/export_methods.py                 # kartu metode halaman Metode -> out/results/methods.json (beberapa detik; SESUDAH export_results.py dan compare_runs.py, karena buktinya dikutip dari keluaran keduanya)
python scripts/compare_runs.py A B               # dua pipeline di out/results: CER + SK bootstrap baris/halaman, recall aksara langka & adeg-adeg -> out/compare/
python scripts/eval_spelling.py real A B          # bentuk ejaan baku (nya, pa cerek, nga lelet, keret, pada pangkat) di baris nyata, SK per halaman
python scripts/eval_slant.py real A B             # CER per kelompok kemiringan baris nyata (tegak / agak miring / miring), SK per halaman
python scripts/eval_slant.py synthetic RUN...     # dosis-respons kemiringan sintetis (javatext, 300 baris val, miring 0-0,7)
python scripts/eval_rare.py --workers 2          # sintetis tertarget aksara langka (split test, font javatext, 3 kondisi x 3 checkpoint; ~1-2 jam CPU)
python scripts/eval_spacing.py --lines 301 --threads 4   # dosis-respons jarak antar suku kata (split val, javatext): spasi palsu per 100 batas suku kata; memakai cache
bash scripts/make_official.sh RUN                # evaluasi resmi G1/G2 10.000 baris untuk RUN, lalu ekspor + impor (OFFICIAL_RUN harus sudah RUN)
.venv/Scripts/python -m uvicorn src.serve:app --host 127.0.0.1 --port 8011   # layanan model untuk Demo
cd web && php artisan aksara:import && php artisan serve --port=8010                  # web UI (lihat web/README)
cd web && php artisan aksara:dictionary                                               # kamus kata halaman Kamus (data jadi di web/database/dictionaries)
cd web && php artisan aksara:datasets                                                 # kartu data halaman Dataset saja (aksara:import ikut mengimpornya bila berkasnya ada)
cd web && php artisan aksara:methods                                                  # kartu metode halaman Metode saja (aksara:import ikut mengimpornya bila berkasnya ada)
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
  **Diulang di atas model resmi 2026-10-12** (`out/beam/fase7_track_o5.json`, checkpoint `fase7_track`, LM dan beam yang
  sama; ~10 menit CPU). Setelan terpilih di dev berubah: **alpha 0,5, beta 1,0** (bobot LM disetel PER checkpoint).
  Dev 2,13% → 1,45% (beam tanpa LM 2,08%); render bersih javatext 0,24% → 0,18% (8 membaik, 2 memburuk);
  **NusaAksara 745 baris: greedy 21,46% → beam 19,35%** (tanpa spasi 20,99% → 18,74%); 344 baris membaik, **176
  memburuk** (koreksi berlebih tetap nyata). Selisih −2,11 poin, SK 95% per halaman [−2,74; −1,51], per baris
  [−2,57; −1,65] (`out/compare/crnn_fase7_track_beam_vs_crnn_fase7_track.md`); pembacaan ke-38 atas data uji itu. Tetap bukan angka gerbang resmi. `scripts/export_results.py::official_beam` mengekspornya sebagai pipeline
  `crnn_<run resmi>_beam` ("Beam + LM fase7_track") bila laporan `out/beam/<run>_o5.json` ada, dengan bobot dari
  laporan itu, dan berhenti bila laporannya bukan milik checkpoint/LM/lebar beam sekarang atau lebih tua dari
  checkpoint; kartu metode membaca bobotnya dari konfigurasi pipeline itu. `src/serve.py` (Demo) memakai alpha 0,5:
  layanan model harus dinyalakan ulang (sudah, 2026-10-12). Tag "beam membaik/memburuk", urutan beam di Penjelajah,
  tabel sintetis Perbandingan, dan titik beam di Ringkasan memakai pasangan milik pipeline resmi
  (`Pipeline::beamPair()`: `<resmi>` dan `<resmi>_beam` bila sudah diekspor, kalau tidak pasangan `fase5_fonts`).
  **Analisis 176 baris yang memburuk (2026-10-13, dari `out/results/predictions.jsonl`, tanpa menjalankan model):**
  kerusakannya kecil dan tersebar, bukan runtuh: 114 dari 176 baris memburuk 1 karakter (total +272 suntingan lawan
  −702 di 344 baris yang membaik). Per karakter label: 842 diperbaiki, 537 dirusak (3,2% dari yang tadinya benar);
  baris persis benar 42 → 47, dan 9 dari 42 baris yang sudah benar diubah. Tiga pola: (1) beam merugikan baris yang
  sudah nyaris benar (CER greedy < 5%: 94 baris, 6 membaik, 29 memburuk, 2,19% → 3,55%) dan membantu sisanya;
  (2) di halaman 87 ke atas (bukan majalah halaman 0–86) nyaris tanpa manfaat (22,96% → 22,25%; 70 membaik, 67
  memburuk) sedangkan di halaman 0–86 21,15% → 18,75%; (3) beam membuang angka palsu (187 → 77 keluaran; label 20)
  tetapi menambah spasi (145 → 187; label 110).
  **Temuan yang lebih penting: korpus latih tidak berejaan baku, dan itu mengenai model DAN model bahasa.** Korpus
  (konvensi `scripts/translit/translit.js`) menulis n di depan c/j dengan na, rê/lê dengan ra/la berpepet, dan keret
  dengan cakra + pepet; cetakan memakai ejaan baku. Di 300.000 baris latih pertama: nya + pangkon + ca/ja **0** kali
  (na + pangkon + ja 14.741, + ca 10.919), pa cerek 33, nga lelet 26, keret 30 (hanya dari sisipan aksara langka),
  sedangkan ra/la berpepet 60.657 dan cakra + pepet 10.378. Di 745 label: pa cerek 44, nga lelet 26, keret 20, semuanya
  terbaca 0%; nya di depan ca/ja 59, greedy benar 31, beam tinggal 13 (LM menggantinya, 17 kali menjadi ba); pada
  pangkat 63, greedy 31, beam 22. Jumlahnya 212 karakter (1,04% label), 150 salah di greedy dan 177 di beam, ditambah
  galat ikutan di sekitarnya. Aturan ejaan bakunya sudah ada dan teruji di `web/app/Support/AksaraWriter.php` (sama
  dengan ejaan kamus di 97,4% lema). Tuas berikutnya: bangun ulang korpus dengan ejaan baku (atau ubah baris korpus
  saat render), lalu latih ulang dan bangun ulang LM; baru sesudah itu beam + LM dinilai lagi sebagai calon jalur resmi.
  Angka-angka ini dibaca dari data uji, jadi dipakai untuk memilih arah, bukan untuk menyetel.
  **Fase 8 (dijalankan 2026-10-13, `scripts/run_fase8.sh`, log `out/fase8_chain.log`; permintaan user):** korpus
  TIDAK dibangun ulang (split, G1/G2, dan kartu data tetap sebanding); baris ditulis ulang dengan ejaan baku saat
  dirender. `src/spelling.py::standardize` = lima aturan pada teks aksara (nya di depan ca/ja di dalam kata; pa cerek
  dan nga lelet kecuali sesudah pangkon; keret; deret angka diapit pada pangkat), dicocokkan dengan ejaan baku
  `AksaraWriter` pada 4.609 kata kamus (sama di semuanya; 291 kata memang berubah). `src.train --standard-prob P`
  (bawaan 0: tidak mengonsumsi angka acak, run lama dan val/test tidak berubah) menerapkannya sebelum sisipan aksara
  langka dan sebelum spasi dibuang; aturan yang glyph-nya tidak ada di font terpilih dilewati, dan hasil yang tidak
  well-formed atau tidak lolos round-trip dibuang. Di 100.000 baris latih: 37,1% baris berubah (pangkat 15,7%, cerek
  8,3%, lelet 8,2%, nya 8,0%, keret 3,4%), semuanya lolos round-trip; kesepuluh font latih punya glyph-nya (lembar
  pembanding diperiksa Claude, belum mata manusia). **Font yang merender dua ejaan KEMBAR selalu memakai ejaan baku**
  untuk aturan itu (`src.dataset.spelling_twins`, dibandingkan lewat render dengan interpreter yang berjalan):
  GumregahNew meligaturkan cakra + pepet menjadi keret; di `.venv` BasaJan juga kembar untuk cerek, lelet, keret
  (di lingkungan training tidak). `src.charlm --standard-prob` membangun model bahasa dengan aturan yang sama
  (berkas berakhiran `_stdP`). Peluang 0,5 dipilih supaya model tetap melihat kedua ejaan (bagian uji korpus G1/G2
  berejaan korpus); nilainya tidak disetel.
  `fase8_std` = `fase7_track` + 1.500 langkah, resep `fase7_track` + `--standard-prob 0.5`; `fase8_ctrl` = sama tanpa
  ejaan baku, titik lanjut ditiru.
  **Aturan analisis fase 8, ditetapkan 2026-10-13 SEBELUM ada angka:** (a) titik akhir utama = recall lima bentuk
  ejaan baku di 745 baris nyata (pa cerek 44, nga lelet 26, keret 20, nya di depan ca/ja 59, pada pangkat 63; n = 212;
  dasar `fase7_track` greedy: 62 benar), `fase8_std` − `fase8_ctrl`, "membantu" hanya bila SK 95% bootstrap per
  halaman tidak memuat 0; (b) efek samping = presisi keluaran kelima bentuk itu (keluaran palsu), CER baris yang tidak
  memuat satu pun bentuk itu, q100 G1/G2 (tidak boleh memburuk di luar ketidakpastian ~1 poin), dan evaluasi
  sintetis javatext pada baris val yang dirender dengan ejaan korpus DAN ejaan baku (skripnya dibuat sebelum membaca
  hasil); (c) G3 keseluruhan dilaporkan dengan SK per halaman sebagai arah (satu run per lengan, derau antar-run
  belum terukur; efek langsung paling banyak ~1 poin); (d) checkpoint dasar berikutnya dipilih dari (a)–(b), bukan
  dari G3; beam + LM dinilai ulang dengan model bahasa berejaan baku, disetel di dev sintetis; mengganti angka resmi
  tetap keputusan user.
  **Jalannya fase 8:** `fase8_std` selesai 2026-10-13 14:21 (val bersih 0,169%): G3 **20,35%** (pad 0,06: 18,95%;
  spasi dibuang 19,85%), q100 G1 0,34%, G2 0,79%. `fase8_ctrl` TERPUTUS di langkah ~1.100: laptop mati 2026-10-13
  18:50:40 (Kernel-Power 41, BugcheckCode 0, tanpa tekan tombol daya): charger dicabut 16:05–17:24 dan lagi sejak
  17:42, lid ditutup, Modern Standby tidak membekukan training, baterai habis. Sesuai protokol (titik lanjut
  `fase8_std` = tidak ada), `fase8_ctrl` diulang dari langkah 0 pada 2026-10-16 11:55; checkpoint langkah 1.000 yang
  terputus disimpan sebagai `out/checkpoints/fase8_ctrl/putus_step1000.pt` (tidak dipakai), dan event `start` proses
  yang terputus tetap di log-nya (dihitung sekali oleh `resume_points`). `run_fase8.sh` kini tidak menyalin ulang
  `last_snapshot.pt` `fase8_std` saat dijalankan ulang (salinan baru lebih muda dari laporan-laporannya).
  **Tinjauan awal, BUKAN titik akhir utama** (2026-10-16, bacaan greedy `fase8_std` di luar ekspor, lawan prediksi
  `fase7_track` di `out/results`; CER sama persis dengan laporan G3): bentuk ejaan baku terbaca **159/212 (75,0%)**
  lawan 62/212 (29,2%), selisih +45,8 poin SK halaman [+37,1; +53,6]; per bentuk nya 47/59, pa cerek 36/44, nga lelet
  21/26, keret 11/20, pada pangkat 44/63; keluaran bentuk baku 187 dengan presisi 81,3% (palsu: pangkat 16, keret 10,
  lelet 7, cerek 2, nya 0); CER −1,11 poin [−1,57; −0,64]; CER 575 baris tanpa bentuk baku −0,61 [−1,11; −0,10]. Selisih
  ini memuat juga +1.500 langkah, jadi efek ejaan baku dibaca dari `fase8_std` − `fase8_ctrl`.
  `scripts/after_fase8.sh` (proses mandiri, log `out/after_fase8.log`) menunggu rantai selesai lalu menjalankan semua
  titik akhir: ekspor hasil, `compare_runs.py` dan `eval_spelling.py real` untuk std−ctrl, std−fase7_track,
  ctrl−fase7_track, kartu data dan metode + impor web (angka resmi tidak diganti), `eval_spelling.py synthetic` (300
  baris val javatext, kedua ejaan), model bahasa `o5_n300000_ds0.5_std0.5.pkl`, lalu `beam_eval.py` untuk `fase8_std`
  dengan model bahasa lama (`fase8_std_o5`) dan yang berejaan baku (`fase8_std_o5std`), keduanya disetel di dev yang
  dirender seperti training-nya (`--dev-standard-prob 0.5`). Kartu metode dan kartu data sudah mengenal perlakuan ini
  (butir "Ejaan baku pada baris latih" dengan bukti dari `out/compare/spelling_<A>_vs_<B>.json` yang hanya dikutip bila
  CER kedua sisinya sama dengan manifest; resep run "ejaan baku p X"; batasan aksara langka menyebut bahwa pa cerek, nga
  lelet, dan keret tidak lagi langka bagi run resmi berejaan baku). 42 mutan terhadap kode fase 8 di `scripts/` dan
  `web/`: semuanya mati kecuali satu yang setara; 25 mutan terhadap `src/spelling.py`, `src/dataset.py`,
  `src/charlm.py`, dan `src/train.py` (dijalankan sesudah training selesai) semuanya mati.
  **Hasil fase 8 (2026-10-16; `out/compare/spelling_crnn_fase8_std_vs_crnn_fase8_ctrl.json`,
  `crnn_fase8_std_vs_crnn_fase8_ctrl.md`, `spelling_synthetic.json`).** `fase8_ctrl` (diulang dari awal) selesai 15:46,
  val bersih 0,178%: G3 **20,81%** (pad 0,06: 19,41%; spasi dibuang 20,18%), q100 G1 0,27%, G2 0,90%.
  (a) Titik akhir utama: bentuk ejaan baku terbaca **159/212 (75,0%) lawan 64/212 (30,2%)**, +44,8 poin, SK halaman
  [+35,8; +53,1] → **membantu**. Per bentuk `fase8_std` / `fase8_ctrl`: nya 47/30 dari 59, pa cerek 36/0 dari 44, nga
  lelet 21/0 dari 26, keret 11/0 dari 20, pada pangkat 44/34 dari 63. (b) Efek samping: presisi keluaran bentuk baku
  81,3% (187 keluaran, 35 palsu: pangkat 16, keret 10, lelet 7, cerek 2) lawan 89,7% (68 keluaran, 7 palsu); CER 575
  baris tanpa bentuk baku 20,45% lawan 20,60% (−0,14 [−0,75; +0,51], tidak memburuk); q100 G1 0,34% lawan 0,27% dan
  G2 0,79% lawan 0,90% (di dalam ketidakpastian); nga lelet dibaca angka dua 21 → 1 kali; sintetis javatext (300
  baris val yang kena aturan, kedua ejaan): bentuk baku terbaca 569/570 lawan 383/570 (`fase8_ctrl` hanya nya dan
  pangkat; cerek, lelet, keret 0), CER baris berejaan korpus 0,34% lawan 0,31% (2 nga lelet palsu). (c) G3 −0,46 poin,
  SK halaman [−1,03; +0,16], per baris [−0,92; −0,01]; 263 baris membaik, 209 memburuk: arah saja. +1.500 langkah tanpa
  ejaan baku (`fase8_ctrl` − `fase7_track`) −0,65 poin [−1,15; −0,20], bentuk baku +0,9 poin [−2,0; +4,3]. (d) Dasar berikutnya =
  `fase8_std`. **Beam + LM** (dev 300 baris dirender seperti training, ejaan baku p 0,5): model bahasa lama dev 2,54% →
  2,04% (α 0,25, β 1,0); model bahasa berejaan baku `o5_n300000_ds0.5_std0.5.pkl` 2,54% → **1,82%** (α 0,25, β 1,0) →
  dipakai; NusaAksara 20,35% → **18,02%** (lama 18,25%), 306 membaik / 135 memburuk (lama 297 / 159); render bersih
  javatext 0,31% → 0,22%. Evaluasi jarak (`spacing_synthetic.json`, semua bacaan dihitung ulang karena sidik kode
  `src/` berubah): baris tanpa spasi pada 0 / 0,1 / 0,2 / 0,3 / 0,45 / 0,6 em `fase8_std` 0 / 0 / 0 / 0,04 / 5,3 / 62,9
  spasi palsu per 100 batas (`fase7_track` 0 / 0 / 0 / 0,02 / 4,2 / 67,5); recall spasi asli 99,8–100%. Terjemahan NLLB dari bacaan greedy `fase8_std`:
  chrF 18,9 / BLEU 0,5 (`fase7_track` 18,5 / 0,4).
  **Angka resmi = `fase8_std` (keputusan user 2026-10-16: "sesuai rekomendasimu saja baiknya"), menggantikan
  `fase7_track`.** Evaluasi resmi 10.000 baris javatext: **G1 0,24%** (0,2353%), **G2 0,98%** (0,9774%), **G3 20,35%**;
  beam + LM resmi 18,02%. Dipilih menurut aturan (a)–(b), bukan G3; data uji tetap dipakai berulang (sekarang paling
  sedikit 44 pembacaan NusaAksara), jadi angkanya sedikit optimistis. Kartu metode mengutip bukti jarak antar suku
  kata, evaluasi jarak, dan sisipan aksara langka dari pasangan di rantai model resmi (`chain_pair`: fase7_track lawan
  fase7_ctrl, fase7_track_rare lawan fase7_track) dengan menyebut run-nya.
  **Analisis galat `fase8_std` (2026-10-16, 745 baris nyata; data uji, jadi hanya untuk memilih arah):** per kelas suntingan:
  aksara 36% (galat 14,6% per karakter), tanda di atas aksara (cecak, wulu, pepet, layar, wignyan) **28,5% (galat 32% per
  tanda)**: cecak hilang 212, wulu hilang 154, wulu ↔ pepet ↔ cecak ↔ layar saling tertukar 260 kali; aksara langka 7,2%
  (galat 84%), taling 6,6%, adeg-adeg 2,7% (0/112 terbaca), spasi 4,5%. Galat di 10% awal baris 27,9% lawan ~19% di tengah;
  baris yang tintanya menyentuh tepi KIRI potongan 26,5% lawan 16,4%. **Temuan utama (DIKOREKSI 2026-10-16 22:50: baris "miring" ini tulisan tangan, bukan cetak miring; lihat adendum fase 9 di bawah): cetakan miring (italic).** Penaksir
  kemiringan gaya Vinciarelli-Luettin (`scripts/eval_slant.py::slant`: tiap geser, jumlah kuadrat panjang kolom tinta yang
  bersambung tanpa celah; render Noto/Tuladha yang dimiringkan 0,2/0,4/0,6 ditaksir persis; penaksir pertama berbasis
  ketajaman profil kolom TIDAK andal: menggolongkan baris tegak tebal sebagai miring) memberi **270 baris miring (≥ 0,3; 20
  halaman, 32% karakter) dengan CER 28,3% lawan 15,7% pada 424 baris tegak**, menyumbang 45% suntingan; potongan bertekstur
  raster kadang tertaksir tegak walau miring. Bila baris miring setara baris tegak, G3 ≈ 16,3%. Menegakkan potongan miring
  sebelum dibaca (diagnosis sekali pada data uji) MEMPERBURUK: 28,3% → 40,0% (huruf italic bukan sekadar huruf tegak yang
  digeser), jadi koreksi saat baca tidak dipakai. Model sangat rentan terhadap kemiringan geometris: render javatext bersih
  150 baris val, miring 0 / 0,2 / 0,35 / 0,5 → CER 0,16% / 1,24% / 4,52% / **15,43%** (gejala tereproduksi di sintetis,
  seperti spasi sebelum fase 7).
  **Fase 9 (dijalankan 2026-10-16 malam, `scripts/run_fase9.sh`, log `out/fase9_chain.log`; `scripts/after_fase9.sh`
  menjalankan titik akhirnya; keputusan user: "setuju" pada rekomendasi analisis galat lalu run berikutnya):**
  `fase9_slant` = `fase8_std` + 1.500 langkah, resep `fase8_std` (jarak 0,5/0,3, ejaan baku 0,5, augmentasi `fase5`,
  drop-space 0,5) + `--font-fixes` + **`--slant-prob 0.5 --slant-max 0.6`** (`src.dataset.oblique`: citra hasil render
  dimiringkan ke kanan seperti italic, geser horizontal per tinggi diundi seragam 0-0,6, sesudah semua undian lain dan
  sebelum augmentasi; peluang 0 tidak mengonsumsi angka acak). `fase9_ctrl` = sama TANPA miring, titik lanjut ditiru.
  **`--font-fixes` (kedua lengan; bawaan mati supaya run lama tidak berubah):** font yang pasangannya tidak menumpuk di
  interpreter yang berjalan dikeluarkan (`src.dataset.shaping_ok`: lebar ka + pangkon + ka dibagi lebar ka < 1,8; terukur
  2026-10-16 di `.venv-xpu`: BasaJan 2,56, font lain 1,00-1,39; di `.venv` BasaJan 1,00), jadi run fase 9 memakai **9 font**
  (tercatat di event `start` log); dan baris yang memuat pangkon + lungsi tidak dirender dengan font yang menggambarnya sama
  dengan pangkon + lingsa (`render_conflicts`; NewKramawirya), font diundi ulang di antara font lain. Ini perbaikan cacat
  data, bukan perlakuan yang diukur (ada di kedua lengan). Rentang 0-0,6 dipilih dari sebaran penaksir di data uji (banyak
  baris miring 0,5-0,8) dan dari dosis-respons sintetis; nilainya tidak disetel.
  **Aturan analisis fase 9, ditetapkan 2026-10-16 SEBELUM ada angka:** (a) titik akhir utama = CER **270 baris miring**
  (penaksir `eval_slant.py`, ≥ 0,3) `fase9_slant` − `fase9_ctrl`, "membantu" hanya bila SK 95% bootstrap per halaman di
  bawah 0, DAN dosis-respons sintetis searah (`eval_slant.py synthetic`, javatext 300 baris val, miring 0 / 0,2 / 0,35 /
  0,5 / 0,7; 0,7 di luar rentang latih); (b) efek samping = CER 424 baris tegak (SK halaman; tidak boleh memburuk nyata),
  CER sintetis miring 0 (tidak boleh naik di luar ~0,1 poin), q100 G1/G2 (di dalam ~1 poin), bentuk ejaan baku
  (`eval_spelling.py real`, tidak boleh turun nyata); (c) G3 keseluruhan dengan SK per halaman sebagai arah; (d) checkpoint
  dasar berikutnya dan usulan angka resmi dipilih dari (a)-(b), bukan dari G3; mengganti angka resmi tetap keputusan user.
  Batasan yang diketahui sebelumnya: kelompok baris berasal dari penaksir yang kadang menggolongkan potongan raster miring
  sebagai tegak (pengenceran, bukan bias ke satu lengan); huruf italic cetakan berbeda bentuk dari huruf tegak yang digeser,
  jadi render miring mungkin hanya menutup sebagian celah (lever berikutnya: font aksara Jawa bergaya italic/kursif).
  **Kode kartu untuk fase 9 (2026-10-16 malam, selama training; `ff47ebd`, `b9fd2d6`):** kartu metode mengenal
  perlakuan `slant` (`CONTROLS["slant"]` = fase9_slant lawan fase9_ctrl). Butir "Render miring seperti huruf italic":
  `official` bila run resmi memakainya, `tested` bila hanya diuji sebagai lanjutan model resmi (kedua lengan dilanjutkan
  dari checkpoint run resmi, `started_from`); buktinya CER baris miring dan tegak dari `out/compare/slant_<A>_vs_<B>.json`
  (perlakuan = A), hanya bila CER semua baris laporan sama dengan manifest. Butir evaluasi "Evaluasi kemiringan sintetis"
  dari `slant_synthetic.json`, hanya bila laporan itu membaca checkpoint yang sekarang. Resep run menyebut render miring
  dan perbaikan font; cacat font yang ditangani `--font-fixes` (`export_datasets.FONT_FIXES_HANDLE`) tidak dihitung untuk
  run yang memakainya, dan font bercacat run sebelumnya yang masih dipakai run resmi disebut "dihindari lewat perbaikan
  font"; bukti ejaan baku diwarisi lewat rantai (`spelling_entry` memakai `chain_pair`). Kartu data dan halaman Dataset
  mencatat `slant_prob`, `slant_max`, `font_fixes` tiap run ("miring p X (hingga Y)", "perbaikan font").
  `scripts/beam_eval.py --dev-from-checkpoint` merender dev dengan resep data training checkpoint itu (argumen dan font
  dari log-nya), dan `export_results.DEV_KEYS` ikut membandingkan render miring dan perbaikan font.
  `make_official.sh fase9_slant` menghitung pembanding dan `eval_slant.py real` untuk ketiga pasangan fase 9. Uji mutasi:
  50 kerusakan terhadap kode itu, semuanya mati (satu baru mati sesudah test kemiringan terbesar ditambahkan).
  Pencarian web 2026-10-16: tidak ditemukan font aksara Jawa bergaya italic (Noto Sans Javanese, Tuladha Jejeg, dan font
  lain yang ditemukan hanya tegak), jadi huruf miring cetakan kemungkinan dibuat dengan memiringkan font biasa (dugaan,
  belum diverifikasi; menegakkan potongan miring tetap memperburuk, lihat analisis galat di atas).
  **Temuan 2026-10-16 22:50 (sebelum ada angka G3 fase 9): separuh data uji nyata adalah TULISAN TANGAN, bukan cetak.**
  Lembar kontak satu baris per halaman untuk 92 halaman, ditambah empat baris acak dari halaman yang meragukan (klasifikasi
  visual Claude, belum diperiksa manusia): 20 halaman (1, 6, 8, 14, 29, 32, 43, 47, 52, 60, 62, 65, 71, 81, 86, 87, 88, 89,
  90, 92) adalah reproduksi naskah tulisan tangan (aksara miring bersambung, banyak di latar raster abu-abu) atau contoh
  aksara yang digambar tangan (87-89), **437 dari 745 baris (58,7%; 51,0% karakter)**; 72 halaman lain (308 baris) huruf cetak
  (halaman 91: cetak dengan jenis huruf lain, berlatar abu-abu). Halaman cetak umumnya hanya 3-4 baris beranotasi, halaman
  tulisan tangan 5-78 baris. Ke-270 baris "miring" analisis galat (20 halaman) SEMUANYA di halaman tulisan tangan, dan
  potongan "bertekstur raster" juga tulisan tangan berlatar raster: cetak miring tidak ditemukan. Itu juga menjelaskan mengapa
  menegakkan potongan miring memperburuk (tulisan tangan bukan huruf cetak yang dimiringkan). Tulisan tangan di luar scope repo
  ini, tetapi G3 menghitungnya. CER cetak / tulisan tangan per pipeline: `fase8_std` **12,63%** / 27,75% (beam + LM 11,05% /
  24,70%), `fase8_ctrl` 12,96% / 28,34%, `fase7_track` 13,80% / 28,81%, `fase7_ctrl` 27,68% / 33,70% (jarak antar suku kata
  terutama menolong cetak), `fase6_ctrl` 27,55% / 36,60%, `fase5_fonts` 34,84% / 39,55%, `fase5_core` 54,98% / 77,74%.
  **Adendum aturan analisis fase 9, ditetapkan SEBELUM ada angka G3 fase 9 (premisnya keliru: yang miring tulisan tangan):**
  titik akhir utama (a) tetap dilaporkan apa adanya, dibaca sebagai "tulisan tangan miring" (di luar scope); (e) baru = CER
  308 baris CETAK `fase9_slant` − `fase9_ctrl` dengan SK 95% per halaman, sebagai efek samping utama (tidak boleh memburuk
  nyata); (f) CER 437 baris tulisan tangan dilaporkan terpisah, di luar scope. Checkpoint dasar berikutnya dipilih dari baris
  cetak dan efek samping (b), bukan dari tulisan tangan. G3 resmi tetap dihitung atas 745 baris sampai user memutuskan lain;
  pembagian cetak / tulisan tangan dilaporkan di sampingnya, bukan menggantikannya (target tidak dilonggarkan diam-diam).
  **Analisis ulang pasangan lama pada baris cetak (2026-10-16 22:40, `eval_slant.py real` dengan pembagian halaman; data
  uji, jadi untuk memilih arah):** selisih CER cetak (308 baris, 72 halaman) / tulisan tangan (437 baris, 20 halaman), SK 95%
  per halaman: jarak antar suku kata `fase7_track` − `fase7_ctrl` **−13,88 [−15,97; −11,50]** / −4,89 [−9,20; −1,33];
  sisipan aksara langka versi perbaikan `fase7_track_rare` − `fase7_track` **−0,80 [−1,32; −0,28]** / +0,23 [−0,84; +1,56]
  (membantu cetak; di G3 keseluruhan −0,27 [−0,87; +0,38] karena tulisan tangan menutupinya); versi pertama `fase6_rare` −
  `fase6_ctrl` +4,20 [+3,36; +4,96] / −0,33; ejaan baku `fase8_std` − `fase8_ctrl` −0,33 [−1,04; +0,42] / −0,59 [−1,51; +0,42].
  Galat `fase8_std` di baris cetak (1.260 suntingan, 9.979 karakter): tanda atas (cecak, layar, wignyan, wulu, pepet) 26,9%
  (20% per tanda; 144 hilang, wulu → cecak 30), aksara umum 25,9% (7%), aksara langka (murda, mahaprana, swara) 12,5% (83%
  dari 189; sa murda → ga 19, na murda → ga 18, ka murda → na 12), taling/tarung 10,7%, pada 8,1% (adeg-adeg → suku 30, →
  taling 13), tanda bawah 7,2%, spasi 5,1%, pangkon 3,6%; persepuluh pertama baris 0,19 suntingan per karakter, sisanya
  0,11-0,14 (sebagian besar adeg-adeg pembuka). `fase7_track_rare` di baris cetak: aksara langka 190 → 108 suntingan, pada
  107 → 76, tetapi aksara umum 386 → 425 dan pangkon 46 → 56.
  Rinci `compare_runs.py --kind cetak` (2026-10-16 23:20) untuk `fase7_track_rare` − `fase7_track` di baris cetak: recall 44
  codepoint langka 58,2% (n = 268) dengan presisi 48,0% (325 keluaran); murda 44,6% (n = 121) presisi 83,1%; adeg-adeg
  pembuka 91,3% (n = 69) presisi 64,3% (35 dari 98 palsu); CER 133 baris cetak TANPA aksara langka +0,87 [+0,12; +1,79]
  (efek samping nyata, tetapi lebih kecil dari manfaatnya). Di baris tulisan tangan presisinya 32,0%.
  **Margin saat membaca, dipisah (2026-10-17 01:00):** `fase8_std` dengan `--pad-ratio 0.06` = cetak 11,23% (tanpa margin
  12,63%; −1,40) / tulisan tangan 26,35% (−1,40) / semua 18,95% (sama dengan laporan `fase8_std_G3_full_pad006`).
  81% baris cetak (251 / 308) tintanya menyentuh tepi atas DAN bawah potongan; di baris itu 21% tanda atas salah baca lawan
  12-14% di baris yang punya ruang (sebagian karena baris bertanda atas memang lebih sering menyentuh tepi atas). Augmentasi
  `tight` hanya memotong rapat ~separuh citra latih dan memangkas paling banyak 2% tinggi. Calon: augmentasi potong rapat
  yang lebih keras, ATAU margin saat membaca (mengubah cara baca resmi = keputusan user; nilainya disetel di dev sintetis,
  bukan di data uji).
  **Margin di dev sintetis (2026-10-17 05:40, `out/compare/pad_dev.json`, `fase9_ctrl`, 300 baris val, 9 font latih,
  augmentasi `fase5`):** baris yang SELALU dipotong rapat 1,52% / 1,34% / 1,39% / 1,40% / 1,60% / 2,01% pada margin 0 / 0,02 /
  0,04 / 0,06 / 0,08 / 0,12; tanpa potong rapat paksa 1,34% / 1,39% / 1,33% / 1,53% / 1,77% / 2,27%. Di dev margin hampir tidak
  berpengaruh (terbaik 0,02: −0,18 poin), jadi keuntungan −1,40 poin di cetak nyata datang dari ciri potongan nyata yang
  belum ditiru data sintetis. Yang terlihat di potongan cetak nyata (lembar kontak 8 potongan acak): **pecahan tinta baris
  tetangga** di tepi atas (tanda bawah / pasangan baris di atasnya) dan tepi bawah, selain tepi rapat, bintik, dan tepi
  goresan bergerigi. Ukuran kasar (>= 2 komponen kecil < 22% tinggi yang menyentuh tepi atas/bawah; ikut menghitung tanda baris
  itu sendiri): 243 / 308 baris cetak, CER `fase9_ctrl` 12,59% lawan 11,34%, sisipan tanda atas 0,28 lawan 0,14 per baris.
  Calon perlakuan: augmentasi pecahan baris tetangga (tempel potongan bawah/atas baris lain di tepi citra latih sebelum
  potong rapat), diuji dengan kontrol seperti biasa.
  **Font Noto Sans Javanese Bold** (rilis resmi notofonts/javanese `NotoSansJavanese-v2.005`, OFL; berbeda versi dari Regular
  di repo): lolos keempat cek otomatis Fase 0 dan lembar kontak 13 kasus tampak benar (diperiksa Claude, belum mata
  manusia). Belum dipakai; calon perlakuan sesudah fase 10, karena huruf cetak majalah jauh lebih tebal dan lebar daripada
  sebagian besar font latih (yang paling dekat: Noto, abmAksaJawa Bold, GumregahNew). Empat font `fonts/extra_review` justru
  bergaya miring kaligrafis (lebih mirip tulisan tangan daripada cetak majalah).
  **`fase9_slant` selesai 2026-10-17 00:06** (val bersih 0,197%): G3 **16,54%** (pad 0,06: 16,08%), q100 G1 0,32%, G2 0,81%.
  Bacaan greedy di luar ekspor (sama dengan laporan G3): cetak **13,09%** / tulisan tangan **19,86%** (`fase8_std` 12,63% /
  27,75%): penurunan G3 berasal dari tulisan tangan; baris cetak sedikit naik. Efek render miring dibaca dari `fase9_slant` −
  `fase9_ctrl` (aturan dan adendum fase 9); `fase9_ctrl` mulai 00:09.
  **Rencana sesudah fase 9 (keputusan user 2026-10-16: "kalau dah selesai lanjut terus ya, gak perlu nunggu perintahku"):**
  pilih dasar dari aturan fase 9, lalu langsung `scripts/run_fase10.sh <dasar>` + `scripts/after_fase10.sh <dasar>`;
  pergantian angka resmi ditunda sampai fase 10 selesai (evaluasi resmi G1/G2 dan beam + LM tidak boleh berjalan bersama
  training). Pertanyaan terbuka untuk user: apakah target G3 < 8% dinilai pada baris cetak saja (`out/cek_tulisan_tangan/`
  berisi lembar kontak untuk memeriksa pembagian halaman).
  **Hasil fase 9 (2026-10-17 04:22; `out/compare/slant_crnn_fase9_slant_vs_crnn_fase9_ctrl.json`,
  `crnn_fase9_slant_vs_crnn_fase9_ctrl.md`).** `fase9_ctrl` selesai 04:00 (val bersih 0,169%): G3 **20,31%** (pad 0,06:
  18,85%), q100 G1 0,38%, G2 0,72%. Selisih `fase9_slant` − `fase9_ctrl`, SK 95% per halaman: (a) 270 baris miring 28,60% →
  19,76%, **−8,84 [−11,78; −6,02]** (membantu, tetapi semuanya tulisan tangan); (b) 424 baris tegak −0,94 [−2,49; +0,43];
  q100 G1 0,32% / 0,38%, G2 0,81% / 0,72%; (e) **308 baris CETAK 12,34% → 13,09%, +0,75 [+0,15; +1,39]: render miring
  memperburuk cetak secara nyata**; (f) 437 baris tulisan tangan 27,96% → 19,86%, −8,10 [−11,33; −5,21]; (c) G3 keseluruhan
  −3,77 [−5,67; −1,73]. Dosis-respons sintetis (`slant_synthetic.json`, javatext 300 baris val, miring 0 / 0,2 / 0,35 / 0,5 /
  0,7): `fase9_slant` 0,28 / 0,25 / 0,31 / 0,49 / 1,23%, `fase9_ctrl` 0,28 / 0,96 / 4,91 / 14,75 / 43,38%, `fase8_std` 0,31 /
  1,34 / 4,94 / 17,00 / 48,21% (searah; miring 0 tidak naik). Bentuk ejaan baku 168/212 lawan 163/212 (SK [−2,7; +6,9]),
  presisi 85,0% lawan 78,4%. Menurut aturan dan adendumnya: **dasar berikutnya = `fase9_ctrl`** (cetak terbaik sejauh ini: 12,34%,
  `fase8_std` 12,63%); render miring tidak dipakai untuk model cetak. Angka resmi tidak diganti sekarang (ditunda sampai
  fase 10; keputusan user).
  **Fase 10 (dimulai 2026-10-17 04:35 sesudah `after_fase9.sh` selesai 04:34; `scripts/run_fase10.sh fase9_ctrl`,
  `scripts/after_fase10.sh fase9_ctrl`, log `out/fase10_chain.log`; argumen di event `start` log diperiksa: 9 font, tanpa
  miring, perbaikan font, ejaan baku 0,5, sisipan 0,3 / 0,15 / 0,9 / tempel; layanan terjemahan 8012 tetap mati selama
  training):** `fase10_rare` = `fase9_ctrl` + 1.500 langkah, resep `fase9_ctrl` (jarak 0,5 /
  0,3, ejaan baku 0,5, augmentasi `fase5`, drop-space 0,5, `--font-fixes`) + sisipan aksara langka versi perbaikan
  (`--rare-insert-prob 0.3 --rare-opener-prob 0.15 --rare-max-similarity 0.9 --rare-attach`, sama persis dengan
  `fase7_track_rare`, jadi nilainya tidak disetel ulang); `fase10_ctrl` = sama tanpa sisipan, titik lanjut ditiru.
  **Aturan analisis fase 10, ditetapkan SEBELUM ada angka:** (a) titik akhir utama = CER **308 baris cetak** `fase10_rare`
  − `fase10_ctrl` (`compare_runs.py --kind cetak` / `eval_slant.py real`), "membantu" hanya bila SK 95% bootstrap per halaman
  di bawah 0 (replikasi temuan fase 7: −0,80 [−1,33; −0,28]); (b) efek samping di baris cetak: presisi keluaran aksara
  langka dan adeg-adeg pembuka palsu (dilaporkan; fase 7: 48,0% dan 35 dari 98), CER baris cetak TANPA aksara langka
  (dilaporkan dengan SK; fase 7 +0,87), q100 G1/G2 (di dalam ~1 poin), bentuk ejaan baku (`eval_spelling.py real`, tidak
  boleh turun nyata), evaluasi sintetis aksara langka (`eval_rare.py`, javatext: teks biasa tidak boleh naik di luar ~0,1
  poin); (c) tulisan tangan dilaporkan terpisah, G3 keseluruhan dengan SK per halaman sebagai arah; (d) dasar berikutnya dan
  usulan angka resmi dipilih dari (a)-(b), bukan dari G3 atau tulisan tangan; mengganti angka resmi tetap keputusan user.
  **Hasil fase 10 (2026-10-17 15:21; `out/compare/crnn_fase10_rare_vs_crnn_fase10_ctrl_cetak.md`,
  `slant_crnn_fase10_rare_vs_crnn_fase10_ctrl.json`, `spelling_crnn_fase10_rare_vs_crnn_fase10_ctrl.json`).** `fase10_rare`
  dievaluasi 09:21 (val bersih 0,131%; dihentikan Claude sesudah last.pt langkah 1.000 karena charger tercabut, dilanjutkan
  07:18): G3 19,69% (pad 0,06: 18,65%), q100 G1 0,32%, G2 0,70%. `fase10_ctrl` selesai 14:40 (val bersih 0,141%; titik
  lanjut 1.000 ditiru; charger tercabut lagi 13:59-15:10, training tetap selesai): G3 21,48% (pad 0,06: 19,53%), q100 G1
  0,29%, G2 0,63%. (a) Titik akhir utama: 308 baris cetak 12,29% lawan 12,73%, **−0,44 [−1,00; +0,07] → TIDAK memenuhi
  aturan** (temuan fase 7 −0,80 [−1,33; −0,28] tidak terulang; per halaman 45 lebih baik, 22 lebih buruk, 5 sama; spasi
  dibuang −0,49 [−1,02; +0,01]). (b) Efek samping di baris cetak: recall 44 codepoint langka 72,8% lawan 14,2% (kontrol
  hanya pa cerek, nga lelet, keret dari ejaan baku), murda 53,7% lawan 0 (presisi 80,2%), adeg-adeg pembuka 64/69 lawan 0;
  presisi aksara langka 48,1% (405 keluaran; da mahaprana 71 keluaran berpresisi 19,7%, pada isen-isen 33 palsu), pembuka
  palsu 10/239 (4,2%); **CER 133 baris cetak tanpa aksara langka 12,11% lawan 10,59%, +1,52 [+0,68; +2,33]** (fase 7: +0,87;
  efek samping nyata dan lebih besar); q100 G1/G2 di dalam 1 poin; bentuk ejaan baku 174/212 lawan 164/212 (+4,7 [+1,4;
  +8,1], tidak turun); evaluasi sintetis aksara langka (`out/compare/fase10/rare_synthetic.md`, javatext, 2.000 baris test):
  teks biasa 0,27% lawan 0,27% (+0,00 [−0,02; +0,03]: tanpa efek samping di sintetis), kondisi bersisipan recall 44
  codepoint 82,6% lawan 7,0% dengan presisi 92,8%, murda 82,7%, pembuka 100%; seperti fase 7, keluaran palsu hanya muncul
  di cetakan nyata (celah bentuk huruf). (c) Tulisan tangan 26,79% lawan 29,88% (−3,09 [−4,75; −1,80]); G3 −1,80 [−2,61; −1,03] (arah saja:
  penurunan G3 terutama dari tulisan tangan). Tambahan 1.500 langkah TANPA sisipan tidak menolong: `fase10_ctrl` −
  `fase9_ctrl` cetak +0,39 [+0,00; +0,85], tulisan tangan +1,92 [+0,05; +3,98]; `fase10_rare` − `fase9_ctrl` cetak −0,05
  [−0,56; +0,50]. (d) Menurut aturan: sisipan aksara langka TIDAK dipakai untuk model cetak (manfaat CER tidak terbukti,
  efek samping pada teks umum nyata). Akibatnya model tanpa sisipan tidak membaca murda dan adeg-adeg sama sekali; bila
  kemampuan itu dianggap lebih penting daripada CER, itu keputusan user.
  **Dasar fase 11 = `fase9_ctrl`** (dipilih Claude SESUDAH melihat angka fase 10, bukan aturan yang ditetapkan sebelumnya):
  resep tanpa sisipan, dan `fase10_ctrl` tidak lebih baik dari induknya di baris cetak (+0,39 [+0,00; +0,85]). Dengan dasar
  itu `fase11_ctrl` mengulang resep `fase10_ctrl` persis (bobot awal, argumen, seed, urutan batch, berhenti dan lanjut di
  langkah 1.000); bedanya hanya acak render dan perangkat, jadi selisih keduanya = ukuran pertama derau antar-run.
  **Fase 11 (pecahan baris tetangga; `scripts/run_fase11.sh fase9_ctrl`, `scripts/after_fase11.sh fase9_ctrl`, log
  `out/fase11_chain.log`):** `fase11_neighbor` = `fase9_ctrl` + 1.500 langkah, resep `fase9_ctrl` (jarak 0,5 / 0,3, ejaan
  baku 0,5, augmentasi `fase5`, drop-space 0,5, `--font-fixes`) + `--neighbor-prob 0.5` (`src/neighbors.py` lewat
  `SyntheticLines`: pita bawah satu baris latih lain ditempel di atas citra dan pita atas baris latih lain di bawahnya, font
  dan ukuran sama, tanpa spasi; tiap sisi kosong dengan peluang 0,3; pita 5-25% tinggi tinta baris tetangga, jarak kosong
  2-15% tinggi tinta baris utama; diundi sesudah semua undian lain dan sebelum augmentasi, jadi `tight` memotong rapat
  sampai pecahan menyentuh tepi; peluang 0 tidak mengonsumsi angka acak); `fase11_ctrl` = sama tanpa pecahan, titik lanjut
  ditiru; kedua lengan berhenti dan lanjut di langkah 1.000. Peluang 0,5 mengikuti perlakuan lain (tidak disetel); rentang
  pita dan jarak dari lembar kontak potongan cetak nyata (data uji, dilihat, bukan disetel pada angka).
  Kode (2026-10-17 16:30, sesudah `after_fase10.sh` selesai): `SyntheticLines(neighbor_prob)` dan `src.train
  --neighbor-prob`; `tests/test_neighbor_option.py` memeriksa bahwa semua opsi bernilai 0 tidak memanggil random() sama
  sekali (perbandingan dua dataset tidak bisa menangkap undian yang selalu terjadi di keduanya), pecahan ada di atas dan di
  bawah pita tinta baris utama yang tetap utuh, baris tetangga diundi dari kumpulan baris latih tanpa spasi dengan font dan
  ukuran baris utama, dan validasi tidak berubah. 10 mutan terhadap kode itu dan 22 mutan terhadap
  `scripts/eval_neighbor.py` semuanya mati (dua yang lolos dari versi pertama test melahirkan dua test terakhir); 409 test
  Python lolos.
  **Aturan analisis fase 11, ditetapkan 2026-10-17 SEBELUM ada angka:** (a) titik akhir utama = CER **308 baris cetak**
  `fase11_neighbor` − `fase11_ctrl` (`compare_runs.py --kind cetak`, `eval_slant.py real`), "membantu" hanya bila SK 95%
  bootstrap per halaman di bawah 0; (b) efek samping: q100 G1/G2 (di dalam ~1 poin), bentuk ejaan baku (`eval_spelling.py
  real`, tidak boleh turun nyata), sintetis javatext 300 baris val (`scripts/eval_neighbor.py synthetic`, citra sama untuk
  semua checkpoint): CER kondisi "bersih" dan "bersih rapat" tidak boleh naik di luar ~0,1 poin; (c) mekanisme, dilaporkan
  dengan SK per halaman (`eval_neighbor.py real`, baris cetak): tanda atas dan tanda bawah yang DISISIPKAN keluaran per baris
  (diharapkan turun), CER baris berpecahan (>= 2 komponen tinta yang menyentuh tepi atas/bawah dan lebih pendek dari 22%
  tinggi potongan; ukuran kasar yang ikut menghitung tanda baris itu sendiri) lawan baris lain, dan CER sintetis kondisi
  "pecahan rapat" (diharapkan turun); (d) tulisan tangan dan G3 keseluruhan dilaporkan terpisah sebagai arah; margin 0,06
  dilaporkan seperti biasa; (e) derau antar-run: `fase11_ctrl` − `fase10_ctrl` (resep sama) dilaporkan dengan SK per
  halaman untuk baris cetak dan semua baris; satu pasang ulangan hanya satu tarikan derau, jadi bila besarnya sebanding
  dengan efek (a), kesimpulan (a) disebut rapuh; (f) dasar berikutnya dan usulan angka resmi dipilih dari (a)-(b), bukan
  dari G3 atau tulisan tangan; mengganti angka resmi tetap keputusan user.
  **Hasil fase 11 (2026-10-18 01:31; `out/compare/crnn_fase11_neighbor_vs_crnn_fase11_ctrl_cetak.md`,
  `slant_crnn_fase11_neighbor_vs_crnn_fase11_ctrl.json`, `neighbor_crnn_fase11_neighbor_vs_crnn_fase11_ctrl.json`,
  `neighbor_synthetic.json`).** `fase11_neighbor` dievaluasi 21:12 (val bersih 0,188%; laptop di-Hibernate dari menu Start
  17:37-18:22, training membeku lalu lanjut sendiri): G3 20,50% (pad 0,06: 18,80%), q100 G1 0,32%, G2 0,70%. `fase11_ctrl`
  selesai 00:55 (val bersih 0,141%): G3 19,99% (pad 0,06: 19,31%), q100 G1 0,29%, G2 0,61%. (a) Titik akhir utama: 308
  baris cetak 13,07% lawan 12,69%, **+0,38 [−0,32; +1,15] → TIDAK memenuhi aturan** (arahnya memburuk). (b) Efek samping:
  q100 di dalam 1 poin; bentuk ejaan baku 167/212 lawan 162/212 (+2,4 [−1,3; +5,5]); sintetis javatext 300 baris val:
  bersih 0,28% lawan 0,28%, bersih rapat 0,31% lawan 0,25% (+0,05, di dalam batas 0,1). (c) Mekanisme: tanda atas yang
  DISISIPKAN di baris cetak 56 lawan 93 (0,18 lawan 0,30 per baris, SK [−0,21; −0,03]: pecahan tepi memang tidak lagi
  dibaca sebagai tanda atas); sintetis "pecahan rapat" 0,39% lawan 1,39% (tanda atas sisipan 2 lawan 124); tetapi CER baris
  cetak berpecahan +0,62 [−0,15; +1,49], tanpa pecahan −0,54 [−1,79; +0,60]. Rincian suntingan baris cetak (penjajaran
  Levenshtein, eksplorasi): tanda atas HILANG 166 lawan 126 (+40), tertukar 141 lawan 165, aksara hilang 113 lawan 89, pada
  tertukar 76 lawan 59; total 1.304 lawan 1.266. Dugaan: model belajar mengabaikan tinta kecil di tepi atas, termasuk tanda
  atas baris itu sendiri yang menyentuh tepi potongan (81% baris cetak menyentuh tepi atas dan bawah), sedangkan di render
  sintetis tanda baris sendiri selalu berjarak dari pecahan. (d) Tulisan tangan +0,64 [−0,77; +2,51]; G3 +0,52 [−0,28;
  +1,47]. (e) **Derau antar-run, pertama kali terukur: `fase11_ctrl` − `fase10_ctrl` (resep, bobot awal, seed, urutan
  batch, titik lanjut sama; beda hanya acak render dan perangkat): cetak −0,04 [−0,59; +0,44], tulisan tangan −2,89
  [−5,37; −1,06], semua −1,50 [−2,66; −0,49]**; val bersih di langkah 500 0,54% lawan 0,16%, di akhir sama. Satu tarikan
  derau: CER baris cetak cukup stabil antar-run, G3 keseluruhan tidak (tulisan tangan bergeser ~3 poin), jadi selisih G3
  1-2 poin di catatan sebelumnya bisa saja derau; titik akhir baris cetak memang pilihan yang tepat. (f) Pecahan baris
  tetangga tidak dipakai. Tambahan 1.500 langkah tanpa perlakuan dari `fase9_ctrl` dua kali sedikit memperburuk baris cetak
  (`fase10_ctrl` +0,39 [+0,00; +0,85], `fase11_ctrl` +0,35 [−0,06; +0,76]).
  **Putaran sintetis dihentikan (2026-10-18):** fase 9-11 menggeser CER cetak paling banyak ~0,5 poin, dan tambahan langkah
  tidak menolong. Calon angka resmi = `fase9_ctrl` (cetak 12,34%, G3 20,31%), tetapi terhadap resmi `fase8_std` (cetak
  12,63%, G3 20,35%) tidak berbeda nyata: cetak −0,29 [−0,93; +0,28], semua −0,03 [−0,59; +0,48]; kelebihannya terutama
  data latih yang lebih bersih (`--font-fixes`). **Evaluasi calon** (2026-10-18 01:31-02:21, `out/candidate_eval.log`,
  tanpa mengganti angka resmi; nama laporan sama dengan `make_official.sh`, jadi skrip itu melewatinya): **G1 0,233%**
  (resmi 0,235%), **G2 0,719%** (resmi 0,977%), G3 20,31% (resmi 20,35%); beam + LM (`out/beam/fase9_ctrl_o5*.json`, dev
  300 baris dirender dengan resep checkpoint itu): model bahasa berejaan baku menang di dev (greedy 1,28% → 1,00%, α 0,25,
  β 0,5; model bahasa lama 1,12%), NusaAksara 20,31% → **17,68%** (resmi 18,02%; 329 baris membaik, 84 memburuk), render
  bersih javatext 0,22% → 0,18%.
  **Angka resmi = `fase9_ctrl` (keputusan user 2026-10-18: "ikuti rekomendasimu"), menggantikan `fase8_std`:**
  `OFFICIAL_RUN` dan `DEFAULT_CHECKPOINT` = `fase9_ctrl`, `src/serve.py` α 0,25 / β 0,5 / model bahasa
  `o5_n300000_ds0.5_std0.5.pkl`, kasus `fase9_ctrl` di `pairs()` `make_official.sh` (pasangannya sendiri, lanjutan fase 10
  dan 11, dan pasangan di rantainya yang dikutip kartu); `bash scripts/make_official.sh fase9_ctrl` dijalankan 08:34.
  Keputusan lain hari itu, juga mengikuti rekomendasi: (1) G3 resmi tetap dihitung atas 745 baris; CER baris cetak (308
  baris, pembagian halaman `eval_slant.HANDWRITTEN_PAGES`, belum diperiksa manusia) dilaporkan DI SAMPINGNYA, bukan
  menggantikannya (`export_results.gate_parts` memasang kunci `parts` ke gerbang G3 manifest: baris cetak dan tulisan
  tangan dengan CER, baris, halaman, dan panjang referensi, dan ekspor berhenti bila rinciannya tidak menjumlah ke G3;
  kolom `gates.parts` json, migrasi `2026_10_18_000001`; `ResultImporter::gateParts` menolak rincian rusak di dalam
  transaksi; ubin G3 Ringkasan menampilkannya dengan catatan bahwa klasifikasi halaman belum diperiksa manusia; 27 mutan
  terhadap kode itu semuanya mati, dua yang lolos dari versi pertama test memperketat datanya); (2) margin saat membaca TIDAK dipakai di jalur resmi: nilai yang disetel di dev sintetis (0,02) hampir
  tanpa efek, dan 0,06 hanya terlihat membantu di data uji; (3) sisipan aksara langka tidak dipakai di model resmi (aturan
  fase 10), jadi model resmi tidak membaca murda dan adeg-adeg. Tuas berikutnya butuh data cetak nyata untuk latihan (43
  baris Commons menunggu pembaca aksara).
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
    Ablasi, Dataset, Metode; Analisis: Penjelajah baris, Kesalahan aksara; Alat: Demo, Kamus, Terjemahan). Mockup disetujui user 2026-09-24:
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
    Salinan statis tidak menjalankan permintaan Livewire. **Uji interaksi di browser sungguhan tanpa menyentuh
    database situs (dipakai 2026-10-05 untuk halaman Kamus):** proses diberi `DB_CONNECTION=sqlite`,
    `DB_DATABASE=<berkas di folder sementara>`, `SESSION_DRIVER=file`, `CACHE_STORE=array` (variabel proses menang
    atas `web/.env` selama tidak ada cache konfigurasi; pastikan dulu lewat tinker bahwa nama databasenya berkas
    itu), lalu `migrate`, akun uji, `aksara:dictionary`; server = `php -S 127.0.0.1:<port>
    ../vendor/laravel/framework/src/Illuminate/Foundation/resources/server.php` DIJALANKAN DARI `web/public`
    (perutenya memuat `index.php` dari folder kerja); Edge `--headless=new --remote-debugging-port` dikemudikan skrip
    Node lewat WebSocket bawaan (masuk, klik, `Input.insertText`, baca DOM). Sebelum memakai sebuah port, cek tidak
    ada proses lama yang masih mendengarkan di situ.
    **Menu Kamus (permintaan user 2026-10-05, "isinya mengenai kamus/dictionary"; isinya keputusan Claude, belum
    dikonfirmasi):** `web/app/Livewire/Pages/Kamus.php` menampilkan tiga rujukan yang memang ada di alur: kamus
    aksara (charset `data/tokenizer.json`, lewat `App\Support\AksaraCatalog`; bacaan Latin = draf Transliterator),
    kamus koreksi OCR (kondisi "kamus" = beam + LM karakter; angka greedy vs beam diambil dari metrik yang diimpor,
    dengan peringatan bila checkpoint-nya bukan pipeline resmi), dan leksikon tingkat tutur (`SpeechLevel::lexicon()`).
    **Menu Terjemahan (permintaan user 2026-10-05: "menerjemahkan dari bahasa indonesia ke teks jawa aksara dan
    sebaliknya"; user bertanya "python atau php", dijawab keduanya menurut tugasnya):** `Pages/Terjemahan.php`.
    *Python hanya untuk model:* satu NLLB-200-distilled-600M melayani dua arah lewat `web/tools/translate_service.py`
    (`POST /translate {texts, source, target}` dengan `jv`/`id`; bentuk lama `{text}` tetap untuk Demo; layanan harus
    dinyalakan ulang sesudah `nllb.py` berubah). *PHP untuk alih aksara dan halaman:* `App\Support\AksaraWriter`
    (Latin -> aksara, aturan sendiri) dan `Transliterator` (arah balik). Indonesia -> Jawa: terjemahan Latin dipulihkan
    tanda é-nya (`TalingRestorer`), boleh disunting, aksara dihitung dari teks Latin yang terlihat. Jawa -> Indonesia:
    aksara dilatinkan dulu. Terukur 2026-10-05:
    (1) alat lama `scripts/translit/translit.js` (honocoroko) TIDAK layak untuk teks umum: "nglakoni" diawali cecak,
    "foto" -> fa-ha-o, "saé" menyisakan huruf Latin, kata ulang kehilangan pangkon (di korpus kata semacam itu tersaring).
    Pada 22.519 kata Wikipedia yang bisa dibandingkan, `AksaraWriter` mode korpus sama dengan alat lama di 99,78%.
    (2) ejaan baku (pa cerek, nga lelet, keret, n di depan c/j ditulis nya, angka diapit pada pangkat) diturunkan dari
    ejaan aksara lema kamus, bukan dari ingatan: sama di 2.173 dari 2.231 lema berejaan baku (97,4%; mode korpus 93,1%; entri varian ejaan tidak dihitung). Angka itu
    konstanta `AksaraWriter::DICTIONARY_AGREEMENT`, dijaga test dan ikut berubah bila kamus dibangun ulang.
    **Aksara murda dan swara untuk nama (permintaan user 2026-10-12: "sepertinya perlu di handling", lalu "sesuai
    baiknya darimu"):** `AksaraWriter::write()` / `fromLatin(names: true)`, kotak centang di halaman Terjemahan,
    bawaannya MATI supaya hasil lama dan `DICTIONARY_AGREEMENT` tidak berubah. Berlapis: (i) nama satu kata di kamus
    yang ejaan aksaranya bermurda/berswara memakai ejaan kamus apa adanya (11 nama: Bima, Suwarnabhumi, Arjuna, Indra,
    ...), juga di awal kalimat; (ii) kata berhuruf besar lain yang bukan kata pertama kalimat = nama: vokal pembuka ->
    aksara swara (pepet tidak punya) dan sesudah itu tanpa murda, selain itu SATU murda pada konsonan pertama yang
    punya bentuk murda dan tidak menutup suku kata (Thomas, Marwan: tanpa murda; diperbaiki sesudah tinjauan keempat).
    Enam dari 11 nama kamus (aditya, agung, arga, arya, bima, ukir) juga kata biasa berhuruf kecil di kamus yang sama:
    di AWAL kalimat kata-kata itu ditulis biasa ("Agung banget omahe" bukan nama), di tengah kalimat tetap ejaan kamus. Lapis (ii) kaidah umum dari pengetahuan Claude, TIDAK diturunkan dari data (kamus hanya punya
    4 entri bermurda dan 20 berswara, pemakaiannya tidak seragam) dan belum diperiksa pembaca aksara; halaman
    menyebut kata mana yang ditulis dengan kaidah itu. Ganti lapis (ii) bila ada pedoman ejaan atau pembaca aksara.
    (3) e tanpa tanda: 53% kata ber-e benar bila semua dibaca pepet; leksikon `web/database/dictionaries/jv-taling.json`
    (8.714 kata, `web/tools/taling_lexicon.py`: 8.615 dari artikel Wikipedia Jawa yang bertanda kuat, 99 dari lema
    kamus Wiktionary bahasa Inggris) menaikkannya ke 90% pada artikel yang ditahan. Aturan "e akhir kata = é" tidak menambah apa-apa di atas leksikon.
    (4) **NLLB tidak mengenal "ê":** "pêkên" -> "ke sana", "sêga" -> "tiga buah"; tanpa tanda -> "pasar", "nasi".
    `TranslationService::forModel()` membuang tanda itu di semua panggilan langsung (Terjemahan, Demo). Batch
    `aksara:translate` BELUM (inputs() masih mengirim ê, juga dari transliterasi manusia), jadi chrF 36,2 / 19,4 yang
    dilaporkan kemungkinan terlalu rendah; menjalankannya ulang ~1 jam CPU dan mengubah angka tahap 3: keputusan user.
    **Batch dijalankan ulang tanpa "ê" 2026-10-12 (keputusan user; `BATCH_KEEPS_PEPET_MARK = false`, 51 menit CPU):**
    dari alih aksara manusia **chrF 38,6 / BLEU 11,6** (dulu 36,2 / 10,5: naik 2,4 poin chrF); dari bacaan greedy
    model resmi `fase7_track` chrF 18,5 / BLEU 0,4 (sejak 2026-10-16 dari `fase8_std`: chrF 18,9 / BLEU 0,5; sejak
    2026-10-18 dari `fase9_ctrl`: chrF 19,1 / BLEU 0,5; tiap kali berkas batch lama dicadangkan
    `web/storage/app/mt/*_sebelum_<tanggal>.*`, lalu `aksara:translate --dump` menulis ulang berkas lengkap dari
    database). Angka OCR lama (19,4 / 1,0) dibuat dari beam + LM `fase5_fonts`
    DAN masih ber-"ê", jadi tidak sebanding; walau G3 turun dari 33,8% ke 21,5%, chrF dari keluaran OCR tidak naik
    (teks 1 dari 5 karakter salah masih terlalu rusak untuk diterjemahkan). Hasil lama dicadangkan di
    `web/storage/app/mt/*_sebelum_20261012.*` (tidak ikut git). Catatan pepet di butir NLLB-200 halaman Metode tidak
    lagi tampil.
    Mutu terjemahan Indonesia -> Jawa belum diukur (tidak ada data uji arah itu); contoh salah yang terlihat: "murah" ->
    "larang regane", "nenek" hilang. Halaman menyebut hasilnya draf.
    **Kamus kata (permintaan user 2026-10-05: "ditambahkan juga kamus bahasa jawa dan kamus bahasa indonesia"):** dua
    bagian baru di atas kamus aksara, isinya DATA PIHAK KETIGA (bukan hasil hitung proyek): tabel `dictionary_entries`
    dan `dictionary_sources`, berkas jadi di `web/database/dictionaries/` (ikut git; sumber & lisensi di `SUMBER.md`),
    dibangun `web/tools/dictionaries.py`, diimpor `php artisan aksara:dictionary`. Sumber (semuanya CC BY-SA 4.0): Wiktionary
    bahasa Inggris lewat kaikki.org untuk bahasa Jawa (3.966 entri, 3.186 lema; arti berbahasa Inggris, ejaan aksara
    di 3.148 entri, ragam di 654, catatan padanan ngoko/krama di 245) dan bahasa Indonesia (39.966 entri, arti
    berbahasa Inggris); **Wikikamus (id.wiktionary) bagian bahasa Jawa** (1.922 entri, 1.822 lema, 593 di antaranya
    juga ada di sumber pertama; arti berbahasa INDONESIA, 942 contoh kalimat berterjemahan). Kamus Jawa jadi 5.888
    entri / 4.415 lema; arti Indonesia didahulukan, dan kata Indonesia menemukan padanan Jawanya lewat pencarian balik
    ("makan" -> mangan, dhahar, madhang). Wikikamus ditambahkan Claude 2026-10-05 untuk memenuhi permintaan kamus
    (user belum menanggapi khusus): dump bertanggal 20261001 diurai `wikt_entries()` (cocok dengan pengurai riset di
    1.835 dari 1.860 lema); dibuang: 47 bagian bertanda impor KBBI, arti yang hanya mengulang lemanya, dan "arti"
    berejaan Jawa bertanda ê/é/è (halaman berjudul kata Indonesia: pijat -> pijêt). Dari Wikikamus TIDAK diambil ragam
    dan ejaan aksara (label ragamnya bertentangan dengan Wiktionary Inggris di 11 dari 60 kata yang punya keduanya,
    mis. dalan "krama inggil"; 3 dari 115 ejaan aksaranya salah); labelnya tetap tampil sebagai teks arti. Leksikon
    pemulih é hanya memakai lema Wiktionary Inggris.
    Pencarian (`DictionarySearch`): kolom turunan `lookup` (tanpa diakritik) dan `gloss_text`, LIKE biasa dan urutan
    byte (`COLLATE "C"` di PostgreSQL) supaya sama di PostgreSQL dan SQLite. Satu kata dicari sebagai katanya (imbuhan
    dengan tanda hubungnya: "-an"; kata ulang yang diketik berspasi "anak anak" = anak, dengan entri anak-anak
    didahulukan; "-rumah" = rumah hanya bila "rumah" memang ada); beberapa kata = frasa utuh seperti diketik ("mau tak
    mau", "ke- -an") lalu arti per kata; kutip pembungkus dibuang (`unquote()` memakai pola /u dan posesif: `trim()`
    dengan daftar karakter non-ASCII memotong per BYTE dan pernah merusak "śrī" dan teks berawalan "—"); kueri dipotong
    di 500 karakter; pencarian balik memakai bentuk kata seperti tersimpan di arti ("son-in-law") dan butuh >= 3
    huruf; entri yang cocok lewat ejaan aksara tidak diulang di daftar balik; kode "U+A9B6" hanya milik kamus aksara;
    leksikon tingkat tutur di halaman memakai bukti `SpeechLevel::classify()` yang sama dengan uraian.
    **Aturan pembangun data yang lahir dari dua tinjauan 2026-10-05 (jangan dikembalikan):** romanisasi dibaca per
    ENTRI dari kepala entrinya (satu halaman beraksara bisa memuat dua kata: ꦲꦗꦶ = haji dan aji), dengan "nyc/nyj"
    hasil transliterasi otomatis ditulis "nc/nj" (jancuk, bukan janycuk); padanan ragam dibaca dari ARGUMEN templat
    kepala (pasangan label-kata di `head`: ngoko, krama, krama ngoko, krama inggil, kawi, ngoko/krama lugu dan alus)
    dan dari tabel "Javanese register set", bukan dari bentuk bertag hasil wiktextract (yang menempelkan "lugu/alus"
    ke kata dan kehilangan kw=); "aksara, latin, ejaan lama" dalam satu ragam = SATU kata; ragam entri dari kepala
    entri ("(krama)", "(krama, ngoko)") lalu dari kolom yang memuat kata itu, dan tanda kurung PERTAMA sesudah judul
    beraksara adalah romanisasi, bukan label ("ꦏꦿꦩ (krama)" = kata "krama"); kata yang sama di ngoko dan krama =
    krama-ngoko; catatan hanya memuat padanan di ragam LAIN, diromanisasi bila sumbernya memberi. Ejaan aksara entri
    berjudul Latin: dari argumen j=/c= dan bentuk beraksara di halaman itu, HANYA bila `reads_as()` (pembanding kasar
    aksara-Latin) menyatakannya terbaca sebagai kata itu; itu menolak ejaan salah tempel di sumber (ngétan <- ꦮꦺꦠꦤ꧀,
    émah <- ꦲꦺꦴꦩꦃ, 7 entri) dan tidak menolak satu pun dari 2.831 ejaan yang sudah ada. Penunjuk "Carakan spelling
    of" dikunci kata persisnya (enèm bukan enem); varian ejaan yang beda diakritik saja dari SATU kata yang
    ditunjuknya mewarisi ejaan aksaranya. Sense bertag romanization yang teksnya "... spelling/form of X" adalah entri
    varian (opo, sego, boekoe: 538 entri varian/penunjuk), dan judul penunjuk romanisasi yang tidak tercapai lewat
    kata kepala (candi untuk candhi) mendapat entri kecil. Arti bersarang wiktextract dirangkai "induk: anak" (tanpa
    itu sekitar 6.000 arti anak di 2.288 entri Indonesia hilang); induk yang lebih panjang dari 40 karakter (tanpa
    labelnya) ditulis sekali dan anaknya menyusul dengan label induknya; sisa templat sumber ("{{place|...") dibuang.
    Angka kecocokan `AksaraWriter` dihitung pada lema berejaan baku saja: entri varian ejaan ("asoe", "opo") memang
    bukan ejaan baku (dengan semua entri beraksara angkanya 91,1%, dan itu bukan ukuran aturannya). Belum beres:
    sekitar 28 kata tanpa ejaan aksara walau halaman beraksaranya ada (sumber tidak memasangkannya), 6 catatan masih
    memuat padanan beraksara, dan kesalahan di sumber yang tidak terdeteksi pembanding kasar ikut terbawa.
    `tests/Feature/DictionaryTest.php` mengimpor berkas jadi yang ikut repo dan memeriksa aturan-aturan ini pada
    datanya, jadi bangun ulang yang merusaknya ketahuan di test.
    **Terukur 2026-10-05 dan JANGAN dilanggar:** KBBI (hak cipta Badan Bahasa, "seluruh isi KBBI Daring dilindungi Hak
    Cipta", tanpa rilis terbuka) dan Kateglo (96% definisinya teks KBBI III) tidak boleh disalin ke repo publik ini;
    Wikikamus (id.wiktionary) punya 71 ribu lema Indonesia tetapi 94% entrinya impor KBBI; kaikki edisi Indonesia tidak
    memuat kosakata dasar; Wikidata hanya 109 leksem Jawa. Halaman hanya MENAUTKAN ke KBBI Daring.
    **Menu Dataset (permintaan user 2026-10-05: "buatkan menu dataset di sidebar"; isinya dari diskusi hari itu:
    pembagian 90/5/5 dipertahankan dan ditampilkan apa adanya, bukan dibagi ulang; data pendukung di tabel kedua;
    pemakaian hanya untuk run resmi):** `Pages/Dataset.php`, kelompok Hasil di bawah Ablasi. Angkanya dari **kartu
    data** `out/results/datasets.json` buatan `scripts/export_datasets.py` (kontrak sendiri, `aksara.dataset_schema`
    1), diimpor `php artisan aksara:datasets` ke tabel `dataset_reports` (satu baris, isi utuh). `aksara:import` ikut
    mengimpornya bila berkasnya ada, dan `scripts/make_official.sh` ikut mengekspornya. Isi kartu: korpus dan
    pembagian (dihitung ulang dari `data/splits`, wajib sama dengan `corpus_stats.json`), pemakaian run resmi dan
    rantai checkpoint yang dilanjutkannya (`args.init` tiap checkpoint, proses dari `log.jsonl`, baris yang dijadwalkan
    dari urutan batch DataLoader `src.train`), karakter langka, daftar dataset, font per peran, data pendukung, dan
    `warnings` (hal yang tidak bisa dipastikan ekspor: log tidak lengkap, run dimundurkan, folder font kosong). Teks
    label NusaAksara tidak pernah ditulis ke kartu (dijaga test). Data pendukung milik web: kamus kata dan anotasi
    dihitung dari tabel web, leksikon é dari berkasnya.
    **Urutan batch training (ditemukan tinjauan independen 2026-10-05, diverifikasi dengan `src.train.make_loader`
    sungguhan; berlaku untuk SEMUA hitungan "baris yang dilihat run"):** dengan worker (`num_workers > 0`) DataLoader
    PyTorch memanggil `iter(batch_sampler)` dua kali sebelum batch pertama, dan `LengthBucketSampler.__iter__`
    menaikkan `epoch` di tiap panggilan, jadi lintasan pertama tiap proses training memakai `random.Random(seed + 1)`,
    bukan `seed + 0` (tanpa worker: `seed + 0`). Semua run di rantai memakai 2 atau 4 worker.
    `Schedule.batches(..., workers)` menirunya; versi pertama kartu memakai panggilan pertama sampler dan salah (20 langkah
    batch 32: hanya 194 dari 640 baris yang sama). Batch terakhir sebuah kelompok panjang bisa lebih kecil dari ukuran
    batch, jadi sampel run `base` 47.968, bukan 1.500 x 32.
    Terukur 2026-10-05, sesudah koreksi urutan: run resmi `fase7_track` menjadwalkan **48.000 baris** berbeda (5,2%
    bagian latih; satu proses, 1.500 langkah x 32) dari kumpulan 100.000 baris. **19.712 adalah angka `fase6_ctrl`**
    (tiga proses), bukan angka run resmi. Seluruh rantai (base -> fase5_quick -> fase5_fonts -> fase6_ctrl ->
    fase7_track): 6.298 langkah, 201.504 sampel, **64.595 baris berbeda** (6,9% bagian latih; `base` 31.984 dari
    kumpulan 50.000). 44 karakter langka masing-masing ada di 100 baris latih, dan hanya di **1-14 baris (rata-rata
    5,9)** dari yang dijadwalkan run resmi. Angka lama 64.612 / 201.536 / 1-12 (rata-rata 5,5) salah. Lebar maju
    CarakanJawa sama dengan javatext di 72 dari 73 aksara/angka/pada (dihitung ulang tiap ekspor; font latih lain 0
    dari 73). Bagian uji korpus tidak "disimpan sampai akhir": di luar gerbang resmi, 24 tes cepat (sampai 100 baris
    pertama; angka 200 di versi sebelumnya menghitung pasangan baris x font) dan 2 evaluasi penuh run lain juga
    membacanya, dan 6 evaluasi di luar `out/eval` (dua laporan aksara langka sintetis 2.000 baris, empat laporan beam +
    LM 300 baris; kunci kartu `usage.test.others`) mengambil barisnya secara ACAK dari seluruh bagian uji, jadi baris di luar 10.000
    pertama pun pernah dibaca. NusaAksara sudah dievaluasi paling sedikit 44 kali (40 laporan G3 di `out/eval` + empat
    laporan beam + LM di `out/beam`, per 2026-10-16; kunci kartu `usage.real.others`; pembacaan yang tidak meninggalkan laporan tidak
    terhitung, karena itu halaman menulis "paling sedikit"); halaman menyebut angka-angka itu. Font tiap run: dari log
    run bila mencatatnya (`src.train.start_record` menulis NAMA font perender data latih di event `start`; run sebelum
    2026-10-06 tidak), kalau tidak dari folder font SEKARANG (`fonts_source`, `font_names`), dan ekspor memperingatkan
    bila ekspor hasil mencatat jumlah lain. Yang dibaca adalah event `start` proses terakhir yang mulai SEBELUM langkah
    checkpoint (proses itulah yang menulisnya; proses yang mulai tepat di langkah itu baru melanjutkannya). Bila log run
    resmi mencatat nama font, peran "latih" di tabel font hanya untuk nama yang tercatat (font inti yang tidak tercatat
    juga tidak diberi peran validasi: berkasnya masuk folder sesudah run dilatih; jumlah font validasi mengikuti log),
    dan kartu memperingatkan font tercatat yang sudah tidak ada di folder serta font di folder yang tidak tercatat.
    Kalimat dan batasan di halaman dipilih dari isi kartu, supaya tetap benar untuk kartu run lain: font uji
    sekeluarga dengan font latih; data nyata tanpa bagian validasi; aksara langka HANYA bila run resmi tidak memakai
    sisipan aksara langka; jumlah run di rantai, seed, kelengkapan rantai. Nilai `null` (jadwal baris run `--overfit` /
    `--real-train`) tampil "tidak dihitung". Baris Commons yang menunggu = draf - terverifikasi, dan perannya mengikuti
    `--real-train` / `--real-val` run resmi. Halaman menampilkan peringatan "Kartu data dan hasil OCR tidak sejalan"
    bila pipeline resmi web bukan run yang dihitung kartu, tanpa menebak sisi mana yang tertinggal.
    **Impor kartu (`App\Services\CardImporter`, dasar `DatasetImporter` dan `MethodImporter`):** dua pengaman supaya
    kartu yang salah ditolak saat impor, bukan menjadi HTTP 500 saat halaman dibuka (tinjauan 2026-10-05: 55 dari 78
    kartu "satu kunci dibuang" diterima versi pertama lalu membuat halaman 500). (1) `shape()`: kunci wajib dan
    tipenya (notasi titik, `*` = setiap butir; tipe `number`, `count` = bilangan tidak negatif yang boleh pecahan
    (rata-rata, jarak dalam em), `int` = bilangan bulat tidak negatif (jumlah baris, langkah, lapis, parameter: "1,5
    baris" atau 1e30 langkah bukan kartu yang sah), `share` = bilangan dari 0 sampai 1 (porsi dan peluang: "bagian
    latih 250%" bukan kartu yang sah), `string`, `text` = teks tidak kosong, `bool`, `list`, `map`; akhiran `?` = boleh
    tidak ada atau null tetapi harus bertipe benar bila ada; wadah didaftar sebelum isinya supaya pesannya menyebut
    kunci yang salah; isi daftar yang dicetak apa adanya dan angka data pendukung ikut bertipe), ditambah `check()`
    tiap kartu (urutan bagian; pasangan pengaturan; bukti harus punya sumber, yang juga dijaga `entry()` di skrip
    ekspor; kunci kelompok dan butir tidak boleh berulang; kelompok "web" dan kunci butir `Metode::WEB_KEYS` milik
    halaman). Yang TIDAK ditolak (kartu disunting tangan): angka sah yang salah nilainya, dan kunci pilihan yang
    dibuang (mis. `shared_with_test`, yang menghilangkan satu batasan dari halaman). (2) uji
    tampil: halaman dirender sekali dengan kartu calon di dalam transaksi (`preview()`, lewat `viewData()` statis
    komponennya); gagal karena apa pun = batal, kartu lama dipertahankan. Pengaman ini menangkap kartu yang RUSAK,
    bukan angka yang salah. `aksara:import` yang hasil OCR-nya berhasil tetapi sebuah kartunya ditolak menulis
    PERINGATAN dengan alasannya dan tetap keluar 0 (hasilnya sudah tersimpan); `aksara:datasets` / `aksara:methods`
    keluar 1. **Galat database bukan kartu yang ditolak** (`CardStorageException`, mis. tabel kartu belum dimigrasi):
    pesannya menunjuk ke `php artisan migrate` dan `aksara:import` pun keluar 1. Ketiga perintah menulis PERINGATAN
    langsung bila kartu yang baru diimpor tidak sejalan dengan hasil OCR yang diimpor (`staleNotes()`).
    **Jebakan PostgreSQL kedua:** kueri yang gagal membatalkan transaksi yang sedang berjalan, dan semua kueri
    sesudahnya ikut gagal. Karena itu semua kueri importir ada di dalam `DB::transaction`-nya (gagal = digulung balik
    ke titik simpan), dan halaman Dataset menanyakan `Schema::hasTable` sebelum membaca kartu metode alih-alih
    menangkap kueri yang gagal. Test "tabel belum dimigrasi" lulus di SQLite walau kodenya salah; jalankan juga
    `-c phpunit.pgsql.xml`.
    **Test halaman diperiksa dengan mutasi (2026-10-06):** tiap test halaman memeriksa isi per baris lewat atribut
    `data-part|bin|run|dataset|font`, bukan `assertSee` pada teks yang muncul di banyak tempat (versi pertama
    `DatasetPageTest` meloloskan 10 dari 16 kerusakan buatan peninjau: pil "disebar" terbalik, penanda resmi terbalik,
    batang kosong). Sesudah diperkuat, 192 kerusakan satu-hal semuanya menggagalkan test: 31 di halaman Dataset, 51
    di halaman Metode (atribut `data-group|method|step|plan`), 37 di `export_datasets.py`, dan 73 di
    `export_methods.py`; untuk dua skrip ekspor itu juga tanpa test yang butuh data lokal. Empat mutan ekspor data dan
    17 mutan ekspor metode lolos dari test versi pertama (sampel dengan batch pendek, `--val-lines` melebihi bagian
    validasi, selisih korpus, peringatan korpus ditulis ulang; pengaturan tiap butir kartu metode tidak diperiksa
    lengkap). Pembantu baca-halaman dipakai bersama lewat `web/tests/Concerns/ReadsPage.php`. Harness mutasinya ada di
    scratchpad sesi, tidak ikut repo. Jebakan saat mengulanginya: folder sementara pytest bernama panjang melewati
    batas jalur 260 karakter Windows dan gagalnya terhitung "mutan terbunuh"; periksa bahwa alasan gagalnya memang
    pernyataan test.
    **Putaran kedua (2026-10-06, sesudah tinjauan kedua):** peninjau menunjukkan dua hal yang tidak tertangkap putaran
    pertama. (1) Pembantu "kartu harus ditolak" di kedua test halaman TIDAK BISA GAGAL: bentuk
    `try { import(); $this->fail(...); } catch (RuntimeException $e) { assert pesan }` menangkap galat `fail()` itu
    sendiri (`AssertionFailedError` PHPUnit turunan `RuntimeException`), jadi kartu yang DITERIMA pun lolos. Gantinya
    `Tests\Concerns\RejectsCards::assertCardRejected()`: pengecualian ditangkap dulu, dinyatakan di luar blok `try`,
    dan kartu yang tersimpan diperiksa tidak berubah. (2) 45 dari 75 mutan baru sisi web dan 14 dari 15 mutan baru di
    dua skrip ekspor lolos, karena mutasi putaran pertama hanya merusak kelas halaman dan tampilan, bukan importir dan
    perintah impor. Sesudah test diperkuat, 526 kerusakan satu-hal terhadap kode sekarang semuanya menggagalkan test:
    131 di `export_methods.py`, 64 di `export_datasets.py`, dan 331 di sisi web (importir kartu 72, perintah impor
    29, `Metode.php` 73, `Dataset.php` 60, `TranslationRun` 6, tampilan Metode 45 dan Dataset 46). Mekanisme sisi web
    dari peninjau: kelas yang dirusak dideklarasikan sebelum autoload, jadi importir, perintah artisan, dan model ikut
    teruji lewat kelas test repo yang tidak diubah. Jebakan baru: `expectsOutputToContain` Laravel mencocokkan satu
    baris keluaran dengan SATU harapan saja (dua harapan untuk baris yang sama = yang kedua tidak pernah terpenuhi);
    berkas daftar mutan yang di-`require` di lingkup berkas lain menimpa variabelnya (`$id`), sehingga semua run
    memakai mutan terakhir: bungkus dengan fungsi; dan run "dasar tanpa mutan" wajib ada untuk menangkap hal semacam
    itu.
    **Putaran ketiga (2026-10-06, tinjauan verifikasi atas `9955ed2`):** 35 mutan baru peninjau terhadap kode yang
    diubah putaran kedua SEMUANYA lolos (15 di dua skrip ekspor, 20 di sisi web): tiap putaran perbaikan menambah kode
    yang test-nya belum pernah diuji mutasi. Penyebab yang berulang: (1) repo palsu test memakai checkpoint kontrol
    tanpa argumen sebenarnya, jadi pasangan kontrol yang tidak sah tidak pernah terlihat (kini `sibling()` menyalin
    argumen run resmi); (2) pembantu baca-halaman memeriksa lebih sedikit daripada yang tampil (kotak catatan kosong,
    nama langkah alur, label pembaca layar); (3) `assertCardRejected` menerima pengecualian turunan, jadi galat
    penyimpanan lolos sebagai "kartu ditolak" (kini kelasnya harus persis sama); (4) aturan yang dipakai dua halaman
    hanya diuji di satu halaman (run terjemahan di Ringkasan); (5) uji tampil saat impor harus dibuktikan merender
    kartu CALON (penyusun tampilan mencatat `generated` yang dirender), bukan sekadar "render gagal = ditolak".
    Sesudah diperkuat, 692 kerusakan satu-hal semuanya menggagalkan test: 197 di `export_methods.py`, 87 di
    `export_datasets.py`, 408 di sisi web (importir kartu 118, perintah impor 30, `Metode.php` 86, `Dataset.php` 67,
    tampilan Metode 49 dan Dataset 46, `TranslationRun` 6, Ringkasan, `LineAnnotation`, `TranslationService`
    masing-masing 2). Dua mutan web (kueri di luar transaksi; `try/catch` pengganti `Schema::hasTable`) hanya mati di
    PostgreSQL: mutan yang lolos di SQLite WAJIB dijalankan dengan `-c phpunit.pgsql.xml` sebelum disebut lolos.
    Jebakan baru: tipe kunci yang diperketat bisa menghidupkan mutan lama (test "angka dalam tanda kutip" untuk tipe
    `count` ikut pindah ke `int`, jadi mutan `count` lolos sampai ada test pada kunci yang masih `count`); laptop yang
    memakai baterai menjalankan run gabungan (2 pytest + 3 PHPUnit) hanya ~2 mutan web per menit, sedangkan penjalan
    yang menjalankan dulu kelas test terdekat dengan berkas yang dirusak dan berhenti di kegagalan pertama ~12 per
    menit (seluruh suite hanya untuk mutan yang lolos tahap itu); proses latar yang dilepas dengan `( ... &)` tidak
    ikut mati saat tugasnya dihentikan: cari dan hentikan lewat baris perintahnya.
    **Putaran keempat (2026-10-12, tinjauan atas perubahan `9955ed2..995a29b`; tidak ada temuan berat):** 9 dari 12
    temuan putaran ketiga yang diperiksa ulang sudah beres, dan 15 dari 42 mutan baru peninjau lolos, hampir semuanya di
    kode beam model resmi yang baru ditambahkan hari itu (pola yang sama: kode baru, test belum diuji mutasi). Yang
    diperbaiki: (1) `control_run` kini juga membandingkan bobot awal (`init`), packing, pemotongan gradien, ukuran
    model, pemecahan batch lebar, dan pengaturan data nyata; kalimat "titik lanjut yang sama" hanya ditulis bila log
    semua pasangan mencatat awal proses di langkah yang sama (`resume_points`). (2) Potongan beam di
    `export_results.main` dipindah ke fungsi kecil yang teruji (`beam_decoders`, `beam_pairs`, `tag_pair`,
    `beam_official`, `beam_delta`, `mark_beam_changes`, `synth_metrics`); pemakaiannya di `main` dijaga lewat sumbernya,
    karena `main` butuh model dan data nyata. `export_methods.beam_weights` berhenti bila bobot pipeline beam run resmi
    tidak terbaca dari konfigurasinya. (3) Halaman Demo tidak lagi menulis α 0,25 (lihat "Angka resmi"). (4) Kamus:
    tabel pasangan lama menunjuk ke tabel beam + LM pipeline resmi, dan tautan "baris yang membaik/memburuk" hanya ada
    di tabel pasangan yang menentukan tag itu (`Pipeline::beamPair()`). (5) Nama: lihat butir murda/swara. 40 mutan
    terhadap perbaikan ini semuanya mati. Jebakan baru: `Http::fake` yang dipanggil dua kali dalam satu test TIDAK
    mengganti tiruan pertama (yang pertama cocok menang), jadi kasus kedua dan seterusnya dalam sebuah loop tidak
    pernah teruji. **Temuan ringannya dikerjakan hari yang sama (permintaan user), 29 mutan mati:**
    (i) kartu metode hanya menulis "model tidak pernah dilatih pada foto" bila tidak satu pun run di rantai memakai
    `--real-train`, dan run tanpa baris sintetis tidak lagi "mengundi dari 0 font"; (ii) `font_facts` mencatat ARAH
    selisih folder font: `missing` (font run hilang) lawan `added` (font ditambahkan sesudah run dilatih, untuk run
    tanpa nama font di log), dan catatan gerbang serta butir render memakai kalimat masing-masing; (iii) font yang hanya
    dipakai proses SEBELUMNYA sebuah run (`export_datasets.dropped_fonts`) menjadi peringatan kartu data dan catatan
    butir render, dan font run di rantai yang tidak bisa diperiksa lengkap disebut; daftar font run tetap milik proses
    terakhirnya; (iv) catatan pepet butir NLLB dibaca dari masukan terjemahan yang TERSIMPAN untuk run yang ditampilkan
    (ada "ê" atau tidak), bukan dari konstanta kode; (v) aturan pemilihan nama kamus dipindah ke
    `AksaraWriter::namesIn()` dan diuji pada entri buatan. Jebakan: test di `tests/Unit` berjalan tanpa aplikasi Laravel
    bila dijalankan dalam suite penuh (`base_path()` gagal), walau lulus bila berkasnya dijalankan sendiri bersama test
    Feature.
    **Menu Metode (permintaan user 2026-10-05: "pada sidebar tambahkan menu method/machine learning, jadi disana list
    method atau machine learning yang digunakan"; isi dan pengelompokannya keputusan Claude, belum dikonfirmasi):**
    `Pages/Metode.php`, kelompok Hasil di bawah Dataset, rute `/metode`. Isinya dari **kartu metode**
    `out/results/methods.json` buatan `scripts/export_methods.py` (kontrak sendiri, `aksara.method_schema` 1), diimpor
    `php artisan aksara:methods` ke tabel `method_reports`; `aksara:import` ikut mengimpornya bila berkasnya ada, dan
    `scripts/make_official.sh` mengekspor dan mengimpornya PALING AKHIR (buktinya dikutip dari manifest hasil, berkas
    pembanding, dan evaluasi sintetis). 26 butir di 5 kelompok (pembaca aksara 7, data latih sintetis 6, pelatihan 5,
    koreksi sesudah baca 1, evaluasi dan statistik 7) + 4 pipeline berstatus "planned"; web menambah kelompok keenam
    "Tahap lanjutan alur" (alih aksara, NLLB-200, chrF/BLEU, tingkat tutur, alih aksara Latin -> aksara, pemulih é)
    dari tabel dan konstanta web (`Metode::webGroup()`), jadi logika tahap 2-4 tetap tidak masuk `src/`. Tiap butir:
    jenis (`model`, `training`, `data`, `statistic`, `rule`, `evaluation`), status (`official` = bagian model resmi,
    `used`, `available`, `tested`, `comparator`), penjelasan, pengaturan, berkas kode, dan bila ada: bukti terukur +
    sumbernya (`evidence`) serta catatan batasan (`note`, kotak "Catatan" di halaman).
    **Angka dan pengaturan di kartu dibaca dari berkas; kalimat penjelasannya ditulis tangan di skrip.** Dua angka
    bersumber "catatan pengukuran di CLAUDE.md" (normalisasi kontras 0,9% -> 74,5%; packing 1,12% vs 0,07%). Model
    dibangun dari `model_config` checkpoint resmi dan dimuati bobotnya (4.590.909 parameter: CNN + proyeksi 1.913.568,
    BiLSTM 2.629.632, keluaran 47.709; 7 lapis konvolusi, tinggi 96 -> 3, langkah lebar 4, 93 kelas), argumen pelatihan
    dari checkpoint. Kalimat tentang kode (pengoptimal, jadwal, rugi, pemotongan gradien di `src.train.main`; persentil
    latar dan tinta di `src.dataset.to_tensor`; rumus penghalusan Witten-Bell di `src.charlm.CharLM.prob`, bukan kata
    di docstring modulnya) dicocokkan dengan sumbernya lewat `CODE_FACTS` (**ekspor BERHENTI bila kode itu berubah**:
    perbarui `CODE_FACTS` dan butirnya). Bukti dari `out/results/manifest.json`, `out/compare/<A>_vs_<B>.json`, dan
    `out/compare/spacing_synthetic.json`; yang berkasnya tidak ada dilewati. Kalimat yang bergantung pada resep run
    (operasi augmentasi, rantai run, perangkat, beam + LM pada checkpoint resmi atau belum) disusun dari argumen dan
    hasilnya, bukan teks tetap. Ekspor juga berhenti bila manifest menetapkan pipeline resmi lain (manifest lama tanpa
    kunci `official` = `crnn_fonts`), dibuat dengan `--limit`, run-nya tidak dikenal `export_results.py`, jumlah kelas
    checkpoint beda dari tokenizer, jumlah parameter di log beda dari model, atau laporan gerbang resmi G1/G2/G3 run
    itu tidak ada atau tidak sah (`gate_reports`, sama dengan kartu data).
    **Aturan kartu metode yang lahir dari tinjauan kedua (2026-10-06, 29 temuan; jangan dikembalikan):**
    (a) selisih antar-run selalu dihitung dari CER manifest; selang kepercayaannya hanya diambil dari berkas
    pembanding yang CER kedua sisinya SAMA dengan manifest sekarang (berkas dari ekspor lama tidak dikutip). Sebuah run
    hanya disebut kontrol untuk perlakuan di barisnya: `CONTROLS` berkunci perlakuan (`fonts`: fase5_fonts lawan
    fase5_core; `tracking`: fase7_track lawan fase7_ctrl; `rare`: fase7_track_rare lawan fase7_track, fase6_rare lawan
    fase6_ctrl) dan hanya berisi CALON: pasangan baru dikutip sesudah `control_run` memeriksa kedua checkpoint (ada di
    mesin ini, langkah sama, perlakuan hidup di run dan mati di kontrol, resep data `data_recipe` sama kecuali
    `TREATMENT_KEYS` perlakuan itu, `TRAINING_KEYS` sama); bukti milik run lain disebut "bukan model resmi".
    (b) Label sintetis tidak disebut "pasti benar": butir render membawa catatan font latih bercacat (BasaJan,
    CarakanJawa, NewKramawirya: `FONT_DEFECTS` + font tanpa glyph sebuah karakter charset), dan butir buang-spasi
    menyebut font berspasi sempit (3 dari 10: spasi selalu dibuang, jadi porsi baris tanpa spasi sekitar 65%, bukan
    50%).
    (c) Gerbang G1/G2 dan evaluasi sintetis tertarget membawa catatan "font uji javatext sekeluarga dengan font latih
    CarakanJawa" (dihitung dari lebar maju, aturan kartu data).
    (d) Ablasi tidak diberi peringkat "penurunan terbesar": yang dikutip jumlah per kelompok (7 operasi preset −2,38
    poin, 4 operasi tiruan pindaian −28,63), dengan catatan satu seed, kumulatif, dan rotate yang menaikkan G3 12,79
    poin.
    (e) Rantai checkpoint yang terpotong (checkpoint leluhur sudah dihapus) tidak dilaporkan "dari bobot acak"
    (`chain_complete`), dan pengantar kelompok pembaca aksara serta ubin halaman hanya menulis "dilatih dari nol" bila
    rantainya utuh; kalimat OneCycle dihitung dari langkah yang DIRENCANAKAN (3 dari 4 run sebelum run resmi
    dihentikan sebelum siklusnya selesai).
    (f) Font tiap run: nama di log -> jumlah di ekspor hasil (hanya bila tidak bertentangan dengan argumen
    checkpoint: run tanpa `--extra-fonts` tidak bisa punya lebih dari font inti) -> folder sekarang, dan bila folder
    sudah berubah itu disebut.
    (g) Evaluasi jarak menyebut juga batasnya (di luar rentang latih 0,6 em: 67,5 spasi palsu per 100 batas).
    **Aturan dari tinjauan verifikasi (putaran ketiga, 2026-10-06: 22 temuan baru, tidak ada yang berat; kebanyakan
    "kartu harus tetap benar untuk run dan mesin lain"):**
    (h) "Font yang pernah dilihat model" = gabungan font SEMUA run di rantai (`chain_fonts`), bukan font run terakhir:
    catatan keluarga font (G1/G2, evaluasi tertarget) dan catatan font bercacat dihitung dari gabungan itu, dan font
    yang hanya dipakai run sebelumnya disebut "(font run X di rantai)".
    (i) Yang tidak bisa dihitung DISEBUT, bukan didiamkan: font uji yang berkasnya tidak ada di mesin ini, font latih
    yang tercatat di log tetapi sudah tidak ada di folder ("cacat font hanya bisa diperiksa pada M dari N font"),
    laporan evaluasi sintetis yang tidak bisa dicocokkan dengan checkpoint sekarang ("angkanya tidak dikutip").
    (j) Laporan `scripts/eval_spacing.py` hanya dikutip sebagai "model resmi" bila `report_matches`: langkah checkpoint
    di laporan sama dengan checkpoint sekarang, dan sidik berkasnya (12 heksadesimal SHA-1) sama bila laporan
    mencatatnya; begitu juga untuk run kontrolnya.
    (k) Aturan "spasi selalu dibuang pada font berspasi sempit" lahir 2026-09-14, SESUDAH `base`, `fase5_quick`,
    `fase5_core`, `fase5_fonts` dilatih (`RUNS_BEFORE_NARROW_SPACE_RULE`): butir buang-spasi tidak boleh membaca aturan
    itu dari kode sekarang untuk run-run tersebut. Kartu sekarang mencatat bahwa `fase5_fonts` di rantai dilatih
    sebelum aturan itu ada (labelnya masih memuat spasi yang nyaris tidak tampak di font berspasi sempit).
    (l) Kalimat pembanding disusun dari nilai, bukan teks tetap: "Yang berubah" tiap run dibandingkan nilai demi nilai
    dengan run sebelumnya (`data_recipe`; "argumen data sama, langkah tambahan" bila memang sama); kalimat OneCycle
    memakai jumlah langkah di jadwal checkpoint (`planned_steps`), juga untuk run `--epochs`; "ablasi satu seed" hanya
    bila seed semua run tercatat dan sama; kalimat sisipan aksara langka mengikuti hasilnya ("tetapi hanya" bila
    presisi < 50%; penutupnya "Karena itu tidak dipakai di run resmi", kecuali bila G3 run bersisipan lebih baik secara
    nyata: saat itu hanya "Run resmi tidak memakainya").
    (m) `src.train` menulis daftar font kosong di event `start` bila proses tidak merender baris sintetis
    (`--train-lines 0` dengan `--real-train`); daftar kosong berarti "tidak ada yang dibandingkan", bukan "tidak ada
    font".
    Sisi web: alur bercabang (arti dan tingkat tutur sama-sama dari teks Latin, bukan berurutan); langkah bermodel
    ditandai warna DAN tulisan "ML"; ubin keempat menghitung model hasil belajar yang DIPAKAI (CRNN, NLLB-200) dan
    menyebut model bahasa n-gram terpisah sebagai "tersedia"; bukti alih aksara menyebut "huruf saja" dan "masukannya
    label aksara"; tingkat tutur tanpa label manusia mendapat catatan "belum dinilai", bukan kotak "Terukur". Butir web
    "NLLB-200" memilih run terjemahannya lewat `TranslationRun::forOcr()` (aturan yang sama dengan Ringkasan: keluaran
    pipeline resmi -> bacaan beam + LM-nya -> run terbaru pipeline lain) dan menyebut "bukan pipeline resmi" selama
    terjemahan dari keluaran OCR dibuat dari pipeline lain (sekarang: beam + LM atas `fase5_fonts`). Halaman
    menampilkan peringatan
    "Kartu metode dan hasil tidak sejalan" bila kartu dibuat untuk run lain atau `results_generated` kartu bukan ekspor
    hasil yang sedang diimpor: sesudah `export_results.py` dijalankan ulang, jalankan ulang `export_methods.py` juga.
    **Jebakan PostgreSQL:** kolom `payload` bertipe `json`, BUKAN `jsonb`. jsonb mengurutkan ulang kunci objek
    (train/val/test tampil sebagai "5 / 5 / 90" dan peran dataset tertukar urutannya), sedangkan SQLite tidak, jadi
    test bawaan tidak menangkapnya. Halaman kini mengurutkan peran sendiri, dan ada test yang mengacak urutan kunci.
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
    `php artisan aksara:translate --dump`). 171 test lolos di SQLite dan PostgreSQL uji; 409 test Python (2026-10-17, sebelum fase 11).
    **Port:** Laravel 8010, layanan model 8011, layanan terjemahan 8012 — 8000/8001, 5173, dan 8020 dipakai proyek
    lain milik user di laptop yang sama (jangan dihentikan; cek dulu siapa yang mendengarkan sebelum memakai sebuah
    port, dan beri skrip sementara port 8030 ke atas). Test tetap bisa di SQLite in-memory
    (`phpunit.xml`); kode dijaga netral: `whereJsonContains` untuk tag, tabel turunan untuk ORDER BY berekspresi
    (PostgreSQL menolak alias kolom di dalam ekspresi). Isi DB kecuali akun & label manusia bisa dibangun ulang
    (`aksara:import`, `aksara:datasets`, `aksara:methods`, `aksara:annotations`, `aksara:translate --import-only`); label tingkat tutur dicadangkan
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
    `src/serve.py` harus run yang sama (dijaga `tests/test_export_results.py`); bawaan Demo = greedy (jalur
    angka resmi). Bobot LM beam di `src/serve.py` (`ALPHA`, `BETA`) dan model bahasanya (`DEFAULT_LM`) milik checkpoint
    bawaan itu (sejak 2026-10-18: model bahasa berejaan baku, 0,25 / 0,5 dari `out/beam/fase9_ctrl_o5std.json`, dev
    dirender dengan resep checkpoint itu; 2026-10-16 sampai 2026-10-18: 0,25 / 1,0 untuk `fase8_std`; test
    mencocokkannya dengan laporan yang dipilih `export_results.official_beam`: di antara `BEAM_REPORTS`
    (`<run>_o5.json` model bahasa lama, `<run>_o5std.json` berejaan baku) yang CER beam-nya di dev TERENDAH, dengan dev
    yang harus sama; laporan yang tidak sah atau dev yang berbeda menghentikan ekspor), dan
    layanan melaporkannya di kunci `beam` jawaban `/health` dan `/predict`: halaman Demo menampilkan bobot dan rumus
    skor dari situ, tidak menuliskannya sendiri. **Mengganti run resmi:**
    `bash scripts/make_official.sh RUN` (evaluasi G1/G2 10.000 baris, ~30 menit per gerbang di CPU; menunggu
    training selesai dulu), ubah `OFFICIAL_RUN` + `DEFAULT_CHECKPOINT`, jalankan `scripts/beam_eval.py` untuk run itu
    dengan kedua model bahasa (`--name <run>_o5` dan `<run>_o5std`, `--dev-standard-prob` sama dengan training run itu)
    dan salin bobot dan model bahasa yang menang di dev ke `src/serve.py`, lalu ekspor dan impor. `make_official.sh`
    menunggu skrip lanjutan rantai mana pun (`after_fase*.sh`), menghitung ulang pembanding per run resmi (`pairs`), dan
    membaca run resmi di evaluasi jarak; log-nya menumpuk antar-run, jadi baca baris sesudah "angka resmi -> RUN" terakhir.
  - **Seluruh alur = scope proyek (keputusan user 2026-09-24):** OCR -> transliterasi -> arti (Indonesia)
    -> tingkat tutur (ngoko / madya / krama / campur). Transliterasi & arti **manusia** untuk 745 label
    tersedia dari NusaAksara (config `Image Transliteration` / `Image Translation`, cocok 745/745, lisensi
    non-komersial sama). Transliterasi draf = aturan deterministik aksara -> Latin di `web/`, berlabel
    "bantu baca, bukan hasil OCR"; **CER draf vs transliterasi manusia 7,1% pada 745 label** (huruf saja;
    7,4% sampai 2026-10-05, lalu nya + pangkon di depan ca/ja dilatinkan "n": panjenengan, bukan panyjenengan;
    kelemahan diketahui: ꦲ tengah baris selalu "h", geminasi ditulis ganda, tanpa spasi kata).
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
    `fase5_fonts` saat itu; beam + LM di atas `fase7_track` baru diukur 2026-10-12 (19,35%, lihat bagian beam search).
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
  arti (terjemahan Indonesia), dan tingkat tutur; sejak 2026-10-05 juga alat Kamus dan Terjemahan (Indonesia ->
  Jawa beraksara Jawa dan sebaliknya). Logika tahap-tahap itu TIDAK boleh masuk `src/`
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
| 6 | `src/real.py`, `--real-train` | CER<8% pada ≥200 baris nyata | test: `data/real/nusaaksara/labels.tsv` 745 baris (lisensi **non-komersial**, HANYA test); fine-tune: 43 baris papan Commons menunggu verifikasi pembaca aksara (`out/verifikasi_aksara.html` → `scripts/import_review.py`); **belum lolos**: terbaik sejauh ini `fase8_std` (masih sintetis saja; `fase7_track` + 1.500 langkah dengan ejaan baku di separuh baris) G3 **20,35%** (beam + LM berejaan baku 18,02%; kontrol `fase8_ctrl` 20,81%); sebelumnya `fase7_track` (`fase6_ctrl` + 1.500 langkah dengan jarak antar suku kata acak) G3 **21,46%** (pad 0,06: 19,99%; spasi dibuang 20,99%) dan `fase7_track_rare` 21,18% (tidak berbeda nyata); kontrol `fase7_ctrl` (langkah sama tanpa jarak) 30,75% → efek jarak −9,29 poin (SK halaman [−11,92; −6,83]); sebelumnya `fase6_ctrl` 32,17%, `fase6_rare` 34,06%, beam+LM atas `fase5_fonts` 33,83%. **Angka resmi sejak 2026-10-18 = `fase9_ctrl`** (keputusan user): G1 0,23%, G2 0,72% (10.000 baris javatext), G3 20,31% (baris cetak 12,34%), beam + LM 17,68%; sebelumnya `fase8_std` 0,24% / 0,98% / 20,35% (2026-10-16 sampai 2026-10-18), `fase7_track` 0,27% / 1,20% / 21,46% (2026-10-04 sampai 2026-10-16) dan `fase5_fonts` 0,34% / 2,01% / 37,25%. Fase 9 (render miring italic, `scripts/run_fase9.sh`) dijalankan 2026-10-16 malam; fase 10 (sisipan aksara langka di atas `fase9_ctrl`) 2026-10-17, tidak memenuhi aturan; fase 11 (pecahan baris tetangga) 2026-10-17/18, juga tidak memenuhi aturan; putaran sintetis dihentikan, calon angka resmi `fase9_ctrl` (cetak 12,34%, tidak berbeda nyata dari `fase8_std`) |

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
