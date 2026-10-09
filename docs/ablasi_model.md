# Studi: ablasi CRNN + CTC lawan model/metode ML lain

Tanggal 2026-10-09. **Status: rencana hasil studi, belum diputuskan user** (pertanyaan keputusan di §12).
Dibuat dari pembacaan kode dan dokumen repo plus riset sumber luar; sesi ini berjalan di kontainer tanpa torch,
data, checkpoint, maupun `fonts/extra`, jadi tidak ada angka baru yang dihitung. Angka biaya dikutip dari CLAUDE.md.

## 0. Ringkasan

- **Yang bisa diablasi secara adil di laptop ini** (iGPU XPU, 32 GB) adalah model yang **tetap memakai pipeline data dan
  CTC yang sama** dan hanya mengganti **satu** hal: blok sekuens (BiLSTM → Transformer-encoder / tanpa rekurensi) atau
  label (urutan visual → urutan logis). Satu keluarga lain yang layak tetapi lebih mahal: **decoder atensi** kecil
  (non-CTC), karena ia menguji klaim PLAN.md §3.5 "CTC, bukan attention".
- **Yang tidak layak di laptop ini**: TrOCR, Donut/Pix2Struct, PARSeq-base, SVTR di PaddleOCR, LoRA VLM (SmolDocling,
  GraniteDocling, Qwen3-VL). Alasannya bukan selera: masukan persegi/kata, tokenizer tanpa blok U+A980, 5-120x
  parameter, kerangka tanpa XPU. Dicatat sebagai kerja lanjutan di cloud, bukan dibuang.
- **Tidak ada baseline eksternal siap pakai.** `jav.traineddata` Tesseract = bahasa Jawa beraksara **Latin**; direktori
  `tessdata/script` (37 model) tidak punya Javanese; EasyOCR dan PP-OCRv5 tidak punya `jv`. NusaAksara (ACL 2025) sudah
  mencoba PP-OCRv3 fine-tune: CER > 1.
- **Kontrol yang sah harus dilatih dari nol dengan resep yang sama**, termasuk CRNN-nya sendiri. Bobot `fase7_track`
  tidak bisa diwarisi arsitektur lain (`--init` memuat state_dict ketat ke kelas `CRNN`, `src/train.py:347`), dan ia
  hasil rantai 6.298 langkah dengan resep yang berubah tiap tahap. Ada opsi hemat (tulang punggung CNN `fase7_track`
  dipakai bersama) dengan kesimpulan yang lebih sempit, lihat §6.
- **Derau antar-run belum pernah diukur** di repo ini (satu run per lengan di semua fase). Ablasi arsitektur tanpa
  lantai derau tidak bisa dibaca, jadi CRNN kontrol dijalankan **dua seed**.
- **Accuracy, precision, recall, F1 di web** (permintaan user 2026-10-09) bisa diturunkan dari alignment karakter yang
  sudah ada (`scripts/compare_runs.py::alignment`), konsisten dengan CER, tanpa mengubah angka resmi. Definisi dan
  perubahan kontrak di §8.
- Urutan yang disarankan: K0 (CRNN dari nol, 2 seed) → K1 (urutan logis) → K2 (Transformer-encoder + CTC) → K3
  (CNN-only + CTC) → K4 (decoder atensi, sesi terpisah). Inti K0-K3 ≈ 28-56 jam XPU (8-12 malam) + ~5 hari kerja.

## 1. Yang sudah ada di repo untuk "ablasi"

Tiga pola ablasi sudah dipakai, masing-masing dengan alat, kontrak data, dan halaman web sendiri:

| Pola | Alat | Keluaran | Halaman web | Keterangan |
|---|---|---|---|---|
| Ablasi augmentasi kumulatif (Fase 5) | `scripts/ablation.py` (`ABLATION_STEPS` di `src/augment.py:27`) | `out/ablation.json` → manifest kunci `ablation` | Ablasi (`AblationRun`: k, added, clean, heavy, g3) | 12 run dari checkpoint 4b yang sama, 600 langkah, satu seed; tabel `ablation_runs` **diganti utuh** saat impor |
| Pasangan perlakuan-kontrol antar run | `scripts/run_fase6_*.sh`, `run_fase7*.sh` (`--stop-step`, titik lanjut ditiru), `scripts/compare_runs.py` | `out/compare/<A>_vs_<B>.json/.md`: SK bootstrap baris dan halaman, uji tanda, recall/presisi aksara langka | Perbandingan (`FOLLOWUP_RUNS`), Metode (`CONTROLS`) | fonts: fase5_fonts vs fase5_core; tracking: fase7_track vs fase7_ctrl; rare: fase7_track_rare vs fase7_track |
| Ablasi cara baca (koreksi pasca-OCR) | `scripts/beam_eval.py`, `src/beam.py`, `src/charlm.py`; wadah `src/gpt.py` | `out/beam/<name>.json` → metrik `synth_dev`, `synth_heldout`, pipeline `crnn_fonts_beam` | Perbandingan, Kamus, Demo | LLM (GPT, Sahabat-AI), VLM fine-tune, "combo" masih `planned` di `PIPELINES` (`scripts/export_results.py:60-80`) |

Semua angka dihitung Python; web hanya menampilkan (`php artisan aksara:import`). Alat statistiknya sudah
model-agnostik: evaluasi berpasangan pada citra identik (`SyntheticLines(deterministic=True)`, `src/evaluate.py:35`;
`scripts/eval_spacing.py`, `scripts/eval_rare.py`) dan SK bootstrap per baris dan per halaman
(`scripts/compare_runs.py:278-318`, murni dari teks prediksi).

## 2. Titik yang mengunci jalur ke kelas `CRNN`

Kontrak yang berlaku untuk semua model sekarang: masukan `(B, 1, 96, W)` float32 tinta = 1 (`src/dataset.py:44-62`,
`to_tensor`), keluaran logits `(B, W // 4, C)` dengan kelas 0 = blank, label **urutan visual**, decode ke logis.

| Tempat | Apa yang diasumsikan | Akibat untuk arsitektur lain |
|---|---|---|
| `src/model.py:18-65` | satu kelas `CRNN`, `WIDTH_STRIDE = 4`, `forward(x, lengths=None)` → `(B, W//4, C)`; packing LSTM dari `lengths` | registry arsitektur belum ada; pengganti wajib menerima `lengths` dan menjadikannya mask (padanan packing), kalau tidak gejala "pada lingsa palsu di frame terakhir" (CLAUDE.md) kembali |
| `src/train.py:329` | `CRNN(tokenizer.n_classes, channels=..., hidden=...)`; `assert_downsample` (`:129-139`) menuntut `W // DOWNSAMPLE` frame; `model_config` (`:215`) hanya `n_classes, channels, hidden`; `--init` (`:347`) memuat state_dict ketat | model dengan stride lain gagal assert; checkpoint tidak menyimpan jenis arsitektur; `--init` lintas arsitektur harus ditolak dengan pesan |
| `src/train.py:141-145`, `:336` | `nn.CTCLoss(blank=BLANK, zero_infinity=True)`; loss dihitung di CPU untuk XPU | decoder atensi butuh rugi dan loop berbeda; literal `nn.CTCLoss(...)` dijaga `CODE_FACTS` kartu metode (`scripts/export_methods.py:92-100`) |
| `src/train.py:148-175` | `split_batch` memecah batch lebar; memori ~105 KiB per kolom B×W (`--max-batch-columns` 64.000 ≈ 6,7 GiB) | heuristik linear terhadap W: TIDAK membatasi memori self-attention O(T²) (W 2.900 px → T 725) |
| `src/dataset.py:30`, `:65-67`, `:185` | `DOWNSAMPLE = 4` konstanta modul: `in_lens = w // DOWNSAMPLE`, filter T ≥ 1,5L | stride lain mengubah `in_lens`, filter, `split_batch`, `src/align.py`, dan test; himpunan baris latih bisa berubah |
| `src/dataset.py:169`, `src/real.py:143`, `src/decode.py:29`, `src/train.py:202`, `src/evaluate.py:50` | label = `encode(to_visual(text))`; keluaran dan **referensi** = `to_logical(decode(...))` | lengan urutan logis harus melewati **kelima** titik ini; `to_logical` pada teks yang sudah logis memindahkan taling (bukan identitas), dan mengosongkan `tokenizer.reorder` BUKAN caranya: `to_visual` jatuh ke `canonical_permutation` (`src/tokenizer.py:232-238`) |
| `src/infer.py:21-30` | `load_checkpoint` membangun `CRNN` dari `model_config`; dipakai evaluate, serve, export_results, beam_eval, eval_rare, eval_spacing | satu tempat untuk `build_model(model_config)`; `predict` (`:59-67`) memakai `greedy_decode` |
| `src/beam.py`, `src/align.py`, `src/serve.py` | `line_log_probs` (T, C) blank 0; `ctc_logp`, `syllable_spans`, kandidat Demo | non-CTC: beam+LM "tidak berlaku", `spans` boleh `null` (`scripts/export_results.py:461`), Demo harus menolak checkpoint non-CTC dengan pesan |
| `scripts/export_methods.py:192-227` | `model_card` membangun `CRNN` dan membaca `cnn/proj/rnn/head`, `lstm_layers`, `bidirectional` | kartu metode hanya untuk run resmi; arsitektur lain butuh cabang `model_card` bila kelak jadi resmi |
| `tests/test_shapes.py`, `tests/test_dataset.py:99,123,192,219,241`, `tests/test_real.py:90`, `tests/test_align.py:67,79` | membangun `CRNN` langsung; `y == encode(to_visual(text))` | parametrisasi atas registry dan atas `label_order`; **pytest wajib lolos sebelum evaluasi berikutnya** (worker Windows mengimpor ulang `src.train`) |
| `scripts/export_results.py:55-105`, `:161-190`, `:204`, `:375-383` | kunci `crnn_<run>`, `kind: "crnn"` ditanam di `optional_pipeline`, `check_optional_run` membaca argumen `src.train`, inferensi CTC untuk semua checkpoint | run arsitektur lain tidak bisa lewat `OPTIONAL_CHECKPOINTS`; butuh daftar sendiri (§8) |
| `tests/test_export_results.py:111-116`, `web/tests/Feature/AksaraPagesTest.php:213-227` | daftar persis 5 run lanjutan dan pola nama | setiap run baru, CRNN sekalipun, menuntut pembaruan dua test; bukan penghalang khusus non-CRNN |
| `scripts/eval_spacing.py:130-133` | cache bacaan di-hash dari `render.py, dataset.py, model.py, infer.py, decode.py, tokenizer.py` | menyentuh `model.py`/`infer.py` membatalkan semua cache (ribuan bacaan checkpoint lama); modul arsitektur baru harus ikut di `CODE_FILES` atau bacaan basi dipakai diam-diam |

## 3. Kandidat: kelayakan (riset sumber luar, lihat §13)

| Kandidat | Keluarga | Parameter | Menangani reorder sendiri? | Kelayakan di laptop | Keputusan |
|---|---|---|---|---|---|
| CRNN + CTC, label urutan **logis** | sama, label beda | 4,59 M | diserahkan ke BiLSTM (CTC monoton terhadap label, bukan glyph) | termurah: nol arsitektur baru | **inti (K1)** |
| CNN (sama) + **Transformer-encoder** + CTC | CTC | blok sekuens 2,4-3,2 M (3-4 lapis d 256, FFN 1024) ≈ BiLSTM 2,63 M | tidak (urutan visual tetap) | hanya `src/model.py`; `lengths` → `key_padding_mask`; atensi O(T²) pada T ≤ 725 belum diuji di XPU | **inti (K2)** |
| **CNN-only** + CTC (Conv1d berdilasi/gated, GFCN) | CTC | 1-3 M | tidak; konteks = receptive field | sangat layak, tanpa packing; latensi Demo CPU | **inti (K3)** |
| CNN + **decoder atensi** kecil (Transformer decoder 2 lapis), opsional hybrid CTC+CE | non-CTC | 4-5 M | ya, native (label logis) | layak; inferensi autoregresif 10-50x lebih lambat; butuh BOS/EOS, loss dan decode sendiri | **inti kedua (K4)**, sesi terpisah |
| CRNN `--hidden 384` / 1 lapis LSTM / BiGRU | kapasitas | 3-8 M | sama | nol kode baru | **opsional (K8)**: kontrol kapasitas supaya K2/K3 tidak terbaca sebagai efek jumlah parameter |
| Focal CTC, label smoothing, kurikulum panjang, masking kolom, TTA | metode | 4,59 M | sama | murah, semua di pipeline sekarang | **opsional (K5)**, desain lanjutan + kontrol langkah sama (pola fase 6/7), bukan ablasi arsitektur |
| Beam + LM karakter di atas `fase7_track` | cara baca | — | — | 0 jam XPU, ~1,5 jam CPU | **opsional (K6)**: baris "Beam + LM" web masih milik `fase5_fonts` |
| Tesseract 5 LSTM+CTC / Kraken dilatih sendiri | alat eksternal | 0,5-4 M | tidak | WSL2 CPU 8-24 jam; butuh dump citra baris ke disk (menyimpang dari aturan "citra tidak disimpan") | **opsional (K7)**, dilabeli "baseline eksternal", bukan ablasi satu faktor |
| CRNN H = 64; stride lebar 8 | invarian 1 dan 5 | sama | sama | mengubah pipeline data (resize, T, filter T ≥ 1,5L) | **hanya atas keputusan user**, paling akhir |
| SVTR-T/S (PaddleOCR) | CTC, ViT | 4,15 M / 8,45 M | tidak | Paddle tanpa XPU; masukan tetap (32×100), harus ditulis ulang di PyTorch untuk baris 96×2.900, lalu hampir sama dengan K2 | ditolak |
| PARSeq | atensi permutasi | 23,8 M (ViT 21,4 M + decoder 2,4 M) | ya | kata (32×128, label ≤ 25), tanpa XPU, 5x CRNN (> 10 jam per 1.500 langkah, perkiraan) | ditolak |
| TrOCR small/base/large | pralatih | 62 / 334 / 558 M | ya | masukan 384×384 persegi (rasio baris 30:1), tokenizer tanpa blok U+A980 (byte fallback atau kosakata baru = pralatihan decoder hilang), ≥ 40 jam per 1.500 langkah (perkiraan) | ditolak di laptop; kandidat cloud |
| Donut, Pix2Struct | pralatih dokumen | 200 M-1,3 B | ya | model halaman utuh | ditolak |
| SmolDocling-256M / GraniteDocling-258M LoRA | VLM kompak | 256-258 M | ya | SigLIP 512×512 patch 16 (masalah rasio yang sama), detik per sampel di iGPU, 50x parameter: bukan pembanding adil | ditolak di laptop; tetap `planned` sebagai "pembaca kedua" (keputusan user 2026-09-24) |
| Qwen3-VL-2B LoRA | VLM | 2 B | ya | realistis hanya di cloud | ditolak di laptop |
| docTR vocab `javanese` (CRNN/SAR/MASTER/ViTSTR/PARSeq/VIPTR) | kerangka | 2-24 M | sebagian | CPU saja, berbasis kata, vocab tanpa spasi, tanpa model pralatih Jawa | ditolak |

Rujukan yang paling relevan untuk memilih K2-K4: Diaz dkk. 2021 (*Rethinking Text Line Recognition Models*):
pada baris teks, encoder self-attention + CTC menang dalam akurasi dan kompleksitas bila ada LM; tanpa LM CTC dan
decoder Transformer bersaing; decoder Transformer memburuk pada citra lebih lebar dari lebar latih; CTC bisa di-chunk
tanpa rugi. Studi CTC untuk 13 bahasa India (CVIT IIIT-H 2022) menyimpulkan CTC cukup untuk abugida tanpa modul khusus
skrip. Literatur OCR aksara Jawa yang ditemukan (SEMNAS INOTEK 2026, INTEK 2024, ETASR 2026) tidak membandingkan
arsitektur; ETASR 2026 mengonfirmasi pipeline segmentasi + k-NN gagal pada tumpukan pasangan (PLAN.md §3.1).

## 4. Lengan inti dan hipotesis yang diuji

| # | Lengan | Klaim PLAN.md yang diuji | Kode baru | Catatan |
|---|---|---|---|---|
| K0 | **CRNN dari nol, seed 0** (kontrol) + **seed 1** (lantai derau) | — | registry saja | tanpa ini tidak ada kontrol sah dan tidak ada ukuran derau |
| K1 | CRNN + CTC, `--label-order logical` | §3.3 / invarian 7: tokenizer urutan visual menyumbang poin? | bendera + simpan di `model_config` | panjang label sama → himpunan baris latih identik; wajib sebelum menafsirkan "atensi menangani reorder sendiri" |
| K2 | CNN (sama) + Transformer-encoder + CTC, parameter blok sekuens disamakan | §3.5 baris 171: "BiLSTM dua arah" | 1 kelas + mask + test | hanya blok sekuens yang berbeda; `lengths` → `key_padding_mask` di training DAN evaluasi |
| K3 | CNN-only + CTC, receptive field ≥ 40 frame (160 px) ditetapkan sebelum run | batas bawah: berapa poin dari BiLSTM 2,63 M | 1 kelas | nilai produk: latensi Demo CPU |
| K4 | CNN + decoder atensi 2 lapis d 256, CE teacher-forced, label **logis**; BOS/EOS/PAD di luar charset (`n_classes + 3`, tokenizer tidak berubah) | §3.5 baris 173: "attention butuh data jauh lebih banyak" | `IS_CTC = False`, `loss()`, `decode()`, 4 cabang call site | satu-satunya keluarga non-CTC; wajib metrik halusinasi (§7) |

Opsional: K5 (metode, desain lanjutan + kontrol), K6 (beam + LM atas `fase7_track`, CPU saja), K7 (Tesseract/Kraken,
butuh pengecualian aturan dump citra), K8 (CRNN kapasitas), H = 64 / stride 8 (invarian; keputusan user).

## 5. Ditolak, dengan alasan

- **`fase7_track` sebagai kontrol** atau membandingkan lengan baru langsung dengannya: 6.298 vs 3.000 langkah, sejarah
  resep berbeda. Ia tetap tampil sebagai **baris rujukan** berlabel "bukan kontrol".
- **`out/checkpoints/base/best_4b_step1500.pt` sebagai tahap 1 kontrol**: dilatih kode lama (sebelum aturan spasi
  sempit 2026-09-14), argumennya tidak terjamin sama.
- **Satu tahap 1.500 langkah dari nol dengan augmentasi `fase5` tanpa bukti konvergensi**: "naikkan anggaran setelah
  lihat angka" melanggar praregistrasi. Dipakai hanya lewat kontingensi yang diputuskan di K0 (§6).
- **Model siap pakai** Tesseract `jav`, EasyOCR, PP-OCRv5: tidak ada model aksara Jawa.
- **TrOCR, Donut, Pix2Struct, PARSeq-base, SVTR Paddle, docTR, LoRA VLM**: alasan di §3.
- **Distilasi / self-training**: butuh guru lebih besar (run tambahan) atau citra nyata tak berlabel di luar NusaAksara
  (belum ada; Commons 43 baris).
- **Masuk web sebelum ada angka**: setiap run baru menuntut pembaruan dua test (§2); kontraknya dirancang sekarang (§8),
  diisi setelah ada lengan yang layak.

## 6. Protokol bersama (satu versi; disalin ke CLAUDE.md bertanggal SEBELUM run pertama)

**Prinsip.** Satu perlakuan per lengan: hanya `--arch` (+ kwargs) atau `--label-order` yang berbeda. Daftar periksa
kesamaan = `TRAINING_KEYS` + `data_recipe` (`scripts/export_methods.py:107-117, 325-351`), ditambah
`TREATMENT_KEYS["arch"] = ("arch", "arch_kwargs", "label_order")`.

**Desain A (utama): semua lengan dari nol, dua tahap.**

- Tahap 1 = resep `base`: `--train-lines 50000 --val-lines 500 --batch-size 32 --lr 1e-3 --steps 1500 --eval-every 250
  --augment none --workers 4 --device xpu --time-budget-hours 0 --max-batch-columns 64000 --seed S`.
- Gerbang antar tahap (kriteria 4b PLAN.md): overfit 32 < 1 % (sebelum tahap 1) dan val bersih tahap 1 < 2 %. Gagal =
  lengan dilaporkan "tidak terjawab pada anggaran ini" dengan angkanya, tanpa tahap 2; gagal overfit 32 =
  "bug/ketidakcocokan kontrak", bukan hasil ablasi (aturan kerja CLAUDE.md).
- Tahap 2 = resep `fase7_track` (`scripts/run_fase7.sh:13-18`, tanpa `--rare-*`): `--init <tahap1 last.pt> --augment
  fase5 --drop-space-prob 0.5 --extra-fonts fonts/extra --train-lines 100000 --val-lines 500 --batch-size 32 --lr 3e-4
  --steps 1500 --eval-every 500 --track-prob 0.5 --track-max 0.3 --workers 4 --seed S`.
- **Kontingensi**, diputuskan di K0 sebelum lengan lain: bila K0 **satu tahap** resep tahap 2 dari nol (`--lr 1e-3`)
  mencapai val bersih < 1 % di `last.pt`, semua lengan memakai satu tahap (biaya separuh). Kalau tidak, semua dua tahap.
  Pilihan dicatat sebelum K1 dimulai.

**Desain B (hemat, kesimpulan lebih sempit): tulang punggung CNN `fase7_track` dipakai bersama.** `cnn.*` + `proj.*`
(1.913.568 parameter) dimuat sebagian ke semua lengan (`--init-partial cnn,proj`, kode baru), hanya blok sekuens/decoder
yang dilatih dari acak; CRNN kontrol = LSTM di-reinit. Satu tahap 1.500 langkah per lengan (resep tahap 2). Hemat
~separuh anggaran, tetapi encoder sudah "melihat" 6.298 langkah bersama BiLSTM, jadi hasilnya hanya tentang blok
sekuens pada ciri yang dibentuk untuk BiLSTM, dan K4 ikut memakai CNN yang sama. Rekomendasi: A; B bila anggaran
malam tidak ada.

**Keadilan hyperparameter.** lr 1e-3/3e-4 disetel untuk CRNN. Pilihan: (i) sapuan kecil yang dipraregistrasi per
arsitektur BARU (3 nilai lr: 3e-4, 1e-3, 3e-3; 300 langkah resep tahap 1; dipilih dari val bersih font inti; aturan
divergensi: NaN/loss naik = nilai itu gugur), CRNN ikut disapu dengan aturan sama; biaya 3 × 0,2 × jam-per-run per
arsitektur ≈ 2-3,3 jam per arsitektur termasuk CRNN; atau (ii) lr tetap dan "hyperparameter disetel untuk CRNN" sebagai
batasan tertulis. lr tahap 2 tetap 3e-4 untuk semua (dicatat sebagai batasan bila (i) dipilih).

**Invarian ablasi (bukan variabel).** RAQM, NFC, reorder dari uharfbuzz, blank = 0 untuk CTC, `in_lens` dari lebar
asli atau mask padanannya, CER pada urutan logis, `to_tensor`, filter T ≥ 1,5L (juga K4, supaya baris latih identik),
NusaAksara hanya test, **fp32 untuk semua lengan** (`src/train.py` tanpa autocast; bf16 hanya eksperimen terpisah untuk
semua lengan sekaligus), **`--max-batch-columns` sama untuk semua lengan** (nilai berbeda = statistik BatchNorm dan
`split_batches` berbeda = faktor kedua; bila K2 memaksa nilai lebih kecil, catat sebagai pengganggu), **CTC di CPU
untuk semua lengan** (pilihan kode `src/train.py:143-144`, bukan keharusan: kernel CTC XPU ada sejak torch-xpu-ops #925),
`--workers > 0` untuk semua (urutan batch lintasan pertama memakai `seed + 1` hanya bila ada worker; jumlah worker 2
atau 4 tidak mengubah urutan).

**Titik lanjut.** Satu proses Windows mandiri tanpa henti per tahap (charger terpasang, Docker/WSL2 mati,
`training_count == 0` sebelum mulai, `Get-WinEvent` sesudahnya). Terputus = tahap itu diulang dari awal; maksimal satu
pengulangan per lengan, sesudah itu lengan dilaporkan terputus. Satu training pada satu waktu; tidak ada evaluasi
besar, ekspor, atau pytest selama training.

**Gerbang per arsitektur baru (sebelum run panjang).** Test kontrak lolos (bentuk `W // 4` pada W = 4, 97, 400, 1601;
`WIDTH_STRIDE == DOWNSAMPLE`; `lengths`; finit; tolak tinggi ≠ 96; **kesetaraan padding**: `model(X_padded, lens)[:,
:len] == model(x_single)` toleransi 1e-5 di CPU), lalu `--overfit 32 --epochs 1000 --eval-every 25` CER < 1 % di
`.venv-xpu` (sekaligus bukti SDPA/LayerNorm jalan di XPU 2.14 dan ukuran ms/sampel + memori puncak). Untuk K4 jalur
overfit dan `evaluate()` di `src/train.py:195-205` harus bercabang ke `model.decode`.

**Checkpoint yang dievaluasi** = `last.pt` → `last_snapshot.pt` tahap terakhir (bukan `best.pt`, yang memihak val bersih).

**Seed.** Satu `--seed` mencampur tiga sumber: subset baris (`src/train.py:264`), urutan batch, dan inisialisasi bobot.
Lantai derau K0 seed 0 vs seed 1 = varians total run, bukan varians inisialisasi saja; K1 (CRNN) berbagi init dan
subset dengan K0 seed 0, K2/K3 tidak, jadi K1−K0 lebih berpasangan daripada K2−K0. Bila ingin dipisah: `--init-seed`
(keputusan user).

## 7. Titik akhir dan aturan pemilihan (ditulis sebelum angka)

- **Dev = split VAL sintetis, font javatext, citra IDENTIK untuk semua checkpoint** (`SyntheticLines(deterministic=True,
  seed=0)`, pola `eval_spacing.py`), dua kondisi: bersih tanpa-spasi 64 px dan augmentasi `fase5`. Jumlah baris dipilih
  dari lebar SK yang diukur pada smoke test (`fase7_track` vs `fase7_ctrl`, citra identik) pada 300 / 1.000 / 3.000
  baris SEBELUM run; aturan bila |Δ seed| melebihi efek terbesar yang diharapkan ditulis saat itu juga ("ulang seed 2
  untuk semua lengan" atau "laporkan tak terbedakan dan berhenti"). Split TEST sintetis disimpan untuk G1/G2 10.000.
  Catat: javatext sekeluarga dengan CarakanJawa.
- **Jalur inferensi dev** = `predict` satu baris (pola `eval_spacing`), bukan batch; selisih batch-vs-satu-baris
  dilaporkan per arsitektur sebagai cek kebenaran mask.
- **Utama:** selisih CER (logis, NFC) lengan − K0 seed 0 pada kondisi bersih tanpa-spasi, SK95 bootstrap berpasangan
  per baris (`compare_runs.bootstrap_diffs`, 10.000 resample) + uji tanda.
- **Sekunder (tidak boleh memburuk dengan SK di luar 0):** CER augmentasi `fase5`; CER berspasi dan recall spasi; spasi
  palsu/100 batas pada 0,1-0,3 em (`eval_spacing.py`; untuk K4 autoregresif 3.612 bacaan × 1-5 dtk = 1-5 jam CPU);
  CER subset baris bertaling U+A9BA/U+A9BB (K1/K4); q100 G1/G2; G1/G2 10.000 hanya untuk lengan yang menang/setara.
- **K4 (praregistrasi metrik halusinasi):** tingkat sisipan = jumlah op insert / karakter referensi (per baris, SK);
  halusinasi = porsi baris dengan |hyp|/|ref| > 1,5 atau n-gram ≥ 4 berulang ≥ 3×; "memburuk berpasangan" = baris yang
  K0 persis benar tetapi K4 salah (uji tanda).
- **Biaya, selalu:** parameter, FLOP per kolom, ms/sampel XPU (`samples_per_s` log), memori puncak, `split_batches`,
  ms/baris `predict` CPU, jam dinding (`hours` event end).
- **G3 745 baris:** SEKALI per lengan, di akhir, sesudah analisis dev tertulis (`out/compare/arch/dev_<lengan>.md`);
  `compare_runs.py` lengan vs K0 (SK halaman, spasi keluaran vs 110, recall aksara langka). Hanya pelapor. Skrip rantai
  TIDAK memakai `evaluate` dari `scripts/fase6_common.sh:56-68` (yang menjalankan G3 otomatis): `evaluate_arch` tanpa
  `--real`, G3 hanya di `after_arch.sh`.
- **Aturan:** "lebih baik" = SK95 utama tidak memuat 0 ∧ |Δ| > lantai derau ∧ tidak ada sekunder memburuk dengan SK di
  luar 0; "setara" = SK memuat 0 ∧ lebar SK < 2 × lantai; selain itu "belum bisa dibedakan". Tiap klaim PLAN.md
  dinyatakan didukung / tidak didukung / tidak terjawab. Dilarang menambah langkah/seed/varian untuk satu lengan sesudah
  melihat angka, mengubah ambang, atau membaca G3 sebelum dev selesai. Pemenang dev TIDAK otomatis resmi (keputusan
  user, `make_official.sh`, `OFFICIAL_RUN`, `DEFAULT_CHECKPOINT`; untuk non-CRNN kartu metode harus digeneralisasi dulu).
- **Praregistrasi yang bisa diperiksa:** commit hash `scripts/eval_arch.py` + tanggal dicatat di CLAUDE.md; sha skrip
  ditulis ke setiap laporan `out/compare/arch/*.json` (pola `file_sha` di `eval_spacing.py` dan `report_matches` kartu
  metode); skrip tidak diubah sampai semua lengan selesai (perbaikan bug = versi baru + semua lengan dievaluasi ulang).

## 8. Accuracy, precision, recall, F1 di web (permintaan user 2026-10-09)

CER tetap metrik gerbang (PLAN.md §1.2). Metrik klasifikasi ditambahkan sebagai **pelengkap** dengan definisi tunggal
yang konsisten dengan CER, dari alignment karakter yang sudah ada (`scripts/compare_runs.py:162`, `alignment()`:
biaya Levenshtein minimum, lalu kecocokan terbanyak, lalu substitusi sekelas; sudah dipakai untuk recall/presisi aksara
langka). Dari alignment satu baris: M cocok, S substitusi, D hapus, I sisip, dengan |ref| = M + S + D dan |hyp| = M + S + I.

| Metrik | Rumus (mikro, dijumlahkan atas semua baris) | Catatan |
|---|---|---|
| CER | (S + D + I) / |ref| | sudah ada (`src/decode.py::cer`); bisa > 1 |
| Recall karakter | M / |ref| | "berapa bagian referensi terbaca benar"; 1 − recall = (S + D)/|ref| |
| Precision karakter | M / |hyp| | turun bila model mengarang (sisipan, halusinasi K4) |
| F1 karakter | 2PR / (P + R) | satu angka untuk tabel |
| Accuracy baris | baris persis / baris | sudah ada (`exact`) |
| Accuracy karakter | M / (M + S + D + I) | porsi langkah alignment yang benar, terikat [0, 1]; opsional, bila user ingin kolom "accuracy" tingkat karakter |
| Per kelas (92 karakter) | TP = M pada c; FN = S + D dengan referensi c; FP = S + I dengan hipotesis c → P, R, F1 per kelas, macro-F1, dan per kelompok (aksara, sandhangan, pangkon, angka, pada) | menjawab "aksara mana yang lemah"; pelengkap tabel kebingungan halaman Kesalahan |

Perubahan yang dibutuhkan (tanpa mengubah angka resmi). **Butir 1-4 DIKERJAKAN 2026-10-09** (commit di cabang ini;
rincian di CLAUDE.md "Metrik karakter"); angkanya baru tampil setelah `php artisan migrate`, `export_results.py`,
dan `aksara:import` dijalankan di laptop.

1. ~~Pindahkan ke `src/align.py`~~ → dipindah ke modul baru **`src/metrics.py`** (tanpa torch): `alignment`,
   `glyph_class`, `char_counts`, `rates`, `char_metrics`, `class_metrics`, `macro_f1`; `compare_runs` mengimpornya
   dan tetap mengekspor namanya.
2. `src/evaluate.py::summarize` menambah `precision`, `recall`, `f1`, `char_accuracy`, `counts` (laporan
   `out/eval/*.json` lama tanpa kunci itu tetap sah; `gate_problem` tidak membacanya).
3. `scripts/export_results.py::aggregate` menambah kunci yang sama ke tiap metrik manifest (`nusaaksara_745`,
   `blind_50`), plus blok `class_metrics` untuk pipeline yang punya prediksi di semua baris. Skema tetap 1 dengan
   kunci opsional.
4. Web: migrasi `2026_10_09_000001_add_character_metrics` (kolom nullable di `metrics`, tabel `class_metrics`),
   `ResultImporter` meneruskannya, model `ClassMetric`. Perbandingan: kolom Presisi / Recall / F1 di tabel 745 baris
   dan uji buta (juga kolom Akurasi karakter; Baris persis = akurasi tingkat baris), definisi di bawah tabel.
   Kesalahan aksara: macro-F1 dan mikro di kepala, 20 aksara F1 terendah (≥ 10 kemunculan di label) untuk pipeline
   resmi; aksara di label yang tidak pernah dikeluarkan model diberi F1 0 (konvensi zero_division=0), bukan
   dikecualikan; daftar kebingungan halaman itu kini dihitung dari penjajaran yang sama. Pemilih pipeline belum
   (semua pipeline sudah ada di tabel `class_metrics`). Test `web/tests/Feature/CharacterMetricsTest.php` membaca
   sel per atribut `data-cell` di dalam baris `data-row|blind|char`, dan bagian `data-part`.
5. **Belum:** untuk ablasi arsitektur: kontrak baru `manifest["architecture_ablation"]` (bukan `ablation_runs`, yang diganti utuh
   saat impor dan milik ablasi augmentasi): satu baris per lengan dan scope (`synth_dev_clean`, `synth_dev_aug`,
   `nusaaksara_745`) dengan cer, precision, recall, f1, exact, params, ms_per_sample, ms_per_line_cpu, diff_vs_control,
   ci_low, ci_high (dari `out/compare/arch/`), kontrol, seed, status klaim. Pipeline lengan masuk `PIPELINES` lewat
   daftar `ARCH_CHECKPOINTS` (kunci `arch_<run>`, `kind` = nama arsitektur, config ≤ 255 karakter), terpisah dari
   `OPTIONAL_CHECKPOINTS` (pola `crnn_<run>` tetap utuh); `Perbandingan::ARCH_RUNS` terpisah dari `FOLLOWUP_RUNS`;
   `Ringkasan.php:33` tidak memasukkan `arch_*` ke sumbu G3; kartu metode: `CONTROLS["arch"]` + `TREATMENT_KEYS["arch"]`
   dan cabang `model_card` per arsitektur (hanya bila sebuah lengan non-CRNN kelak jadi resmi). Jalur termurah bila
   kontrak baru ditunda: entri statis `PIPELINES` + berkas prediksi (pola `vlm_zeroshot`), ~2 jam.

Tampilan: tabel satu-faktor "lengan vs K0" di bagian baru halaman Ablasi (Ablasi arsitektur), kolom CER, P, R, F1, baris
persis, Δ CER vs K0 dengan SK, parameter, ms/sampel; baris `fase7_track` sebagai rujukan berlabel "bukan kontrol";
kalimat batasan baku dari §11 ditampilkan di bawah tabel.

## 9. Perubahan kode minimal, berurutan

1. **Registry** `src/archs.py`: `ARCHS = {"crnn": CRNN, ...}`, `build_model(model_config)` membaca
   `model_config.get("arch", "crnn")` (checkpoint lama tetap CRNN), kwargs sisanya diteruskan. `src/infer.py:27-28` →
   `build_model`; anotasi `CRNN` → `nn.Module` (`infer.py:21,59`, `train.py:129`). `src/train.py`: `--arch` (choices
   ARCHS) + `--arch-kwargs` JSON (pertahankan `--channels/--hidden` untuk crnn), konstruksi `:328-329` lewat
   `build_model`, `model_config` (`:215`) menyimpan `arch` + kwargs + `label_order`, `--init` (`:347`) menolak `arch`
   beda, `assert_downsample` membandingkan `DOWNSAMPLE` dengan `type(model).WIDTH_STRIDE`, `--no-pack` ditolak untuk
   arsitektur yang tidak mendefinisikannya. Kontrak kelas: `forward(x, lengths=None) -> (B, W//4, C)`, `WIDTH_STRIDE =
   4`, `height = 96`, `IS_CTC = True`. Literal `nn.CTCLoss(blank=BLANK, zero_infinity=True)` tetap (CODE_FACTS).
   Tambahkan modul arsitektur ke `CODE_FILES` `eval_spacing.py:133` dan naikkan `CACHE_VERSION`; jadwalkan satu hitung
   ulang cache saat GPU idle, lalu jangan menyentuh `model.py`/`infer.py` lagi. (0,5 hari)
2. **`--label-order {visual,logical}`** sebagai bendera `visual_order` di objek `Tokenizer` yang dibaca
   `load_checkpoint` dari `model_config.label_order` dan membuat `to_visual`/`to_logical` identitas, sehingga
   `dataset.py:169`, `real.py:143`, `decode.py:29`, `train.py:202`, `evaluate.py:50`, `eval_rare.py`, `eval_spacing.py`,
   `align.py:114` mengikut tanpa diubah satu per satu. JANGAN mengosongkan `tokenizer.reorder`. Charset/reorder tetap
   disimpan supaya `eval_rare.py:183-186` tidak menolak. `beam_eval`/`serve` menolak checkpoint logical dengan pesan.
   (0,5 hari)
3. **Test**: parametrisasi `tests/test_shapes.py` atas ARCHS (kontrak bersama + kesetaraan padding), `feat_height`
   dan rentang parameter 3-7 M hanya CRNN; `tests/test_dataset.py` dan `tests/test_real.py` atas `label_order`;
   `build_model` tanpa `arch` → CRNN; `--init` lintas arsitektur ditolak. `py_compile` + impor di `.venv-xpu` + pytest
   sebelum training berikutnya. (0,5 hari)
4. **`scripts/eval_arch.py` + `tests/test_eval_arch.py`** (dev berpasangan, biaya, subset bertaling, metrik K4; untuk
   `IS_CTC=False` memakai `model.decode`; menulis sha skrip ke laporan), **`scripts/run_arch.sh`** (fungsi
   `fase6_common.sh`: `wait_no_training`, `run_train`; overfit → sapuan lr → tahap 1 → gerbang → tahap 2 →
   eval_arch/eval_spacing/q100 **tanpa `--real`**) + `scripts/after_arch.sh` (G1/G2 10k, G3 sekali, compare_runs).
   Dibekukan sebelum run; smoke test pada `fase7_track`/`fase7_ctrl` (hasil tidak dipakai menyetel). (1,5 hari)
5. **K2 `TransformerCTC`** (CNN+proj CRNN dipakai ulang; `nn.TransformerEncoder` d 256 × 3-4 lapis × 4 head, FFN 1024,
   posisi sinusoidal, `key_padding_mask` dari `lengths`; opsi atensi berjendela/`max_frames` karena O(T²)) dan **K3
   `ConvCTC`** (Conv1d depthwise berdilasi + GLU + residual, receptive field dicatat). Overfit 32 masing-masing. (1,5-2 hari)
6. **Metrik P/R/F1** (§8 butir 1-4). (1 hari, bisa paralel dengan training)
7. **K4 `AttnReader`** (`loss()`, `decode()`; cabang `train.py:376-381` dan `:195-205`, `evaluate.py:46-51`,
   `infer.py:64-66`, `export_results.py:374-387` → `predict()` + `spans=None`; `serve.resources()` menolak
   `IS_CTC=False`). Test overfit kecil (8 sampel, CPU) sebelum sesi XPU. (2-3 hari, sesi tersendiri)
8. **K6** bila dipilih: `export_results.py:384-386` beam untuk kunci resmi, bukan hanya `crnn_fonts`;
   `TranslationRun::forOcr` ikut. (2 jam, tanpa training)
9. **Web ablasi arsitektur** (§8 butir 5) setelah ada lengan yang layak dilaporkan. (1 hari)

## 10. Biaya (jam dinding; 3-5,5 jam per 1.500 langkah CRNN, periksa `hours` di `out/checkpoints/*/log.jsonl`)

| Tahap | XPU | CPU serial (tidak bersamaan training) | Kerja |
|---|---|---|---|
| Sekali di awal: registry, label-order, test, eval_arch, run_arch.sh, smoke test, hitung ulang cache eval_spacing | — | ~1-2 jam | 2,5-3 hari |
| K0 seed 0 + seed 1 (desain A: 2 × 2 tahap) | 12-22 jam | dev + G1/G2 10k kedua seed + G3 ≈ 4 jam | skrip 2 jam |
| K1 (tahap 1; tahap 2 bila lolos) | 3-11 jam | ~1,5 jam | 0,5 hari |
| K2 (+ overfit, sapuan lr bila dipilih) | 8-14 jam | ~1,5 jam | 1 hari |
| K3 | 5-9 jam | ~1 jam | 0,5 hari |
| Sapuan lr (bila dipilih), 4 arsitektur termasuk CRNN | 8-13 jam | — | — |
| **Inti K0-K3** | **28-56 jam (8-12 malam)** | ~8 jam | ~5 hari kerja |
| K4 | 10-14 jam | dev autoregresif + eval_spacing 2-6 jam; G1/G2 hanya q100 + 1.000 baris | 2-3 hari |
| K6 / K5 / K7 | 0 / 6-11 jam / 0 | 1,5 jam / 1 jam / 8-24 jam WSL2 | 2 jam / 2 jam / 1-2 hari |
| Cadangan insiden (Hibernate, baterai, OOM; 3 insiden tercatat di rantai fase6/7) | +25 % | | |

Desain B (tulang punggung bersama, satu tahap): K0-K3 ≈ 15-28 jam XPU (4-6 malam).

## 11. Batasan yang wajib disebut di laporan

- Satu run per lengan; lantai derau dari dua seed CRNN = satu sampel varians, bukan distribusi.
- Angka "dari nol pada anggaran 3.000 langkah" (atau 1.500 satu tahap): tidak sebanding dengan angka resmi
  `fase7_track` (rantai 6.298 langkah); `fase7_track` hanya rujukan.
- Citra latih antar lengan tidak berpasangan (render worker acak, `src/dataset.py:117-120`); hanya evaluasi yang berpasangan.
- Hyperparameter disetel untuk CRNN (atau: disapu kecil dengan aturan sama; lr tahap 2 tetap).
- "Parameter sama" ≠ FLOP sama; laporkan keduanya + ms/sampel; fp32 untuk semua lengan.
- Font uji javatext sekeluarga dengan font latih CarakanJawa; tiga cacat font latih (BasaJan tanpa shaping di
  `.venv-xpu`, CarakanJawa tanpa O, NewKramawirya) berlaku untuk semua lengan.
- G3 = test set NusaAksara, dibaca sekali per lengan, SK per halaman, hanya arah; sudah dievaluasi ≥ 37 kali oleh run
  sebelumnya.
- K2: memori atensi O(T²) dan dukungan SDPA XPU diuji baru di overfit 32 (Core Ultra Seri 1 tidak ada di daftar
  validasi dokumen XPU walau terbukti jalan); K3: hasil bergantung receptive field yang dipilih; K4: beam + LM, spans,
  skor kandidat Demo, kartu metode "tidak berlaku"; K5/K6/H/stride bila dijalankan: desain berbeda, dilaporkan terpisah.
- Metrik P/R/F1 bergantung pada pilihan alignment di antara penjajaran berbiaya sama (kecocokan terbanyak); CER tidak.
- Angka resmi, Demo, dan kartu metode tidak berubah oleh ablasi ini.

## 12. Pertanyaan untuk user (jawab sebelum sesi 1), dengan rekomendasi

1. Lengan: K0-K3 inti (8-12 malam) saja, atau juga K4 (+2-3 hari kode, +2 malam)? **Rekomendasi: K0-K3 dulu, K4 sesi
   berikutnya.** K5/K6/K7/K8 dan H = 64 / stride 8 (menguji invarian 1 dan 5): masuk atau tidak?
2. Desain A (semua dari nol, dua tahap) atau B (tulang punggung CNN `fase7_track` bersama)? **Rekomendasi: A.**
3. Bersedia membayar K0 seed kedua (6-11 jam) untuk lantai derau? **Rekomendasi: ya; tanpa ini tidak ada yang bisa
   disimpulkan.**
4. Sapuan lr kecil per arsitektur (~2-3 jam per arsitektur, termasuk CRNN) atau lr tetap + batasan tertulis?
   **Rekomendasi: lr tetap pada iterasi pertama, batasan ditulis.**
5. `--max-batch-columns` 64.000 dan fp32 sebagai invarian ablasi? **Rekomendasi: ya.**
6. BOS/EOS K4 di luar charset (`n_classes + 3`) **(rekomendasi)** atau memakai ulang indeks 0?
7. Hasil cukup di `out/eval` + `out/compare/arch/` dulu, web menyusul; metrik P/R/F1 (§8 butir 1-4) dikerjakan sekarang
   untuk pipeline yang sudah ada? **Rekomendasi: ya, karena tidak bergantung pada training.**
8. Ablasi ini dihitung "fase" baru (satu fase per sesi) dan praregistrasinya ditulis di mana: CLAUDE.md (sudah ~900
   baris) atau dokumen ini dengan tanggal + commit hash? **Rekomendasi: dokumen ini, satu baris penunjuk di CLAUDE.md.**
9. Angka jam dinding per 1.500 langkah untuk anggaran: 3-3,5 jam atau ~5,3 jam (`hours` di `log.jsonl`)?

## 13. Sumber

Terverifikasi dari berkas sumber (github.com / raw.githubusercontent.com, satu-satunya host yang bisa dibuka dari
kontainer ini) atau kutipan teks primer di hasil pencarian:

- Tesseract: `tessdata/script` (37 model, tanpa Javanese) <https://github.com/tesseract-ocr/tessdata/tree/main/script>;
  `langdata_lstm/jav/jav.unicharset` dan `okfonts.txt` (Latin saja); tesstrain <https://github.com/tesseract-ocr/tesstrain>;
  blog jrenslin.de/post/42 ("limited to Javanese texts using latin script").
- Kraken: `docs/user_guide/training_recognition.rst`, `models.rst` (mittagessen/kraken).
- docTR `doctr/datasets/vocabs.py` (vocab `javanese`), `references/recognition/README.md`; EasyOCR `easyocr/config.py`;
  PaddleOCR `PP-OCRv5_multi_languages.en.md`, `algorithm_rec_svtr.en.md`.
- PARSeq README (baudm/parseq: 23,833 M); TrOCR README (microsoft/unilm: 62/334/558 M, `--input-size 384`); Donut dan
  Pix2Struct README; Qwen3-VL README; docling `docs/usage/vision_models.md`; transformers `training_args.py` (xpu);
  PyTorch `docs/source/notes/get_start_xpu.md`; torch-xpu-ops #925 (kernel CTC).
- NusaAksara, Adilazuarda dkk., ACL 2025, arXiv 2502.18148 (PP-OCRv3 fine-tune CER > 1; GPT-4o CER > 1 di sebagian
  besar aksara). Diaz dkk. 2021, arXiv 2104.07787. Coquenet dkk. ICFHR 2020 (GFCN), arXiv 2012.04961. Michael dkk.
  2019 (CE + CTC λ 0,5), arXiv 1903.07377. Kim, Hori, Watanabe ICASSP 2017 (hybrid CTC/attention), arXiv 1609.06773.
  CVIT IIIT-H 2022, arXiv 2205.06740. SVTR, arXiv 2205.00159 (T 4,15 M, S 8,45 M). GraniteDocling: pengumuman IBM
  (Apache 2.0, 258 M).

Belum terverifikasi (sumber sekunder atau tidak terjangkau): ketiadaan model Kraken aksara Jawa di Zenodo; model
publik Transkribus PyLaia aksara Jawa cetak (disebut hasil pencarian, CER validasi ~5,7-6,3 % pada datanya sendiri;
layanan awan, cek lisensi NusaAksara sebelum memakai); dukungan SDPA/`nn.TransformerEncoder` di XPU mesin user;
tokenizer TrOCR-small untuk blok U+A980; angka IndicSTR12 (PARSeq vs CRNN); semua perkiraan jam (proporsional terhadap
140 ms/sampel model 4,6 M) sampai diukur di overfit 32.
