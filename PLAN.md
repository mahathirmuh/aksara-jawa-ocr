# Aksara Jawa OCR — Rencana Kerja

Pengenalan teks aksara Jawa tingkat baris berbasis CRNN + CTC, dilatih dari data sintetis hasil render font Unicode.

---

## 1. Goals

### 1.1 Tujuan utama

Membangun sistem OCR yang menerima **citra satu baris teks aksara Jawa cetak** dan mengeluarkan **string Unicode aksara Jawa** (blok U+A980–U+A9DF) yang benar secara logis dan ternormalisasi NFC.

Input: `Image (H×W, grayscale)` → Output: `str` (Unicode aksara Jawa)

### 1.2 Kriteria keberhasilan — terukur

Proyek dianggap selesai ketika **keempat** target ini tercapai:

| # | Metrik | Target | Diukur pada |
|---|---|---|---|
| G1 | CER | < 2% | 10k baris sintetis bersih (held-out, font & teks tak terlihat saat training) |
| G2 | CER | < 5% | 10k baris sintetis dengan augmentasi berat |
| G3 | CER | < 8% | ≥200 baris nyata (foto papan nama / scan buku cetak) |
| G4 | Round-trip | 100% | `decode(encode(s)) == NFC(s)` di seluruh korpus |

CER dihitung pada **urutan logis Unicode**, setelah normalisasi NFC dan canonical ordering sandhangan, bukan pada urutan visual.

### 1.3 Definisi "selesai" per komponen

- **Tokenizer**: bijektif, teruji pada ≥100k baris, reorder visual↔logis 100% round-trip
- **Renderer**: shaping terverifikasi manual untuk 11 kasus uji, di 2 font
- **Korpus**: ≥200k baris Unicode valid, round-trip CER < 5% pada ≥95% baris
- **Model**: memenuhi G1–G3, checkpoint tersimpan, inference script jalan standalone
- **Reproducibility**: `python -m src.train` dari repo bersih menghasilkan model yang memenuhi G1

---

## 2. Scope

### 2.1 Termasuk

- Teks aksara Jawa **cetak** dan **teks digital ter-render** (papan nama, buku cetak, dokumen digital)
- Tingkat **baris tunggal** (single text line)
- Output Unicode aksara Jawa logis

### 2.2 TIDAK termasuk — jangan dikerjakan di repo ini

| Di luar scope | Alasan | Kapan dipertimbangkan |
|---|---|---|
| Tulisan tangan | Render bukan proses generatif sebenarnya; synthetic-only akan gagal | Setelah G1–G3 tercapai, butuh anotasi manual besar |
| Naskah lontar / manuskrip | Butuh layout analysis + Jawa Kuno; gunakan Kraken/eScriptorium | Proyek terpisah |
| Layout analysis / deteksi baris | Masalah berbeda (segmentasi halaman) | Fase lanjutan, pakai Kraken BLLA |
| Transliterasi ke Latin | Deterministik, sudah ada library | Repo terpisah, ~2 hari kerja |
| Machine translation ke Indonesia | Off-the-shelf (NLLB / Sahabat-AI) | Repo terpisah |
| Deteksi tingkat tutur | Sudah ada baseline leksikon + Unggah-Ungguh | Repo terpisah |

**Aturan keras:** repo ini hanya mengerjakan citra baris → Unicode. Empat baris terakhir tabel di atas adalah proyek terpisah dengan repo sendiri. Membangunnya sekarang akan mengaburkan sumber error dan memperlambat semuanya.

---

## 3. Metode

### 3.0 Identifikasi metode

**Nama lengkap:** *Segmentation-free line-level sequence recognition* menggunakan arsitektur **CRNN (Convolutional Recurrent Neural Network)** yang dilatih dengan **CTC loss (Connectionist Temporal Classification)**, dengan strategi data **synthetic pretraining + supervised fine-tuning**.

Singkatnya: **CRNN + CTC**.

Ini arsitektur standar, bukan metode baru. Yang spesifik untuk kasus aksara Jawa adalah penanganan reordering (§3.3), pilihan `H=96`, dan skema tokenisasi — bukan arsitekturnya.

**Referensi dasar:**
- CRNN — Shi, Bai & Yao (2015), *An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition*
- CTC — Graves, Fernández, Gomez & Schmidhuber (2006), *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*

#### Empat komponen, masing-masing menyelesaikan satu masalah

| Komponen | Kategori ML | Menyelesaikan |
|---|---|---|
| CNN | Supervised feature learning | Ekstraksi fitur visual dari piksel mentah |
| BiLSTM | Recurrent sequence modeling | Konteks dua arah sepanjang baris |
| **CTC** | **Weakly-supervised alignment** | **Belajar tanpa label per-karakter** |
| Synthetic data | Data-centric / domain randomization | Mengatasi ketiadaan dataset |

**Komponen yang paling menentukan adalah CTC.** Alasannya spesifik: kita punya label "baris ini berbunyi X" tapi **tidak punya** label "kolom piksel 40–70 adalah karakter ka". CTC memungkinkan training dari label lemah itu dengan memarginalkan seluruh alignment yang mungkin. Tanpa CTC, posisi setiap karakter di ratusan ribu baris harus dilabeli manual — mustahil.

#### Posisi dalam taksonomi ML

| Aspek | Klasifikasi |
|---|---|
| Paradigma | Supervised, spesifiknya **weakly-supervised** (alignment tidak dilabeli) |
| Task | **Sequence-to-sequence transduction** (citra → urutan simbol), bukan klasifikasi |
| Loss | CTC — marginalisasi atas ruang alignment monotonik |
| Strategi data | Synthetic generation + domain randomization, lalu fine-tuning pada data nyata |
| Inference | Greedy CTC decoding; opsional beam search + n-gram LM |

> **Catatan supaya tidak salah kaprah:** ini **bukan** transfer learning dan **bukan** fine-tuning model pretrained. Model dilatih dari nol. Yang berpindah adalah dari domain sintetis ke domain nyata — itu **domain adaptation**, bukan transfer learning dalam arti memakai bobot ImageNet.

#### Metode yang TIDAK dipakai, dan alasannya

Literatur OCR aksara Jawa lokal hampir seluruhnya memakai baris pertama tabel ini. Itu sebabnya hasilnya tidak bertahan di data nyata.

| Metode | Kenapa tidak dipakai |
|---|---|
| CNN klasifikasi karakter (GoogLeNet, Xception, ResNet) | Butuh segmentasi karakter dulu — mustahil pada pasangan yang menumpuk vertikal |
| k-NN / SVM + feature extraction manual | Sama, plus kapasitas tidak cukup untuk 600+ kelas karakter |
| Attention encoder-decoder (TrOCR, Donut) | Menangani reordering secara native, tapi butuh data nyata jauh lebih banyak. **Kandidat upgrade setelah Fase 6** |
| VLM end-to-end (GPT-4o, Qwen-VL) | Terbukti gagal — chrF++ mendekati nol untuk terjemahan langsung dari citra aksara Nusantara |
| Self-supervised pretraining | Tidak ada korpus citra aksara Jawa berskala besar untuk di-pretrain |

### 3.1 Kenapa bebas-segmentasi

Pendekatan "potong karakter lalu klasifikasi" mengasumsikan ada garis potong vertikal antar karakter. **Asumsi ini tidak berlaku untuk aksara Jawa:**

- **Pasangan** menumpuk vertikal (`C1 + U+A9C0 pangkon + C2` → C2 di bawah C1)
- **Sandhangan** mengelilingi aksara dasar dari 4 sisi
- **Taling-tarung** mengapit aksara dasar (circumfix)

Satu kolom piksel bisa memuat 3 codepoint berbeda. Error segmentasi tidak bisa dipulihkan di tahap klasifikasi. Karena itu: **model membaca seluruh baris sekaligus, tanpa segmentasi.**

### 3.2 CTC — menyelesaikan alignment tanpa melabelinya

Citra baris → CNN → T kolom fitur. Target L karakter. Kita tidak tahu kolom mana milik karakter mana, dan melabelinya manual mustahil.

CTC menambah token **blank (∅)** dan mendefinisikan aturan collapse: gabungkan token berulang bersebelahan, lalu buang blank.

```
k  k  ∅  a  a  ∅   →  "ka"
∅  k  ∅  ∅  a  a   →  "ka"
k  ∅  a  ∅  ∅  ∅   →  "ka"
```

CTC loss = −log dari total probabilitas **semua** jalur yang collapse ke target, dihitung dengan forward-backward DP. Model belajar sendiri di kolom mana ia menaruh apa.

### 3.3 Dua batasan CTC — keduanya menggigit di sini

**Batasan A — alignment harus monoton.** Taling (U+A9BA) digambar **di kiri** aksara dasar tapi disimpan **sesudahnya** di Unicode. Urutan visual `[taling][ka]`, urutan logis `[ka][taling]`. Monotonisitas dilanggar.

> **Solusi:** latih pada **urutan visual**, terapkan **reorder deterministik saat decode** untuk kembali ke urutan logis. Aturan reorder diturunkan otomatis dari posisi glyph HarfBuzz, bukan ditulis manual.

**Batasan B — T ≥ L.** Tumpukan pasangan memampatkan banyak codepoint ke sedikit kolom. Suku kata tumpuk-tiga bisa 5 codepoint dalam ~40px; setelah downsample W/4 hanya 10 frame untuk 5 target.

> **Solusi:** tinggi normalisasi **96px** (bukan 32), pooling lebar dihentikan setelah 2 tahap (total stride lebar = 4), dan filter dataset `T ≥ 1.5 × L`.

### 3.4 Kenapa synthetic data bukan kompromi di sini

Untuk teks **cetak**, proses render **adalah** proses generatif yang sebenarnya — teks cetak Jawa memang dihasilkan dari font. Render dengan Tuladha Jejeg mereproduksi bentuk glyph asli, bukan mengaproksimasi. Label sempurna, gratis, tak terbatas.

Gap yang tersisa hanya **kanal degradasi** (kertas, tinta, scan, cahaya) — itulah yang dimodelkan augmentasi. Gap ini jauh lebih sempit daripada kalau konten harus disintesis.

Konsekuensi: untuk **tulisan tangan**, argumen ini tidak berlaku. Karena itu tulisan tangan di luar scope.

### 3.5 Arsitektur

```
Citra baris (1 × 96 × W)
   │
   ├─ CNN          ekstraksi fitur
   │               tinggi 96 → 1  (dipampatkan habis)
   │               lebar  W  → W/4 (SENGAJA dipertahankan)
   │
   ├─ BiLSTM ×2    konteks dua arah sepanjang baris
   │
   ├─ Linear       → n_classes (termasuk blank di indeks 0)
   │
   └─ CTC loss
```

| Keputusan | Alasan |
|---|---|
| Tinggi dipampatkan habis, lebar dijaga | Lebar = sumbu "waktu" CTC. Memampatkannya melanggar T ≥ L |
| BiLSTM **dua arah** | Makna sandhangan bergantung pada aksara dasar yang bisa ada di kanan (kasus taling). Model harus bisa melihat ke depan |
| H = 96, bukan 32 | Tumpukan pasangan butuh ruang vertikal. 32px menghancurkan sandhangan |
| CTC, bukan attention decoder | Attention menangani non-monotonisitas secara native tapi butuh data jauh lebih banyak. CTC punya bias induktif monotonik yang menguntungkan di sini |

Ukuran model ≈ 5M parameter (~20MB fp32). Kecil — bottleneck training adalah rendering di CPU, bukan GPU.

---

## 4. Requirements

### 4.1 Sistem

| Item | Kebutuhan | Catatan |
|---|---|---|
| OS | Linux / macOS | Windows via WSL2 |
| Python | 3.10 – 3.12 | |
| GPU | ≥8GB VRAM | Opsional untuk Fase 0–4; wajib untuk Fase 5 |
| RAM | ≥16GB | Rendering on-the-fly di worker |
| CPU | ≥8 core | **Ini bottleneck sesungguhnya**, bukan GPU |
| Disk | ~20GB | Korpus + checkpoint. Citra tidak disimpan (render on-the-fly) |

### 4.2 System packages

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y libraqm-dev libharfbuzz-dev libfribidi-dev \
                    libfreetype6-dev fonts-noto-core

# macOS
brew install libraqm harfbuzz fribidi freetype
```

### 4.3 Python packages

```
# requirements.txt
torch>=2.1
torchvision>=0.16
Pillow>=10.2
uharfbuzz>=0.39
numpy>=1.24
albumentations>=1.4
jiwer>=3.0            # CER/WER
rapidfuzz>=3.6
tqdm
tensorboard
pytest
```

Instalasi:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 4.4 Verifikasi kritis — jalankan SEBELUM apa pun

```bash
python - <<'EOF'
from PIL import ImageFont, features
print("RAQM  :", ImageFont.core.HAVE_RAQM)
print("HB    :", features.version("raqm"))
assert ImageFont.core.HAVE_RAQM, "RAQM TIDAK AKTIF — shaping tidak akan jalan"
EOF
```

> **Ini gerbang paling penting di seluruh proyek.**
>
> Kalau RAQM tidak aktif, Pillow tetap merender tanpa error apa pun — hanya saja pasangan tidak menumpuk dan taling di posisi salah. **Kegagalannya senyap.** Anda baru sadar setelah menghabiskan berjam-jam training.
>
> Kalau `False`: install `libraqm-dev` lalu rebuild Pillow:
> ```bash
> pip uninstall -y pillow
> pip install --no-binary :all: --no-cache-dir Pillow
> ```

### 4.5 Font — aset generator, wajib ≥2

Font bukan sekadar dependency. **Font ADALAH generator data training Anda.** Kualitas shaping-nya menentukan kualitas seluruh dataset.

| Font | Sumber | Peran |
|---|---|---|
| **Tuladha Jejeg** | Situs R.S. Wihananto / `bennylin.github.io/jawa-fonts.html` | Shaping OpenType paling lengkap → **referensi kebenaran** |
| **Noto Sans Javanese** | Google Fonts / repo `notofonts` | Bentuk glyph berbeda → variasi training |

Simpan di `fonts/`. Tambah font ketiga/keempat kalau tersedia — **variasi bentuk glyph adalah augmentasi paling berharga** untuk teks cetak, lebih berdampak daripada noise atau blur.

> Setiap font baru wajib lolos verifikasi Fase 0 sebelum dipakai. Font yang shaping-nya salah akan meracuni seluruh dataset secara senyap.

---

### 4.6 Dataset

Proyek ini butuh **tiga jenis data yang berbeda**. Membingungkan ketiganya adalah kesalahan umum — perhatikan bahwa hanya jenis C yang berupa citra nyata.

| | Jenis | Bentuk | Volume | Untuk fase |
|---|---|---|---|---|
| **A** | Korpus teks | Teks Latin → dikonversi ke Unicode aksara | ≥200k baris | 2 |
| **B** | Font | File `.ttf` | ≥2 | 0, 3, 5 |
| **C** | Citra nyata + anotasi | PNG/JPG + string Unicode | 300–1000 baris | 6 |

**A + B menghasilkan data training.** Citra training tidak pernah disimpan ke disk — dirender on-the-fly di DataLoader worker dari korpus teks (A) memakai font (B). Ini menghemat puluhan GB dan memungkinkan augmentasi tak terbatas.

**C tidak bisa disintesis.** Ini satu-satunya bagian yang butuh kerja manual, dan ini yang menentukan apakah model Anda benar-benar bekerja atau hanya bagus di angka sintetis.

---

#### A. Korpus teks

Tujuan: menghasilkan ≥200k baris teks **aksara Jawa Unicode**. Sumbernya teks Latin, dikonversi lewat transliterator.

| Sumber | Isi | Lisensi | Prioritas |
|---|---|---|---|
| **Wikipedia Jawa** (`jvwiki`) | Puluhan ribu artikel, domain umum | CC-BY-SA | **Utama** — volume terbesar |
| **sastra.org** | ~6.000 dokumen, >15 juta kata naskah abad XIX–XX, sudah dialihaksarakan ke Latin | Cek per-dokumen | **Utama** — kosakata sastra/klasik |
| **NusaTranslation** | Javanese: 9.449 kalimat + 10.188 paragraf, domain beragam | Cek repo IndoNLP | Sekunder |
| **NusaX** | ~1.000 kalimat/bahasa, domain review, kualitas tinggi (penutur asli) | CC-BY-SA | Sekunder — volume kecil |

```bash
# Wikipedia Jawa
wget https://dumps.wikimedia.org/jvwiki/latest/jvwiki-latest-pages-articles.xml.bz2 \
     -P data/raw/
```

**Kenapa perlu lebih dari satu sumber:** Wikipedia memberi volume tapi kosakatanya modern dan agak seragam. sastra.org memberi kosakata klasik, nama diri, dan konstruksi yang tidak muncul di Wikipedia. Model yang hanya dilatih Wikipedia akan lemah pada teks sastra — dan sebaliknya.

**Transliterator Latin→Aksara:**
- `@naandalist/honocoroko` (npm, TypeScript) — bidirectional
- Port Python dari `bennylin/transliterasijawa` → `jrenslin/transliteration-javanese`

**Spesifikasi output** (`data/corpus_jv.txt`):

```
Format   : satu baris teks per baris file, UTF-8, NFC
Panjang  : 20–80 karakter Unicode (pecah di batas kata)
Filter   : buang baris yang round-trip Latin→Aksara→Latin CER > 5%
Dedup    : ya, exact match
Coverage : semua codepoint charset harus muncul ≥100×
```

> **Cek coverage codepoint langka.** Aksara murda, aksara swara, aksara rekan, dan angka Jawa jarang muncul di teks alami. Kalau ada codepoint yang muncul < 100×, model tidak akan mempelajarinya. Solusi: inject baris sintetis yang sengaja memakai codepoint tersebut.

---

#### C. Citra nyata + anotasi

Ini bagian yang tidak bisa didelegasikan dan tidak bisa disintesis.

**Target volume:**

| Split | Jumlah baris | Aturan |
|---|---|---|
| Test | **≥200** | **Tidak pernah dipakai training. Sekali pun.** |
| Train (fine-tune) | 300–800 | |

**Matriks kondisi** — usahakan tersebar, jangan 500 baris dari satu papan nama yang sama:

| Dimensi | Variasi yang dikejar |
|---|---|
| Sumber | Papan nama jalan, papan nama instansi, plakat, buku cetak, kemasan produk |
| Pencahayaan | Siang terang, teduh, dalam ruangan, ada bayangan |
| Sudut | Frontal, miring ringan (≤15°) |
| Kondisi | Bersih, pudar, kotor, ada pantulan |
| Font/gaya | Sebanyak mungkin gaya berbeda |

> Yogyakarta dan Surakarta punya banyak signage aksara Jawa karena kebijakan daerah — ini sumber pengumpulan paling efisien.

**Spesifikasi teknis citra:**

```
Tinggi baris  : ≥40 px setelah dipotong (di bawah ini, sandhangan hilang)
Format        : PNG (lossless) atau JPG kualitas ≥90
Warna         : boleh RGB, dikonversi grayscale saat load
Isi           : SATU baris teks per file, sudah dipotong
```

**Format anotasi** (`data/real/labels.tsv`):

```tsv
image_path	text	source_id	condition
images/0001.png	ꦲꦤꦕꦫꦏ	sign_malioboro_01	outdoor_bright
images/0002.png	ꦢꦠꦱꦮꦭ	book_serat_p12	scan_clean
```

- Kolom `text`: Unicode aksara Jawa, **ternormalisasi NFC**, urutan logis
- Kolom `source_id`: **wajib** — dipakai untuk mencegah kebocoran split

**Aturan split yang kritis:** pisahkan train/test **berdasarkan `source_id`, bukan berdasarkan baris.**

> Kalau satu papan nama menyumbang 8 baris dan Anda split acak per-baris, 6 baris masuk train dan 2 masuk test — dari papan yang sama, font yang sama, pencahayaan yang sama. CER test Anda akan terlihat bagus secara palsu. **Split per sumber, bukan per baris.**

**Anotasi praktis:** pakai transliterator terbalik sebagai bantuan — ketik bacaan Latin-nya, konversi ke aksara, lalu verifikasi visual terhadap citra. Jauh lebih cepat daripada mengetik Unicode aksara langsung, dan lebih sedikit salah.

---

#### Dataset yang sudah ada — apa yang bisa dipakai

Penting diketahui supaya Anda tidak menghabiskan waktu mencari sesuatu yang tidak ada.

| Dataset | Isi | Bisa dipakai? |
|---|---|---|
| **Hanacaraka** | ~20 aksara dasar, ~75 citra/kelas, tulisan tangan | ❌ Level karakter tunggal, bukan baris |
| **Baksara** (Roboflow) | 164 citra, anotasi object detection | ❌ Terlalu kecil, format salah |
| `hermansh-id/Aksara-Jawa-OCR` (Kaggle) | Dataset OCR aksara Jawa | ⚠️ Cek isinya — mungkin berguna sebagai tambahan Fase 6 |
| **NusaAksara** | Benchmark multimodal aksara Nusantara, ada citra Jawa + transliterasi | ✅ **Berguna sebagai eval set eksternal** |
| **British Library** (Bollinger 120 mss, Lohia 76 mss) | ~30.000 citra naskah | ❌ Tidak ada transkripsi level baris; juga naskah = di luar scope |
| **AMADI_LontarSet** | Lontar Bali | ❌ Aksara Bali, bukan Jawa |

**Kesimpulan penting:** tidak ada satu pun dataset publik yang menyediakan pasangan **citra baris → Unicode aksara Jawa** dalam skala yang cukup. Itulah alasan fundamental kenapa proyek ini memakai synthetic data — bukan karena malas mencari, tapi karena datanya memang tidak ada.

NusaAksara layak dipakai sebagai **evaluasi eksternal** di akhir Fase 6: menunjukkan performa pada benchmark pihak ketiga jauh lebih meyakinkan daripada hanya pada test set buatan sendiri.

---

#### Ringkasan checklist dataset

- [ ] ≥2 font terverifikasi lolos Fase 0
- [ ] Dump Wikipedia Jawa terunduh & terekstrak
- [ ] Teks sastra.org terkumpul
- [ ] `corpus_jv.txt` ≥200k baris, round-trip ≥95%, coverage codepoint terverifikasi
- [ ] Split korpus train/val/test tanpa kebocoran teks
- [ ] ≥200 baris citra nyata untuk test, dari sumber yang **berbeda** dari train
- [ ] 300–800 baris citra nyata untuk fine-tune
- [ ] `labels.tsv` dengan kolom `source_id` terisi
- [ ] Semua string anotasi ternormalisasi NFC

### 4.7 Pengetahuan yang diasumsikan

- PyTorch: custom Dataset, DataLoader, training loop, AMP
- Konsep CTC (dijelaskan di §3.2, tidak perlu implementasi dari nol — `nn.CTCLoss` sudah ada)
- Unicode: normalisasi NFC, combining marks, kategori karakter
- **Tidak** perlu bisa membaca aksara Jawa — semua verifikasi bisa dilakukan lewat round-trip test dan perbandingan visual dengan gambar referensi

---

## 5. Struktur repo

```
aksara-ocr/
├── CLAUDE.md                 # instruksi untuk Claude Code
├── PLAN.md                   # dokumen ini
├── requirements.txt
├── fonts/
│   ├── TuladhaJejeg.ttf
│   └── NotoSansJavanese-Regular.ttf
├── data/
│   ├── raw/                  # dump wikipedia, sastra.org
│   ├── corpus_jv.txt         # output Fase 2
│   └── real/                 # citra nyata + anotasi (Fase 6)
│       ├── images/
│       └── labels.tsv
├── src/
│   ├── render.py             # Fase 0
│   ├── tokenizer.py          # Fase 1
│   ├── corpus.py             # Fase 2
│   ├── dataset.py            # Fase 3
│   ├── model.py              # Fase 4
│   ├── train.py              # Fase 4
│   ├── decode.py             # greedy + beam
│   └── infer.py              # inference standalone
├── tests/
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   └── test_shapes.py
├── scripts/
│   └── verify_shaping.py     # Fase 0
└── out/                      # PNG verifikasi, checkpoint, log
```

---

## 6. Invariant global — jangan diubah tanpa alasan kuat

Salin ini ke `CLAUDE.md`. Ini adalah keputusan desain yang sudah diargumentasikan di §3, bukan preferensi.

1. **`H = 96`.** Bukan 32, bukan 64. Tumpukan pasangan butuh ruang vertikal.
2. **Rendering wajib `ImageFont.Layout.RAQM`.** Assert, jangan asumsikan. Kegagalannya senyap.
3. **`blank = 0`.** Charset 1-indexed.
4. **`input_lengths` dari lebar citra ASLI**, bukan lebar setelah padding. Ini bug CTC paling umum.
5. **`DOWNSAMPLE` harus cocok dengan stride lebar CNN.** Verifikasi lewat assert, jangan asumsikan.
6. **Semua string ternormalisasi NFC** sebelum encode dan sebelum hitung CER.
7. **Training pada urutan visual**, decode mengembalikan ke urutan logis. Jangan campur.
8. **Aturan reorder diturunkan dari uharfbuzz**, jangan tulis manual dari ingatan.

---

## 7. Fase kerja

> **Aturan:** satu fase per sesi kerja. Jangan lanjut sebelum kriteria lulus terpenuhi. Ini bukan formalitas — ini yang mencegah Anda men-debug arsitektur padahal bug-nya di data.

---

### Fase 0 — Verifikasi rendering
**Estimasi: 2–3 hari** (mayoritas untuk beresin libraqm)

**Tujuan:** memastikan shaping OpenType benar sebelum menulis kode lain.

**Deliverable:**
- `src/render.py` — `render_line(text, font_path, size, pad) -> PIL.Image`, dengan assert RAQM
- `scripts/verify_shaping.py` — merender 11 kasus uji × 2 font ke `out/`

**Kasus uji wajib:**

| Nama | Codepoint | Yang harus terlihat |
|---|---|---|
| nglegena | `A98F A9A5 A9A9` | tiga aksara sejajar |
| taling_prebase | `A98F A9BA` | taling di **KIRI** ka |
| taling_tarung | `A98F A9BA A9B4` | mengapit ka kiri-kanan |
| pasangan | `A98F A9C0 A9A5` | ta **menumpuk di bawah** ka |
| pasangan_susun3 | `A98F A9C0 A9A5 A9C0 A9A9` | tiga tingkat, tidak tumpang tindih |
| sandhangan_atas | `A98F A9B6` | wulu di atas |
| sandhangan_bawah | `A98F A9B8` | suku di bawah |
| layar | `A98F A982` | di atas kanan |
| cakra | `A98F A9BF` | di bawah kanan |
| angka | `A9D1 A9D2 A9D3` | tiga digit |
| pada | `A9C8 A9C9` | pada lingsi & lungsi |

**Kriteria lulus:**
- [ ] `ImageFont.core.HAVE_RAQM == True`
- [ ] Ke-11 PNG dibuka dan **diperiksa manusia**
- [ ] Taling di kiri, pasangan menumpuk vertikal, tumpuk-tiga tidak tumpang tindih
- [ ] Pangkon terlihat di akhir kata, tidak terlihat di tengah kata

> **Checkpoint ini tidak bisa diotomatiskan.** Claude Code tidak bisa memverifikasi bentuk glyph. Anda harus membuka gambarnya sendiri. Kalau ada yang salah: ganti font, cek versi libraqm/HarfBuzz. **Jangan lanjut.**

---

### Fase 1 — Tokenizer
**Estimasi: 3–4 hari**

**Tujuan:** pemetaan bijektif antara string Unicode dan urutan indeks, plus reorder visual↔logis.

**Deliverable:**
- `src/tokenizer.py` — charset, `encode()`, `decode()`, `to_visual()`, `to_logical()`
- `tests/test_tokenizer.py`

**Pendekatan:**
1. Bangun charset dari korpus (bukan dari tabel Unicode manual) — codepoint yang benar-benar muncul
2. Reserve indeks 0 untuk blank
3. Turunkan urutan visual dari **uharfbuzz**: shape string, urutkan glyph berdasarkan posisi x, baca cluster-nya
4. Bangun tabel reorder empiris dari pasangan (cluster logis → cluster visual) di seluruh korpus
5. `to_logical()` = inversi tabel tersebut

**Kriteria lulus:**
- [ ] `decode(encode(s)) == NFC(s)` untuk **100%** dari ≥100k baris korpus
- [ ] `to_logical(to_visual(s)) == NFC(s)` untuk **100%** baris
- [ ] Charset terdokumentasi: berapa kelas, codepoint apa saja
- [ ] Test lolos di CI/pytest

> Kegagalan round-trip di sini akan muncul sebagai CER floor yang tidak bisa ditembus di Fase 4. Selesaikan sekarang.

---

### Fase 2 — Korpus
**Estimasi: 1 minggu**

**Tujuan:** ≥200k baris teks aksara Jawa Unicode yang valid.

**Deliverable:**
- `src/corpus.py` — pipeline ekstraksi + transliterasi + filter
- `data/corpus_jv.txt` — satu baris per baris teks

**Langkah:**
1. Unduh dump Wikipedia Jawa, ekstrak teks bersih
2. Transliterasi Latin → Aksara (honocoroko / port bennylin)
3. Potong jadi baris 20–80 karakter Unicode, pecah di batas kata
4. **Filter round-trip:** buang baris yang Latin→Aksara→Latin CER-nya > 5%
5. Deduplikasi
6. Split train / val / test (jaga agar tidak ada kebocoran teks antar split)

**Kriteria lulus:**
- [ ] ≥200k baris
- [ ] ≥95% baris lolos filter round-trip
- [ ] Distribusi panjang terdokumentasi (histogram)
- [ ] Distribusi karakter terdokumentasi — cek codepoint langka yang mungkin under-represented
- [ ] Split tidak bocor

---

### Fase 3 — Dataset & collate
**Estimasi: 2–3 hari**

**Tujuan:** pipeline data yang benar. **Di sinilah bug CTC paling sering bersembunyi.**

**Deliverable:**
- `src/dataset.py` — Dataset class (render on-the-fly), collate function, filter T≥L
- `tests/test_dataset.py`, `tests/test_shapes.py`

**Poin kritis:**

```python
DOWNSAMPLE = 4   # HARUS cocok dengan total stride lebar CNN

def collate(batch):
    xs, ys = zip(*batch)
    widths = [x.shape[-1] for x in xs]
    W = max(widths)
    X = torch.zeros(len(xs), 1, H, W)
    for i, x in enumerate(xs):
        X[i, :, :, :x.shape[-1]] = x
    # dari lebar ASLI, BUKAN W
    in_lens  = torch.tensor([max(1, w // DOWNSAMPLE) for w in widths])
    tgt_lens = torch.tensor([len(y) for y in ys])
    Y = torch.cat(ys)              # CTC menerima target 1D terkonkatenasi
    return X, Y, in_lens, tgt_lens
```

Filter di dataset, jangan tunggu meledak saat training:
```python
def is_trainable(text, img_width, tok, downsample=DOWNSAMPLE):
    return (img_width // downsample) >= len(tok.encode(text)) * 1.5
```

**Kriteria lulus:**
- [ ] Assert: `model(dummy).shape[1] == W // DOWNSAMPLE` — **verifikasi, jangan asumsikan**
- [ ] Satu batch keluar dengan shape benar
- [ ] `in_lens` terbukti berasal dari lebar asli (test dengan batch lebar campur)
- [ ] Nol sampel melanggar T ≥ 1.5L setelah filter
- [ ] Rendering on-the-fly, citra tidak disimpan ke disk

---

### Fase 4 — Model & training
**Estimasi: 3–5 hari**

**Tujuan:** model yang terbukti bisa belajar.

**Deliverable:**
- `src/model.py` — CRNN
- `src/train.py` — training loop dengan AMP, OneCycleLR, grad clipping
- `src/decode.py` — greedy decode + collapse + reorder ke logis
- Evaluasi CER

**Urutan kerja — jangan dibalik:**

**4a. Overfit sanity check.** Latih di **32 sampel** sampai CER mendekati 0.

> Model yang benar akan menghafal 32 sampel dalam beberapa menit. Yang tidak bisa punya bug struktural. **Kalau gagal di sini, jangan lanjut ke dataset penuh.** Langkah ini menghemat berhari-hari.

**4b. Training kecil.** 50k sampel, **tanpa augmentasi**, ~20 epoch.

**Kriteria lulus:**
- [ ] 4a: CER < 1% pada 32 sampel
- [ ] 4b: **CER < 2% pada synthetic bersih held-out** ← gerbang terpenting
- [ ] Checkpoint tersimpan & bisa dimuat ulang
- [ ] `infer.py` jalan standalone pada satu PNG

> **Kalau 4b tidak tercapai, JANGAN tambah data dan JANGAN ganti arsitektur.** Bug-nya ada di salah satu dari empat tempat:
>
> 1. `DOWNSAMPLE` tidak cocok dengan stride CNN sebenarnya
> 2. `in_lens` dihitung dari lebar padded
> 3. Indeks blank bukan 0, atau charset tidak 1-indexed
> 4. Tokenizer tidak round-trip
>
> **Gejala diagnostik:** loss turun tapi CER mentok ~90% → model mengeluarkan blank terus → hampir selalu nomor 1 atau 2.

---

### Fase 5 — Skala & augmentasi
**Estimasi: 1–2 minggu** (mayoritas waktu GPU)

**Tujuan:** ketahanan terhadap degradasi dunia nyata.

**Deliverable:** model terlatih di 300k–1jt baris dengan augmentasi, plus tabel ablation.

**Metodenya — penting:** tambahkan **satu jenis augmentasi per iterasi**, ukur dampaknya terhadap CER. Kalau semua sekaligus, Anda tidak akan tahu mana yang membantu dan mana yang merusak.

Urutan yang disarankan:
1. Variasi ukuran font & font berbeda (baseline)
2. Rotasi ringan (±1.5°) — kemiringan baseline
3. Noise Gaussian
4. Blur (motion + gaussian)
5. Elastic transform
6. Brightness / contrast
7. Kompresi JPEG
8. Tekstur kertas / background

**Kriteria lulus:**
- [ ] CER < 5% pada synthetic ter-augmentasi berat (G2)
- [ ] CER pada synthetic bersih tidak memburuk > 1 poin dari Fase 4
- [ ] Tabel ablation: kontribusi tiap augmentasi terdokumentasi

---

### Fase 6 — Fine-tune data nyata
**Estimasi: 2–3 minggu** (mayoritas untuk anotasi)

**Tujuan:** memenuhi G3 pada citra nyata.

**Deliverable:** 300–1000 baris nyata teranotasi + model ter-fine-tune.

**Langkah:**
1. Kumpulkan citra: foto papan nama (Yogyakarta/Solo punya banyak signage aksara Jawa berdasarkan perda), scan buku cetak
2. Potong jadi baris tunggal
3. Anotasi manual → Unicode aksara Jawa, ternormalisasi NFC
4. Split: ≥200 untuk test (jangan pernah dipakai training), sisanya train
5. Fine-tune: **learning rate 10× lebih kecil**, jangan bekukan layer apa pun
6. Evaluasi pada test set nyata

**Kriteria lulus:**
- [ ] ≥200 baris test nyata teranotasi, tidak pernah tersentuh training
- [ ] CER < 8% pada test nyata (G3)
- [ ] Analisis error: kategorikan 50 kesalahan teratas

> Fine-tune pada beberapa ratus baris nyata biasanya menurunkan CER **3–5×** dibanding synthetic-only. ROI-nya jauh lebih tinggi daripada menambah 500k synthetic lagi.

---

## 8. Ringkasan timeline

| Fase | Isi | Estimasi | Kumulatif |
|---|---|---|---|
| 0 | Verifikasi rendering | 2–3 hari | 3 hari |
| 1 | Tokenizer | 3–4 hari | 1 minggu |
| 2 | Korpus | 1 minggu | 2 minggu |
| 3 | Dataset & collate | 2–3 hari | 2.5 minggu |
| 4 | Model & training | 3–5 hari | 3.5 minggu |
| 5 | Skala & augmentasi | 1–2 minggu | 5 minggu |
| 6 | Fine-tune data nyata | 2–3 minggu | **7 minggu** |

**Perhatikan:** Fase 0–2 memakan ~2 minggu dan **tidak ada training sama sekali** di situ. Itu wajar dan itu memang di mana proyek ini menang atau kalah.

---

## 9. Failure mode yang sudah diketahui

| Gejala | Penyebab paling mungkin | Cara cek |
|---|---|---|
| Loss turun, CER mentok ~90% | Model mengeluarkan blank terus → `in_lens` salah atau `DOWNSAMPLE` tidak cocok | Print `in_lens` vs lebar citra asli; assert shape output model |
| CER punya lantai yang tidak bisa ditembus (~5–10%) | Tokenizer tidak round-trip, atau normalisasi NFC tidak konsisten | Jalankan test round-trip Fase 1 |
| Training bagus di synthetic, hancur di nyata | Augmentasi tidak menutup gap domain | Bandingkan histogram: kontras, ketebalan stroke, noise |
| Model tidak bisa overfit 32 sampel | Bug struktural (LR, gradien, shape) | Cek grad norm; turunkan LR; cek `zero_infinity=True` |
| CER tinggi hanya pada teks padat | Pelanggaran T ≥ L | Cek filter dataset; naikkan ukuran render atau kurangi stride lebar |
| Semua taling salah posisi | RAQM tidak aktif | `ImageFont.core.HAVE_RAQM` |
| Loss = inf / NaN | Target lebih panjang dari input | `zero_infinity=True` + filter T≥1.5L |

---

## 10. Bekerja dengan Claude Code

### 10.1 `CLAUDE.md` — buat sebelum apa pun

```markdown
# Aksara Jawa OCR — CRNN + CTC

## Baca dulu
Baca PLAN.md sebelum mengerjakan apa pun. Kerjakan SATU fase per sesi.

## Invariant yang TIDAK BOLEH diubah tanpa diskusi
- H = 96. BUKAN 32. Tumpukan pasangan butuh ruang vertikal.
- Rendering WAJIB ImageFont.Layout.RAQM. Assert-nya. Tanpa RAQM, shaping
  OpenType gagal SECARA SENYAP — gambar tetap keluar tapi pasangan tidak
  menumpuk dan taling di posisi salah.
- blank = indeks 0. Charset 1-indexed.
- input_lengths dari lebar citra ASLI, bukan lebar setelah padding.
- DOWNSAMPLE harus konsisten dengan stride lebar CNN. Verifikasi dengan
  assert, jangan asumsikan.
- Semua string NFC sebelum encode dan sebelum hitung CER.
- Training pada urutan visual, decode ke urutan logis. Jangan campur.
- Aturan reorder diturunkan dari uharfbuzz, jangan tulis manual.

## Scope
Repo ini HANYA mengerjakan: citra baris → Unicode aksara Jawa.
JANGAN bangun transliterasi ke Latin, machine translation, atau deteksi
tingkat tutur. Itu proyek terpisah dengan repo sendiri.

## Aturan kerja
- Tulis test dulu untuk tokenizer dan collate sebelum menulis model.
- Setiap fase harus lulus kriteria di PLAN.md sebelum lanjut.
- Sebelum training penuh, overfit 32 sampel dulu. Kalau tidak bisa
  overfit 32 sampel, ada bug — laporkan, jangan lanjut.

## Konteks aksara
Aksara Jawa = abugida, blok Unicode U+A980–U+A9DF.
- Pasangan: C1 + U+A9C0 (pangkon/virama) + C2 → C2 menumpuk di bawah C1
- Taling U+A9BA: pre-base — digambar di KIRI aksara dasar tapi disimpan
  SESUDAHNYA di Unicode. Ini melanggar monotonisitas CTC.
- Taling-tarung (U+A9BA + U+A9B4): circumfix, mengapit aksara dasar
- Sandhangan mengelilingi aksara dasar dari 4 sisi
```

### 10.2 Yang akan salah kalau tidak dicegah

| Kesalahan | Kenapa terjadi | Pencegahan |
|---|---|---|
| Pakai H=32 | Semua tutorial CRNN memakai 32 | Sudah di CLAUDE.md |
| `ImageDraw.text` tanpa RAQM | Default Pillow, **gagal senyap** | Assert eksplisit |
| `in_lens` dari lebar padded | Bug CTC paling umum di internet | Test khusus |
| Bangun 4 tahap sekaligus | Konteks menggoda | Aturan scope |
| Skip verifikasi shaping | Claude Code tidak bisa melihat gambar | Anda yang periksa |
| Lupa cek T ≥ L | Tidak intuitif | Filter di dataset |

### 10.3 Urutan prompt

Jangan minta "buatkan OCR aksara Jawa". Satu sesi per fase:

**Sesi 1 (Fase 0)**
> Baca PLAN.md dan CLAUDE.md. Kerjakan Fase 0 saja. Buat `src/render.py` dengan `render_line` yang memakai RAQM (assert kalau tidak tersedia), plus `scripts/verify_shaping.py` yang merender 11 kasus uji dari tabel Fase 0 ke `out/`. Jangan buat file lain.

→ **Buka PNG-nya, periksa sendiri.** Jangan lanjut sebelum yakin.

**Sesi 2 (Fase 1)**
> Fase 1: tokenizer. Turunkan urutan visual dari uharfbuzz, jangan tulis aturan reorder manual dari ingatan. Tulis test dulu: `decode(encode(s)) == NFC(s)` dan round-trip reorder, jalankan pada 10.000 baris. Laporkan tingkat kelulusan sebelum lanjut.

**Sesi 3 (Fase 2)** — korpus + filter round-trip
**Sesi 4 (Fase 3)** — dataset + collate + test shape
**Sesi 5 (Fase 4)** — model + training, dengan tambahan:
> Sebelum training penuh, overfit model pada 32 sampel sampai CER mendekati 0. Kalau tidak bisa, ada bug — laporkan, jangan lanjut ke dataset penuh.

### 10.4 Yang tetap harus Anda kerjakan sendiri

- **Verifikasi visual shaping** (Fase 0) — mutlak, tidak bisa didelegasikan
- **Keputusan skema tokenisasi** — visual vs logis vs cluster HarfBuzz
- **Anotasi data nyata** (Fase 6)
- **Menolak tawaran "sekalian saya buatkan tahap MT-nya"** — ini akan terjadi

---

## 11. Setelah selesai

Setelah G1–G3 tercapai, tiga tahap berikutnya masing-masing repo terpisah:

| Tahap | Metode | Estimasi |
|---|---|---|
| Transliterasi → Latin | Rule-based (honocoroko / port bennylin). **Jangan pakai ML** — deterministik | ~2 hari |
| Terjemahan → Indonesia | NLLB-200-distilled-600M (int8/CTranslate2) atau Sahabat-AI | ~3 hari |
| Deteksi tingkat tutur | Leksikon + fitur afiks; upgrade ke encoder fine-tuned di Unggah-Ungguh | ~1 minggu |

Catatan lisensi: NLLB-200 CC-BY-NC, dataset Unggah-Ungguh CC-BY-NC 4.0. Kalau target Anda komersial, dua komponen ini terkunci non-komersial — baseline berbasis leksikon bebas dari masalah itu.

Arah pengembangan lanjutan setelah G3: perluas ke tulisan tangan (butuh anotasi manual besar), naskah lontar (Kraken/eScriptorium + layout analysis), atau ganti CTC dengan attention decoder (TrOCR) setelah punya cukup data nyata.
