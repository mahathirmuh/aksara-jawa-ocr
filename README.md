# Aksara Jawa OCR (CRNN + CTC)

OCR satu baris aksara Jawa cetak: citra potongan baris → teks Unicode (blok U+A980–U+A9DF, urutan logis, NFC).
Model CRNN (CNN → BiLSTM → CTC) dilatih sepenuhnya dari baris sintetis yang dirender dengan HarfBuzz/RAQM.
Folder `web/` berisi aplikasi Laravel lokal untuk melihat hasil dan alur lanjutannya: transliterasi Latin, arti
(Indonesia), dan tingkat tutur (ngoko/madya/krama).

Proyek pribadi dan tugas kuliah; masih berjalan.

## Status (2026-10-04)

Angka resmi = checkpoint `fase7_track` (greedy).

| Gerbang | Data | CER | Target |
|---|---|---:|---:|
| G1 | render bersih, font Javanese Text (10.000 baris) | 0,27% | < 2% |
| G2 | render + augmentasi berat, font yang sama (10.000 baris) | 1,20% | < 5% |
| G3 | 745 baris cetak nyata (NusaAksara) | 21,46% | < 8% |
| G4 | round-trip tokenizer (1.034.357 baris) | 100% lolos | 100% |

Catatan G1/G2: Javanese Text tidak dipakai merender data training, tetapi salah satu font training tambahan
(CarakanJawa) ternyata satu keluarga huruf dengannya, jadi angka ini bukan lagi uji font yang belum pernah dilihat.

G3 belum tercapai. `fase7_track` melatih model pada baris sintetis yang dirender dengan jarak antar suku kata
acak, sehingga model berhenti membaca renggang antar-aksara di cetakan sebagai spasi kata (spasi keluaran
2.201 → 145; referensi 110). Terhadap run kontrol dengan jumlah langkah yang sama tanpa jarak (`fase7_ctrl`,
30,75%), selisihnya −9,3 poin (selang kepercayaan 95% per halaman −11,9 sampai −6,8). Sisipan aksara langka
membuat murda dan adeg-adeg mulai terbaca (recall 0% → 50% pada baris nyata) tetapi juga menambah keluaran palsu,
sehingga tidak mengubah G3 (21,18%). Sisa galat sekarang hampir seluruhnya salah baca bentuk aksara cetak. Rincian,
keputusan desain, dan riwayat eksperimen ada di [`CLAUDE.md`](CLAUDE.md); rencana fase di [`PLAN.md`](PLAN.md).

## Isi

| Folder | Isi |
|---|---|
| `src/` | render (RAQM wajib), tokenizer urutan visual, dataset sintetis, augmentasi, model, training, evaluasi, beam + LM, layanan FastAPI |
| `scripts/` | verifikasi shaping font, ablasi, ekspor hasil untuk web, pembanding run, evaluasi aksara langka |
| `tests/` | pytest (termasuk round-trip tokenizer 1 juta baris, ~6 menit) |
| `web/` | Laravel 12 + Livewire; lihat [`web/README.md`](web/README.md) |

Cara menjalankan: bagian "Cara menjalankan" di `CLAUDE.md` dan `web/README.md`. Dependensi Python di
`requirements.txt`; render wajib memakai Pillow dengan RAQM aktif.

## Yang tidak ikut di repo

- `data/`: korpus Wikipedia Jawa dan split (besar; dibangun ulang dengan `python -m src.corpus`), kecuali
  `data/tokenizer.json`.
- Data uji NusaAksara (citra, label, anotasi): berlisensi non-komersial dan hanya dipakai lokal sebagai test.
  Karena itu fixture test web (`web/tests/Fixtures/`) dan notebook eksplorasi juga tidak disertakan.
- `out/`: checkpoint dan hasil evaluasi.
- Font tambahan `fonts/extra*`: hanya untuk training riset lokal; sumbernya dicatat di `fonts/extra/SOURCES.md`.

## Pihak ketiga

- Noto Sans Javanese (The Noto Project Authors) dan Tuladha Jejeg OT (R.S. Wihananto; konversi OpenType Fadhl
  Haqq): SIL Open Font License 1.1, keterangan lisensi ada di metadata font.
- Korpus: Wikipedia bahasa Jawa (CC BY-SA), tidak disertakan.
- Terjemahan di `web/tools/`: NLLB-200-distilled-600M (CC-BY-NC), diunduh saat dipakai, tidak disertakan.
- Leksikon pemulih tanda é (`web/database/dictionaries/jv-taling.json`): daftar ejaan kata dari Wikipedia bahasa Jawa
  snapshot 20231101 (CC BY-SA 4.0).
- Kamus kata halaman Kamus (`web/database/dictionaries/`): entri bahasa Jawa dan bahasa Indonesia dari Wiktionary
  bahasa Inggris, lewat ekstraksi kaikki.org (wiktextract), dan bagian bahasa Jawa Wikikamus (Wiktionary bahasa
  Indonesia) untuk arti berbahasa Indonesia; CC BY-SA 4.0. Rincian di
  [`web/database/dictionaries/SUMBER.md`](web/database/dictionaries/SUMBER.md).
- Antarmuka web: huruf Inter (SIL OFL 1.1, paket npm `inter-ui`) dan Tabler Icons (MIT). Lambang aplikasi
  (`web/resources/brand/logo.png`) disediakan pemilik repo.
- Foto halaman masuk (`web/public/img/login/`), dari Wikimedia Commons, dipotong dan diperkecil: naskah beraksara
  Jawa koleksi Museum Sonobudoyo (Candramawa99, CC0), Serat Damar Wulan (British Library MSS Jav 89, domain publik),
  gerbang Donopratono Kraton Yogyakarta (Chainwit., CC BY 4.0). Tautan sumber di
  [`web/public/img/login/SUMBER.md`](web/public/img/login/SUMBER.md).
