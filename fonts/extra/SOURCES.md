# Font tambahan (kandidat Fase 5)

Diunduh 2026-09-13 dari aksaradinusantara.com. Dipakai HANYA untuk merender citra training
riset non-komersial; file font tidak disebarkan. Semua lolos cek otomatis awal Fase 0.

- `fonts/extra/`: tampak benar pada pemeriksaan visual Claude (`out/shaping_extra/contact_sheet_*.png`).
  **2026-09-13: user mendelegasikan keputusan ("rekomendasikan"); dipakai di run `fase5_fonts`.**
  Pemeriksaan mata oleh manusia BELUM dilakukan — sebutkan ini di laporan.
- `fonts/extra_review/`: meragukan, butuh keputusan manusia; tidak dipakai training.
- `fonts/extra_rejected/`: ditolak; tidak dipakai.
- Budyamarsudi tidak diunduh ke sini: render penuh artefak kotak.
- **Lebar spasi (2026-09-14, spasi / lebar ka @64px; Noto 0,21, javatext 0,21):** abmAksaJawa
  Regular & Bold 0,10, ARDemak 0,14, BasaJan 0,35, CarakanJawa 0,21, **GumregahNew 0,00** (celah kata
  hasil shaping 0,09), NewKramawirya 0,17, nykNgayogyanJejeg 0,21. Spasi yang nyaris tak tampak membuat
  label berisi spasi tanpa celah di citra (derau label). Tidak ditangani di run `fase5_fonts`/`fase5_core`;
  sejak 2026-09-14 `src/dataset.py` selalu membuang spasi (teks + label) untuk font dengan celah kata
  < 0,12 ka (GumregahNew, abmAksaJawa Regular & Bold).
- **Cakupan glyph karakter langka (cmap, 2026-09-14):** dari 44 codepoint langka (murda, mahaprana,
  swara, pada, dll.) semua font di sini lengkap, kecuali **CarakanJawa tidak punya aksara O
  (U+A98E)**. `src/text_augment.py` memeriksa cmap sebelum menyisipkan karakter.

| File | Folder | Sumber | Lisensi | Status visual |
|---|---|---|---|---|
| `abmAksaJawa-Regular.ttf` | `extra` | https://aksaradinusantara.com/fonta/abm-aksajawa.font | tidak tercantum | tampak benar (pra-periksa Claude) |
| `abmAksaJawa-Bold.ttf` | `extra` | https://aksaradinusantara.com/fonta/abm-aksajawa.font | tidak tercantum | tampak benar (pra-periksa Claude) |
| `ARDemak-Regular.ttf` | `extra` | https://aksaradinusantara.com/fonta/ar-demak.font | tidak tercantum | tampak benar, gaya kaligrafis |
| `BasaJan.ttf` | `extra` | https://aksaradinusantara.com/fonta/basa-jan.font | tidak tercantum | tampak benar |
| `CarakanJawa.otf` | `extra` | https://aksaradinusantara.com/fonta/carakan-jawa.font | tidak tercantum | tampak benar |
| `GumregahNew.ttf` | `extra` | https://aksaradinusantara.com/fonta/gumregah.font | tidak tercantum | tampak benar |
| `NewKramawirya.ttf` | `extra` | https://aksaradinusantara.com/fonta/new-kramawirya.font | tidak tercantum | tampak benar |
| `nykNgayogyanJejeg.ttf` | `extra` | https://aksaradinusantara.com/fonta/nyk-ngayogyan-jejeg.font | tidak tercantum | tampak benar |
| `AB-Bakul.ttf` | `extra_review` | https://aksaradinusantara.com/fonta/ab-bakul.font | tidak tercantum | MERAGUKAN: susun-3 tidak jelas menumpuk |
| `AB-Wulang.otf` | `extra_review` | https://aksaradinusantara.com/fonta/ab-wulang.font | CC BY-NC-ND | MERAGUKAN: susun-3 berjajar, tanda pada janggal |
| `Damarwulan.ttf` | `extra_review` | https://aksaradinusantara.com/fonta/damarwulan.font | tidak tercantum | MERAGUKAN: susun-3 bertabrakan |
| `Sehulbari.ttf` | `extra_review` | https://aksaradinusantara.com/fonta/sehulbari.font | tidak tercantum | MERAGUKAN: sapuan cakra/susun-3 sangat lebar |
| `Istaka.ttf` | `extra_rejected` | https://aksaradinusantara.com/fonta/istaka.font | tidak tercantum | DITOLAK: wulu & layar tergeser ke kiri ka |
| `Nawatura.ttf` | `extra_rejected` | https://aksaradinusantara.com/fonta/nawatura.font | tidak tercantum | DITOLAK: wulu & layar tergeser ke kiri ka |
