# Sumber kamus kata

Berkas di folder ini adalah data pihak ketiga yang diolah `web/tools/dictionaries.py` untuk halaman Kamus.
Bukan hasil hitung proyek ini, dan lisensinya mengikuti sumbernya, bukan lisensi kode repo.

| Berkas | Isi | Sumber | Lisensi |
|---|---|---|---|
| `jv.jsonl.gz` | kamus bahasa Jawa: kata, ejaan aksara Jawa, kelas kata, ragam (ngoko, krama, krama inggil), padanan antar-ragam, arti berbahasa Inggris | entri bahasa Jawa [Wiktionary bahasa Inggris](https://en.wiktionary.org/wiki/Category:Javanese_language), lewat ekstraksi [kaikki.org](https://kaikki.org/dictionary/Javanese/) (wiktextract) | CC BY-SA 4.0 (juga GFDL) |
| `id.jsonl.gz` | kamus bahasa Indonesia: kata, kelas kata, arti berbahasa Inggris | entri bahasa Indonesia [Wiktionary bahasa Inggris](https://en.wiktionary.org/wiki/Category:Indonesian_language), lewat ekstraksi [kaikki.org](https://kaikki.org/dictionary/Indonesian/) | CC BY-SA 4.0 (juga GFDL) |
| `sources.json` | nama, lisensi, tautan, dan tanggal pengambilan tiap sumber; ditampilkan di kaki tiap kamus | – | – |

Atribusi: teks entri ditulis para penyunting Wiktionary; tiap entri di halaman Kamus bertaut ke halaman asalnya.
Ekstraksi terstruktur oleh Tatu Ylonen, *Wiktextract: Wiktionary as Machine-Readable Structured Data*, LREC 2022.

Perubahan terhadap sumber: hanya kata, kelas kata, arti, dan bentuk di kepala entri yang diambil; kelas kata
diterjemahkan ke istilah Indonesia; halaman berjudul aksara Jawa digabung dengan ejaan Latinnya; halaman penunjuk
(romanisasi, "Carakan spelling of") dilebur ke entri utamanya; ragam diturunkan dari tabel ragam dan label di sumber.
Karena lisensinya berbagi-serupa, berkas turunan di folder ini juga berlisensi CC BY-SA 4.0.

Yang sengaja TIDAK dipakai: isi KBBI (hak cipta Badan Bahasa; halaman Kamus hanya menautkan ke KBBI Daring) dan
kamus yang isinya salinan KBBI.

Membangun ulang (dari akar repo): `.venv/Scripts/python web/tools/dictionaries.py`, lalu
`cd web && php artisan aksara:dictionary`. Unduhan mentah disimpan di `web/storage/app/dictionaries/` (tidak ikut git).
