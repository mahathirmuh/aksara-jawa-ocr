# Sumber kamus kata

Berkas di folder ini adalah data pihak ketiga yang diolah `web/tools/dictionaries.py` untuk halaman Kamus.
Bukan hasil hitung proyek ini, dan lisensinya mengikuti sumbernya, bukan lisensi kode repo.

| Berkas | Isi | Sumber | Lisensi |
|---|---|---|---|
| `jv.jsonl.gz`, entri `source: enwiktionary` | kamus bahasa Jawa: kata, ejaan aksara Jawa, kelas kata, ragam (ngoko, krama, krama inggil), padanan antar-ragam, arti berbahasa Inggris | entri bahasa Jawa [Wiktionary bahasa Inggris](https://en.wiktionary.org/wiki/Category:Javanese_language), lewat ekstraksi [kaikki.org](https://kaikki.org/dictionary/Javanese/) (wiktextract) | CC BY-SA 4.0 (juga GFDL) |
| `jv.jsonl.gz`, entri `source: idwiktionary` | kamus bahasa Jawa: kata, kelas kata, arti berbahasa Indonesia (dengan label ragam dan dialek dari sumbernya), contoh kalimat berterjemahan, sinonim | bagian bahasa Jawa [Wikikamus](https://id.wiktionary.org/) (Wiktionary bahasa Indonesia), dump [idwiktionary 2026-10-01](https://dumps.wikimedia.org/idwiktionary/), diurai `web/tools/dictionaries.py` | CC BY-SA 4.0 (juga GFDL) |
| `id.jsonl.gz` | kamus bahasa Indonesia: kata, kelas kata, arti berbahasa Inggris | entri bahasa Indonesia [Wiktionary bahasa Inggris](https://en.wiktionary.org/wiki/Category:Indonesian_language), lewat ekstraksi [kaikki.org](https://kaikki.org/dictionary/Indonesian/) | CC BY-SA 4.0 (juga GFDL) |
| `jv-taling.json` | leksikon pemulih tanda é/è halaman Terjemahan: kata tanpa tanda → ejaan bertandanya, plus hasil uji tahan di kunci `meta` | ejaan kata di artikel [Wikipedia bahasa Jawa](https://jv.wikipedia.org/) snapshot 20231101 yang menandai é/è, ditambah lema kamus di atas; dibangun `web/tools/taling_lexicon.py` | CC BY-SA 4.0 |
| `sources.json` | nama, lisensi, tautan, dan tanggal pengambilan tiap sumber; ditampilkan di kaki tiap kamus | – | – |

Atribusi: teks entri ditulis para penyunting Wiktionary bahasa Inggris dan Wikikamus; tiap entri di halaman Kamus
bertaut ke halaman asalnya. Ekstraksi terstruktur Wiktionary bahasa Inggris oleh Tatu Ylonen, *Wiktextract:
Wiktionary as Machine-Readable Structured Data*, LREC 2022.

Perubahan terhadap Wiktionary bahasa Inggris: hanya kata, kelas kata, arti, dan bentuk di kepala entri yang diambil;
kelas kata diterjemahkan ke istilah Indonesia; halaman berjudul aksara Jawa digabung dengan ejaan Latinnya; halaman
penunjuk ("Carakan spelling of", romanisasi) dilebur ke entri utamanya, kecuali judul penunjuk yang tidak tercapai
lewat kata kepala (mis. "candi" untuk candhi), yang menjadi entri kecil; romanisasi otomatis "nyc/nyj" ditulis
"nc/nj"; ragam dan padanan antar-ragam diturunkan dari argumen kepala entri, tabel ragam, dan label di sumber; arti
bersarang dirangkai dengan arti induknya ("house: abode"; induk yang panjang ditulis sekali); tanda baca penutup arti
dibuang; tanda aksara yang terlepas dari aksaranya di ekstraksi dirapatkan lagi; varian ejaan yang hanya beda diakritik
dari kata yang ditunjuknya ("dheweke" = "dhèwèké") diberi ejaan aksara kata itu; ejaan aksara yang oleh pembanding
kasar tidak terbaca sebagai katanya tidak diambil (mis. `ngétan` yang di sumbernya berejaan ꦮꦺꦠꦤ꧀ "wétan").

Perubahan terhadap Wikikamus: hanya bagian bahasa Jawa (judul `{{bahasa|jv}}` dan bentuk lamanya) dari halaman
berjudul huruf Latin; markup wiki dibuang, templat label (ragam, dialek, bidang) ditulis dalam kurung di depan arti;
contoh kalimat dan terjemahannya digabung ("kalimat — terjemahan"); arti yang hanya mengulang lemanya dan "arti" yang
berupa ejaan Jawa bertanda ê/é/è dibuang; **bagian bertanda impor KBBI tidak diambil**; ragam dan ejaan aksara tidak
diambil (di sumber ini tidak andal).

Yang diketahui kurang: sekitar 28 kata tidak membawa ejaan aksara walau halaman beraksaranya ada di sumber; kesalahan
di sumber ikut terbawa (di Wikikamus mis. `dalan` berlabel "krama inggil" padahal ngoko, dan kelas kata sebagian
entri); mutu entri Wikikamus tidak seragam (banyak berasal dari acara komunitas).
Karena lisensinya berbagi-serupa, berkas turunan di folder ini juga berlisensi CC BY-SA 4.0.

Yang sengaja TIDAK dipakai: isi KBBI (hak cipta Badan Bahasa; halaman Kamus hanya menautkan ke KBBI Daring) dan
kamus yang isinya salinan KBBI.

Membangun ulang (dari akar repo): `.venv/Scripts/python web/tools/dictionaries.py`, lalu
`cd web && php artisan aksara:dictionary`. Unduhan mentah disimpan di `web/storage/app/dictionaries/` (tidak ikut git).
Wikikamus dibaca dari dump bertanggal (`IDWIKT_DUMP` di skrip itu, sekitar 43 MB) supaya hasilnya bisa diulang; Wikimedia
menghapus dump lama sesudah beberapa bulan, jadi `--refresh` di kemudian hari mungkin perlu tanggal dump yang baru, dan
keluarannya perlu diperiksa lagi (format halaman Wikikamus sedang dirombak).
