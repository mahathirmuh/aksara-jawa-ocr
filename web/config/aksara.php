<?php

return [

    /*
    | Repo OCR (Python) tempat out/results/ dan citra NusaAksara berada. Default: folder induk web/.
    | Semua angka OCR dihitung di sana (scripts/export_results.py); web hanya menampilkan.
    */
    'ocr_repo' => env('OCR_REPO_PATH') ?: dirname(base_path()),

    'results_dir' => 'out/results',

    'schema' => 1,

    /* Layanan model untuk halaman Demo (src/serve.py, FastAPI). */
    'service_url' => env('OCR_SERVICE_URL', 'http://127.0.0.1:8011'),
    'service_timeout' => 90,

    /* Tahap 3 (arti): Python untuk tools/translate_batch.py (NLLB lokal) dan layanan terjemahan Demo. */
    'python' => env('OCR_PYTHON') ?: (env('OCR_REPO_PATH') ?: dirname(base_path())).DIRECTORY_SEPARATOR
        .(PHP_OS_FAMILY === 'Windows' ? '.venv\\Scripts\\python.exe' : '.venv/bin/python'),
    'translate_service_url' => env('TRANSLATE_SERVICE_URL', 'http://127.0.0.1:8012'),

    /*
    | Berkas batch terjemahan (input.jsonl, output.jsonl, summary.json). Selain database, hanya di sini hasil
    | NLLB (~1 jam CPU) tersimpan: `aksara:translate --import-only` membacanya, `--dump` menulisnya ulang dari
    | database. Test mengalihkannya ke folder sementara (tests/TestCase.php).
    */
    'mt_dir' => storage_path('app/mt'),

    /* Transliterasi & terjemahan manusia NusaAksara (config Image Transliteration / Image Translation). */
    'annotations_path' => storage_path('app/nusaaksara/annotations.json'),

    /* Status fase pengerjaan repo OCR (dari CLAUDE.md; bukan hasil hitung). */
    'phases' => [
        ['0', 'Render & verifikasi shaping', 'wait', 'Mata manusia belum', 'cek otomatis 4/4 lolos · out/shaping/contact_sheet.png'],
        ['1', 'Tokenizer urutan visual', 'ok', 'Lulus', 'round-trip 1.034.357/1.034.357'],
        ['2', 'Korpus Wikipedia Jawa', 'ok', 'Lulus', '1.034.357 baris · 95,82% lolos round-trip'],
        ['3', 'Dataset & collate', 'ok', 'Lulus', '0/300 melanggar T ≥ 1,5L'],
        ['4', 'Model, training, decode', 'ok', 'Lulus', '4a 0,07% @32 · 4b val 0,38%'],
        ['5', 'Augmentasi & ablasi', 'ok', 'Lulus', 'G2 @10.000 baris · ablasi 12 run'],
        ['6', 'Data nyata & fine-tune', 'no', 'Belum', 'menunggu verifikasi 43 baris Commons'],
    ],

];
