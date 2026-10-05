<?php

namespace App\Console\Commands;

use App\Services\AnnotationImporter;
use App\Services\DatasetImporter;
use App\Services\MethodImporter;
use App\Services\ResultImporter;
use Illuminate\Console\Command;
use RuntimeException;

class ImportResults extends Command
{
    protected $signature = 'aksara:import {--path= : folder hasil (default: <repo OCR>/out/results)}';

    protected $description = 'Impor hasil eksperimen OCR (manifest.json, lines.jsonl, predictions.jsonl, dan kartu datasets.json / methods.json bila ada) ke database';

    public function handle(ResultImporter $results, AnnotationImporter $annotations, DatasetImporter $datasets, MethodImporter $methods): int
    {
        $path = $this->option('path') ?: config('aksara.ocr_repo').DIRECTORY_SEPARATOR.config('aksara.results_dir');
        try {
            $counts = $results->import($path);
        } catch (RuntimeException $e) {
            $this->error($e->getMessage());

            return self::FAILURE;
        }
        $this->info("Diimpor dari {$path}: {$counts['lines']} baris, {$counts['predictions']} prediksi, "
            ."{$counts['metrics']} metrik, {$counts['pipelines']} pipeline.");
        if ($draft = $annotations->scoreDraftTransliteration()) {
            $this->line('Transliterasi draf vs manusia: CER '.pct($draft['cer']).' pada '.$draft['lines'].' baris.');
        }

        // Kartu halaman Dataset dan Metode ikut diimpor bila ada di folder yang sama; ekspornya skrip tersendiri.
        // Kartu yang ditolak tidak menggagalkan perintah ini: hasil OCR di atas sudah tersimpan, dan tiap kartu punya
        // perintah sendiri (aksara:datasets, aksara:methods) yang keluar dengan kode gagal.
        $cards = [
            [$datasets, ImportDatasets::class, 'Dataset', 'scripts/export_datasets.py', 'aksara:datasets'],
            [$methods, ImportMethods::class, 'Metode', 'scripts/export_methods.py', 'aksara:methods'],
        ];
        foreach ($cards as [$importer, $command, $page, $exporter, $artisan]) {
            if (! is_file($importer::path($path))) {
                $this->line('Kartu halaman '.$page.' ('.$importer::FILE.') tidak ada di folder itu; halaman '.$page.' tidak diubah. '
                    .'Buat dengan: python '.$exporter);

                continue;
            }
            try {
                $this->info($command::summary($importer->import($path)));
            } catch (RuntimeException $e) {
                $this->warn('PERINGATAN: hasil OCR sudah diimpor, tetapi kartu halaman '.$page.' ditolak dan kartu lamanya '
                    .'dipertahankan. '.$e->getMessage().' Perbaiki lalu jalankan: php artisan '.$artisan);
            }
        }

        return self::SUCCESS;
    }
}
