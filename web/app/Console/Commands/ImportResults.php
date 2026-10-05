<?php

namespace App\Console\Commands;

use App\Services\AnnotationImporter;
use App\Services\CardStorageException;
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
        // Kartu yang DITOLAK (isinya salah) tidak menggagalkan perintah ini: hasil OCR di atas sudah tersimpan, dan
        // tiap kartu punya perintah sendiri yang keluar dengan kode gagal. Kartu yang tidak bisa disimpan karena
        // keadaan database (tabel belum dimigrasi) menggagalkannya, karena itu bukan soal kartu.
        $cards = [
            [$datasets, ImportDatasets::class, 'Dataset', 'scripts/export_datasets.py', 'aksara:datasets'],
            [$methods, ImportMethods::class, 'Metode', 'scripts/export_methods.py', 'aksara:methods'],
        ];
        $status = self::SUCCESS;
        foreach ($cards as [$importer, $command, $page, $exporter, $artisan]) {
            if (! is_file($importer::path($path))) {
                $this->line('Kartu halaman '.$page.' ('.$importer::FILE.') tidak ada di folder itu; halaman '.$page.' tidak diubah. '
                    .'Buat dengan: python '.$exporter);

                continue;
            }
            try {
                $this->info($command::summary($importer->import($path)));
                self::warnIfStale($this, $importer, $page, $exporter, $artisan);
            } catch (CardStorageException $e) {
                $this->error('Kartu halaman '.$page.' tidak diimpor. '.$e->getMessage());
                $status = self::FAILURE;
            } catch (RuntimeException $e) {
                $this->warn('PERINGATAN: hasil OCR sudah diimpor, tetapi kartu halaman '.$page.' ditolak. '.$e->getMessage()
                    .' Perbaiki lalu jalankan: php artisan '.$artisan);
            }
        }

        return $status;
    }

    /**
     * Kartu yang baru diimpor tetapi tidak sejalan dengan hasil OCR yang diimpor web (dibuat untuk run lain, atau
     * buktinya dikutip dari ekspor hasil yang lain): halamannya akan menandainya, dan perintah impor mengatakannya
     * saat itu juga supaya tidak baru ketahuan ketika halaman dibuka.
     */
    public static function warnIfStale(Command $command, DatasetImporter|MethodImporter $importer, string $page, string $exporter, string $artisan): void
    {
        foreach ($importer->staleNotes() as $note) {
            $command->warn('PERINGATAN: kartu halaman '.$page.' tidak sejalan dengan hasil OCR yang diimpor. '.$note
                .' Jalankan ulang: python '.$exporter.', lalu php artisan '.$artisan);
        }
    }
}
