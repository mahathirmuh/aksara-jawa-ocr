<?php

namespace App\Console\Commands;

use App\Services\MatrixImporter;
use Illuminate\Console\Command;
use RuntimeException;

class ImportMatrix extends Command
{
    protected $signature = 'aksara:matrix {--path= : folder hasil (default: <repo OCR>/out/results)}';

    protected $description = 'Impor matriks metode (matrix.json dari scripts/method_matrix.py export) untuk halaman Matriks';

    public function handle(MatrixImporter $importer): int
    {
        $path = $this->option('path') ?: config('aksara.ocr_repo').DIRECTORY_SEPARATOR.config('aksara.results_dir');
        try {
            $counts = $importer->import($path);
        } catch (RuntimeException $e) {
            $this->error($e->getMessage());

            return self::FAILURE;
        }
        $this->info("Matriks diimpor: {$counts['methods']} metode, {$counts['architecture']} arsitektur, "
            ."{$counts['ablation']} pasangan ablasi, {$counts['augment']} langkah augmentasi.");

        return self::SUCCESS;
    }
}
