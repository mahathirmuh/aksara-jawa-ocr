<?php

namespace App\Console\Commands;

use App\Services\DatasetImporter;
use Illuminate\Console\Command;
use RuntimeException;

class ImportDatasets extends Command
{
    protected $signature = 'aksara:datasets {--path= : folder hasil (default: <repo OCR>/out/results)}';

    protected $description = 'Impor kartu data (datasets.json dari scripts/export_datasets.py) untuk halaman Dataset';

    public function handle(DatasetImporter $importer): int
    {
        $path = $this->option('path') ?: config('aksara.ocr_repo').DIRECTORY_SEPARATOR.config('aksara.results_dir');
        try {
            $counts = $importer->import($path);
        } catch (RuntimeException $e) {
            $this->error($e->getMessage());

            return self::FAILURE;
        }
        $this->info(self::summary($counts));

        return self::SUCCESS;
    }

    public static function summary(array $counts): string
    {
        return "Kartu data diimpor: {$counts['datasets']} dataset, {$counts['fonts']} font, korpus "
            .nfmt($counts['lines'])." baris, pemakaian run {$counts['run']}.";
    }
}
