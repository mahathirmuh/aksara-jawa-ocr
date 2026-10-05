<?php

namespace App\Console\Commands;

use App\Services\MethodImporter;
use Illuminate\Console\Command;
use RuntimeException;

class ImportMethods extends Command
{
    protected $signature = 'aksara:methods {--path= : folder hasil (default: <repo OCR>/out/results)}';

    protected $description = 'Impor kartu metode (methods.json dari scripts/export_methods.py) untuk halaman Metode';

    public function handle(MethodImporter $importer): int
    {
        $path = $this->option('path') ?: config('aksara.ocr_repo').DIRECTORY_SEPARATOR.config('aksara.results_dir');
        try {
            $counts = $importer->import($path);
        } catch (RuntimeException $e) {
            $this->error($e->getMessage());

            return self::FAILURE;
        }
        $this->info(self::summary($counts));
        ImportResults::warnIfStale($this, $importer, 'Metode', 'scripts/export_methods.py', 'aksara:methods');

        return self::SUCCESS;
    }

    public static function summary(array $counts): string
    {
        return "Kartu metode diimpor: {$counts['methods']} butir di {$counts['groups']} kelompok, {$counts['planned']} direncanakan, "
            .'model '.nfmt($counts['parameters'])." parameter, run {$counts['run']}.";
    }
}
