<?php

namespace App\Console\Commands;

use App\Services\AnnotationImporter;
use Illuminate\Console\Command;
use RuntimeException;

class ImportAnnotations extends Command
{
    protected $signature = 'aksara:annotations
        {--fetch : unduh ulang dari HuggingFace (NusaAksara, lambat: beberapa menit)}
        {--path= : berkas JSON anotasi (default: storage/app/nusaaksara/annotations.json)}';

    protected $description = 'Impor transliterasi & arti buatan manusia NusaAksara (tahap 2 & 3 alur)';

    public function handle(AnnotationImporter $importer): int
    {
        $path = $this->option('path') ?: config('aksara.annotations_path');
        if ($this->option('fetch')) {
            $this->info('Mengunduh dari datasets-server HuggingFace...');
            $importer->fetch($path, fn ($config, $done, $total) => $this->line("  {$config}: {$done}/{$total}"));
        }
        try {
            $result = $importer->import($path);
        } catch (RuntimeException $e) {
            $this->error($e->getMessage());

            return self::FAILURE;
        }
        $this->info("{$result['annotations']} anotasi diimpor.");
        if ($draft = $result['translit_draft']) {
            $this->line('Transliterasi draf vs manusia: CER '.pct($draft['cer']).' pada '.$draft['lines'].' baris.');
        }

        return self::SUCCESS;
    }
}
