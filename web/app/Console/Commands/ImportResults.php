<?php

namespace App\Console\Commands;

use App\Services\AnnotationImporter;
use App\Services\ResultImporter;
use Illuminate\Console\Command;
use RuntimeException;

class ImportResults extends Command
{
    protected $signature = 'aksara:import {--path= : folder hasil (default: <repo OCR>/out/results)}';

    protected $description = 'Impor hasil eksperimen OCR (manifest.json, lines.jsonl, predictions.jsonl) ke database';

    public function handle(ResultImporter $results, AnnotationImporter $annotations): int
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

        return self::SUCCESS;
    }
}
