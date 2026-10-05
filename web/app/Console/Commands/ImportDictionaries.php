<?php

namespace App\Console\Commands;

use App\Models\DictionaryEntry;
use App\Services\DictionaryImporter;
use Illuminate\Console\Command;
use JsonException;
use RuntimeException;

class ImportDictionaries extends Command
{
    protected $signature = 'aksara:dictionary
        {--path= : folder berisi sources.json dan jv.jsonl / id.jsonl (boleh .gz); default: database/dictionaries}
        {--only= : hanya kamus ini: jv atau id}';

    protected $description = 'Impor kamus kata bahasa Jawa dan bahasa Indonesia untuk halaman Kamus (data dari tools/dictionaries.py)';

    public function handle(DictionaryImporter $importer): int
    {
        $only = $this->option('only');
        if ($only !== null && ! isset(DictionaryEntry::DICTIONARIES[$only])) {
            $this->error('--only harus jv atau id.');

            return self::FAILURE;
        }
        try {
            $result = $importer->import($this->option('path') ?: config('aksara.dictionary_dir'), $only ? [$only] : null);
        } catch (RuntimeException|JsonException $e) {
            $this->error($e->getMessage());

            return self::FAILURE;
        }
        foreach ($result as $dictionary => $count) {
            $sources = collect($count['sources'])->map(fn (int $n, string $key) => "{$key} ".nfmt($n))->implode(', ');
            $this->info(DictionaryEntry::DICTIONARIES[$dictionary].': '.nfmt($count['entries'])." entri ({$sources}).");
        }

        return self::SUCCESS;
    }
}
