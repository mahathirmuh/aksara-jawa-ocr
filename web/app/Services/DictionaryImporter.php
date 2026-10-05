<?php

namespace App\Services;

use App\Models\DictionaryEntry;
use App\Models\DictionarySource;
use Illuminate\Support\Facades\DB;
use Normalizer;
use RuntimeException;

/**
 * Mengisi kamus kata dari berkas buatan tools/dictionaries.py:
 *   <dir>/sources.json      {"jv": [{"key","name","license","url","retrieved"}], "id": [...]}
 *   <dir>/jv.jsonl[.gz]     satu entri per baris: {"word","glosses":[...],"gloss_lang","source", "pos"?, "aksara"?, "register"?, "examples"?, "note"?, "url"?}
 *   <dir>/id.jsonl[.gz]
 * Satu kamus diganti seutuhnya dalam satu transaksi; baris yang rusak menggagalkan impor kamus itu (datanya buatan
 * mesin, jadi baris rusak berarti pembuatnya salah, bukan sesuatu yang boleh dilewati diam-diam).
 */
class DictionaryImporter
{
    private const CHUNK = 500;

    /** @return array<string, array{entries: int, sources: array<string, int>}> jumlah per kamus yang diimpor */
    public function import(string $dir, ?array $only = null): array
    {
        $sourcesFile = $dir.DIRECTORY_SEPARATOR.'sources.json';
        if (! is_file($sourcesFile)) {
            throw new RuntimeException("Berkas kamus tidak ada: {$sourcesFile}. Bangun dulu dengan tools/dictionaries.py (lihat web/README).");
        }
        $declared = json_decode(file_get_contents($sourcesFile), true, flags: JSON_THROW_ON_ERROR);
        $result = [];
        foreach (array_keys(DictionaryEntry::DICTIONARIES) as $dictionary) {
            if ($only && ! in_array($dictionary, $only, true)) {
                continue;
            }
            $file = $this->entriesFile($dir, $dictionary);
            if ($file === null) {
                continue;
            }
            $sources = collect($declared[$dictionary] ?? [])->keyBy('key');
            if ($sources->isEmpty()) {
                throw new RuntimeException("sources.json tidak menyebut sumber untuk kamus '{$dictionary}', padahal ".basename($file).' ada.');
            }
            $result[$dictionary] = DB::transaction(fn () => $this->replace($dictionary, $file, $sources));
        }
        if ($result === []) {
            throw new RuntimeException("Tidak ada berkas kamus (jv.jsonl / id.jsonl, boleh .gz) di {$dir}.");
        }

        return $result;
    }

    private function entriesFile(string $dir, string $dictionary): ?string
    {
        foreach (['.jsonl.gz', '.jsonl'] as $suffix) {
            if (is_file($file = $dir.DIRECTORY_SEPARATOR.$dictionary.$suffix)) {
                return $file;
            }
        }

        return null;
    }

    private function replace(string $dictionary, string $file, $sources): array
    {
        DictionaryEntry::where('dictionary', $dictionary)->delete();
        DictionarySource::where('dictionary', $dictionary)->delete();

        $counts = [];
        $rows = [];
        $name = basename($file);
        // gzopen membaca berkas biasa maupun .gz.
        $handle = gzopen($file, 'rb');
        if ($handle === false) {
            throw new RuntimeException("Tidak bisa membuka {$file}.");
        }
        try {
            for ($number = 1; ($line = gzgets($handle)) !== false; $number++) {
                if (trim($line) === '') {
                    continue;
                }
                $rows[] = $this->row($dictionary, $line, "{$name} baris {$number}", $sources);
                $source = end($rows)['source'];
                $counts[$source] = ($counts[$source] ?? 0) + 1;
                if (count($rows) === self::CHUNK) {
                    DictionaryEntry::insert($rows);
                    $rows = [];
                }
            }
        } finally {
            gzclose($handle);
        }
        if ($rows) {
            DictionaryEntry::insert($rows);
        }
        if ($counts === []) {
            throw new RuntimeException("{$name} kosong.");
        }
        foreach ($sources as $key => $source) {
            foreach (['name', 'license', 'url'] as $field) {
                if (! is_string($source[$field] ?? null) || $source[$field] === '') {
                    throw new RuntimeException("sources.json: sumber '{$key}' kamus '{$dictionary}' tidak punya '{$field}'.");
                }
            }
            if ($this->link($source['url']) === null) {
                throw new RuntimeException("sources.json: 'url' sumber '{$key}' kamus '{$dictionary}' harus berawalan http:// atau https://.");
            }
            DictionarySource::create([
                'dictionary' => $dictionary, 'key' => $key, 'name' => $source['name'], 'license' => $source['license'],
                'url' => $source['url'], 'retrieved' => $source['retrieved'] ?? null, 'entries' => $counts[$key] ?? 0,
            ]);
        }

        return ['entries' => array_sum($counts), 'sources' => $counts];
    }

    private function row(string $dictionary, string $line, string $where, $sources): array
    {
        $entry = json_decode($line, true);
        if (! is_array($entry)) {
            throw new RuntimeException("{$where}: bukan JSON.");
        }
        $word = is_string($entry['word'] ?? null) ? trim($this->nfc($entry['word'])) : '';
        $glosses = array_values(array_filter(array_map(
            fn ($gloss) => is_string($gloss) ? trim($this->nfc($gloss)) : '', (array) ($entry['glosses'] ?? [])), fn ($gloss) => $gloss !== ''));
        $lookup = DictionaryEntry::normalize($word);
        if ($lookup === '' || mb_strlen($word) > 190) {
            throw new RuntimeException("{$where}: 'word' kosong, tanpa huruf, atau lebih dari 190 karakter.");
        }
        if ($glosses === []) {
            throw new RuntimeException("{$where}: '{$word}' tidak punya arti (glosses).");
        }
        if (! $sources->has($entry['source'] ?? '')) {
            throw new RuntimeException("{$where}: sumber '".($entry['source'] ?? '')."' tidak ada di sources.json.");
        }
        if (! isset(DictionaryEntry::GLOSS_LANGUAGES[$entry['gloss_lang'] ?? ''])) {
            throw new RuntimeException("{$where}: gloss_lang harus salah satu dari ".implode(', ', array_keys(DictionaryEntry::GLOSS_LANGUAGES)).'.');
        }

        return [
            'dictionary' => $dictionary,
            'source' => $entry['source'],
            'word' => $word,
            'lookup' => $lookup,
            'pos' => $this->optional($entry, 'pos', 60),
            'aksara' => $this->optional($entry, 'aksara', 190),
            'register' => $this->optional($entry, 'register', 60),
            'gloss_lang' => $entry['gloss_lang'],
            'glosses' => json_encode($glosses, JSON_UNESCAPED_UNICODE | JSON_THROW_ON_ERROR),
            'gloss_text' => DictionaryEntry::glossText($glosses),
            'examples' => $this->examples($entry),
            'note' => $this->optional($entry, 'note', 500),
            'url' => $this->link($this->optional($entry, 'url', 500)),
        ];
    }

    /** Contoh pemakaian: paling banyak tiga, yang kepanjangan dibuang; null bila tidak ada (kolom JSON). */
    private function examples(array $entry): ?string
    {
        $examples = [];
        foreach ((array) ($entry['examples'] ?? []) as $example) {
            $example = is_string($example) ? trim($this->nfc($example)) : '';
            if ($example !== '' && mb_strlen($example) <= 300 && count($examples) < 3) {
                $examples[] = $example;
            }
        }

        return $examples ? json_encode($examples, JSON_UNESCAPED_UNICODE | JSON_THROW_ON_ERROR) : null;
    }

    /** Tautan dari berkas data dipakai sebagai href: hanya http(s) yang diterima (bukan javascript: dsb.). */
    private function link(?string $url): ?string
    {
        return $url !== null && preg_match('#^https?://#i', $url) ? $url : null;
    }

    private function optional(array $entry, string $field, int $max): ?string
    {
        $value = is_string($entry[$field] ?? null) ? trim($this->nfc($entry[$field])) : '';

        return $value === '' || mb_strlen($value) > $max ? null : $value;
    }

    private function nfc(string $text): string
    {
        return Normalizer::normalize($text, Normalizer::FORM_C) ?: $text;
    }
}
