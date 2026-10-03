<?php

namespace App\Services;

use App\Models\Line;
use App\Models\LineAnnotation;
use App\Models\Metric;
use App\Support\TextMetrics;
use App\Support\Transliterator;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;
use RuntimeException;

/**
 * Transliterasi & arti buatan manusia dari NusaAksara (HuggingFace, config Image Transliteration /
 * Image Translation, script jawa), plus metrik tahap 2: transliterasi draf web vs transliterasi manusia.
 */
class AnnotationImporter
{
    private const ROWS_URL = 'https://datasets-server.huggingface.co/filter';

    /** Unduh dari HuggingFace ke berkas JSON {"transliteration": {file: teks}, "translation": {...}}. */
    public function fetch(string $path, ?callable $progress = null): array
    {
        $out = [];
        foreach (['Image Transliteration' => 'transliteration', 'Image Translation' => 'translation'] as $config => $field) {
            $got = [];
            $offset = 0;
            do {
                $body = Http::retry(4, 3000)->timeout(90)->get(self::ROWS_URL, [
                    'dataset' => 'NusaAksara/NusaAksara', 'config' => $config, 'split' => 'train',
                    'where' => "\"script\"='jawa'", 'offset' => $offset, 'length' => 100,
                ])->throw()->json();
                foreach ($body['rows'] as $row) {
                    $got[basename($row['row']['image'])] = $row['row'][$field];
                }
                $offset += count($body['rows']);
                $progress && $progress($config, $offset, $body['num_rows_total']);
            } while ($body['rows'] !== [] && $offset < $body['num_rows_total']);
            $out[$field] = $got;
        }
        if (! is_dir(dirname($path))) {
            mkdir(dirname($path), 0775, true);
        }
        file_put_contents($path, json_encode($out, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT));

        return $out;
    }

    public function import(string $path): array
    {
        if (! is_file($path)) {
            throw new RuntimeException("Berkas anotasi tidak ada: {$path}. Jalankan: php artisan aksara:annotations --fetch");
        }
        $data = json_decode(file_get_contents($path), true, flags: JSON_THROW_ON_ERROR);
        $translit = $data['transliteration'] ?? [];
        $translation = $data['translation'] ?? [];
        $files = array_unique(array_merge(array_keys($translit), array_keys($translation)));

        DB::transaction(function () use ($files, $translit, $translation) {
            foreach ($files as $file) {
                LineAnnotation::updateOrCreate(['dataset' => 'nusaaksara', 'external_id' => $file], [
                    'transliteration' => $translit[$file] ?? null,
                    'translation' => $translation[$file] ?? null,
                    'source' => 'NusaAksara (manusia)',
                ]);
            }
        });

        return ['annotations' => count($files), 'translit_draft' => $this->scoreDraftTransliteration()];
    }

    /** Tahap 2: CER transliterasi draf (dari label aksara) terhadap transliterasi manusia, huruf saja. */
    public function scoreDraftTransliteration(): ?array
    {
        $refs = $hyps = [];
        Line::with('annotation')->where('dataset', 'nusaaksara')->each(function (Line $line) use (&$refs, &$hyps) {
            $human = $line->annotation?->transliteration;
            if ($human) {
                $refs[] = TextMetrics::lettersOnly($human);
                $hyps[] = TextMetrics::lettersOnly(Transliterator::toLatin($line->reference));
            }
        });
        if ($refs === []) {
            return null;
        }
        $metric = ['lines' => count($refs), 'cer' => TextMetrics::cer($refs, $hyps),
            'exact' => count(array_filter(array_map(fn ($r, $h) => $r === $h, $refs, $hyps))) / count($refs)];
        // updateOrCreate tidak bisa: "pipeline = NULL" di SQL tidak pernah cocok.
        Metric::where('scope', 'translit_draft')->delete();
        Metric::create(['scope' => 'translit_draft', 'pipeline' => null, ...$metric]);

        return $metric;
    }
}
