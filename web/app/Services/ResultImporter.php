<?php

namespace App\Services;

use App\Models\AblationRun;
use App\Models\Confusion;
use App\Models\Gate;
use App\Models\Line;
use App\Models\Metric;
use App\Models\Pipeline;
use App\Models\Prediction;
use App\Models\ResultImport;
use Illuminate\Support\Facades\DB;
use RuntimeException;

/**
 * Impor kontrak data dari repo OCR (scripts/export_results.py, skema 1).
 *
 * Web tidak menghitung ulang angka OCR: semua CER, segmen beda, dan kolom citra datang dari Python.
 * Baris di-upsert per (dataset, external_id) supaya id stabil; tabel hasil lain diganti utuh.
 */
class ResultImporter
{
    public function import(string $dir): array
    {
        $manifest = $this->readJson($dir.'/manifest.json');
        $expected = config('aksara.schema');
        if (($manifest['schema'] ?? null) !== $expected) {
            throw new RuntimeException("Skema manifest {$manifest['schema']} tidak didukung (butuh {$expected}).");
        }
        $lines = iterator_to_array($this->readJsonl($dir.'/lines.jsonl'), false);
        $predictions = iterator_to_array($this->readJsonl($dir.'/predictions.jsonl'), false);

        // Pipeline resmi: kunci "official" di manifest; ekspor lama tidak punya kunci itu dan berarti crnn_fonts.
        $official = $manifest['official'] ?? Pipeline::DEFAULT_OFFICIAL;
        if (! in_array($official, array_column($manifest['pipelines'], 'key'), true)) {
            throw new RuntimeException("Pipeline resmi {$official} tidak ada di daftar pipeline manifest.");
        }
        // Halaman Kesalahan aksara menampilkan tabel ini atas nama pipeline resmi.
        $confused = $manifest['confusion']['pipeline'] ?? $official;
        if ($confused !== $official) {
            throw new RuntimeException("Kesalahan aksara di manifest dihitung dari {$confused}, bukan pipeline resmi {$official}.");
        }

        return DB::transaction(function () use ($dir, $manifest, $lines, $predictions, $official) {
            foreach ($manifest['pipelines'] as $p) {
                Pipeline::updateOrCreate(['key' => $p['key']], [
                    'label' => $p['label'], 'config' => $p['config'], 'kind' => $p['kind'],
                    'status' => $p['status'], 'sort' => $p['sort'] ?? 0, 'official' => $p['key'] === $official,
                ]);
            }
            Pipeline::whereNotIn('key', array_column($manifest['pipelines'], 'key'))->delete();

            $ids = [];
            foreach ($lines as $l) {
                $line = Line::updateOrCreate(
                    ['dataset' => $l['dataset'], 'external_id' => $l['id']],
                    ['image_path' => $l['image'], 'width' => $l['width'], 'height' => $l['height'],
                        'reference' => $l['reference'], 'condition' => $l['condition'] ?? null,
                        'source_id' => $l['source_id'] ?? null, 'tags' => $l['tags'] ?? []],
                );
                $ids[$l['id']] = $line->id;
            }
            Line::whereNotIn('id', array_values($ids))->delete();

            Prediction::query()->delete();
            $rows = [];
            foreach ($predictions as $p) {
                if (! isset($ids[$p['line']])) {
                    throw new RuntimeException("Prediksi untuk baris yang tidak ada di lines.jsonl: {$p['line']}");
                }
                $rows[] = [
                    'line_id' => $ids[$p['line']], 'pipeline' => $p['pipeline'], 'text' => $p['text'],
                    'cer' => $p['cer'], 'segments' => json_encode($p['segments'], JSON_UNESCAPED_UNICODE),
                    'spans' => isset($p['spans']) ? json_encode($p['spans']) : null,
                    'confidence' => $p['confidence'] ?? null,
                ];
            }
            foreach (array_chunk($rows, 200) as $chunk) {
                Prediction::insert($chunk);
            }

            Metric::where('scope', '!=', 'translit_draft')->delete();
            foreach ($manifest['metrics'] as $m) {
                Metric::create(array_intersect_key($m, array_flip(
                    ['scope', 'pipeline', 'lines', 'cer', 'cer_no_space', 'exact', 'better', 'worse'])));
            }

            Gate::query()->delete();
            foreach ($manifest['gates'] as $g) {
                Gate::create($g);
            }

            AblationRun::query()->delete();
            foreach ($manifest['ablation'] as $a) {
                AblationRun::create([
                    'k' => $a['k'], 'added' => $a['added'], 'augment' => $a['augment'], 'clean' => $a['clean'],
                    'heavy' => $a['heavy'], 'g3' => $a['G3'] ?? null, 'g3_no_space' => $a['G3_no_space'] ?? null,
                ]);
            }

            Confusion::query()->delete();
            foreach (array_chunk($manifest['confusion']['items'], 200) as $chunk) {
                Confusion::insert(array_map(fn ($c) => [
                    'kind' => $c['kind'], 'ref' => $c['ref'], 'hyp' => $c['hyp'], 'count' => $c['count'],
                ], $chunk));
            }

            $counts = ['lines' => count($lines), 'predictions' => count($rows), 'metrics' => count($manifest['metrics']),
                'pipelines' => count($manifest['pipelines'])];
            ResultImport::create([
                'schema' => $manifest['schema'], 'generated_at' => $manifest['generated'], 'source_path' => $dir,
                'limited' => (bool) ($manifest['limited'] ?? false), 'counts' => $counts,
            ]);

            return $counts;
        });
    }

    private function readJson(string $path): array
    {
        if (! is_file($path)) {
            throw new RuntimeException("Berkas tidak ditemukan: {$path}. Jalankan dulu: python scripts/export_results.py");
        }

        return json_decode(file_get_contents($path), true, flags: JSON_THROW_ON_ERROR);
    }

    private function readJsonl(string $path): \Generator
    {
        if (! is_file($path)) {
            throw new RuntimeException("Berkas tidak ditemukan: {$path}");
        }
        foreach (new \SplFileObject($path) as $line) {
            if (trim($line) !== '') {
                yield json_decode($line, true, flags: JSON_THROW_ON_ERROR);
            }
        }
    }
}
