<?php

namespace App\Services;

use App\Models\DatasetReport;
use Illuminate\Support\Facades\DB;
use JsonException;
use RuntimeException;

/**
 * Impor kartu data dari repo OCR (scripts/export_datasets.py, kontrak dataset skema 1) untuk halaman Dataset.
 *
 * Web tidak menghitung apa pun dari data OCR: jumlah baris, pembagian, pemakaian run resmi, dan font datang dari
 * Python. Isinya disimpan utuh; yang diperiksa di sini hanya bentuknya, supaya berkas yang salah ditolak saat impor
 * dan bukan menjadi galat saat halaman dibuka.
 */
class DatasetImporter
{
    public const FILE = 'datasets.json';

    /** Kunci yang dibaca halaman Dataset (notasi titik); semuanya wajib ada. */
    private const REQUIRED = [
        'generated', 'official.run', 'corpus.lines', 'corpus.articles', 'corpus.length_histogram', 'splits',
        'usage.train.distinct_lines', 'usage.train.pool', 'usage.train.samples', 'usage.val.lines', 'usage.test.lines',
        'usage.test.gates', 'usage.real.lines', 'lineage.runs', 'lineage.distinct_lines', 'rare.codepoints',
        'datasets', 'fonts', 'support', 'warnings',
    ];

    private const SPLITS = ['train', 'val', 'test'];

    public static function path(string $dir): string
    {
        return rtrim($dir, '/\\').DIRECTORY_SEPARATOR.self::FILE;
    }

    public function import(string $dir): array
    {
        $path = self::path($dir);
        if (! is_file($path)) {
            throw new RuntimeException("Berkas tidak ditemukan: {$path}. Jalankan dulu: python scripts/export_datasets.py");
        }
        try {
            $card = json_decode(file_get_contents($path), true, flags: JSON_THROW_ON_ERROR);
        } catch (JsonException $e) {
            throw new RuntimeException("{$path} bukan JSON yang sah: {$e->getMessage()}");
        }
        if (! is_array($card)) {
            throw new RuntimeException("{$path} bukan kartu data: isinya harus objek JSON.");
        }
        $expected = config('aksara.dataset_schema');
        if (($card['schema'] ?? null) !== $expected) {
            throw new RuntimeException('Skema kartu data '.json_encode($card['schema'] ?? null)." tidak didukung (butuh {$expected}).");
        }
        foreach (self::REQUIRED as $key) {
            if (data_get($card, $key) === null) {
                throw new RuntimeException("Kartu data tidak lengkap: kunci {$key} tidak ada di {$path}.");
            }
        }
        if (array_column($card['splits'], 'key') !== self::SPLITS) {
            throw new RuntimeException('Kartu data tidak lengkap: splits harus berisi train, val, test (urutan itu).');
        }

        return DB::transaction(function () use ($card, $dir) {
            DatasetReport::query()->delete();
            DatasetReport::create([
                'schema' => $card['schema'], 'generated_at' => $card['generated'],
                'official_run' => $card['official']['run'], 'source_path' => $dir, 'payload' => $card,
            ]);

            return ['run' => $card['official']['run'], 'lines' => $card['corpus']['lines'],
                'datasets' => count($card['datasets']), 'fonts' => count($card['fonts'])];
        });
    }
}
