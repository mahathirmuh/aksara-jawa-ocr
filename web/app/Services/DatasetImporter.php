<?php

namespace App\Services;

use App\Livewire\Pages\Dataset;
use App\Models\DatasetReport;
use Illuminate\Database\Eloquent\Model;
use RuntimeException;

/**
 * Impor kartu data dari repo OCR (scripts/export_datasets.py, kontrak dataset skema 1) untuk halaman Dataset.
 *
 * Web tidak menghitung apa pun dari data OCR: jumlah baris, pembagian, pemakaian run resmi, dan font datang dari
 * Python. Isinya disimpan utuh. Bentuknya diperiksa `CardImporter` menurut `shape()` di bawah, lalu halaman Dataset
 * dirender sekali dengan kartu itu sebelum kartu lama diganti.
 */
class DatasetImporter extends CardImporter
{
    public const FILE = 'datasets.json';

    private const SPLITS = ['train', 'val', 'test'];

    protected function model(): string
    {
        return DatasetReport::class;
    }

    protected function label(): string
    {
        return 'kartu data';
    }

    protected function schemaConfig(): string
    {
        return 'aksara.dataset_schema';
    }

    protected function exporter(): string
    {
        return 'scripts/export_datasets.py';
    }

    protected function shape(): array
    {
        $typed = fn (string $type, string $prefix, array $keys) => array_fill_keys(array_map(fn ($key) => $prefix.$key, $keys), $type);

        return $typed('number', 'corpus.', ['articles', 'candidates', 'roundtrip_passed', 'roundtrip_pass_rate', 'duplicates', 'injected',
            'lines', 'leaked_lines'])
            + ['corpus.length_histogram' => 'list', 'corpus.length_histogram.*.range' => 'string', 'corpus.length_histogram.*.lines' => 'number',
                'splits' => 'list', 'splits.*.key' => 'string', 'splits.*.lines' => 'number', 'splits.*.share' => 'number']
            + $typed('number', 'usage.train.', ['steps', 'batch_size', 'samples', 'pool', 'seed'])
            // Jadwal baris sintetis tidak dihitung untuk run --overfit / --real-train: ekspor menulis null.
            + ['usage.train.distinct_lines' => 'number?', 'lineage.distinct_lines' => 'number?', 'lineage.runs.*.distinct_lines' => 'number?',
                'usage.val.lines' => 'number', 'usage.val.steps' => 'list', 'usage.val.steps.*' => 'number', 'usage.val.fonts' => 'number',
                'usage.test.lines' => 'number', 'usage.test.gates' => 'list', 'usage.test.gates.*.code' => 'string',
                'usage.real.lines' => 'number',
                'lineage.complete' => 'bool', 'lineage.runs' => 'list', 'lineage.runs.*.run' => 'string', 'lineage.runs.*.augment' => 'string']
            + $typed('number', 'lineage.', ['steps', 'samples', 'pool'])
            + $typed('number', 'lineage.runs.*.', ['steps', 'samples', 'pool', 'seed', 'fonts', 'drop_space_prob', 'track_prob', 'track_max',
                'rare_insert_prob', 'rare_opener_prob'])
            + ['rare.codepoints' => 'number', 'rare.javanese' => 'number', 'rare.train_lines' => 'map', 'rare.scheduled_lines' => 'map',
                'datasets' => 'list', 'datasets.*.count' => 'number', 'datasets.*.shareable' => 'bool', 'datasets.*.roles' => 'map',
                'datasets.*.roles.*' => 'number', 'datasets.*.gates' => 'list', 'datasets.*.gates.*' => 'string']
            + $typed('string', 'datasets.*.', ['key', 'name', 'content', 'unit', 'source', 'license', 'share_note'])
            // Wadah (daftar) diperiksa sebelum isinya, supaya pesannya menyebut kunci yang salah bentuk.
            + ['fonts' => 'list']
            + $typed('string', 'fonts.*.', ['file', 'group', 'license'])
            + ['fonts.*.roles' => 'list', 'fonts.*.roles.*' => 'string', 'fonts.*.in_repo' => 'bool',
                'support' => 'list', 'support.*.key' => 'string', 'warnings' => 'list', 'warnings.*' => 'string'];
    }

    protected function check(array $card): void
    {
        if (array_column($card['splits'], 'key') !== self::SPLITS) {
            throw new RuntimeException('Kartu data tidak lengkap: splits harus berisi train, val, test (urutan itu).');
        }
        if ($card['lineage']['runs'] === [] || $card['corpus']['length_histogram'] === []) {
            throw new RuntimeException('Kartu data tidak lengkap: lineage.runs dan corpus.length_histogram tidak boleh kosong.');
        }
    }

    protected function preview(Model $report): void
    {
        view('livewire.pages.dataset', Dataset::viewData($report))->render();
    }

    protected function summary(array $card): array
    {
        return ['run' => $card['official']['run'], 'lines' => $card['corpus']['lines'],
            'datasets' => count($card['datasets']), 'fonts' => count($card['fonts'])];
    }
}
