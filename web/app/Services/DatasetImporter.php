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

        // "count" = bilangan yang tidak negatif (jumlah baris, langkah, peluang); akhiran "?" = kunci pilihan, yang bila
        // ada tetap harus bertipe benar (halaman memilih kalimatnya dari nilai-nilai itu).
        return $typed('count', 'corpus.', ['articles', 'candidates', 'roundtrip_passed', 'roundtrip_pass_rate', 'duplicates', 'injected',
            'lines', 'leaked_lines'])
            + ['corpus.other' => 'number?', 'corpus.max_roundtrip_cer' => 'count?', 'corpus.split_buckets' => 'map?',
                'corpus.split_buckets.*' => 'count?', 'corpus.min_count' => 'map?', 'corpus.min_count.*' => 'count?',
                'corpus.length_histogram' => 'list', 'corpus.length_histogram.*.range' => 'text', 'corpus.length_histogram.*.lines' => 'count',
                'splits' => 'list', 'splits.*.key' => 'string', 'splits.*.lines' => 'count', 'splits.*.share' => 'count']
            + $typed('count', 'usage.train.', ['steps', 'batch_size', 'samples', 'pool', 'seed'])
            // Jadwal baris sintetis tidak dihitung untuk run --overfit / --real-train: ekspor menulis null.
            + ['usage.train.distinct_lines' => 'count?', 'lineage.distinct_lines' => 'count?', 'lineage.runs.*.distinct_lines' => 'count?',
                'usage.train.real_train' => 'bool?', 'usage.train.real_val' => 'bool?',
                'usage.train.rare_insert_prob' => 'count?', 'usage.train.rare_opener_prob' => 'count?',
                'usage.val.lines' => 'count', 'usage.val.steps' => 'list', 'usage.val.steps.*' => 'count', 'usage.val.fonts' => 'count',
                'usage.test.lines' => 'count', 'usage.test.gates' => 'list', 'usage.test.gates.*.code' => 'text',
                'usage.test.quick' => 'count?', 'usage.test.quick_max_lines' => 'count?', 'usage.test.full' => 'count?',
                'usage.test.others' => 'list?', 'usage.test.others.*.kind' => 'text?', 'usage.test.others.*.lines' => 'count?',
                'usage.real.lines' => 'count', 'usage.real.reports' => 'count?',
                'lineage.complete' => 'bool', 'lineage.runs' => 'list', 'lineage.runs.*.run' => 'text', 'lineage.runs.*.augment' => 'text',
                'lineage.runs.*.real_train' => 'bool?']
            + $typed('count', 'lineage.', ['steps', 'samples', 'pool'])
            + $typed('count', 'lineage.runs.*.', ['steps', 'samples', 'pool', 'seed', 'fonts', 'drop_space_prob', 'track_prob', 'track_max',
                'rare_insert_prob', 'rare_opener_prob'])
            + ['rare.codepoints' => 'count', 'rare.javanese' => 'count', 'rare.train_lines' => 'map', 'rare.train_lines.*' => 'count?',
                'rare.scheduled_lines' => 'map', 'rare.scheduled_lines.*' => 'count?',
                'datasets' => 'list', 'datasets.*.count' => 'count', 'datasets.*.shareable' => 'bool', 'datasets.*.roles' => 'map',
                'datasets.*.roles.*' => 'count', 'datasets.*.gates' => 'list', 'datasets.*.gates.*' => 'string',
                'datasets.*.status' => 'string?', 'datasets.*.pending' => 'count?', 'datasets.*.verified' => 'count?',
                'datasets.*.articles' => 'count?', 'datasets.*.pages' => 'count?', 'datasets.*.source_url' => 'string?',
                'datasets.*.planned' => 'list?']
            + $typed('text', 'datasets.*.', ['key', 'name', 'unit'])
            + $typed('string', 'datasets.*.', ['content', 'source', 'license', 'share_note'])
            // Wadah (daftar) diperiksa sebelum isinya, supaya pesannya menyebut kunci yang salah bentuk.
            + ['fonts' => 'list']
            + $typed('text', 'fonts.*.', ['file', 'group'])
            + ['fonts.*.license' => 'string', 'fonts.*.roles' => 'list', 'fonts.*.roles.*' => 'string', 'fonts.*.in_repo' => 'bool',
                'fonts.*.available' => 'bool?', 'fonts.*.drops_space' => 'bool?', 'fonts.*.missing' => 'list?', 'fonts.*.notes' => 'list?',
                'fonts.*.review' => 'string?', 'fonts.*.shared_with_test' => 'map?', 'fonts.*.shared_with_test.family' => 'bool?',
                'fonts.*.shared_with_test.font' => 'string?', 'fonts.*.shared_with_test.same' => 'count?',
                'fonts.*.shared_with_test.of' => 'count?',
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

    public function staleNotes(): array
    {
        $report = DatasetReport::current();
        $stale = $report ? Dataset::stale($report->payload) : null;

        return $stale ? ['Kartu data ini dihitung untuk run '.$stale['run'].', sedangkan angka resmi web sekarang milik '.$stale['official'].'.'] : [];
    }

    protected function summary(array $card): array
    {
        return ['run' => $card['official']['run'], 'lines' => $card['corpus']['lines'],
            'datasets' => count($card['datasets']), 'fonts' => count($card['fonts'])];
    }
}
