<?php

namespace App\Services;

use App\Livewire\Pages\Metode;
use App\Models\MethodReport;
use Illuminate\Database\Eloquent\Model;
use RuntimeException;

/**
 * Impor kartu metode dari repo OCR (scripts/export_methods.py, kontrak metode skema 1) untuk halaman Metode.
 *
 * Penjelasan, pengaturan, dan bukti tiap butir ditulis dan dihitung Python dari checkpoint, kode, dan hasil evaluasi;
 * web menampilkannya apa adanya dan hanya menambah butir tahap lanjutan alur miliknya sendiri.
 */
class MethodImporter extends CardImporter
{
    public const FILE = 'methods.json';

    protected function model(): string
    {
        return MethodReport::class;
    }

    protected function label(): string
    {
        return 'kartu metode';
    }

    protected function schemaConfig(): string
    {
        return 'aksara.method_schema';
    }

    protected function exporter(): string
    {
        return 'scripts/export_methods.py';
    }

    protected function shape(): array
    {
        $method = 'groups.*.methods.*.';

        return [
            // Waktu ekspor hasil yang dikutip buktinya: halaman membandingkannya dengan hasil yang sedang diimpor.
            'results_generated' => 'string',
            'model.classes' => 'number', 'model.height' => 'number', 'model.conv_layers' => 'number', 'model.lstm_layers' => 'number',
            'model.parameters.total' => 'number', 'model.parameters.cnn' => 'number', 'model.parameters.proj' => 'number',
            'model.parameters.rnn' => 'number', 'model.parameters.head' => 'number',
            'groups' => 'list', 'groups.*.key' => 'string', 'groups.*.title' => 'string', 'groups.*.intro' => 'string',
            'groups.*.methods' => 'list',
            $method.'key' => 'string', $method.'name' => 'string', $method.'kind' => 'string', $method.'status' => 'string',
            $method.'summary' => 'string', $method.'settings' => 'list', $method.'files' => 'list', $method.'files.*' => 'string',
            'planned' => 'list', 'planned.*.label' => 'string', 'planned.*.config' => 'string',
        ];
    }

    /** Pengaturan = pasangan [label, nilai] berupa teks; bukti dan sumbernya teks atau kosong. */
    protected function check(array $card): void
    {
        foreach ($card['groups'] as $group) {
            foreach ($group['methods'] as $method) {
                foreach ($method['settings'] as $pair) {
                    if (! is_array($pair) || ! array_is_list($pair) || count($pair) !== 2 || ! is_string($pair[0]) || ! is_string($pair[1])) {
                        throw new RuntimeException("Kartu metode tidak sah: pengaturan butir {$method['key']} harus pasangan [label, nilai] berupa teks.");
                    }
                }
                foreach (['evidence', 'evidence_source'] as $key) {
                    if (isset($method[$key]) && ! is_string($method[$key])) {
                        throw new RuntimeException("Kartu metode tidak sah: {$key} butir {$method['key']} harus teks.");
                    }
                }
            }
        }
    }

    protected function preview(Model $report): void
    {
        view('livewire.pages.metode', Metode::viewData($report))->render();
    }

    protected function summary(array $card): array
    {
        return ['run' => $card['official']['run'], 'groups' => count($card['groups']),
            'methods' => array_sum(array_map(fn ($group) => count($group['methods']), $card['groups'])),
            'planned' => count($card['planned']), 'parameters' => $card['model']['parameters']['total']];
    }
}
