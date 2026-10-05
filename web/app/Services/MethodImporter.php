<?php

namespace App\Services;

use App\Livewire\Pages\Metode;
use App\Models\MethodReport;
use Illuminate\Database\Eloquent\Model;
use RuntimeException;

/**
 * Impor kartu metode dari repo OCR (scripts/export_methods.py, kontrak metode skema 1) untuk halaman Metode.
 *
 * Penjelasan, pengaturan, bukti, dan catatan tiap butir ditulis dan dihitung Python dari checkpoint, kode, dan hasil
 * evaluasi; web menampilkannya apa adanya dan hanya menambah butir tahap lanjutan alur miliknya sendiri.
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
            'results_generated' => 'text',
            'model.classes' => 'count', 'model.height' => 'count', 'model.conv_layers' => 'count', 'model.lstm_layers' => 'count',
            'model.bidirectional' => 'bool?',
            'model.parameters.total' => 'count', 'model.parameters.cnn' => 'count', 'model.parameters.proj' => 'count',
            'model.parameters.rnn' => 'count', 'model.parameters.head' => 'count',
            'groups' => 'list', 'groups.*.key' => 'text', 'groups.*.title' => 'text', 'groups.*.intro' => 'string',
            'groups.*.methods' => 'list',
            $method.'key' => 'text', $method.'name' => 'text', $method.'kind' => 'text', $method.'status' => 'text',
            $method.'summary' => 'string', $method.'settings' => 'list', $method.'files' => 'list', $method.'files.*' => 'string',
            // Bukti, sumbernya, dan catatan: pilihan, tetapi harus teks bila ada.
            $method.'evidence' => 'string?', $method.'evidence_source' => 'string?', $method.'note' => 'string?',
            'planned' => 'list', 'planned.*.label' => 'text', 'planned.*.config' => 'string', 'planned.*.key' => 'string?',
        ];
    }

    /** Pengaturan = pasangan [label, nilai] berupa teks; kunci kelompok dan kunci butir tidak boleh berulang. */
    protected function check(array $card): void
    {
        $groups = array_column($card['groups'], 'key');
        if (count($groups) !== count(array_unique($groups)) || in_array('web', $groups, true)) {
            throw new RuntimeException('Kartu metode tidak sah: kunci kelompok berulang, atau memakai kunci "web" milik halaman.');
        }
        $seen = [];
        foreach ($card['groups'] as $group) {
            foreach ($group['methods'] as $method) {
                if (isset($seen[$method['key']])) {
                    throw new RuntimeException("Kartu metode tidak sah: kunci butir {$method['key']} berulang.");
                }
                $seen[$method['key']] = true;
                foreach ($method['settings'] as $pair) {
                    if (! is_array($pair) || ! array_is_list($pair) || count($pair) !== 2 || ! is_string($pair[0]) || ! is_string($pair[1])) {
                        throw new RuntimeException("Kartu metode tidak sah: pengaturan butir {$method['key']} harus pasangan [label, nilai] berupa teks.");
                    }
                }
            }
        }
    }

    protected function preview(Model $report): void
    {
        view('livewire.pages.metode', Metode::viewData($report))->render();
    }

    public function staleNotes(): array
    {
        $report = MethodReport::current();

        return $report ? Metode::stale($report->payload) : [];
    }

    protected function summary(array $card): array
    {
        return ['run' => $card['official']['run'], 'groups' => count($card['groups']),
            'methods' => array_sum(array_map(fn ($group) => count($group['methods']), $card['groups'])),
            'planned' => count($card['planned']), 'parameters' => $card['model']['parameters']['total']];
    }
}
