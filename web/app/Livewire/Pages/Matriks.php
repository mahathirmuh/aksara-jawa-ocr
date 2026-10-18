<?php

namespace App\Livewire\Pages;

use App\Models\MethodMatrix;
use Illuminate\Support\Facades\Schema;
use Livewire\Attributes\Layout;
use Livewire\Attributes\Title;
use Livewire\Component;

/**
 * Matriks metode: metode kita lawan metode OCR lain pada baris nyata yang sama, ablasi komponen, dan pembanding
 * arsitektur. Semua angka dari out/results/matrix.json (scripts/method_matrix.py); halaman hanya menampilkan.
 */
#[Layout('components.layouts.app')]
#[Title('Matriks metode')]
class Matriks extends Component
{
    public const GROUP_LABELS = ['kita' => 'Metode kita', 'vlm' => 'VLM zero-shot', 'umum' => 'OCR umum'];

    public const GROUP_STATUS = ['kita' => 'status-blue', 'vlm' => 'status-yellow', 'umum' => 'status-gray'];

    public const ABLATION_GROUPS = ['data' => 'Data latih', 'baca' => 'Cara membaca', 'rujukan' => 'Rujukan'];

    public static function viewData(?array $m): array
    {
        if ($m === null) {
            return ['matrix' => null];
        }
        $scored = array_values(array_filter($m['methods'], fn ($r) => $r['cetak'] !== null));

        return [
            'matrix' => $m,
            'methods' => $m['methods'],
            'chart' => [
                'labels' => array_map(fn ($r) => $r['label'], $scored),
                'values' => array_map(fn ($r) => round($r['cetak']['ned'], 5), $scored),
                'configs' => array_map(fn ($r) => self::GROUP_LABELS[$r['group']], $scored),
                // NED selalu 0-100%; sumbu tetap dibulatkan ke atas per 20 poin bila nilai lain dipakai.
                'max' => max(1.0, ceil(max([0, ...array_map(fn ($r) => $r['cetak']['ned'], $scored)]) * 5) / 5),
            ],
            'ablation' => collect($m['ablation'])->groupBy('group')->all(),
            'stages' => $m['architecture_stages'] ?? [],
        ];
    }

    public function render()
    {
        $matrix = Schema::hasTable('method_matrices') ? MethodMatrix::current() : null;

        return view('livewire.pages.matriks', self::viewData($matrix?->payload));
    }
}
