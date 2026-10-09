<?php

namespace App\Livewire\Pages;

use App\Models\ClassMetric;
use App\Models\Confusion;
use App\Models\Metric;
use App\Models\Pipeline;
use Livewire\Attributes\Layout;
use Livewire\Attributes\Title;
use Livewire\Component;

#[Layout('components.layouts.app')]
#[Title('Kesalahan aksara')]
class Kesalahan extends Component
{
    /** Cakupan metrik per aksara yang ditampilkan (manifest class_metrics.scope). */
    public const SCOPE = 'nusaaksara_745';

    /** Aksara dengan kemunculan di label lebih sedikit dari ini tidak masuk daftar terlemah: recall-nya terlalu kasar. */
    public const MIN_REF = 10;

    /** Panjang daftar aksara terlemah (F1 terendah). */
    public const WEAKEST = 20;

    public function render()
    {
        $limits = ['sub' => 14, 'del' => 10, 'ins' => 10];
        $pipeline = Pipeline::official();
        $classes = ClassMetric::where('scope', self::SCOPE)->where('pipeline', $pipeline?->key ?? '')->get();
        // Macro-F1 = rata-rata F1 aksara yang ada di label; F1 null (tidak pernah dikeluarkan dan tidak ada di label)
        // tidak ikut. Mikro (presisi, recall, F1 seluruh karakter) dari tabel metrics, pipeline yang sama.
        $scored = $classes->filter(fn ($c) => $c->ref > 0 && $c->f1 !== null);

        return view('livewire.pages.kesalahan', [
            'lists' => collect($limits)->map(fn ($n, $kind) => Confusion::where('kind', $kind)->orderByDesc('count')->limit($n)->get()),
            'totals' => Confusion::selectRaw('kind, sum(count) as total')->groupBy('kind')->pluck('total', 'kind'),
            // Ekspor menghitung kesalahan aksara dari pipeline resmi (manifest: confusion.pipeline = official).
            'pipeline' => $pipeline,
            'micro' => Metric::where('scope', self::SCOPE)->where('pipeline', $pipeline?->key ?? '')->first(),
            'macroF1' => $scored->isEmpty() ? null : $scored->avg('f1'),
            'classCount' => $scored->count(),
            'weakest' => $classes->filter(fn ($c) => $c->ref >= self::MIN_REF && $c->f1 !== null)
                ->sortBy([['f1', 'asc'], ['ref', 'desc'], ['code', 'asc']])->take(self::WEAKEST)->values(),
            'hasClasses' => $classes->isNotEmpty(),
        ]);
    }
}
