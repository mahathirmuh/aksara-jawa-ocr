<?php

namespace App\Livewire\Pages;

use App\Models\Metric;
use App\Models\Pipeline;
use App\Models\ResultImport;
use App\Models\TranslationRun;
use Livewire\Attributes\Layout;
use Livewire\Attributes\Title;
use Livewire\Component;

#[Layout('components.layouts.app')]
#[Title('Perbandingan')]
class Perbandingan extends Component
{
    public function render()
    {
        $pipelines = Pipeline::orderBy('sort')->get()->keyBy('key');
        // toBase(): only()/keyBy pada Eloquent Collection bekerja dengan primary key model, bukan kunci array.
        $metrics = Metric::whereNotNull('pipeline')->get()->groupBy('scope')->toBase();
        $order = fn ($rows) => $rows->sortBy(fn ($m) => $pipelines[$m->pipeline]->sort ?? 99)->values();
        $blind = $order($metrics->get('blind_50', collect()));

        return view('livewire.pages.perbandingan', [
            'hasData' => ResultImport::exists(),
            'pipelines' => $pipelines,
            'full' => $order($metrics->get('nusaaksara_745', collect())),
            'blind' => $blind,
            'blindChart' => [
                'labels' => $blind->map(fn ($m) => $pipelines[$m->pipeline]->label ?? $m->pipeline)->all(),
                'values' => $blind->map(fn ($m) => round($m->cer, 5))->all(),
                'configs' => $blind->map(fn ($m) => $pipelines[$m->pipeline]->config ?? '')->all(),
            ],
            'synth' => $metrics->only(['synth_dev', 'synth_heldout'])->map(fn ($rows) => $rows->keyBy('pipeline')),
            'translit' => Metric::where('scope', 'translit_draft')->first(),
            'mtRuns' => TranslationRun::latest('id')->get()->unique('source')->values(),
        ]);
    }
}
