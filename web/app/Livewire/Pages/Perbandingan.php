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
    /**
     * Run lanjutan (kunci pipeline => catatan di tabel 745 baris), urut seperti OPTIONAL_CHECKPOINTS di
     * scripts/export_results.py. Satu run per kondisi dan diukur lagi pada test set yang sama: pembanding, bukan
     * angka resmi (angka G3 resmi tetap crnn_fonts). Ringkasan memakai kuncinya untuk titik "Jarak G3 ke target".
     */
    public const FOLLOWUP_RUNS = [
        'crnn_fase6_rare' => 'fase5_fonts + 1.500 langkah dengan sisipan aksara langka',
        'crnn_fase6_ctrl' => 'pembanding: langkah dan titik lanjut sama, tanpa aksara langka',
        'crnn_fase7_track' => 'fase6_ctrl + 1.500 langkah dengan jarak antar-aksara acak',
        'crnn_fase7_track_rare' => 'sama dengan fase7_track + sisipan aksara langka yang disaring',
        'crnn_fase7_ctrl' => 'pembanding: langkah sama tanpa jarak tambahan',
    ];

    public function render()
    {
        $pipelines = Pipeline::orderBy('sort')->get()->keyBy('key');
        // toBase(): only()/keyBy pada Eloquent Collection bekerja dengan primary key model, bukan kunci array.
        $metrics = Metric::whereNotNull('pipeline')->get()->groupBy('scope')->toBase();
        $order = fn ($rows) => $rows->sortBy(fn ($m) => $pipelines[$m->pipeline]->sort ?? 99)->values();
        $full = $order($metrics->get('nusaaksara_745', collect()));
        $blind = $order($metrics->get('blind_50', collect()));

        return view('livewire.pages.perbandingan', [
            'hasData' => ResultImport::exists(),
            'pipelines' => $pipelines,
            'full' => $full,
            'runNotes' => self::FOLLOWUP_RUNS,
            // Peringatan "satu run per kondisi" hanya menyebut run lanjutan yang ada di tabel: bisa sebagian saja,
            // mis. run fase7 berikutnya belum selesai dilatih sehingga belum diekspor.
            'followups' => $full->filter(fn ($m) => isset(self::FOLLOWUP_RUNS[$m->pipeline]))
                ->map(fn ($m) => $pipelines[$m->pipeline]->label ?? $m->pipeline)->values(),
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
