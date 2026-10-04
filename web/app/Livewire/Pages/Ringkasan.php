<?php

namespace App\Livewire\Pages;

use App\Models\Gate;
use App\Models\Line;
use App\Models\LineAnnotation;
use App\Models\Metric;
use App\Models\Pipeline;
use App\Models\ResultImport;
use App\Models\TranslationRun;
use App\Support\SpeechLevel;
use Livewire\Attributes\Layout;
use Livewire\Attributes\Title;
use Livewire\Component;

#[Layout('components.layouts.app')]
#[Title('Ringkasan')]
class Ringkasan extends Component
{
    public function render()
    {
        // toBase(): only() pada Eloquent Collection menyaring primary key model, bukan kunci "pipeline".
        $g3 = Metric::where('scope', 'nusaaksara_745')->get()->keyBy('pipeline')->toBase();
        $labels = Pipeline::pluck('label', 'key');
        // Nilai bisa berdekatan (fonts, beam, fase6 di 32–38%) dan lebar sumbu ikut layar, jadi label tidak ditaruh
        // di samping setiap titik: hanya titik resmi yang berlabel, semua nilai ada di daftar urut CER (terbaik dulu,
        // sama dengan arah sumbu). x = posisi di sumbu, dipotong ke 0–100% (CRNN 4b > 100%). Run lanjutan (fase6,
        // fase7) ikut bila sudah diekspor; daftarnya satu sumber dengan catatan di halaman Perbandingan.
        $points = collect(['crnn_4b', 'crnn_core', 'crnn_fonts', 'crnn_fonts_beam'])
            ->merge(array_keys(Perbandingan::FOLLOWUP_RUNS))
            ->filter(fn ($k) => $g3->has($k))
            ->map(fn ($k) => ['key' => $k, 'label' => $labels[$k] ?? $k, 'value' => $g3[$k]->cer,
                'x' => round(min(100, max(0, $g3[$k]->cer * 100)), 2), 'official' => $k === 'crnn_fonts'])
            ->sortBy('value')
            ->values();

        $lines = Line::count();

        // Tahap 4: prediksi leksikon dari transliterasi manusia, dan akurasinya terhadap label manusia.
        $annotations = LineAnnotation::whereIn('external_id', Line::select('external_id'))
            ->whereNotNull('transliteration')->get(['external_id', 'transliteration', 'speech_level']);
        $auto = $annotations->map(fn ($a) => SpeechLevel::classify($a->transliteration)['level'] ?? 'tak tentu');
        // Per baris rata-rata hanya ~1 kata penanda; per halaman (source_id) penandanya cukup banyak.
        $pageOf = Line::pluck('source_id', 'external_id');
        $perPage = $annotations->groupBy(fn ($a) => $pageOf[$a->external_id] ?? '?')
            ->map(fn ($rows) => SpeechLevel::classify($rows->pluck('transliteration')->join(' '))['level'] ?? 'tak tentu');
        $speechEval = SpeechLevel::evaluate($annotations->whereNotNull('speech_level')
            ->map(fn ($a) => [$a->speech_level, SpeechLevel::classify($a->transliteration)['level']])->values()->all());

        return view('livewire.pages.ringkasan', [
            'import' => ResultImport::latest('id')->first(),
            'gates' => Gate::orderBy('code')->get(),
            'points' => $points,
            'best' => $g3->only(['crnn_fonts', 'crnn_fonts_beam']),
            'translit' => Metric::where('scope', 'translit_draft')->first(),
            'lines' => $lines,
            'humanTranslit' => LineAnnotation::whereNotNull('transliteration')->count(),
            'humanTranslation' => LineAnnotation::whereNotNull('translation')->count(),
            'mtRuns' => TranslationRun::latest('id')->get()->unique('source')->keyBy('source'),
            'speechAuto' => $auto->countBy()->sortDesc(),
            'speechPages' => $perPage->countBy()->sortDesc(),
            'speechEval' => $speechEval,
            'phases' => config('aksara.phases'),
        ]);
    }
}
