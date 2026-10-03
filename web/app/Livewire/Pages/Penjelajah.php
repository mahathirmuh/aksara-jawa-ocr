<?php

namespace App\Livewire\Pages;

use App\Models\Line;
use App\Models\LineAnnotation;
use App\Models\MachineTranslation;
use App\Models\Pipeline;
use App\Models\Prediction;
use App\Support\SpeechLevel;
use Illuminate\Database\Eloquent\Builder;
use Livewire\Attributes\Layout;
use Livewire\Attributes\Title;
use Livewire\Attributes\Url;
use Livewire\Component;
use Livewire\WithPagination;

#[Layout('components.layouts.app')]
#[Title('Penjelajah baris')]
class Penjelajah extends Component
{
    use WithPagination;

    public const TAGS = ['uji buta', 'beam membaik', 'beam memburuk', 'aksara klasik', 'adeg-adeg'];

    /** Saringan untuk pemberi label tingkat tutur. */
    public const UNLABELED = 'tutur belum dilabel';

    public const SORTS = [
        'fonts_desc' => 'CER fase5_fonts, terburuk dulu',
        'fonts_asc' => 'CER fase5_fonts, terbaik dulu',
        'gain' => 'Perbaikan beam terbesar',
        'loss' => 'Kerusakan beam terbesar',
        'name' => 'Nama berkas',
    ];

    #[Url(as: 'baris')]
    public string $line = '';

    #[Url]
    public string $tag = 'semua';

    #[Url(as: 'urut')]
    public string $sort = 'fonts_desc';

    #[Url(as: 'p')]
    public array $pipes = ['crnn_core', 'crnn_fonts', 'crnn_fonts_beam', 'vlm_zeroshot'];

    #[Url]
    public bool $latin = false;

    public function updatedTag(): void
    {
        $this->resetPage();
    }

    public function updatedSort(): void
    {
        $this->resetPage();
    }

    public function select(string $id): void
    {
        $this->line = $id;
    }

    /** Label tingkat tutur dari manusia (data uji tahap 4). null = hapus label. */
    public function setSpeechLevel(string $externalId, ?string $level): void
    {
        abort_unless($level === null || in_array($level, SpeechLevel::LEVELS, true), 422);
        $line = Line::where('external_id', $externalId)->firstOrFail();
        LineAnnotation::updateOrCreate(
            ['dataset' => $line->dataset, 'external_id' => $line->external_id],
            ['speech_level' => $level, 'labeled_by' => $level ? auth()->id() : null, 'labeled_at' => $level ? now() : null,
                'source' => LineAnnotation::where('external_id', $line->external_id)->value('source') ?? 'label manusia (web)'],
        );
        $this->line = $externalId;
    }

    public function togglePipe(string $key): void
    {
        $this->pipes = in_array($key, $this->pipes, true)
            ? (count($this->pipes) > 1 ? array_values(array_diff($this->pipes, [$key])) : $this->pipes)
            : [...$this->pipes, $key];
    }

    /** Subquery CER satu pipeline untuk baris yang sedang dipilih query utama. */
    private function cerOf(string $pipeline): Builder
    {
        return Prediction::query()->select('cer')
            ->whereColumn('predictions.line_id', 'lines.id')
            ->where('pipeline', $pipeline)
            ->limit(1);
    }

    private function lines(): Builder
    {
        $inner = Line::query()->select('lines.*')
            ->selectSub($this->cerOf('crnn_fonts'), 'fonts_cer')
            ->selectSub($this->cerOf('crnn_fonts_beam'), 'beam_cer')
            ->selectSub($this->cerOf('vlm_zeroshot'), 'vlm_cer');
        if (in_array($this->tag, self::TAGS, true)) {
            $inner->whereJsonContains('tags', $this->tag);
        } elseif ($this->tag === self::UNLABELED) {
            $inner->whereDoesntHave('annotation', fn ($q) => $q->whereNotNull('speech_level'));
        }
        // Tabel turunan: PostgreSQL tidak mengizinkan alias kolom di dalam ekspresi ORDER BY.
        $query = Line::query()->fromSub($inner, 'lines');

        return match ($this->sort) {
            'fonts_asc' => $query->orderBy('fonts_cer')->orderBy('external_id'),
            'gain' => $query->orderByRaw('(fonts_cer - beam_cer) desc')->orderBy('external_id'),
            'loss' => $query->orderByRaw('(beam_cer - fonts_cer) desc')->orderBy('external_id'),
            'name' => $query->orderBy('external_id'),
            default => $query->orderByDesc('fonts_cer')->orderBy('external_id'),
        };
    }

    public function render()
    {
        $page = $this->lines()->paginate(40);
        $current = $this->line !== '' ? Line::with('annotation.labeler')->where('external_id', $this->line)->first() : null;
        $current ??= $page->first()?->load('annotation.labeler');
        $pipelines = Pipeline::orderBy('sort')->get();
        $done = $pipelines->where('status', 'done');
        $predictions = $current
            ? $current->predictions()->whereIn('pipeline', $this->pipes)->get()->keyBy('pipeline')
            : collect();

        return view('livewire.pages.penjelajah', [
            'page' => $page,
            'current' => $current,
            'machine' => $current
                ? MachineTranslation::where('dataset', $current->dataset)->where('external_id', $current->external_id)->get()->keyBy('source')
                : collect(),
            'pipelines' => $pipelines,
            'shown' => $done->filter(fn ($p) => $predictions->has($p->key))->values(),
            'missing' => $done->filter(fn ($p) => in_array($p->key, $this->pipes, true) && ! $predictions->has($p->key))->values(),
            'predictions' => $predictions,
            'tagCounts' => collect(self::TAGS)->mapWithKeys(fn ($t) => [$t => Line::whereJsonContains('tags', $t)->count()])
                ->put(self::UNLABELED, Line::whereDoesntHave('annotation', fn ($q) => $q->whereNotNull('speech_level'))->count()),
            'total' => Line::count(),
        ]);
    }
}
