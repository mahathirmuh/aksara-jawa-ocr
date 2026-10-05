<?php

namespace App\Livewire\Pages;

use App\Models\Line;
use App\Models\Metric;
use App\Models\Pipeline;
use App\Support\AksaraCatalog;
use App\Support\SpeechLevel;
use App\Support\Transliterator;
use Livewire\Attributes\Layout;
use Livewire\Attributes\Title;
use Livewire\Attributes\Url;
use Livewire\Component;

/**
 * Tiga "kamus" yang dipakai alur, dalam satu halaman rujukan:
 * aksara yang dibaca model (charset tokenizer), kamus koreksi OCR (model bahasa tingkat karakter untuk beam search),
 * dan leksikon penanda tingkat tutur (tahap 4). Tidak ada angka yang dihitung di sini selain hitungan kemunculan
 * aksara di label uji; CER berasal dari hasil ekspor yang sudah diimpor.
 */
#[Layout('components.layouts.app')]
#[Title('Kamus')]
class Kamus extends Component
{
    /** Cakupan metrik yang membandingkan greedy dengan beam + LM, dari data nyata ke sintetis. */
    public const SCOPES = [
        'nusaaksara_745' => 'Baris cetak nyata (NusaAksara)',
        'blind_50' => 'Uji buta',
        'synth_dev' => 'Dev sintetis (aug fase5)',
        'synth_heldout' => 'Font held-out bersih',
    ];

    #[Url(as: 'q', except: '')]
    public string $q = '';

    public function render()
    {
        $catalog = AksaraCatalog::load();
        $counts = AksaraCatalog::countIn(Line::pluck('reference'));
        $query = trim($this->q);
        $needle = mb_strtolower($query);
        $aksaraQuery = AksaraCatalog::hasJavanese($query);

        $entries = collect($catalog['entries'])
            ->map(fn (array $e) => $e + ['count' => $counts[$e['char']] ?? 0])
            ->filter(fn (array $e) => $needle === '' || ($aksaraQuery
                ? mb_strpos($query, $e['char']) !== false
                : str_contains($e['name'], $needle) || $e['latin'] === $needle || str_contains(mb_strtolower($e['code']), $needle)));

        $lexicon = collect(SpeechLevel::lexicon())
            ->map(fn (array $words) => collect($words)->filter(fn (string $w) => $needle === '' || (! $aksaraQuery && str_contains($w, $needle)))->values());

        return view('livewire.pages.kamus', [
            'catalog' => $catalog,
            'kinds' => AksaraCatalog::KINDS,
            'groups' => $entries->groupBy('kind'),
            'shown' => $entries->count(),
            'total' => count($catalog['entries']),
            'lines' => Line::count(),
            // Alasan kamus koreksi berupa model bahasa karakter, bukan daftar kata: label tanpa spasi tidak punya batas kata.
            'unspaced' => Line::whereJsonContains('tags', 'label tanpa spasi')->count(),
            'lexicon' => $lexicon,
            'lexiconTotal' => collect(SpeechLevel::lexicon())->map(fn (array $words) => count($words)),
            'probe' => $query === '' ? null : $this->probe($query, $aksaraQuery, collect($catalog['entries'])->keyBy('char')),
            'lm' => $this->languageModels(),
            'official' => Pipeline::official(),
        ]);
    }

    /**
     * Uraian teks yang dicari: aksara diurai per codepoint dan ditransliterasikan (draf); teks Latin dicocokkan
     * dengan leksikon tingkat tutur.
     */
    private function probe(string $query, bool $aksara, $byChar): array
    {
        $latin = $aksara ? Transliterator::toLatin($query) : $query;

        return [
            'aksara' => $aksara,
            'latin' => $latin,
            'chars' => $aksara
                ? collect(mb_str_split($query))->map(fn (string $ch) => $byChar->get($ch) ?? ['char' => $ch, 'name' => aksara_name($ch), 'code' => sprintf('U+%04X', mb_ord($ch))])->all()
                : [],
            'speech' => SpeechLevel::classify($latin),
        ];
    }

    /**
     * Pipeline beam + LM ("<dasar>_beam") dengan pembanding greedy-nya per cakupan. Hanya cakupan yang punya
     * kedua angka yang ditampilkan.
     */
    private function languageModels()
    {
        $pipelines = Pipeline::orderBy('sort')->get()->keyBy('key');
        $metrics = Metric::whereIn('scope', array_keys(self::SCOPES))->get()->groupBy('pipeline')->toBase();

        return $pipelines
            ->filter(fn (Pipeline $p) => str_ends_with($p->key, '_beam') && $p->isDone())
            ->map(function (Pipeline $beam) use ($pipelines, $metrics) {
                $base = $pipelines->get(substr($beam->key, 0, -strlen('_beam')));
                $rows = [];
                foreach (self::SCOPES as $scope => $label) {
                    $greedy = $base ? $metrics->get($base->key)?->firstWhere('scope', $scope) : null;
                    $withLm = $metrics->get($beam->key)?->firstWhere('scope', $scope);
                    if ($greedy && $withLm) {
                        $rows[] = [
                            'label' => $label,
                            'lines' => $withLm->lines,
                            'greedy' => $greedy->cer,
                            'beam' => $withLm->cer,
                            'better' => $withLm->better,
                            'worse' => $withLm->worse,
                        ];
                    }
                }

                return ['beam' => $beam, 'base' => $base, 'rows' => $rows];
            })
            ->filter(fn (array $lm) => $lm['rows'] !== [])
            ->values();
    }
}
