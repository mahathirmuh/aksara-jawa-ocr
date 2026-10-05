<?php

namespace App\Livewire\Pages;

use App\Models\DictionaryEntry;
use App\Models\DictionarySource;
use App\Models\Line;
use App\Models\Metric;
use App\Models\Pipeline;
use App\Support\AksaraCatalog;
use App\Support\DictionarySearch;
use App\Support\SpeechLevel;
use App\Support\Transliterator;
use Livewire\Attributes\Layout;
use Livewire\Attributes\Locked;
use Livewire\Attributes\Title;
use Livewire\Attributes\Url;
use Livewire\Component;
use Normalizer;

/**
 * Semua "kamus" alur dalam satu halaman rujukan:
 * dua kamus kata (bahasa Jawa dan bahasa Indonesia, data pihak ketiga yang diimpor `aksara:dictionary`),
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

    /** Kata contoh di kamus kata sebelum ada pencarian; hanya yang memang ada di data yang ditampilkan. */
    private const EXAMPLES = [
        'jv' => ['omah', 'griya', 'banyu', 'toya', 'mangan', 'nedha', 'dhahar', 'kula', 'aksara', 'sinau'],
        'id' => ['rumah', 'air', 'makan', 'aksara', 'bahasa', 'kamus', 'baca', 'tulis'],
    ];

    private const PAGE = 20;

    private const MAX_SHOWN = 200;

    #[Url(as: 'q', except: '')]
    public string $q = '';

    /** Berapa entri yang ditampilkan per kamus kata; hanya berubah lewat more(). */
    #[Locked]
    public array $limits = ['jv' => self::PAGE, 'id' => self::PAGE];

    public function updatedQ(): void
    {
        $this->limits = ['jv' => self::PAGE, 'id' => self::PAGE];
    }

    public function more(string $dictionary): void
    {
        if (isset(DictionaryEntry::DICTIONARIES[$dictionary])) {
            $this->limits[$dictionary] = min(self::MAX_SHOWN, $this->limits[$dictionary] + self::PAGE);
        }
    }

    public function render()
    {
        $catalog = AksaraCatalog::load();
        $counts = AksaraCatalog::countIn(Line::pluck('reference'));
        $query = trim($this->q);
        $needle = mb_strtolower($query);
        $aksaraQuery = AksaraCatalog::hasJavanese($query);
        $latin = $aksaraQuery ? Transliterator::toLatin($query) : $query;

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
            'probe' => $query === '' ? null : $this->probe($query, $latin, $aksaraQuery, collect($catalog['entries'])->keyBy('char')),
            'dictionaries' => $this->dictionaries($query, $latin, $aksaraQuery),
            'lm' => $this->languageModels(),
            'official' => Pipeline::official(),
        ]);
    }

    /**
     * Uraian teks yang dicari: aksara diurai per codepoint dan ditransliterasikan (draf); teks Latin dicocokkan
     * dengan leksikon tingkat tutur.
     */
    private function probe(string $query, string $latin, bool $aksara, $byChar): array
    {
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
     * Keadaan dua kamus kata untuk kueri sekarang. Satu kata: daftar hasil berperingkat + pencarian balik di arti.
     * Beberapa kata (mis. baris transliterasi yang ditempel): arti per kata, didahului frasa utuhnya bila ada.
     * Kueri beraksara dicari lewat bacaan Latin drafnya, dan di kamus Jawa juga lewat ejaan aksaranya.
     */
    private function dictionaries(string $query, string $latin, bool $aksara): array
    {
        $sources = DictionarySource::orderBy('id')->get()->groupBy('dictionary');
        $words = DictionarySearch::words($latin);
        $out = [];
        foreach (DictionaryEntry::DICTIONARIES as $key => $label) {
            $mine = $sources->get($key, collect());
            $state = ['label' => $label, 'sources' => $mine, 'entries' => (int) $mine->sum('entries'), 'mode' => 'idle',
                'limit' => $this->limits[$key] ?? self::PAGE, 'max' => self::MAX_SHOWN];
            // Bahasa arti yang ada di kamus ini: menentukan keterangan di anak judul dan perlu tidaknya tanda per entri.
            $state['languages'] = $state['entries'] === 0 ? []
                : DictionaryEntry::where('dictionary', $key)->distinct()->orderBy('gloss_lang')->pluck('gloss_lang')->all();
            if ($state['entries'] === 0) {
                $state['mode'] = 'missing';
            } elseif ($query === '') {
                $found = DictionaryEntry::where('dictionary', $key)->whereIn('lookup', self::EXAMPLES[$key])->distinct()->pluck('lookup')->all();
                $state['examples'] = array_values(array_intersect(self::EXAMPLES[$key], $found));
            } elseif (count($words) > 1) {
                $state['mode'] = 'words';
                $state['phrase'] = DictionarySearch::gloss($key, [implode(' ', $words)])[0]['entries'];
                $state['rows'] = DictionarySearch::gloss($key, $words);
                $state['found'] = collect($state['rows'])->filter(fn (array $row) => $row['entries']->isNotEmpty())->count() + ($state['phrase']->isNotEmpty() ? 1 : 0);
            } else {
                $state['mode'] = 'search';
                $state['term'] = $words[0] ?? '';
                $spelling = $aksara && $key === 'jv' ? (Normalizer::normalize($query, Normalizer::FORM_C) ?: $query) : null;
                $state += DictionarySearch::search($key, $state['term'], $state['limit'], $spelling);
                $state['found'] = $state['total'] + $state['reverse_total'];
            }
            $out[$key] = $state;
        }

        return $out;
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
