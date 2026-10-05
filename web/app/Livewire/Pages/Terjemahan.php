<?php

namespace App\Livewire\Pages;

use App\Services\TranslationService;
use App\Support\AksaraCatalog;
use App\Support\AksaraWriter;
use App\Support\DictionarySearch;
use App\Support\SpeechLevel;
use App\Support\TalingRestorer;
use App\Support\Transliterator;
use Livewire\Attributes\Layout;
use Livewire\Attributes\Title;
use Livewire\Attributes\Url;
use Livewire\Component;
use RuntimeException;

/**
 * Terjemahan dua arah antara bahasa Indonesia dan bahasa Jawa beraksara Jawa.
 *
 * Dua pekerjaan yang berbeda, sengaja dipisah supaya salahnya bisa dilacak:
 *  - menerjemahkan (Indonesia <-> Jawa beraksara Latin): model NLLB lokal lewat layanan tools/translate_service.py;
 *  - mengganti sistem tulisan (Latin <-> aksara Jawa): aturan pasti di PHP (AksaraWriter, Transliterator), tanpa layanan.
 * Indonesia -> Jawa: terjemahan Latin dipulihkan tanda é-nya (TalingRestorer), boleh disunting, lalu aksaranya dihitung
 * dari teks Latin yang terlihat itu. Jawa -> Indonesia: aksara dilatinkan dulu (draf), lalu diterjemahkan.
 * Semua hasil di sini draf mesin, bukan bagian dari angka OCR.
 */
#[Layout('components.layouts.app')]
#[Title('Terjemahan')]
class Terjemahan extends Component
{
    /** Arah => [bahasa asal, bahasa hasil]. */
    public const DIRECTIONS = [
        'id-jv' => ['Bahasa Indonesia', 'Bahasa Jawa beraksara Jawa'],
        'jv-id' => ['Bahasa Jawa (aksara atau Latin)', 'Bahasa Indonesia'],
    ];

    public const MAX_CHARS = 1000;

    public const MAX_SENTENCES = 12;

    /** Di atas panjang ini model memotong masukan (128 token), jadi terjemahannya bisa tidak lengkap. */
    public const LONG_SENTENCE = 350;

    #[Url(as: 'arah', except: 'id-jv')]
    public string $direction = 'id-jv';

    public string $text = '';

    /** Bahasa Jawa beraksara Latin: hasil terjemahan Indonesia -> Jawa, boleh disunting; aksaranya dihitung dari sini. */
    public string $javanese = '';

    public string $indonesian = '';

    /** Kata yang tanda é/è-nya dipulihkan pada langkah terakhir (asal => hasil). */
    public array $restored = [];

    /** Keterangan terjemahan terakhir: jumlah kalimat, waktu, model, kalimat panjang. */
    public ?array $meta = null;

    public ?string $error = null;

    public ?array $service = null;

    public function mount(TranslationService $mt): void
    {
        $this->direction = isset(self::DIRECTIONS[$this->direction]) ? $this->direction : 'id-jv';
        // ?q= mengisi kotak masukan dan ?jawa= kotak bahasa Jawa Latin (tautan dari halaman lain); tidak disinkronkan balik ke alamat.
        $this->text = mb_substr(trim((string) request()->query('q', '')), 0, self::MAX_CHARS);
        $this->javanese = $this->direction === 'id-jv' ? mb_substr(trim((string) request()->query('jawa', '')), 0, 2 * self::MAX_CHARS) : '';
        $this->service = $mt->health();
    }

    public function checkService(TranslationService $mt): void
    {
        $this->service = $mt->health();
    }

    public function updatedDirection(): void
    {
        $this->direction = isset(self::DIRECTIONS[$this->direction]) ? $this->direction : 'id-jv';
        $this->reset('javanese', 'indonesian', 'restored', 'meta', 'error');
    }

    /** Tukar arah; hasil yang ada menjadi masukan arah sebaliknya. */
    public function swap(): void
    {
        $output = $this->direction === 'id-jv' ? AksaraWriter::fromLatin($this->javanese) : $this->indonesian;
        $this->direction = $this->direction === 'id-jv' ? 'jv-id' : 'id-jv';
        $this->text = trim($output) !== '' ? $output : $this->text;
        $this->reset('javanese', 'indonesian', 'restored', 'meta', 'error');
    }

    public function clear(): void
    {
        $this->reset('text', 'javanese', 'indonesian', 'restored', 'meta', 'error');
    }

    /** Pulihkan tanda é/è pada bahasa Jawa yang diketik sendiri di kotak Latin. */
    public function restoreTaling(): void
    {
        ['text' => $this->javanese, 'changed' => $this->restored] = TalingRestorer::restore($this->javanese);
    }

    public function translate(TranslationService $mt): void
    {
        $this->reset('restored', 'meta', 'error');
        $this->validate(['text' => 'required|string|max:'.self::MAX_CHARS], attributes: ['text' => 'teks']);
        [$source, $target] = explode('-', $this->direction);
        // Bahasa Jawa beraksara: dilatinkan dulu; model terjemahan hanya mengenal bahasa Jawa beraksara Latin.
        $input = $source === 'jv' && AksaraCatalog::hasJavanese($this->text) ? Transliterator::toLatin($this->text) : $this->text;
        $sentences = self::sentences($input);
        if ($sentences === []) {
            $this->error = 'Tidak ada kalimat untuk diterjemahkan.';

            return;
        }
        if (count($sentences) > self::MAX_SENTENCES) {
            $this->error = 'Teks berisi '.count($sentences).' kalimat; paling banyak '.self::MAX_SENTENCES.' sekali jalan (model lokal butuh 1-3 detik per kalimat).';

            return;
        }
        try {
            $result = $mt->translateMany(array_column($sentences, 'text'), $source, $target);
        } catch (RuntimeException $e) {
            $this->error = $e->getMessage();

            return;
        }
        if ($result === null) {
            $this->service = null;
            $this->error = 'Layanan terjemahan belum berjalan.';

            return;
        }
        $this->service ??= $mt->health();
        $output = self::join($sentences, $result['translations']);
        if ($target === 'jv') {
            ['text' => $this->javanese, 'changed' => $this->restored] = TalingRestorer::restore($output);
        } else {
            $this->indonesian = $output;
        }
        $this->meta = [
            'sentences' => count($sentences),
            'elapsed_ms' => (float) ($result['elapsed_ms'] ?? 0),
            'model' => (string) ($result['model'] ?? ''),
            'long' => count(array_filter($sentences, fn (array $s) => mb_strlen($s['text']) > self::LONG_SENTENCE)),
        ];
    }

    /**
     * Pecah teks menjadi kalimat (model menerjemahkan per kalimat); `break` = kalimat itu mengakhiri baris.
     *
     * @return list<array{text: string, break: bool}>
     */
    public static function sentences(string $text): array
    {
        $out = [];
        foreach (preg_split('/\R/u', $text) ?: [] as $line) {
            $parts = array_values(array_filter(array_map('trim', preg_split('/(?<=[.!?…])\s+/u', $line) ?: []), 'strlen'));
            foreach ($parts as $i => $part) {
                $out[] = ['text' => $part, 'break' => $i === count($parts) - 1];
            }
        }

        return $out;
    }

    /** Susun kembali terjemahan per kalimat dengan pemisah baris seperti teks asalnya. */
    private static function join(array $sentences, array $translations): string
    {
        $out = '';
        foreach ($sentences as $i => $sentence) {
            $out .= trim((string) ($translations[$i] ?? '')).($sentence['break'] ? "\n" : ' ');
        }

        return trim($out);
    }

    public function render()
    {
        $toJavanese = $this->direction === 'id-jv';
        $inputIsAksara = ! $toJavanese && AksaraCatalog::hasJavanese($this->text);
        // Teks Jawa beraksara Latin yang sedang dikerjakan: hasil terjemahan (id-jv) atau masukan (jv-id).
        $latin = $toJavanese ? $this->javanese : ($inputIsAksara ? Transliterator::toLatin($this->text) : $this->text);
        $hasLatin = trim($latin) !== '';

        return view('livewire.pages.terjemahan', [
            'directions' => self::DIRECTIONS,
            'toJavanese' => $toJavanese,
            'inputIsAksara' => $inputIsAksara,
            'latinDraft' => $inputIsAksara ? $latin : null,
            'aksara' => $toJavanese && $hasLatin ? AksaraWriter::fromLatin($this->javanese) : '',
            'speech' => $hasLatin ? SpeechLevel::classify($latin) : null,
            // Arti per kata di halaman Kamus: kata-kata pertama teks Jawa.
            'kamusQuery' => $hasLatin ? implode(' ', DictionarySearch::words($latin)) : '',
            'lexicon' => TalingRestorer::meta(),
            'agreement' => AksaraWriter::DICTIONARY_AGREEMENT,
        ]);
    }
}
