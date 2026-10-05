<?php

namespace App\Livewire\Pages;

use App\Models\LineAnnotation;
use App\Models\MethodReport;
use App\Models\Metric;
use App\Models\Pipeline;
use App\Models\ResultImport;
use App\Models\TranslationRun;
use App\Support\AksaraWriter;
use App\Support\SpeechLevel;
use App\Support\TalingRestorer;
use Illuminate\Support\Collection;
use Illuminate\Support\HtmlString;
use Livewire\Attributes\Layout;
use Livewire\Attributes\Title;
use Livewire\Component;

/**
 * Halaman Metode: daftar metode dan machine learning yang dipakai, per tahap, dengan pengaturan dan, bila ada, bukti
 * terukur serta catatan batasannya.
 *
 * Butir milik repo OCR (model, data sintetis, pelatihan, decoding, evaluasi) datang utuh dari kartu metode buatan
 * Python (scripts/export_methods.py, diimpor `php artisan aksara:methods`): penjelasan, pengaturan, bukti, dan
 * catatannya tidak ditulis ulang di sini. Yang disusun di sini dari tabel dan konstanta web sendiri: butir tahap
 * lanjutan alur (alih aksara, terjemahan, tingkat tutur), empat ubin ringkas, dan gambar alurnya.
 */
#[Layout('components.layouts.app')]
#[Title('Metode')]
class Metode extends Component
{
    /** Jenis butir. */
    public const KINDS = [
        'model' => 'model deep learning',
        'training' => 'teknik pelatihan',
        'data' => 'teknik data',
        'statistic' => 'statistik',
        'rule' => 'aturan atau algoritme',
        'evaluation' => 'evaluasi',
    ];

    /**
     * Status butir: label dan kelas pil. "official" = bagian dari model resmi (arsitektur, data, pelatihan, cara baca);
     * "used" = dipakai untuk mengukur atau di tahap web, yang tidak punya jalur angka resmi.
     */
    public const STATUSES = [
        'official' => ['dipakai model resmi', 'status-blue'],
        'used' => ['dipakai', 'status-blue'],
        'available' => ['tersedia, bukan jalur resmi', 'status-gray'],
        'tested' => ['diuji, tidak dipakai', 'status-yellow'],
        'comparator' => ['pembanding', 'status-gray'],
    ];

    /** Status yang berarti butir itu benar-benar dipakai (bukan sekadar tersedia, diuji, atau pembanding). */
    private const IN_USE = ['official', 'used'];

    private const NLLB = 'facebook/nllb-200-distilled-600M';

    public function render()
    {
        return view('livewire.pages.metode', self::viewData(MethodReport::current()));
    }

    /**
     * Data tampilan untuk sebuah kartu. Dipakai halaman ini dan impor kartu (MethodImporter merendernya sekali
     * sebelum menyimpan, supaya kartu yang tidak bisa ditampilkan ditolak saat impor).
     */
    public static function viewData(?MethodReport $report): array
    {
        if (! $report) {
            return ['report' => null];
        }
        $card = $report->payload;
        $groups = array_merge($card['groups'], [self::webGroup()]);

        return [
            'report' => $report,
            'card' => $card,
            'stale' => self::stale($card),
            'tiles' => self::tiles($card['model'], collect($groups)),
            'flow' => self::flow($card['model']),
            'groups' => $groups,
            'planned' => $card['planned'],
        ];
    }

    /**
     * Peringatan bila kartu bukan untuk pipeline resmi web, atau buktinya dari ekspor hasil yang lain. Kalimatnya
     * tidak menebak sisi mana yang tertinggal: kartu bisa lebih baru daripada hasil yang diimpor, atau sebaliknya.
     * Kartu untuk run yang tidak dikenal ekspor hasil (kunci pipeline kosong) juga tidak cocok dengan pipeline resmi.
     * Dipakai halaman ini dan `aksara:import` (yang memperingatkan langsung sesudah mengimpor kartu yang tidak cocok).
     *
     * @return list<string>
     */
    public static function stale(array $card): array
    {
        $notes = [];
        $official = Pipeline::official();
        $key = $card['official']['pipeline'] ?? null;
        if ($official && $official->key !== $key) {
            $notes[] = 'Kartu metode ini dibuat untuk run '.$card['official']['run'].', sedangkan angka resmi web sekarang milik '
                .$official->label.'.';
        }
        $results = ResultImport::latest('id')->first();
        if ($results && ($card['results_generated'] ?? null) && $results->generated_at !== $card['results_generated']) {
            $notes[] = 'Bukti di halaman ini dibaca dari ekspor hasil '.$card['results_generated']
                .', sedangkan hasil yang diimpor web berasal dari ekspor '.$results->generated_at.'.';
        }

        return $notes;
    }

    /** Empat ubin: arsitektur, ukuran model, banyaknya butir per kelompok, dan model hasil belajar yang dipakai. */
    private static function tiles(array $model, Collection $groups): array
    {
        $parameters = $model['parameters'];
        $methods = $groups->flatMap(fn ($group) => $group['methods'])->keyBy('key');
        // Model hasil belajar: pembaca aksara (satu model, tiga butir: CNN, LSTM, lapisan keluaran), model terjemahan,
        // dan model bahasa n-gram. Yang berstatus "tersedia" disebut terpisah, tidak dihitung sebagai dipakai.
        $models = array_filter([
            'cnn' => 'CRNN (dilatih dari nol)',
            'nllb' => 'NLLB-200 (pralatih, tidak dilatih ulang)',
            'beam_lm' => 'model bahasa n-gram karakter (statistik)',
        ], fn ($key) => $methods->has($key), ARRAY_FILTER_USE_KEY);
        $used = array_filter($models, fn ($key) => in_array($methods[$key]['status'], self::IN_USE, true), ARRAY_FILTER_USE_KEY);
        $other = array_diff_key($models, $used);
        $direction = ($model['bidirectional'] ?? true) ? 'BiLSTM' : 'LSTM';

        return [
            ['icon' => 'ti-brain', 'title' => 'Model pembaca', 'value' => 'CRNN + CTC', 'note' => '',
                'hint' => 'CNN '.$model['conv_layers'].' lapis → '.$direction.' '.$model['lstm_layers'].' lapis → '.$model['classes'].' kelas, dilatih dari nol'],
            ['icon' => 'ti-adjustments-horizontal', 'title' => 'Parameter', 'value' => nfmt($parameters['total']), 'note' => '',
                'hint' => 'CNN '.self::compact($parameters['cnn'] + $parameters['proj']).' · '.$direction.' '.self::compact($parameters['rnn'])
                    .' · keluaran '.self::compact($parameters['head'])],
            ['icon' => 'ti-list-search', 'title' => 'Butir metode', 'value' => nfmt($methods->count()), 'note' => 'di halaman ini',
                'hint' => $groups->map(fn ($group) => $group['title'].' '.count($group['methods']))->join(' · ')],
            ['icon' => 'ti-flask', 'title' => 'Model hasil belajar', 'value' => nfmt(count($used)), 'note' => 'dipakai',
                'hint' => implode(' · ', $used).($other ? ' · tersedia, bukan jalur resmi: '.implode(', ', $other) : '')],
        ];
    }

    /**
     * Alur dari citra ke arti dan tingkat tutur. Tiap langkah [nama, rincian, memakai model hasil belajar]; langkah
     * terakhir bercabang: arti dan tingkat tutur sama-sama dihitung dari teks Latin, bukan yang satu dari yang lain.
     */
    private static function flow(array $model): array
    {
        $bidirectional = $model['bidirectional'] ?? true;

        return [
            'steps' => [
                ['Citra baris', 'tinggi '.$model['height'].' piksel', false],
                ['CNN', $model['conv_layers'].' lapis konvolusi', true],
                [$bidirectional ? 'BiLSTM' : 'LSTM', $model['lstm_layers'].' lapis'.($bidirectional ? ', dua arah' : ''), true],
                ['CTC', $model['classes'].' kelas per kolom', true],
                ['Teks aksara', 'urutan visual ke Unicode', false],
                ['Latin', 'aturan tetap', false],
            ],
            'branches' => [
                ['Arti', 'NLLB-200', true],
                ['Tingkat tutur', 'leksikon penanda', false],
            ],
        ];
    }

    /** Tahap sesudah OCR: milik web, diukur dengan data ujinya sendiri. */
    private static function webGroup(): array
    {
        $translit = Metric::where('scope', 'translit_draft')->first();
        $runs = TranslationRun::latestBySource();
        $human = $runs->get('label');
        // Terjemahan "dari keluaran OCR" dibuat dari satu pipeline tertentu, dipilih dengan aturan yang sama dengan
        // Ringkasan; bila itu bukan pipeline resmi (mis. masih keluaran checkpoint lama), halaman menyebutnya supaya
        // angkanya tidak dibaca sebagai milik model resmi.
        $official = Pipeline::official();
        $ocr = TranslationRun::forOcr($official?->key, $runs);
        $ocrName = $ocr ? (Pipeline::where('key', $ocr->source)->value('label') ?? $ocr->source)
            .($official && $official->key !== $ocr->source ? ', bukan pipeline resmi' : '') : null;
        $markers = array_map('count', SpeechLevel::lexicon());
        [$same, $of] = AksaraWriter::DICTIONARY_AGREEMENT;
        $taling = TalingRestorer::meta();
        $labeled = LineAnnotation::whereNotNull('speech_level')->whereNotNull('transliteration')->get(['transliteration', 'speech_level']);
        $speech = SpeechLevel::evaluate($labeled->map(
            fn ($row) => [$row->speech_level, SpeechLevel::classify($row->transliteration)['level']])->all());
        $score = fn ($run) => 'chrF '.number_format($run->chrf, 1, ',', '.').' / BLEU '.number_format($run->bleu, 1, ',', '.')
            .' pada '.nfmt($run->lines).' baris';
        $translated = array_filter([
            $human ? 'Dari alih aksara manusia: '.$score($human).'.' : null,
            $ocr ? 'Dari keluaran OCR ('.$ocrName.'): '.$score($ocr).'.' : null,
        ]);

        $methods = [
            self::entry('translit', 'Alih aksara ke Latin', 'rule',
                'Tiap aksara, sandhangan, dan pasangan dipetakan ke huruf Latin dengan aturan tetap. Hasilnya draf bantu baca, '
                .'bukan bagian dari angka OCR.',
                [], $translit ? 'CER '.pct($translit->cer).' terhadap alih aksara manusia pada '.nfmt($translit->lines).' baris, huruf saja '
                    .'(spasi, tanda baca, dan huruf besar tidak dibandingkan). Masukannya label aksara, bukan keluaran OCR.' : null,
                'tabel metrik web', ['web/app/Support/Transliterator.php']),
            self::entry('nllb', 'NLLB-200 (model terjemahan pralatih)', 'model',
                'Model Transformer pralatih untuk 200 bahasa, versi 600 juta parameter, dijalankan lokal tanpa mengirim data '
                .'keluar. Dipakai apa adanya, tanpa dilatih ulang, untuk bahasa Jawa ke Indonesia dan sebaliknya.',
                [['Model', $human?->model ?? $ocr?->model ?? self::NLLB], ['Lisensi', 'CC-BY-NC 4.0, hanya non-komersial']],
                $translated ? implode(' ', $translated) : null,
                'tabel terjemahan web', ['web/tools/nllb.py', 'web/tools/translate_batch.py']),
            self::entry('mt_scores', 'chrF dan BLEU', 'evaluation',
                'Mengukur terjemahan mesin terhadap terjemahan manusia: chrF dari tumpang-tindih potongan karakter, BLEU dari '
                .'tumpang-tindih kata. Dihitung pustaka sacrebleu atas semua baris sekaligus.',
                [], null, null, ['web/tools/translate_batch.py']),
            self::entry('speech', 'Tingkat tutur dengan leksikon penanda', 'rule',
                'Kata penanda ngoko, madya, dan krama serta imbuhan krama (-ipun, dipun-) dihitung per teks, lalu tingkat dengan '
                .'penanda terbanyak yang dipilih. Hasilnya campur bila penanda ngoko dan penanda madya atau krama sama-sama '
                .'cukup banyak (masing-masing paling sedikit dua kata dan seperempat dari semua penanda). Kata buktinya '
                .'ditampilkan di Penjelajah baris dan Kamus.',
                [['Kata penanda', collect($markers)->map(fn ($n, $level) => $level.' '.nfmt($n))->join(' · ')]],
                $speech['lines'] ? 'Akurasi '.pct($speech['accuracy']).' pada '.nfmt($speech['lines']).' baris berlabel manusia.' : null,
                'label di Penjelajah baris', ['web/app/Support/SpeechLevel.php'],
                $speech['lines'] ? null : 'Belum dinilai: belum ada label manusia.'),
            self::entry('aksara_writer', 'Alih aksara Latin ke aksara Jawa', 'rule',
                'Kebalikan dari alih aksara ke Latin: suku kata Latin disusun menjadi aksara, sandhangan, dan pasangan menurut '
                .'aturan ejaan. Hanya mengganti sistem tulisan, tidak menerjemahkan.',
                [], 'Sama dengan ejaan aksara kamus pada '.nfmt($same).' dari '.nfmt($of).' lema berejaan baku ('.pct($of ? $same / $of : 0).').',
                'kamus kata bahasa Jawa', ['web/app/Support/AksaraWriter.php']),
        ];
        if ($taling) {
            $held = $taling['held_out'] ?? null;
            $origin = isset($taling['from_wikipedia'], $taling['from_dictionary'])
                ? nfmt($taling['from_wikipedia']).' dari Wikipedia bahasa Jawa, '.nfmt($taling['from_dictionary']).' dari lema kamus' : null;
            $methods[] = self::entry('taling', 'Pemulih tanda é', 'statistic',
                'Teks Latin sehari-hari menulis é dan e dengan huruf yang sama, padahal aksara Jawa membedakannya. Leksikon kata '
                .'bertanda, yang sebagian besar dihitung dari Wikipedia bahasa Jawa, mengembalikan tandanya per kata.',
                array_values(array_filter([['Kata di leksikon', nfmt($taling['words'])], $origin ? ['Asal kata', $origin] : null,
                    isset($taling['rule']) ? ['Syarat masuk', $taling['rule']] : null])),
                $held ? 'Kata ber-e yang ejaannya benar naik dari '.pct($held['accuracy_without'], 0).' ke '.pct($held['accuracy_with'], 0)
                    .' pada '.nfmt($held['articles']).' artikel yang ditahan.' : null,
                'leksikon jv-taling.json', ['web/app/Support/TalingRestorer.php', 'web/tools/taling_lexicon.py']);
        }

        return [
            'key' => 'web',
            'title' => 'Tahap lanjutan alur',
            'intro' => 'Sesudah aksara terbaca: alih aksara, arti, dan tingkat tutur. Tahap ini hidup di aplikasi web dan diukur '
                .'dengan data ujinya sendiri. Hanya terjemahan yang memakai model hasil belajar; sisanya aturan dan leksikon.',
            'methods' => $methods,
        ];
    }

    private static function entry(string $key, string $name, string $kind, string $summary, array $settings, ?string $evidence,
        ?string $source, array $files, ?string $note = null): array
    {
        return ['key' => $key, 'name' => $name, 'kind' => $kind, 'status' => 'used', 'summary' => $summary, 'settings' => $settings,
            'evidence' => $evidence, 'evidence_source' => $evidence ? $source : null, 'note' => $note, 'files' => $files];
    }

    /**
     * Jalur berkas yang boleh dipatahkan sesudah tiap garis miring: kolom pengaturan sempit, dan tanpa ini jalur
     * panjang terpotong di tengah nama berkas ("Transliterator.\nphp").
     */
    public static function breakable(string $path): HtmlString
    {
        return new HtmlString(str_replace('/', '/<wbr>', e($path)));
    }

    /** Bilangan besar ringkas: 1913568 -> "1,9 juta", 47709 -> "47,7 ribu". */
    private static function compact(int $n): string
    {
        return match (true) {
            $n >= 1_000_000 => self::dec($n / 1_000_000, 1).' juta',
            $n >= 1_000 => self::dec($n / 1_000, 1).' ribu',
            default => (string) $n,
        };
    }

    /** Pecahan gaya Indonesia tanpa nol di belakang: 36.2 -> "36,2", 10.0 -> "10". */
    private static function dec(float $x, int $decimals = 2): string
    {
        $text = number_format($x, $decimals, ',', '.');

        return str_contains($text, ',') ? rtrim(rtrim($text, '0'), ',') : $text;
    }
}
