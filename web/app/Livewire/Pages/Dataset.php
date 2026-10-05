<?php

namespace App\Livewire\Pages;

use App\Models\DatasetReport;
use App\Models\DictionarySource;
use App\Models\Line;
use App\Models\LineAnnotation;
use App\Models\MethodReport;
use App\Models\Pipeline;
use App\Support\TalingRestorer;
use Illuminate\Support\Collection;
use Livewire\Attributes\Layout;
use Livewire\Attributes\Title;
use Livewire\Component;

/**
 * Halaman Dataset: daftar dataset, pembagian latih/validasi/uji, pemakaian oleh run resmi, font per peran, batasan.
 *
 * Semua angka data OCR datang dari kartu data buatan Python (scripts/export_datasets.py, diimpor
 * `php artisan aksara:datasets`); di sini hanya disusun untuk tampilan. Kalimat yang bergantung pada isi kartu
 * (rantai run, seed, batasan) dipilih dari nilai kartu, supaya tetap benar untuk kartu run lain. Data pendukung milik
 * web dihitung dari tabel web (kamus kata, anotasi) dan dari berkas leksikon é.
 */
#[Layout('components.layouts.app')]
#[Title('Dataset')]
class Dataset extends Component
{
    /** Peran sebuah dataset atau font. */
    public const ROLES = ['train' => 'latih', 'val' => 'validasi', 'test' => 'uji'];

    /** Arti tiga bagian korpus: penjelasan umum, bukan hasil hitung. */
    public const PARTS = [
        'train' => ['Latih', 'Bahan belajar. Model melihat baris ini, dirender menjadi citra, lalu memperbaiki dirinya. Ibarat buku latihan soal.'],
        'val' => ['Validasi', 'Dicek berkala selama training; model tidak belajar darinya. Dipakai memantau training dan menyetel pengaturan. Ibarat try-out.'],
        'test' => ['Uji', 'Tidak pernah dilatihkan. Dipakai untuk angka yang dilaporkan. Ibarat ujian akhir.'],
    ];

    /** Kenapa sekelompok font tidak dipakai run resmi. */
    private const UNUSED_FONTS = [
        'core' => 'Font inti yang tidak dipakai run resmi',
        'extra' => 'Font tambahan yang tidak dipakai run resmi',
        'review' => 'Tidak dipakai karena meragukan',
        'rejected' => 'Tidak dipakai karena ditolak',
    ];

    public function render()
    {
        return view('livewire.pages.dataset', self::viewData(DatasetReport::current()));
    }

    /**
     * Data tampilan untuk sebuah kartu. Dipakai halaman ini dan impor kartu (DatasetImporter merendernya sekali
     * sebelum menyimpan, supaya kartu yang tidak bisa ditampilkan ditolak saat impor).
     */
    public static function viewData(?DatasetReport $report): array
    {
        if (! $report) {
            return ['report' => null];
        }
        $card = $report->payload;
        $fonts = collect($card['fonts']);
        $trainFonts = $fonts->filter(fn ($f) => in_array('train', $f['roles'], true));
        $testFonts = $fonts->filter(fn ($f) => in_array('test', $f['roles'], true));
        $datasets = collect($card['datasets']);
        $trainLines = $card['splits'][0]['lines'];
        $seen = $card['lineage']['distinct_lines'];

        return [
            'report' => $report,
            'card' => $card,
            'stale' => self::stale($card),
            'tiles' => self::tiles($card, $datasets, $trainFonts, $testFonts),
            'parts' => self::parts($card, $testFonts),
            'footer' => self::footer($card, $trainLines),
            'funnel' => self::funnel($card['corpus']),
            'histogram' => self::histogram($card['corpus']),
            'runs' => array_map(fn ($run) => $run + ['recipe' => self::recipe($run), 'official' => $run['run'] === $card['official']['run']],
                $card['lineage']['runs']),
            'chainIntro' => self::chainIntro($card, $trainLines),
            'seenShare' => $seen !== null && $trainLines ? $seen / $trainLines : null,
            'datasets' => $datasets->map(fn ($d) => [
                'roles' => self::byRole($d['roles']),
                'link' => preg_match('#^https?://#i', (string) ($d['source_url'] ?? '')) ? $d['source_url'] : null,
                'waiting' => self::waiting($d),
            ] + $d),
            'usedFonts' => $fonts->filter(fn ($f) => $f['roles'])->map(fn ($f) => $f + ['remarks' => self::remarks($f)])->values(),
            'unusedFonts' => self::unusedFonts($fonts),
            'limits' => self::limits($card, $datasets, $trainFonts),
            'support' => self::support($card),
        ];
    }

    /**
     * Kartu data dan hasil OCR yang diimpor dihitung untuk run yang berbeda: pemakaian di halaman ini bukan milik
     * pipeline resmi web. Kalimatnya tidak menebak sisi mana yang tertinggal (kartu bisa lebih baru daripada hasil,
     * atau sebaliknya). Kartu untuk run yang tidak dikenal ekspor hasil (kunci pipeline kosong) juga tidak cocok.
     */
    private static function stale(array $card): ?array
    {
        $official = Pipeline::official();
        if (! $official || $official->key === ($card['official']['pipeline'] ?? null)) {
            return null;
        }

        return ['run' => $card['official']['run'], 'official' => $official->label];
    }

    private static function tiles(array $card, Collection $datasets, Collection $trainFonts, Collection $testFonts): array
    {
        $corpus = $card['corpus'];
        $real = $datasets->filter(fn ($d) => $d['key'] !== 'corpus');
        $tested = $real->first(fn ($d) => isset($d['roles']['test']));
        $trained = (int) $real->sum(fn ($d) => $d['roles']['train'] ?? 0);
        $extra = $trainFonts->where('group', 'extra')->count();
        // Rasio rancangan = keranjang hash id artikel per bagian; porsi menurut jumlah baris sedikit berbeda.
        $buckets = array_map(fn ($split) => $corpus['split_buckets'][$split['key']] ?? round($split['share'] * 100), $card['splits']);
        $shares = array_map(fn ($split) => number_format($split['share'] * 100, 1, ',', '.'), $card['splits']);

        return [
            ['icon' => 'ti-database', 'title' => 'Korpus teks', 'value' => nfmt($corpus['lines']), 'note' => 'baris',
                'hint' => nfmt($corpus['articles']).' artikel Wikipedia bahasa Jawa, dialihaksarakan ke aksara Jawa'],
            ['icon' => 'ti-chart-pie', 'title' => 'Pembagian', 'value' => implode(' / ', $buckets), 'note' => '% artikel',
                'hint' => 'Latih / validasi / uji. Menurut jumlah baris '.implode(' / ', $shares).'% · teks yang sama di dua bagian: '
                    .nfmt($corpus['leaked_lines']).' baris'],
            ['icon' => 'ti-scan', 'title' => 'Data nyata', 'value' => nfmt($card['usage']['real']['lines']), 'note' => 'baris uji',
                'hint' => ($tested && isset($tested['pages']) ? nfmt($tested['pages']).' halaman pindaian · ' : '')
                    .($trained ? nfmt($trained).' baris nyata lain dipakai melatih' : 'semuanya untuk uji, tidak untuk latih')],
            ['icon' => 'ti-typography', 'title' => 'Font latih', 'value' => nfmt($trainFonts->count()), 'note' => 'font',
                'hint' => nfmt($trainFonts->count() - $extra).' inti + '.nfmt($extra).' tambahan · font uji: '
                    .($testFonts->pluck('file')->map(fn ($f) => pathinfo($f, PATHINFO_FILENAME))->join(', ') ?: 'tidak ada')],
        ];
    }

    /** Tiga bagian korpus: besar, arti, dan seberapa banyak yang dipakai run resmi (null = tidak dihitung). */
    private static function parts(array $card, Collection $testFonts): array
    {
        $usage = $card['usage'];
        $train = $usage['train'];
        $run = $card['official']['run'];
        $fontNames = $testFonts->pluck('file')->map(fn ($f) => pathinfo($f, PATHINFO_FILENAME))->join(', ') ?: 'font uji';
        $gates = implode(' dan ', array_column($usage['test']['gates'], 'code'));
        $checks = count($usage['val']['steps']);
        $quick = $usage['test']['quick'] ?? 0;
        $full = $usage['test']['full'] ?? 0;
        $others = array_filter([
            $quick ? nfmt($quick).' tes cepat penentu arah (sampai '.nfmt($usage['test']['quick_max_lines'] ?? 0).' baris pertama)' : null,
            $full ? nfmt($full).' evaluasi penuh untuk run lain' : null,
        ]);
        // Sampel = baris di semua batch; batch terakhir sebuah kelompok panjang bisa lebih kecil dari ukuran batch.
        $samples = $train['samples'] === $train['steps'] * $train['batch_size']
            ? nfmt($train['steps']).' langkah × '.$train['batch_size'].' = '.nfmt($train['samples']).' sampel'
            : nfmt($train['steps']).' langkah, '.nfmt($train['samples']).' sampel';

        $used = [
            'train' => $train['distinct_lines'] === null
                ? [null, 'baris sintetis: tidak dihitung', 'Run '.$run.' dilatih dengan data nyata atau dalam mode menghafal, jadi jadwal baris sintetisnya tidak dihitung.']
                : [$train['distinct_lines'], 'baris dijadwalkan run '.$run,
                    'Dari kumpulan '.nfmt($train['pool']).' baris acak (seed '.$train['seed'].'): '.$samples
                    .'. Tiap baris dirender dengan font, ukuran, dan augmentasi acak.'],
            'val' => ($train['real_val'] ?? false)
                ? [null, 'tidak dipakai run '.$run, 'Run ini divalidasi dengan data nyata, bukan dengan bagian ini.']
                : [$usage['val']['lines'], 'baris pertama, tiap pemeriksaan',
                    ($checks ? 'Diperiksa '.$checks.' kali (langkah '.implode(', ', array_map('nfmt', $usage['val']['steps'])).')' : 'Diperiksa berkala')
                    .' dengan '.$usage['val']['fonts'].' font inti, tanpa augmentasi.'],
            'test' => [$usage['test']['lines'], 'baris pertama untuk '.($gates ?: 'gerbang sintetis'),
                'Dirender dengan '.$fontNames.'; baris yang sama untuk tiap gerbang.'
                .($others ? ' Di luar gerbang resmi, '.implode(' dan ', $others).' juga membaca bagian ini.' : '')],
        ];

        return array_map(function ($split) use ($used) {
            [$lines, $caption, $detail] = $used[$split['key']];

            return [
                'key' => $split['key'], 'label' => self::PARTS[$split['key']][0], 'meaning' => self::PARTS[$split['key']][1],
                'lines' => $split['lines'], 'share' => $split['share'], 'used' => $lines, 'caption' => $caption, 'detail' => $detail,
                'used_share' => $lines !== null && $split['lines'] ? min(1, $lines / $split['lines']) : null,
            ];
        }, $card['splits']);
    }

    /** Kenapa bukan 80/20: angkanya dari kartu, dan kalimat "baru sekian persen yang dilihat" hanya bila memang begitu. */
    private static function footer(array $card, int $trainLines): string
    {
        $seen = $card['lineage']['distinct_lines'];
        $text = 'Kenapa bukan 80/20 atau 70/15/15? Rasio itu untuk data berjumlah ribuan. Di sini bagian uji saja sudah '
            .nfmt($card['splits'][2]['lines']).' baris, padahal gerbang hanya memakai '.nfmt($card['usage']['test']['lines']);
        if ($seen !== null && $trainLines && $seen < $trainLines) {
            $text .= ', dan sepanjang rantai run resmi baru '.pct($seen / $trainLines).' bagian latih yang pernah dilihat';
        }

        return $text.'. Membagi ulang juga memindahkan teks yang sudah dilatih ke bagian uji, sehingga angka lama tidak bisa lagi '
            .'dibandingkan dengan yang baru.';
    }

    /** Dari artikel ke baris korpus: tiap langkah pembangunan dengan jumlahnya. */
    private static function funnel(array $corpus): array
    {
        $rows = [
            ['Artikel Wikipedia bahasa Jawa', nfmt($corpus['articles']), null],
            ['Baris kandidat', nfmt($corpus['candidates']), 'paragraf dialihaksarakan ke aksara Jawa, lalu dipotong per baris'],
            ['Lolos uji bolak-balik', nfmt($corpus['roundtrip_passed']), pct($corpus['roundtrip_pass_rate'], 2).' kandidat · Latin → aksara → Latin'
                .(isset($corpus['max_roundtrip_cer']) ? ' dengan CER ≤ '.pct($corpus['max_roundtrip_cer'], 0) : '')],
            ['Duplikat dibuang', '−'.nfmt($corpus['duplicates']), null],
            ['Baris sisipan aksara langka', '+'.nfmt($corpus['injected']), isset($corpus['min_count']['train'])
                ? 'supaya tiap karakter aksara Jawa muncul paling sedikit '.nfmt($corpus['min_count']['train']).' kali di bagian latih' : null],
        ];
        if ($corpus['other'] ?? 0) {
            $rows[] = ['Penyesuaian lain', ($corpus['other'] > 0 ? '+' : '−').nfmt(abs($corpus['other'])), null];
        }

        return $rows;
    }

    private static function histogram(array $corpus): array
    {
        $max = max(1, ...array_column($corpus['length_histogram'], 'lines'));

        return array_map(fn ($bin) => [
            'range' => str_replace('-', '–', $bin['range']), 'lines' => $bin['lines'],
            'share' => $corpus['lines'] ? $bin['lines'] / $corpus['lines'] : 0, 'width' => round($bin['lines'] / $max * 100, 2),
        ], $corpus['length_histogram']);
    }

    /** Kalimat pembuka tabel rantai, dipilih dari isi kartu: jumlah run, kelengkapan rantai, seed, dan porsi terlihat. */
    private static function chainIntro(array $card, int $trainLines): string
    {
        $runs = $card['lineage']['runs'];
        $before = count($runs) - 1;
        $run = $card['official']['run'];
        $complete = $card['lineage']['complete'];
        $text = match (true) {
            $before > 0 => 'Checkpoint '.$run.' melanjutkan bobot '.($complete ? '' : 'sedikitnya ').$before.' run sebelumnya.',
            $complete => 'Checkpoint '.$run.' dilatih dalam satu run, dari bobot acak.',
            default => 'Checkpoint '.$run.' melanjutkan run lain yang checkpoint-nya sudah tidak ada.',
        };
        $seeds = array_values(array_unique(array_column($runs, 'seed')));
        if ($before > 0) {
            $text .= count($seeds) === 1
                ? ' Semua run mengambil barisnya dari kumpulan acak yang sama (seed '.$seeds[0].').'
                : ' Run-run itu memakai seed yang berbeda, jadi kumpulan barisnya tidak sama.';
        }
        $seen = $card['lineage']['distinct_lines'];

        return $text.($seen === null || ! $trainLines ? ' Jumlah baris berbeda di sepanjang rantai tidak dihitung.'
            : ' Sepanjang rantai, model melihat '.pct($seen / $trainLines).' bagian latih.');
    }

    /**
     * Nilai per peran dalam urutan latih, validasi, uji. Urutan kunci objek dari kartu data tidak bisa dipegang:
     * penyimpanan JSON boleh mengurutkannya ulang (jsonb PostgreSQL menaruh "val" dan "test" sebelum "train").
     */
    private static function byRole(array $values): array
    {
        $ordered = [];
        foreach (array_keys(self::ROLES) as $role) {
            if (array_key_exists($role, $values)) {
                $ordered[$role] = $values[$role];
            }
        }

        return $ordered + $values;
    }

    /** Baris sebuah dataset yang masih menunggu verifikasi (0 bila tidak ada atau bukan dataset yang menunggu). */
    private static function waiting(array $dataset): int
    {
        if (($dataset['status'] ?? '') !== 'pending') {
            return 0;
        }

        return (int) ($dataset['pending'] ?? max(0, $dataset['count'] - ($dataset['verified'] ?? 0)));
    }

    /** Resep data sebuah run di rantai checkpoint, dari argumen training di checkpoint-nya. */
    private static function recipe(array $run): string
    {
        $parts = [$run['augment'] === 'none' ? 'tanpa augmentasi' : 'augmentasi '.$run['augment']];
        if ($run['drop_space_prob'] > 0) {
            $parts[] = 'spasi dibuang p '.self::dec($run['drop_space_prob']);
        }
        if ($run['track_prob'] > 0) {
            $parts[] = 'jarak antar suku kata p '.self::dec($run['track_prob']).' (hingga '.self::dec($run['track_max']).' em)';
        }
        if ($run['rare_insert_prob'] > 0 || $run['rare_opener_prob'] > 0) {
            $parts[] = 'sisipan aksara langka';
        }
        if ($run['real_train'] ?? false) {
            $parts[] = 'data nyata ikut dilatih';
        }

        return implode(' · ', $parts);
    }

    /** Catatan sebuah font: yang dihitung ekspor (sekeluarga dengan font uji, glyph tidak ada, spasi dibuang) lalu catatan repo. */
    private static function remarks(array $font): array
    {
        $remarks = [];
        $shared = $font['shared_with_test'] ?? [];
        if (($shared['family'] ?? false) && isset($shared['font'], $shared['same'], $shared['of'])) {
            $remarks[] = 'Sekeluarga dengan font uji '.pathinfo($shared['font'], PATHINFO_FILENAME).': lebar '
                .$shared['same'].' dari '.$shared['of'].' aksara, angka, dan pada persis sama.';
        }
        if ($font['missing'] ?? []) {
            $remarks[] = 'Tidak punya '.implode(', ', array_map(
                fn ($code) => $code.' ('.aksara_name(mb_chr(hexdec(substr((string) $code, 2)))).')', $font['missing'])).'.';
        }
        if ($font['drops_space'] ?? false) {
            $remarks[] = 'Spasinya nyaris tak tampak, jadi spasi selalu dibuang dari teks dan label.';
        }
        if (! ($font['available'] ?? true)) {
            $remarks[] = 'Berkasnya tidak ditemukan saat ekspor.';
        }

        return array_merge($remarks, array_map('strval', $font['notes'] ?? []));
    }

    /** Font yang tidak dipakai run resmi, per alasan, dengan catatan pemeriksaannya ditulis terlihat. */
    private static function unusedFonts(Collection $fonts): array
    {
        return $fonts->reject(fn ($f) => $f['roles'])->groupBy('group')->map(fn ($items, $group) => [
            'label' => self::UNUSED_FONTS[$group] ?? 'Tidak dipakai ('.$group.')',
            'fonts' => $items->map(function ($font) {
                // "MERAGUKAN: susun-3 bertabrakan" -> "susun-3 bertabrakan": labelnya sudah ada di kalimat.
                $reason = trim((string) preg_replace('/^[A-Z ]+:\s*/u', '', (string) ($font['review'] ?? '')));

                return pathinfo($font['file'], PATHINFO_FILENAME).($reason !== '' ? ' ('.$reason.')' : '');
            })->values()->all(),
        ])->values()->all();
    }

    /** Batasan data, hanya yang syaratnya terpenuhi di kartu data. */
    private static function limits(array $card, Collection $datasets, Collection $trainFonts): array
    {
        $limits = [];
        $twin = $trainFonts->first(fn ($f) => ($f['shared_with_test']['family'] ?? false) && isset($f['shared_with_test']['font']));
        if ($twin) {
            $shared = $twin['shared_with_test'];
            $gates = implode(' dan ', array_column($card['usage']['test']['gates'], 'code')) ?: 'Gerbang sintetis';
            $limits[] = ['Font uji bukan font yang benar-benar baru',
                'Font uji '.pathinfo($shared['font'], PATHINFO_FILENAME).' sekeluarga dengan font latih '.pathinfo($twin['file'], PATHINFO_FILENAME)
                .': lebar '.($shared['same'] ?? '?').' dari '.($shared['of'] ?? '?').' aksara, angka, dan pada persis sama. '.$gates
                .' karena itu mengukur generalisasi di dalam satu keluarga huruf, bukan ke huruf yang belum pernah dilihat model.'];
        }

        $real = $datasets->filter(fn ($d) => $d['key'] !== 'corpus');
        $tested = $real->first(fn ($d) => isset($d['roles']['test']));
        if ($tested && ! $real->contains(fn ($d) => isset($d['roles']['val']))) {
            $reports = $card['usage']['real']['reports'] ?? 0;
            $pending = $real->first(fn ($d) => self::waiting($d) > 0);
            $limits[] = ['Data nyata tidak punya bagian validasi',
                'Semua '.nfmt($tested['count']).' baris '.$tested['name'].' dipakai sebagai data uji, dan tidak ada data nyata untuk validasi. '
                .'Keputusan seperti memilih run resmi ikut melihat data uji, jadi angka G3 sedikit terlalu bagus'
                .($reports > 1 ? ' (data itu sudah dievaluasi '.nfmt($reports).' kali sepanjang proyek)' : '').'.'
                // Catatan proyek tentang NusaAksara (CLAUDE.md, keputusan 2026-09-13), bukan nilai kartu.
                .($tested['key'] === 'nusaaksara' ? ' Data itu tidak dibelah karena sebagian besar halamannya berasal dari satu terbitan dengan '
                    .'huruf yang sama: membaginya per halaman akan membocorkan bentuk huruf ke bagian uji.' : '')
                .($pending ? ' Perbaikannya butuh data nyata lain untuk latih dan validasi: '.nfmt(self::waiting($pending)).' baris '
                    .mb_strtolower(mb_substr($pending['name'], 0, 1)).mb_substr($pending['name'], 1).' masih menunggu verifikasi pembaca aksara.' : '')];
        }

        // Sisipan aksara langka saat render adalah obat keterbatasan ini: bila run resmi memakainya, batasannya tidak berlaku.
        $rare = $card['rare'];
        $train = $card['usage']['train'];
        $inserted = ($train['rare_insert_prob'] ?? 0) > 0 || ($train['rare_opener_prob'] ?? 0) > 0;
        $seen = $rare['scheduled_lines'] ?? [];
        if ($rare['codepoints'] > 0 && ! $inserted && $train['distinct_lines'] !== null && isset($seen['min'], $seen['max'], $seen['mean'])) {
            $each = $rare['train_lines'];
            $trial = collect(MethodReport::current()?->payload['groups'] ?? [])->flatMap(fn ($g) => $g['methods'] ?? [])->contains('key', 'rare');
            $limits[] = ['Aksara langka nyaris tidak terlihat saat training',
                nfmt($rare['codepoints']).' dari '.nfmt($rare['javanese']).' karakter aksara Jawa di charset ada di '
                .(($each['min'] ?? null) === ($each['max'] ?? null) ? 'hanya '.nfmt($each['max'] ?? 0) : nfmt($each['min'] ?? 0).'–'.nfmt($each['max'] ?? 0))
                .' baris latih masing-masing: aksara murda dan mahaprana, aksara swara, beberapa sandhangan dan pada. Di antara '
                .nfmt($train['distinct_lines']).' baris yang dijadwalkan run resmi, tiap karakter itu muncul di '.nfmt($seen['min']).'–'.nfmt($seen['max'])
                .' baris, rata-rata '.self::dec($seen['mean'], 1).'.'
                .($trial ? ' Sisipan aksara langka saat render sudah diuji; hasilnya ada di halaman Metode.' : '')];
        }

        return $limits;
    }

    /** Data pendukung: milik repo OCR (dari kartu data) lalu milik web (tabel web dan berkas leksikon). */
    private static function support(array $card): array
    {
        $rows = [];
        foreach ($card['support'] as $item) {
            $rows[] = match (true) {
                $item['key'] === 'charset' && isset($item['characters'], $item['classes']) => ['Charset tokenizer', 'Kelas keluaran model',
                    nfmt($item['characters']).' karakter + blank = '.nfmt($item['classes']).' kelas', 'Dibangun dari korpus ('.($item['file'] ?? 'data/tokenizer.json').')'],
                $item['key'] === 'charlm' && isset($item['order'], $item['lines']) => ['Model bahasa karakter', 'Kondisi "kamus": beam search + LM',
                    'order '.$item['order'].' · '.nfmt($item['lines']).' baris', 'Bagian latih korpus'],
                $item['key'] === 'vlm_blind' && isset($item['lines']) => ['Uji buta VLM', 'Pembanding VLM zero-shot', nfmt($item['lines']).' baris', 'Sampel acak data nyata'],
                default => null,
            };
        }
        // Anotasi yang cocok dengan baris uji yang diimpor (anotasi NusaAksara lebih banyak daripada baris uji);
        // sebelum hasil OCR diimpor, semua anotasi dihitung.
        $matched = Line::exists();
        $count = fn (string $column) => LineAnnotation::whereNotNull($column)
            ->when($matched, fn ($query) => $query->whereIn('external_id', Line::select('external_id')))->count();
        $amount = fn (int $n, string $unit) => $n ? nfmt($n).' '.$unit : 'belum diimpor';
        $rows[] = ['Transliterasi manusia', 'Menilai transliterasi draf', $amount($count('transliteration'), 'baris'), 'NusaAksara (non-komersial)'];
        $rows[] = ['Terjemahan manusia', 'Menilai terjemahan mesin', $amount($count('translation'), 'baris'), 'NusaAksara (non-komersial)'];
        $labels = $count('speech_level');
        $rows[] = ['Label tingkat tutur', 'Menilai tebakan ngoko, madya, krama', $labels ? nfmt($labels).' baris' : 'belum ada label', 'Diisi manusia di Penjelajah baris'];
        $names = ['jv' => 'Kamus kata bahasa Jawa', 'id' => 'Kamus kata bahasa Indonesia'];
        $sources = DictionarySource::orderBy('id')->get()->groupBy('dictionary');
        foreach ($names as $key => $name) {
            $group = $sources->get($key, collect());
            $rows[] = [$name, 'Halaman Kamus', $amount((int) $group->sum('entries'), 'entri'),
                $group->isEmpty() ? 'php artisan aksara:dictionary' : $group->pluck('name')->join(' + ').' ('.$group->pluck('license')->unique()->join(', ').')'];
        }
        $taling = TalingRestorer::meta();
        $rows[] = ['Leksikon tanda é', 'Halaman Terjemahan', $amount((int) ($taling['words'] ?? 0), 'kata'),
            $taling['source'] ?? 'Wikipedia bahasa Jawa dan kamus kata'];

        return array_values(array_filter($rows));
    }

    /** Pecahan gaya Indonesia tanpa nol di belakang: 0.5 -> "0,5". */
    private static function dec(float $x, int $decimals = 2): string
    {
        $text = number_format($x, $decimals, ',', '.');

        return str_contains($text, ',') ? rtrim(rtrim($text, '0'), ',') : $text;
    }
}
