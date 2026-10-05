<?php

namespace App\Livewire\Pages;

use App\Models\DatasetReport;
use App\Models\DictionarySource;
use App\Models\Line;
use App\Models\LineAnnotation;
use App\Models\Pipeline;
use App\Support\TalingRestorer;
use Livewire\Attributes\Layout;
use Livewire\Attributes\Title;
use Livewire\Component;

/**
 * Halaman Dataset: daftar dataset, pembagian latih/validasi/uji, pemakaian oleh run resmi, font per peran, batasan.
 *
 * Semua angka data OCR datang dari kartu data buatan Python (scripts/export_datasets.py, diimpor
 * `php artisan aksara:datasets`); di sini hanya disusun untuk tampilan. Data pendukung milik web (kamus kata,
 * leksikon, anotasi) dihitung dari tabel web sendiri.
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
        'test' => ['Uji', 'Disimpan sampai akhir, hanya untuk angka yang dilaporkan. Ibarat ujian akhir.'],
    ];

    private const FONT_GROUPS = ['core' => 'inti', 'extra' => 'tambahan', 'test' => 'uji', 'review' => 'meragukan', 'rejected' => 'ditolak'];

    public function render()
    {
        $report = DatasetReport::current();
        if (! $report) {
            return view('livewire.pages.dataset', ['report' => null]);
        }
        $card = $report->payload;
        $train = $card['usage']['train'];
        $fonts = collect($card['fonts']);
        $trainFonts = $fonts->filter(fn ($f) => in_array('train', $f['roles'], true));
        $testFonts = $fonts->filter(fn ($f) => in_array('test', $f['roles'], true));
        $datasets = collect($card['datasets']);

        return view('livewire.pages.dataset', [
            'report' => $report,
            'card' => $card,
            'train' => $train,
            'stale' => $this->stale($card),
            'tiles' => $this->tiles($card, $datasets, $trainFonts, $testFonts),
            'parts' => $this->parts($card, $testFonts),
            'funnel' => $this->funnel($card['corpus']),
            'histogram' => $this->histogram($card['corpus']),
            'runs' => array_map(fn ($run) => $run + ['recipe' => self::recipe($run), 'official' => $run['run'] === $card['official']['run']],
                $card['lineage']['runs']),
            'trainLines' => $card['splits'][0]['lines'],
            'datasets' => $datasets->map(fn ($d) => ['roles' => self::byRole($d['roles'])] + $d),
            'usedFonts' => $fonts->filter(fn ($f) => $f['roles'])->map(fn ($f) => $f + ['remarks' => self::remarks($f)])->values(),
            'unusedFonts' => $fonts->reject(fn ($f) => $f['roles'])->groupBy(fn ($f) => self::FONT_GROUPS[$f['group']] ?? $f['group']),
            'limits' => $this->limits($card, $datasets, $trainFonts),
            'support' => $this->support($card),
        ]);
    }

    /** Kartu data dihitung untuk run lain daripada pipeline resmi web sekarang: pemakaiannya tidak lagi berlaku. */
    private function stale(array $card): ?array
    {
        $official = Pipeline::official();
        $key = $card['official']['pipeline'] ?? null;
        if (! $official || ! $key || $official->key === $key) {
            return null;
        }

        return ['run' => $card['official']['run'], 'official' => $official->label];
    }

    private function tiles(array $card, $datasets, $trainFonts, $testFonts): array
    {
        $corpus = $card['corpus'];
        $real = $datasets->first(fn ($d) => isset($d['roles']['test']) && $d['key'] !== 'corpus');
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
            ['icon' => 'ti-scan', 'title' => 'Data nyata', 'value' => nfmt($card['usage']['real']['lines']), 'note' => 'baris cetak',
                'hint' => ($real && isset($real['pages']) ? nfmt($real['pages']).' halaman pindaian · ' : '').'semuanya untuk uji, tidak untuk latih'],
            ['icon' => 'ti-typography', 'title' => 'Font latih', 'value' => nfmt($trainFonts->count()), 'note' => 'font',
                'hint' => nfmt($trainFonts->count() - $extra).' inti + '.nfmt($extra).' tambahan · font uji: '
                    .($testFonts->pluck('file')->map(fn ($f) => pathinfo($f, PATHINFO_FILENAME))->join(', ') ?: 'tidak ada')],
        ];
    }

    /** Tiga bagian korpus: besar, arti, dan seberapa banyak yang dipakai run resmi. */
    private function parts(array $card, $testFonts): array
    {
        $usage = $card['usage'];
        $train = $usage['train'];
        $run = $card['official']['run'];
        $fontNames = $testFonts->pluck('file')->map(fn ($f) => pathinfo($f, PATHINFO_FILENAME))->join(', ') ?: 'font uji';
        $gates = implode(' dan ', array_column($usage['test']['gates'], 'code'));
        $checks = count($usage['val']['steps']);
        $used = [
            'train' => [$train['distinct_lines'], 'baris dijadwalkan run '.$run,
                'Dari kumpulan '.nfmt($train['pool']).' baris acak (seed '.$train['seed'].'): '.nfmt($train['steps']).' langkah × '
                .$train['batch_size'].' = '.nfmt($train['samples']).' sampel. Tiap baris dirender dengan font, ukuran, dan augmentasi acak.'],
            'val' => [$usage['val']['lines'], 'baris pertama, tiap pemeriksaan',
                ($checks ? 'Diperiksa '.$checks.' kali (langkah '.implode(', ', array_map('nfmt', $usage['val']['steps'])).')' : 'Diperiksa berkala')
                .' dengan '.$usage['val']['fonts'].' font inti, tanpa augmentasi.'],
            'test' => [$usage['test']['lines'], 'baris pertama untuk '.($gates ?: 'gerbang sintetis'),
                'Dirender dengan '.$fontNames.'; baris yang sama untuk kedua gerbang.'],
        ];

        return array_map(function ($split) use ($used) {
            [$lines, $caption, $detail] = $used[$split['key']];

            return [
                'key' => $split['key'], 'label' => self::PARTS[$split['key']][0], 'meaning' => self::PARTS[$split['key']][1],
                'lines' => $split['lines'], 'share' => $split['share'], 'used' => $lines, 'caption' => $caption, 'detail' => $detail,
                'used_share' => $split['lines'] ? min(1, $lines / $split['lines']) : 0,
            ];
        }, $card['splits']);
    }

    /** Dari artikel ke baris korpus: tiap langkah pembangunan dengan jumlahnya. */
    private function funnel(array $corpus): array
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

    private function histogram(array $corpus): array
    {
        $max = max(1, ...array_column($corpus['length_histogram'], 'lines'));

        return array_map(fn ($bin) => [
            'range' => str_replace('-', '–', $bin['range']), 'lines' => $bin['lines'],
            'share' => $corpus['lines'] ? $bin['lines'] / $corpus['lines'] : 0, 'width' => round($bin['lines'] / $max * 100, 2),
        ], $corpus['length_histogram']);
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

        return implode(' · ', $parts);
    }

    /** Catatan sebuah font: yang dihitung ekspor (sekeluarga dengan font uji, glyph tidak ada, spasi dibuang) lalu catatan repo. */
    private static function remarks(array $font): array
    {
        $remarks = [];
        $shared = $font['shared_with_test'] ?? null;
        if ($shared && $shared['family']) {
            $remarks[] = 'Sekeluarga dengan font uji '.pathinfo($shared['font'], PATHINFO_FILENAME).': lebar '
                .$shared['same'].' dari '.$shared['of'].' aksara, angka, dan pada persis sama.';
        }
        if ($font['missing'] ?? []) {
            $remarks[] = 'Tidak punya '.implode(', ', array_map(
                fn ($code) => $code.' ('.aksara_name(mb_chr(hexdec(substr($code, 2)))).')', $font['missing'])).'.';
        }
        if ($font['drops_space'] ?? false) {
            $remarks[] = 'Spasinya nyaris tak tampak, jadi spasi selalu dibuang dari teks dan label.';
        }
        if (! ($font['available'] ?? true)) {
            $remarks[] = 'Berkasnya tidak ditemukan saat ekspor.';
        }

        return array_merge($remarks, $font['notes'] ?? []);
    }

    /** Batasan data, hanya yang syaratnya terpenuhi di kartu data (hilang sendiri begitu datanya diperbaiki). */
    private function limits(array $card, $datasets, $trainFonts): array
    {
        $limits = [];
        $twin = $trainFonts->first(fn ($f) => $f['shared_with_test']['family'] ?? false);
        if ($twin) {
            $shared = $twin['shared_with_test'];
            $gates = implode(' dan ', array_column($card['usage']['test']['gates'], 'code')) ?: 'Gerbang sintetis';
            $limits[] = ['Font uji bukan font yang benar-benar baru',
                'Font uji '.pathinfo($shared['font'], PATHINFO_FILENAME).' sekeluarga dengan font latih '.pathinfo($twin['file'], PATHINFO_FILENAME)
                .': lebar '.$shared['same'].' dari '.$shared['of'].' aksara, angka, dan pada persis sama. '.$gates
                .' karena itu mengukur generalisasi di dalam satu keluarga huruf, bukan ke huruf yang belum pernah dilihat model.'];
        }

        $real = $datasets->filter(fn ($d) => $d['key'] !== 'corpus');
        $tested = $real->first(fn ($d) => isset($d['roles']['test']));
        if ($tested && ! $real->contains(fn ($d) => isset($d['roles']['val']))) {
            $pending = $real->first(fn ($d) => ($d['status'] ?? '') === 'pending');
            $limits[] = ['Data nyata tidak punya bagian validasi',
                'Semua '.nfmt($tested['count']).' baris nyata masuk bagian uji. Keputusan seperti memilih run resmi ikut melihat data uji, '
                .'jadi angka G3 sedikit terlalu bagus. Data itu tidak dibelah karena sebagian besar halamannya dari satu majalah dengan '
                .'huruf yang sama: membaginya per halaman akan membocorkan bentuk huruf ke bagian uji.'
                .($pending ? ' Perbaikannya butuh data nyata lain untuk latih dan validasi: '.nfmt($pending['count'])
                    .' baris '.mb_strtolower(mb_substr($pending['name'], 0, 1)).mb_substr($pending['name'], 1).' masih menunggu verifikasi pembaca aksara.' : '')];
        }

        $rare = $card['rare'];
        if ($rare['codepoints'] > 0 && ($rare['scheduled_lines'] ?? [])) {
            $each = $rare['train_lines'];
            $seen = $rare['scheduled_lines'];
            $limits[] = ['Aksara langka nyaris tidak terlihat saat training',
                nfmt($rare['codepoints']).' dari '.nfmt($rare['javanese']).' karakter aksara Jawa di charset ada di '
                .($each['min'] === $each['max'] ? 'hanya '.nfmt($each['max']) : nfmt($each['min']).'–'.nfmt($each['max']))
                .' baris latih masing-masing: aksara murda dan mahaprana, aksara swara, beberapa sandhangan dan pada. Di antara '
                .nfmt($card['usage']['train']['distinct_lines'])
                .' baris yang dijadwalkan run resmi, tiap karakter itu muncul di '.nfmt($seen['min']).'–'.nfmt($seen['max'])
                .' baris, rata-rata '.self::dec($seen['mean'], 1).'. Model resmi karena itu hampir tidak pernah mengeluarkannya.'];
        }

        return $limits;
    }

    /** Data pendukung: milik repo OCR (dari kartu data) lalu milik web (dihitung dari tabel web). */
    private function support(array $card): array
    {
        $rows = [];
        foreach ($card['support'] as $item) {
            $rows[] = match ($item['key']) {
                'charset' => ['Charset tokenizer', 'Kelas keluaran model',
                    nfmt($item['characters']).' karakter + blank = '.nfmt($item['classes']).' kelas', 'Dibangun dari korpus ('.$item['file'].')'],
                'charlm' => ['Model bahasa karakter', 'Kondisi "kamus": beam search + LM',
                    'order '.$item['order'].' · '.nfmt($item['lines']).' baris', 'Bagian latih korpus'],
                'vlm_blind' => ['Uji buta VLM', 'Pembanding VLM zero-shot', nfmt($item['lines']).' baris', 'Sampel acak data nyata'],
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
