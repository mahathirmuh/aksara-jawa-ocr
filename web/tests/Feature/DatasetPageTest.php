<?php

namespace Tests\Feature;

use App\Models\DatasetReport;
use App\Models\DictionarySource;
use App\Models\Line;
use App\Models\LineAnnotation;
use App\Models\MethodReport;
use App\Models\Pipeline;
use App\Models\User;
use App\Services\DatasetImporter;
use Illuminate\Filesystem\Filesystem;
use Illuminate\Foundation\Testing\RefreshDatabase;
use RuntimeException;
use Tests\Concerns\ReadsPage;
use Tests\TestCase;

class DatasetPageTest extends TestCase
{
    use ReadsPage;
    use RefreshDatabase;

    private string $dir;

    protected function setUp(): void
    {
        parent::setUp();
        $this->dir = sys_get_temp_dir().DIRECTORY_SEPARATOR.'aksara-datasets-'.uniqid();
        mkdir($this->dir);
    }

    protected function tearDown(): void
    {
        (new Filesystem)->deleteDirectory($this->dir);
        parent::tearDown();
    }

    /** Satu run di rantai checkpoint, bentuknya seperti keluaran scripts/export_datasets.py. */
    private function chainRun(string $name, ?string $init, int $steps, int $pool, ?int $distinct, int $fonts, array $extra = []): array
    {
        return $extra + ['run' => $name, 'init' => $init, 'checkpoint' => "out/checkpoints/{$name}/last_snapshot.pt",
            'steps' => $steps, 'batch_size' => 32, 'samples' => $steps * 32, 'pool' => $pool, 'seed' => 0, 'processes' => 1,
            'segments' => [[0, $steps]], 'distinct_lines' => $distinct, 'real_train' => false, 'real_val' => false,
            'fonts' => $fonts, 'extra_fonts' => max(0, $fonts - 2), 'augment' => 'fase5', 'drop_space_prob' => 0.5,
            'track_prob' => 0.0, 'track_max' => 0.0, 'rare_insert_prob' => 0.0, 'rare_opener_prob' => 0.0, 'val_lines' => 500,
            'val_steps' => [500, 1000, 1500]];
    }

    private function font(string $file, string $group, array $roles, array $extra = []): array
    {
        return $extra + ['file' => $file, 'group' => $group, 'roles' => $roles, 'license' => 'tidak tercantum',
            'source' => 'https://contoh.test/'.$file, 'in_repo' => $group === 'core', 'available' => true, 'notes' => []];
    }

    /** Kartu data kecil dengan angka korpus sebenarnya (2026-10-06), tanpa teks label apa pun. */
    private function card(): array
    {
        $official = $this->chainRun('fase7_track', 'fase6_ctrl', 1500, 100000, 48000, 10, ['track_prob' => 0.5, 'track_max' => 0.3]);
        $none = ['font' => 'javatext.ttf', 'same' => 0, 'of' => 73, 'family' => false];

        return [
            'schema' => 1, 'generated' => '2026-10-06T00:12:00', 'source' => 'metopenv5',
            'official' => ['run' => 'fase7_track', 'pipeline' => 'crnn_fase7_track',
                'checkpoint' => 'out/checkpoints/fase7_track/last_snapshot.pt', 'step' => 1500],
            'corpus' => ['articles' => 70923, 'candidates' => 1158012, 'roundtrip_passed' => 1109569, 'roundtrip_pass_rate' => 0.958167,
                'duplicates' => 80492, 'injected' => 5280, 'other' => 0, 'lines' => 1034357, 'leaked_lines' => 0,
                'split_buckets' => ['train' => 90, 'val' => 5, 'test' => 5], 'max_roundtrip_cer' => 0.05,
                'min_count' => ['train' => 100, 'val' => 10, 'test' => 10], 'built' => '2026-09-13',
                'length_histogram' => [['range' => '20-29', 'lines' => 279814], ['range' => '30-80', 'lines' => 754543]]],
            'splits' => [
                ['key' => 'train', 'file' => 'data/splits/train.txt', 'lines' => 931958, 'injected' => 4400, 'share' => 0.9010022651753699],
                ['key' => 'val', 'file' => 'data/splits/val.txt', 'lines' => 51756, 'injected' => 440, 'share' => 0.05003688281705446],
                ['key' => 'test', 'file' => 'data/splits/test.txt', 'lines' => 50643, 'injected' => 440, 'share' => 0.048960852007575724],
            ],
            'usage' => [
                'train' => $official,
                'val' => ['lines' => 500, 'steps' => [500, 1000, 1500], 'fonts' => 2],
                'test' => ['lines' => 10000, 'quick' => 24, 'quick_max_lines' => 200, 'full' => 2, 'gates' => [
                    ['code' => 'G1', 'lines' => 10000, 'fonts' => ['javatext.ttf'], 'split' => 'test', 'augment' => 'none', 'source' => 'out/eval/fase7_track_G1_10k.json'],
                    ['code' => 'G2', 'lines' => 10000, 'fonts' => ['javatext.ttf'], 'split' => 'test', 'augment' => 'heavy', 'source' => 'out/eval/fase7_track_G2_10k.json']]],
                'real' => ['lines' => 745, 'reports' => 36, 'gates' => [['code' => 'G3', 'lines' => 745, 'fonts' => [], 'split' => null, 'augment' => null, 'source' => 'out/eval/fase7_track_G3_full.json']]],
            ],
            'lineage' => ['complete' => true, 'steps' => 3000, 'samples' => 96000, 'pool' => 100000, 'distinct_lines' => 64595, 'runs' => [
                $this->chainRun('fase6_ctrl', null, 1500, 100000, 19712, 10, ['processes' => 3, 'segments' => [[0, 500], [500, 884], [884, 1500]]]),
                $official,
            ]],
            'rare' => ['codepoints' => 44, 'javanese' => 91, 'charset' => 92, 'max_line_fraction' => 0.001,
                'train_lines' => ['min' => 100, 'max' => 100, 'mean' => 100.0], 'pool_lines' => ['min' => 4, 'max' => 20, 'mean' => 11.98],
                'scheduled_lines' => ['min' => 1, 'max' => 14, 'mean' => 5.8636], 'never_scheduled' => 0],
            'datasets' => [
                ['key' => 'corpus', 'name' => 'Korpus teks sintetis', 'content' => 'Paragraf Wikipedia bahasa Jawa yang dialihaksarakan.',
                    'count' => 1034357, 'unit' => 'baris teks', 'articles' => 70923, 'roles' => ['train' => 931958, 'val' => 51756, 'test' => 50643],
                    'gates' => ['G1', 'G2'], 'status' => 'used', 'source' => 'Wikipedia bahasa Jawa, snapshot 20231101',
                    'source_url' => 'https://huggingface.co/datasets/wikimedia/wikipedia', 'license' => 'CC BY-SA 4.0', 'shareable' => true,
                    'in_repo' => false, 'share_note' => 'Dengan atribusi dan lisensi yang sama.'],
                ['key' => 'nusaaksara', 'name' => 'NusaAksara, bagian aksara Jawa', 'content' => 'Potongan baris dari pindaian buku cetak.',
                    'count' => 745, 'unit' => 'baris citra', 'pages' => 92, 'without_space' => 650, 'roles' => ['test' => 745], 'gates' => ['G3'],
                    'status' => 'used', 'source' => 'NusaAksara, ACL 2025', 'source_url' => 'https://huggingface.co/datasets/NusaAksara/NusaAksara',
                    'license' => 'non-komersial', 'shareable' => false, 'in_repo' => false, 'share_note' => 'Lisensi non-komersial. Dipakai lokal.'],
                ['key' => 'commons', 'name' => 'Papan nama Wikimedia Commons', 'content' => 'Foto papan nama beraksara Jawa.', 'count' => 43,
                    'unit' => 'baris citra', 'files' => 34, 'verified' => 0, 'pending' => 43, 'roles' => [], 'planned' => ['train', 'val'], 'gates' => [],
                    'status' => 'pending', 'source' => 'Wikimedia Commons, 34 berkas foto', 'license' => 'CC BY-SA 3.0, CC BY-SA 4.0',
                    'shareable' => true, 'in_repo' => false, 'share_note' => 'Lisensi bebas per berkas, atribusi wajib.'],
            ],
            'fonts' => [
                $this->font('NotoSansJavanese-Regular.ttf', 'core', ['train', 'val'], ['license' => 'SIL OFL 1.1', 'missing' => [], 'drops_space' => false, 'shared_with_test' => $none]),
                $this->font('CarakanJawa.otf', 'extra', ['train'], ['missing' => ['U+A98E'], 'drops_space' => false, 'review' => 'tampak benar',
                    'shared_with_test' => ['font' => 'javatext.ttf', 'same' => 72, 'of' => 73, 'family' => true]]),
                $this->font('GumregahNew.ttf', 'extra', ['train'], ['missing' => [], 'drops_space' => true, 'shared_with_test' => $none]),
                $this->font('BasaJan.ttf', 'extra', ['train'], ['missing' => [], 'drops_space' => false, 'shared_with_test' => $none,
                    'notes' => ['Di lingkungan training aturan GSUB font ini tidak diterapkan.']]),
                $this->font('javatext.ttf', 'test', ['test'], ['license' => 'bawaan Windows (Microsoft), tidak boleh disebarkan', 'missing' => []]),
                $this->font('Damarwulan.ttf', 'review', [], ['review' => 'MERAGUKAN: susun-3 bertabrakan']),
                $this->font('Istaka.ttf', 'rejected', [], ['review' => 'DITOLAK: wulu tergeser']),
            ],
            'support' => [
                ['key' => 'charset', 'file' => 'data/tokenizer.json', 'characters' => 92, 'classes' => 93],
                ['key' => 'charlm', 'file' => 'data/charlm/o5_n300000_ds0.5.pkl', 'order' => 5, 'lines' => 300000, 'drop_space_prob' => 0.5, 'split' => 'train'],
                ['key' => 'vlm_blind', 'file' => 'out/eval/vlm_blind_50.json', 'lines' => 50],
                ['key' => 'jenis_baru', 'lines' => 1],
            ],
            'warnings' => [],
        ];
    }

    private function write(array $card): void
    {
        file_put_contents($this->dir.'/datasets.json', json_encode($card, JSON_UNESCAPED_UNICODE));
    }

    private function import(array $card): void
    {
        $this->write($card);
        $this->artisan('aksara:datasets', ['--path' => $this->dir])->assertSuccessful();
    }

    public function test_guests_are_redirected_and_an_empty_page_explains_the_export(): void
    {
        $this->get('/dataset')->assertRedirect('/login');

        $this->actingAs(User::factory()->create());
        $this->get('/dataset')->assertOk()
            ->assertSee('Kartu data belum diimpor')->assertSee('scripts/export_datasets.py')->assertSee('php artisan aksara:datasets')
            ->assertSee('href="'.route('dataset').'"', false)   // menu sidebar
            ->assertDontSee('Daftar dataset');
    }

    public function test_command_imports_the_card_and_the_page_shows_it(): void
    {
        $this->write($this->card());
        $this->artisan('aksara:datasets', ['--path' => $this->dir])
            ->expectsOutputToContain('Kartu data diimpor: 3 dataset, 7 font, korpus 1.034.357 baris, pemakaian run fase7_track.')
            ->assertSuccessful();
        $report = DatasetReport::sole();
        $this->assertSame(['fase7_track', '2026-10-06T00:12:00', 1], [$report->official_run, $report->generated_at, $report->schema]);

        $this->actingAs(User::factory()->create());
        $page = $this->get('/dataset')->assertOk();
        $html = $page->getContent();

        // Empat ubin, masing-masing dengan angkanya sendiri.
        $this->assertSame('Korpus teks 1.034.357 baris 70.923 artikel Wikipedia bahasa Jawa, dialihaksarakan ke aksara Jawa', $this->tile($html, 'Korpus teks'));
        $this->assertSame('Pembagian 90 / 5 / 5 % artikel Latih / validasi / uji. Menurut jumlah baris 90,1 / 5,0 / 4,9% · teks yang sama di dua bagian: 0 baris',
            $this->tile($html, 'Pembagian'));
        $this->assertSame('Data nyata 745 baris uji 92 halaman pindaian · semuanya untuk uji, tidak untuk latih', $this->tile($html, 'Data nyata'));
        $this->assertSame('Font latih 4 font 1 inti + 3 tambahan · font uji: javatext', $this->tile($html, 'Font latih'));

        // Pembagian: batang berskala, arti tiap bagian, dan pemakaian run resmi (48.000 / 931.958 = 5,15%; 500 / 51.756 =
        // 0,966%; 10.000 / 50.643 = 19,746%).
        $page->assertSee('Latih 90,1%, Validasi 5,0%, Uji 4,9%')->assertSee('width: 90.1%', false);
        $train = $this->part($html, 'data-part', 'train');
        $this->assertStringContainsString('Latih 90,1% 931.958 baris', $train);
        $this->assertStringContainsString('Ibarat buku latihan soal', $train);
        $this->assertStringContainsString('48.000 baris dijadwalkan run fase7_track', $train);
        $this->assertStringContainsString('5,2% bagian latih. Dari kumpulan 100.000 baris acak (seed 0): 1.500 langkah × 32 = 48.000 sampel.', $train);
        $val = $this->part($html, 'data-part', 'val');
        $this->assertStringContainsString('Validasi 5,0% 51.756 baris', $val);
        $this->assertStringContainsString('500 baris pertama, tiap pemeriksaan', $val);
        $this->assertStringContainsString('1,0% bagian validasi. Diperiksa 3 kali (langkah 500, 1.000, 1.500) dengan 2 font inti, tanpa augmentasi.', $val);
        $test = $this->part($html, 'data-part', 'test');
        $this->assertStringContainsString('Uji 4,9% 50.643 baris Tidak pernah dilatihkan. Dipakai untuk angka yang dilaporkan.', $test);
        $this->assertStringContainsString('10.000 baris pertama untuk G1 dan G2', $test);
        $this->assertStringContainsString('19,7% bagian uji. Dirender dengan javatext; baris yang sama untuk tiap gerbang. Di luar gerbang resmi, '
            .'24 tes cepat penentu arah (sampai 200 baris pertama) dan 2 evaluasi penuh untuk run lain juga membaca bagian ini.', $test);
        $widths = fn (string $key) => $this->xpath($html)->evaluate("string(//*[@data-part='{$key}']//div[contains(@class, 'meter')]/span/@style)");
        $this->assertSame(['width: 5.15%', 'width: 0.966%', 'width: 19.746%'], [$widths('train'), $widths('val'), $widths('test')]);
        $page->assertSee('bagian uji saja sudah 50.643 baris, padahal gerbang hanya memakai 10.000, dan sepanjang rantai run resmi baru 6,9% '
            .'bagian latih yang pernah dilihat.');

        // Asal korpus dan sebaran panjang (279.814 / 1.034.357 = 27%; 754.543 = 73%).
        $page->assertSee('1.158.012')->assertSee('95,82% kandidat')->assertSee('CER ≤ 5%')->assertSee('−80.492')->assertSee('+5.280')
            ->assertSee('paling sedikit 100 kali')->assertDontSee('Penyesuaian lain');
        $this->assertSame('20–29 279.814 · 27%', $this->part($html, 'data-bin', '20–29'));
        $this->assertSame('30–80 754.543 · 73%', $this->part($html, 'data-bin', '30–80'));

        // Rantai checkpoint: hanya run resmi yang bertanda.
        $page->assertSee('Checkpoint fase7_track melanjutkan bobot 1 run sebelumnya. Semua run mengambil barisnya dari kumpulan acak yang sama '
            .'(seed 0). Sepanjang rantai, model melihat 6,9% bagian latih.')->assertDontSee('rantai di atas tidak lengkap');
        $this->assertSame('fase6_ctrl 1.500 48.000 100.000 19.712 10 augmentasi fase5 · spasi dibuang p 0,5', $this->part($html, 'data-run', 'fase6_ctrl'));
        $this->assertSame('fase7_track resmi 1.500 48.000 100.000 48.000 10 augmentasi fase5 · spasi dibuang p 0,5 · jarak antar suku kata p 0,5 (hingga 0,3 em)',
            $this->part($html, 'data-run', 'fase7_track'));
        $page->assertSeeInOrder(['Seluruh rantai', '3.000', '96.000', '100.000', '64.595', '6,9% bagian latih pernah dijadwalkan']);

        // Daftar dataset: peran, status, dan boleh tidaknya disebar per baris.
        $corpus = $this->part($html, 'data-dataset', 'corpus');
        $this->assertStringContainsString('1.034.357 baris teks 70.923 artikel latih 931.958 validasi 51.756 uji 50.643 gerbang G1, G2', $corpus);
        $this->assertStringContainsString('Lisensi: CC BY-SA 4.0 Boleh disebar', $corpus);
        $this->assertStringNotContainsString('Tidak disebarkan', $corpus);
        $this->assertStringNotContainsString('menunggu verifikasi', $corpus);
        $real = $this->part($html, 'data-dataset', 'nusaaksara');
        $this->assertStringContainsString('745 baris citra 92 halaman uji 745 gerbang G3', $real);
        $this->assertStringContainsString('Lisensi: non-komersial Tidak disebarkan', $real);
        $this->assertStringNotContainsString('Boleh disebar', $real);
        $this->assertStringNotContainsString('menunggu verifikasi', $real);
        $commons = $this->part($html, 'data-dataset', 'commons');
        $this->assertStringContainsString('43 baris citra 0 terverifikasi 43 baris menunggu verifikasi calon latih dan validasi', $commons);
        $this->assertStringContainsString('Boleh disebar', $commons);
        $page->assertSee('href="https://huggingface.co/datasets/NusaAksara/NusaAksara"', false);

        // Font: peran dan catatan per baris.
        $this->assertSame('NotoSansJavanese-Regular ada di repo latih validasi SIL OFL 1.1 –', $this->part($html, 'data-font', 'NotoSansJavanese-Regular.ttf'));
        $this->assertSame('CarakanJawa latih tidak tercantum Sekeluarga dengan font uji javatext: lebar 72 dari 73 aksara, angka, dan pada persis sama. '
            .'Tidak punya U+A98E (o).', $this->part($html, 'data-font', 'CarakanJawa.otf'));
        $this->assertStringContainsString('Spasinya nyaris tak tampak', $this->part($html, 'data-font', 'GumregahNew.ttf'));
        $this->assertStringContainsString('aturan GSUB font ini tidak diterapkan', $this->part($html, 'data-font', 'BasaJan.ttf'));
        $this->assertSame('javatext uji bawaan Windows (Microsoft), tidak boleh disebarkan –', $this->part($html, 'data-font', 'javatext.ttf'));
        $page->assertSee('Tidak dipakai karena meragukan: Damarwulan (susun-3 bertabrakan).')
            ->assertSee('Tidak dipakai karena ditolak: Istaka (wulu tergeser).');

        // Batasan dan data pendukung.
        $page->assertSee('Font uji javatext sekeluarga dengan font latih CarakanJawa: lebar 72 dari 73 aksara, angka, dan pada persis sama. G1 dan G2 karena itu')
            ->assertSee('Semua 745 baris NusaAksara, bagian aksara Jawa dipakai sebagai data uji, dan tidak ada data nyata untuk validasi.')
            ->assertSee('(data itu sudah dievaluasi 36 kali sepanjang proyek)')->assertSee('berasal dari satu terbitan dengan huruf yang sama')
            ->assertSee('43 baris papan nama Wikimedia Commons masih menunggu verifikasi pembaca aksara.')
            ->assertSee('44 dari 91 karakter aksara Jawa di charset ada di hanya 100 baris latih masing-masing')
            ->assertSee('Di antara 48.000 baris yang dijadwalkan run resmi, tiap karakter itu muncul di 1–14 baris, rata-rata 5,9.')
            ->assertDontSee('hasilnya ada di halaman Metode')
            ->assertSee('92 karakter + blank = 93 kelas')->assertSee('order 5 · 300.000 baris')->assertSee('Uji buta VLM')
            ->assertSee('belum diimpor')->assertSee('belum ada label')
            ->assertDontSee('tidak sejalan')->assertDontSee('jenis_baru');
    }

    public function test_reimport_replaces_the_card_and_web_side_support_counts_come_from_the_database(): void
    {
        $this->import($this->card());
        $card = $this->card();
        $card['corpus']['lines'] = 2000000;
        $card['datasets'][0]['count'] = 2000000;
        $card['corpus']['other'] = -12;
        $card['lineage']['complete'] = false;
        $card['warnings'] = ['Log run fase7_track tidak menjelaskan semua 1500 langkahnya.'];
        $this->import($card);
        $this->assertSame(1, DatasetReport::count());

        LineAnnotation::create(['dataset' => 'nusaaksara', 'external_id' => 'a.png', 'transliteration' => 'sugeng', 'translation' => 'selamat',
            'speech_level' => 'krama', 'source' => 'uji']);
        LineAnnotation::create(['dataset' => 'nusaaksara', 'external_id' => 'b.png', 'transliteration' => 'rawuh', 'source' => 'uji']);
        DictionarySource::create(['dictionary' => 'jv', 'key' => 'satu', 'name' => 'Sumber Satu', 'license' => 'CC BY-SA 4.0', 'url' => 'https://contoh.test/1', 'entries' => 3000]);
        DictionarySource::create(['dictionary' => 'jv', 'key' => 'dua', 'name' => 'Sumber Dua', 'license' => 'CC BY-SA 4.0', 'url' => 'https://contoh.test/2', 'entries' => 2888]);

        $this->actingAs(User::factory()->create());
        $this->get('/dataset')->assertOk()
            ->assertSee('2.000.000')->assertDontSee('1.034.357')
            ->assertSee('Penyesuaian lain')->assertSee('−12')
            ->assertSee('melanjutkan bobot sedikitnya 1 run sebelumnya')->assertSee('rantai di atas tidak lengkap')
            ->assertSee('Log run fase7_track tidak menjelaskan semua 1500 langkahnya.')
            ->assertSeeInOrder(['Transliterasi manusia', '2 baris', 'Terjemahan manusia', '1 baris', 'Label tingkat tutur', '1 baris'])
            ->assertSeeInOrder(['Kamus kata bahasa Jawa', '5.888 entri', 'Sumber Satu + Sumber Dua (CC BY-SA 4.0)', 'Kamus kata bahasa Indonesia', 'belum diimpor']);

        // Begitu baris uji diimpor, hanya anotasi yang cocok dengan baris itu yang dihitung (b.png tidak punya baris).
        Line::create(['dataset' => 'nusaaksara', 'external_id' => 'a.png', 'image_path' => 'data/real/nusaaksara/images/a.png',
            'width' => 100, 'height' => 20, 'reference' => 'ꦏ', 'tags' => []]);
        $this->get('/dataset')->assertOk()
            ->assertSeeInOrder(['Transliterasi manusia', '1 baris', 'Terjemahan manusia', '1 baris', 'Label tingkat tutur', '1 baris']);
    }

    public function test_page_does_not_depend_on_the_key_order_of_the_stored_card(): void
    {
        // jsonb PostgreSQL mengurutkan ulang kunci objek (menurut panjang lalu byte: val, test, train). Urutan abjad di
        // sini (test, train, val) bukan urutan jsonb, tetapi sama-sama bukan urutan ekspor: cukup untuk menguji bahwa
        // halaman mengurutkan peran sendiri.
        $sorted = function (array $value) use (&$sorted): array {
            if (! array_is_list($value)) {
                ksort($value);
            }

            return array_map(fn ($item) => is_array($item) ? $sorted($item) : $item, $value);
        };
        $card = $sorted($this->card());
        $this->assertSame(['test', 'train', 'val'], array_keys($card['datasets'][0]['roles']));
        $this->assertSame(['test', 'train', 'val'], array_keys($card['corpus']['split_buckets']));
        $this->import($card);

        $this->actingAs(User::factory()->create());
        $html = $this->get('/dataset')->assertOk()->assertSee('Latih 90,1%, Validasi 5,0%, Uji 4,9%')->getContent();
        $this->assertStringContainsString('Pembagian 90 / 5 / 5 % artikel Latih / validasi / uji. Menurut jumlah baris 90,1 / 5,0 / 4,9%', $this->tile($html, 'Pembagian'));
        $this->assertStringContainsString('latih 931.958 validasi 51.756 uji 50.643 gerbang G1, G2', $this->part($html, 'data-dataset', 'corpus'));
        $this->assertStringContainsString('rata-rata 5,9', $html);
    }

    public function test_limits_and_sentences_follow_the_card(): void
    {
        // Font uji tidak sekeluarga, data nyata punya bagian validasi, dan run resmi menyisipkan aksara langka: ketiga
        // batasan tidak berlaku lagi.
        $card = $this->card();
        $card['fonts'][1]['shared_with_test'] = ['font' => 'javatext.ttf', 'same' => 3, 'of' => 73, 'family' => false];
        $card['datasets'][2] = ['roles' => ['train' => 30, 'val' => 13], 'status' => 'used', 'verified' => 43, 'pending' => 0, 'planned' => []] + $card['datasets'][2];
        $card['usage']['train']['rare_insert_prob'] = 0.3;
        $card['lineage']['runs'][1]['rare_insert_prob'] = 0.3;
        $this->import($card);

        $this->actingAs(User::factory()->create());
        $page = $this->get('/dataset')->assertOk()->assertSee('Daftar dataset')
            ->assertDontSee('Batasan data')->assertDontSee('Font uji bukan font yang benar-benar baru')
            ->assertDontSee('Sekeluarga dengan font uji')->assertDontSee('menunggu verifikasi')->assertDontSee('calon')
            ->assertDontSee('Aksara langka nyaris tidak terlihat');
        $html = $page->getContent();
        $this->assertStringContainsString('43 baris citra 43 terverifikasi latih 30 validasi 13', $this->part($html, 'data-dataset', 'commons'));
        $this->assertSame('Data nyata 745 baris uji 92 halaman pindaian · 30 baris nyata lain dipakai melatih', $this->tile($html, 'Data nyata'));
        $this->assertStringContainsString('sisipan aksara langka', $this->part($html, 'data-run', 'fase7_track'));

        // Sebagian draf sudah diverifikasi: yang menunggu = sisanya, bukan seluruh draf. Kartu metode punya butir uji
        // sisipan aksara langka, jadi batasannya menunjuk ke halaman Metode.
        $card = $this->card();
        $card['datasets'][2] = ['verified' => 40, 'pending' => 3] + $card['datasets'][2];
        MethodReport::create(['schema' => 1, 'generated_at' => 'x', 'official_run' => 'fase7_track', 'source_path' => 'x',
            'payload' => ['groups' => [['methods' => [['key' => 'rare']]]]]]);
        $this->import($card);
        $page = $this->get('/dataset')->assertOk()
            ->assertSee('3 baris papan nama Wikimedia Commons masih menunggu verifikasi pembaca aksara.')
            ->assertSee('rata-rata 5,9. Sisipan aksara langka saat render sudah diuji; hasilnya ada di halaman Metode.');
        $this->assertStringContainsString('43 baris citra 40 terverifikasi 3 baris menunggu verifikasi', $this->part($page->getContent(), 'data-dataset', 'commons'));
    }

    public function test_values_the_export_did_not_compute_are_shown_as_not_computed(): void
    {
        // Run resmi dilatih dan divalidasi dengan data nyata: jadwal baris sintetis tidak dihitung (null), dan hanya ada
        // satu run di rantai.
        $card = $this->card();
        $official = ['distinct_lines' => null, 'real_train' => true, 'real_val' => true, 'seed' => 7] + $card['usage']['train'];
        $card['usage']['train'] = $official;
        $card['lineage'] = ['complete' => true, 'steps' => 1500, 'samples' => 48000, 'pool' => 100000, 'distinct_lines' => null, 'runs' => [$official]];
        $card['rare']['scheduled_lines'] = [];
        $card['rare']['never_scheduled'] = null;
        $card['warnings'] = ['Run fase7_track memakai --overfit atau --real-train; baris berbeda tidak dihitung.'];
        $this->import($card);

        $this->actingAs(User::factory()->create());
        $page = $this->get('/dataset')->assertOk();
        $html = $page->getContent();
        $train = $this->part($html, 'data-part', 'train');
        $this->assertStringContainsString('Dipakai run resmi baris sintetis: tidak dihitung Run fase7_track dilatih dengan data nyata', $train);
        $this->assertStringNotContainsString('bagian latih.', $train);
        $this->assertSame('', $this->xpath($html)->evaluate("string(//*[@data-part='train']//div[contains(@class, 'meter')]/span/@style)"));
        $this->assertStringContainsString('tidak dipakai run fase7_track Run ini divalidasi dengan data nyata, bukan dengan bagian ini.', $this->part($html, 'data-part', 'val'));
        $this->assertStringContainsString('1.500 48.000 100.000 tidak dihitung 10', $this->part($html, 'data-run', 'fase7_track'));
        $this->assertStringContainsString('data nyata ikut dilatih', $this->part($html, 'data-run', 'fase7_track'));
        $page->assertSee('Checkpoint fase7_track dilatih dalam satu run, dari bobot acak. Jumlah baris berbeda di sepanjang rantai tidak dihitung.')
            ->assertSee('padahal gerbang hanya memakai 10.000. Membagi ulang')
            ->assertDontSee('pernah dijadwalkan')->assertDontSee('Aksara langka nyaris tidak terlihat')->assertDontSee('kumpulan acak yang sama');

        // Dua run dengan seed berbeda, dan seluruh bagian latih terjadwal.
        $card = $this->card();
        $card['lineage']['runs'][0]['seed'] = 5;
        $card['lineage']['distinct_lines'] = 931958;
        $this->import($card);
        $this->get('/dataset')->assertOk()
            ->assertSee('Run-run itu memakai seed yang berbeda, jadi kumpulan barisnya tidak sama. Sepanjang rantai, model melihat 100,0% bagian latih.')
            ->assertSee('padahal gerbang hanya memakai 10.000. Membagi ulang')->assertDontSee('baru 100,0%');
    }

    public function test_a_card_for_another_run_than_the_official_pipeline_is_flagged(): void
    {
        $this->import($this->card());
        $this->actingAs(User::factory()->create());
        $pipeline = Pipeline::create(['key' => 'crnn_fase7_track', 'label' => 'CRNN fase7_track', 'config' => 'crnn:fase7_track@1500 · greedy',
            'kind' => 'crnn', 'status' => 'done', 'sort' => 0, 'official' => true]);
        $this->get('/dataset')->assertOk()->assertDontSee('tidak sejalan');

        $pipeline->update(['official' => false]);
        Pipeline::create(['key' => 'crnn_fase8_x', 'label' => 'CRNN fase8_x', 'config' => 'crnn:fase8_x@1500 · greedy',
            'kind' => 'crnn', 'status' => 'done', 'sort' => 1, 'official' => true]);
        $this->get('/dataset')->assertOk()->assertSee('Kartu data dan hasil OCR tidak sejalan')
            ->assertSee('dihitung untuk run fase7_track, sedangkan angka resmi web sekarang milik CRNN fase8_x')
            ->assertSee('scripts/export_results.py')->assertSee('scripts/export_datasets.py');

        // Kartu untuk run yang tidak dikenal ekspor hasil (kunci pipeline kosong) juga tidak cocok dengan pipeline resmi.
        Pipeline::where('key', 'crnn_fase8_x')->delete();
        $pipeline->update(['official' => true]);
        $card = $this->card();
        $card['official']['pipeline'] = null;
        $this->import($card);
        $this->get('/dataset')->assertOk()->assertSee('Kartu data dan hasil OCR tidak sejalan');
    }

    public function test_a_source_link_is_only_rendered_for_http_urls(): void
    {
        $card = $this->card();
        $card['datasets'][0]['source_url'] = 'javascript:alert(document.domain)';
        $card['datasets'][1]['source_url'] = '//contoh.test/x';
        $this->import($card);

        $this->actingAs(User::factory()->create());
        $html = $this->get('/dataset')->assertOk()->assertSee('Wikipedia bahasa Jawa, snapshot 20231101')->assertSee('NusaAksara, ACL 2025')->getContent();
        $this->assertStringNotContainsString('javascript:', $html);
        $this->assertStringNotContainsString('href="//contoh.test', $html);
        $this->assertSame(0, $this->xpath($html)->query('//*[@data-dataset]//a')->length);
    }

    public function test_importer_rejects_cards_it_cannot_show_and_keeps_the_previous_one(): void
    {
        $this->import($this->card());
        $importer = app(DatasetImporter::class);
        $rejects = function (string $expected, ?array $card = null, ?string $raw = null) use ($importer) {
            file_put_contents($this->dir.'/datasets.json', $raw ?? json_encode($card, JSON_UNESCAPED_UNICODE));
            try {
                $importer->import($this->dir);
                $this->fail("Kartu data seharusnya ditolak: {$expected}");
            } catch (RuntimeException $e) {
                $this->assertStringContainsString($expected, $e->getMessage());
            }
        };
        $changed = function (callable $change): array {
            $card = $this->card();
            $change($card);

            return $card;
        };

        $rejects('bukan JSON yang sah', raw: '{"schema": 1,');
        $rejects('Skema kartu data "1" tidak didukung', raw: '{"schema": "1"}');
        $rejects('Skema kartu data 2 tidak didukung (butuh 1)', ['schema' => 2] + $this->card());
        $rejects('Skema kartu data null tidak didukung', raw: '{"corpus": {}}');
        $rejects('bukan kartu data: isinya harus objek JSON', raw: '"teks"');
        $rejects('bukan kartu data: isinya harus objek JSON', raw: '[{"schema": 1}]');
        // Kunci yang dibaca halaman tanpa nilai cadangan: hilang satu saja ditolak, dengan nama kuncinya.
        $rejects('kunci usage.train.pool tidak ada', $changed(function (&$c) {
            unset($c['usage']['train']['pool']);
        }));
        $rejects('kunci rare.codepoints tidak ada', $changed(function (&$c) {
            unset($c['rare']);
        }));
        $rejects('kunci corpus.leaked_lines tidak ada', $changed(function (&$c) {
            unset($c['corpus']['leaked_lines']);
        }));
        $rejects('kunci lineage.runs.*.fonts tidak ada', $changed(function (&$c) {
            unset($c['lineage']['runs'][0]['fonts']);
        }));
        $rejects('kunci datasets.*.share_note tidak ada', $changed(function (&$c) {
            unset($c['datasets'][1]['share_note']);
        }));
        $rejects('kunci fonts.*.roles tidak ada', $changed(function (&$c) {
            unset($c['fonts'][2]['roles']);
        }));
        $rejects('kunci usage.val.steps tidak ada', $changed(function (&$c) {
            unset($c['usage']['val']['steps']);
        }));
        // Tipe yang salah.
        $rejects('kunci corpus.lines harus angka', $changed(function (&$c) {
            $c['corpus']['lines'] = 'sejuta';
        }));
        $rejects('kunci fonts harus daftar', $changed(function (&$c) {
            $c['fonts'] = 'bukan daftar';
        }));
        $rejects('kunci datasets.*.roles.* harus angka', $changed(function (&$c) {
            $c['datasets'][0]['roles']['train'] = 'banyak';
        }));
        $rejects('kunci datasets.*.shareable harus benar/salah', $changed(function (&$c) {
            $c['datasets'][0]['shareable'] = 'ya';
        }));
        $rejects('kunci warnings.* harus teks', $changed(function (&$c) {
            $c['warnings'] = [['bukan teks']];
        }));
        $rejects('kunci official.run lebih panjang dari 255 karakter', $changed(function (&$c) {
            $c['official']['run'] = str_repeat('x', 300);
        }));
        // Bentuk yang tidak bisa dinyatakan daftar kunci.
        $rejects('splits harus berisi train, val, test', $changed(function (&$c) {
            $c['splits'] = array_reverse($c['splits']);
        }));
        $rejects('lineage.runs dan corpus.length_histogram tidak boleh kosong', $changed(function (&$c) {
            $c['corpus']['length_histogram'] = [];
        }));
        // Lolos daftar kunci tetapi tidak bisa ditampilkan: uji tampil saat impor menolaknya.
        $rejects('Kartu data tidak bisa ditampilkan halamannya', $changed(function (&$c) {
            $c['fonts'][1]['missing'] = [['bukan', 'kode']];
        }));
        $rejects('Kartu data tidak bisa ditampilkan halamannya', $changed(function (&$c) {
            $c['rare']['scheduled_lines'] = ['min' => 'satu', 'max' => 2, 'mean' => 1.5];
        }));

        unlink($this->dir.'/datasets.json');
        $this->artisan('aksara:datasets', ['--path' => $this->dir])
            ->expectsOutputToContain('python scripts/export_datasets.py')->assertFailed();

        $this->assertSame(1, DatasetReport::count());
        $this->assertSame(1034357, DatasetReport::current()->payload['corpus']['lines']);
        $this->actingAs(User::factory()->create());
        $this->get('/dataset')->assertOk()->assertSee('1.034.357');
    }

    /** Folder hasil OCR terkecil yang diterima aksara:import: satu pipeline, satu baris buatan, tanpa data NusaAksara. */
    private function writeResults(): void
    {
        file_put_contents($this->dir.'/manifest.json', json_encode([
            'schema' => 1, 'generated' => '2026-10-06T00:00:00', 'limited' => false, 'official' => 'crnn_fase7_track',
            'pipelines' => [['key' => 'crnn_fase7_track', 'label' => 'CRNN fase7_track', 'config' => 'crnn:fase7_track@1500 · greedy',
                'kind' => 'crnn', 'status' => 'done', 'sort' => 0]],
            'gates' => [], 'metrics' => [], 'ablation' => [], 'confusion' => ['pipeline' => 'crnn_fase7_track', 'items' => []],
        ], JSON_UNESCAPED_UNICODE));
        file_put_contents($this->dir.'/lines.jsonl', json_encode(['dataset' => 'contoh', 'id' => 'a.png', 'image' => 'contoh/a.png', 'width' => 100,
            'height' => 20, 'reference' => 'ꦏ', 'condition' => 'contoh', 'source_id' => 'hal_0', 'tags' => []], JSON_UNESCAPED_UNICODE)."\n");
        file_put_contents($this->dir.'/predictions.jsonl', json_encode(['line' => 'a.png', 'pipeline' => 'crnn_fase7_track', 'text' => 'ꦏ', 'cer' => 0.0,
            'segments' => [['t' => 'ꦏ', 's' => 'eq']]], JSON_UNESCAPED_UNICODE)."\n");
    }

    public function test_results_import_picks_up_the_cards_next_to_the_manifest(): void
    {
        $this->writeResults();
        $this->artisan('aksara:import', ['--path' => $this->dir])
            ->expectsOutputToContain('Kartu halaman Dataset (datasets.json) tidak ada di folder itu')
            ->expectsOutputToContain('Kartu halaman Metode (methods.json) tidak ada di folder itu')->assertSuccessful();
        $this->assertSame(0, DatasetReport::count());

        $this->write($this->card());
        $this->artisan('aksara:import', ['--path' => $this->dir])
            ->expectsOutputToContain('Kartu data diimpor: 3 dataset, 7 font')->assertSuccessful();
        $this->assertSame('fase7_track', DatasetReport::sole()->official_run);
        $this->assertSame('crnn_fase7_track', Pipeline::official()->key);

        // Kartu yang ditolak tidak menggagalkan impor hasil OCR: hasilnya sudah tersimpan, kartu lama dipertahankan.
        $this->write(['schema' => 9] + $this->card());
        $this->artisan('aksara:import', ['--path' => $this->dir])
            ->expectsOutputToContain('PERINGATAN: hasil OCR sudah diimpor, tetapi kartu halaman Dataset ditolak')->assertSuccessful();
        $this->assertSame('fase7_track', DatasetReport::sole()->official_run);
        // Perintah kartunya sendiri tetap keluar dengan kode gagal.
        $this->artisan('aksara:datasets', ['--path' => $this->dir])->expectsOutputToContain('Skema kartu data 9 tidak didukung')->assertFailed();
    }

    public function test_the_card_exported_on_this_machine_imports_and_renders(): void
    {
        $dir = config('aksara.ocr_repo').DIRECTORY_SEPARATOR.config('aksara.results_dir');
        if (! is_file(DatasetImporter::path($dir))) {
            $this->markTestSkipped('out/results/datasets.json belum diekspor di mesin ini.');
        }
        $this->artisan('aksara:datasets', ['--path' => $dir])->assertSuccessful();
        $card = DatasetReport::sole()->payload;

        $this->actingAs(User::factory()->create());
        $this->get('/dataset')->assertOk()
            ->assertSee(nfmt($card['corpus']['lines']))->assertSee($card['official']['run'])
            ->assertSee('Daftar dataset')->assertSee('Font per peran')->assertSee('Data pendukung');
    }
}
