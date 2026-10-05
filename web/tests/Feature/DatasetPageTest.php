<?php

namespace Tests\Feature;

use App\Models\DatasetReport;
use App\Models\DictionarySource;
use App\Models\Line;
use App\Models\LineAnnotation;
use App\Models\Pipeline;
use App\Models\User;
use App\Services\DatasetImporter;
use Illuminate\Filesystem\Filesystem;
use Illuminate\Foundation\Testing\RefreshDatabase;
use RuntimeException;
use Tests\TestCase;

class DatasetPageTest extends TestCase
{
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
    private function chainRun(string $name, ?string $init, int $steps, int $pool, int $distinct, int $fonts, array $extra = []): array
    {
        return $extra + ['run' => $name, 'init' => $init, 'checkpoint' => "out/checkpoints/{$name}/last_snapshot.pt",
            'steps' => $steps, 'batch_size' => 32, 'samples' => $steps * 32, 'pool' => $pool, 'seed' => 0, 'processes' => 1,
            'segments' => [[0, $steps]], 'distinct_lines' => $distinct, 'fonts' => $fonts, 'extra_fonts' => max(0, $fonts - 2),
            'augment' => 'fase5', 'drop_space_prob' => 0.5, 'track_prob' => 0.0, 'track_max' => 0.0, 'rare_insert_prob' => 0.0,
            'rare_opener_prob' => 0.0, 'val_lines' => 500, 'val_steps' => [500, 1000, 1500]];
    }

    private function font(string $file, string $group, array $roles, array $extra = []): array
    {
        return $extra + ['file' => $file, 'group' => $group, 'roles' => $roles, 'license' => 'tidak tercantum',
            'source' => 'https://contoh.test/'.$file, 'in_repo' => $group === 'core', 'available' => true, 'notes' => []];
    }

    /** Kartu data kecil dengan angka korpus sebenarnya (2026-10-05), tanpa teks label apa pun. */
    private function card(): array
    {
        $official = $this->chainRun('fase7_track', 'fase6_ctrl', 1500, 100000, 48000, 10, ['track_prob' => 0.5, 'track_max' => 0.3]);
        $none = ['font' => 'javatext.ttf', 'same' => 0, 'of' => 73, 'family' => false];

        return [
            'schema' => 1, 'generated' => '2026-10-05T22:56:12', 'source' => 'metopenv5',
            'official' => ['run' => 'fase7_track', 'pipeline' => 'crnn_fase7_track',
                'checkpoint' => 'out/checkpoints/fase7_track/last_snapshot.pt', 'step' => 1500],
            'corpus' => ['articles' => 70923, 'candidates' => 1158012, 'roundtrip_passed' => 1109569, 'roundtrip_pass_rate' => 0.958167,
                'duplicates' => 80492, 'injected' => 5280, 'other' => 0, 'lines' => 1034357, 'leaked_lines' => 0,
                'split_buckets' => ['train' => 90, 'val' => 5, 'test' => 5], 'max_roundtrip_cer' => 0.05,
                'min_count' => ['train' => 100, 'val' => 10, 'test' => 10], 'built' => '2026-09-13',
                'length_histogram' => [['range' => '20-29', 'lines' => 279814], ['range' => '30-39', 'lines' => 754543]]],
            'splits' => [
                ['key' => 'train', 'file' => 'data/splits/train.txt', 'lines' => 931958, 'injected' => 4400, 'share' => 0.9010022651753699],
                ['key' => 'val', 'file' => 'data/splits/val.txt', 'lines' => 51756, 'injected' => 440, 'share' => 0.05003688281705446],
                ['key' => 'test', 'file' => 'data/splits/test.txt', 'lines' => 50643, 'injected' => 440, 'share' => 0.048960852007575724],
            ],
            'usage' => [
                'train' => $official,
                'val' => ['lines' => 500, 'steps' => [500, 1000, 1500], 'fonts' => 2],
                'test' => ['lines' => 10000, 'gates' => [
                    ['code' => 'G1', 'lines' => 10000, 'fonts' => ['javatext.ttf'], 'split' => 'test', 'augment' => 'none', 'source' => 'out/eval/fase7_track_G1_10k.json'],
                    ['code' => 'G2', 'lines' => 10000, 'fonts' => ['javatext.ttf'], 'split' => 'test', 'augment' => 'heavy', 'source' => 'out/eval/fase7_track_G2_10k.json']]],
                'real' => ['lines' => 745, 'gates' => [['code' => 'G3', 'lines' => 745, 'fonts' => [], 'split' => null, 'augment' => null, 'source' => 'out/eval/fase7_track_G3_full.json']]],
            ],
            'lineage' => ['complete' => true, 'steps' => 3000, 'samples' => 96000, 'pool' => 100000, 'distinct_lines' => 64612, 'runs' => [
                $this->chainRun('fase6_ctrl', null, 1500, 100000, 19712, 10, ['processes' => 3, 'segments' => [[0, 500], [500, 884], [884, 1500]]]),
                $official,
            ]],
            'rare' => ['codepoints' => 44, 'javanese' => 91, 'charset' => 92, 'max_line_fraction' => 0.001,
                'train_lines' => ['min' => 100, 'max' => 100, 'mean' => 100.0], 'pool_lines' => ['min' => 4, 'max' => 20, 'mean' => 11.98],
                'scheduled_lines' => ['min' => 1, 'max' => 12, 'mean' => 5.5], 'never_scheduled' => 0],
            'datasets' => [
                ['key' => 'corpus', 'name' => 'Korpus teks sintetis', 'content' => 'Paragraf Wikipedia bahasa Jawa yang dialihaksarakan.',
                    'count' => 1034357, 'unit' => 'baris teks', 'articles' => 70923, 'roles' => ['train' => 931958, 'val' => 51756, 'test' => 50643],
                    'gates' => ['G1', 'G2'], 'status' => 'used', 'source' => 'Wikipedia bahasa Jawa, snapshot 20231101',
                    'source_url' => 'https://huggingface.co/datasets/wikimedia/wikipedia', 'license' => 'CC BY-SA 4.0', 'shareable' => true,
                    'in_repo' => false, 'share_note' => 'Boleh disebar dengan atribusi dan lisensi yang sama.'],
                ['key' => 'nusaaksara', 'name' => 'NusaAksara, bagian aksara Jawa', 'content' => 'Potongan baris dari pindaian buku cetak.',
                    'count' => 745, 'unit' => 'baris citra', 'pages' => 92, 'without_space' => 650, 'roles' => ['test' => 745], 'gates' => ['G3'],
                    'status' => 'used', 'source' => 'NusaAksara, ACL 2025', 'source_url' => 'https://huggingface.co/datasets/NusaAksara/NusaAksara',
                    'license' => 'non-komersial', 'shareable' => false, 'in_repo' => false, 'share_note' => 'Tidak boleh disebar. Dipakai lokal.'],
                ['key' => 'commons', 'name' => 'Papan nama Wikimedia Commons', 'content' => 'Foto papan nama beraksara Jawa.', 'count' => 43,
                    'unit' => 'baris citra', 'files' => 34, 'verified' => 0, 'roles' => [], 'planned' => ['train', 'val'], 'gates' => [],
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
        $this->assertSame(['fase7_track', '2026-10-05T22:56:12', 1], [$report->official_run, $report->generated_at, $report->schema]);

        $this->actingAs(User::factory()->create());
        $page = $this->get('/dataset')->assertOk();
        // Angka pokok dan pembagian.
        $page->assertSee('1.034.357')->assertSee('70.923 artikel')->assertSee('90 / 5 / 5')
            ->assertSee('Menurut jumlah baris 90,1 / 5,0 / 4,9%')
            ->assertSee('90,1%')->assertSee('931.958 baris')->assertSee('5,0%')->assertSee('4,9%')
            ->assertSee('Latih 90,1%, Validasi 5,0%, Uji 4,9%')->assertSee('width: 90.1%', false)
            ->assertSee('Ibarat buku latihan soal')->assertSee('Ibarat try-out')->assertSee('Ibarat ujian akhir');
        // Pemakaian run resmi: 48.000 dari 931.958 baris latih = 5,2%; 500 dari 51.756 = 1,0%; 10.000 dari 50.643 = 19,7%.
        $page->assertSee('baris dijadwalkan run fase7_track')->assertSee('5,2%')->assertSee('1,0%')->assertSee('19,7%')
            ->assertSee('1.500 langkah × 32 = 48.000 sampel')->assertSee('Diperiksa 3 kali (langkah 500, 1.000, 1.500)')
            ->assertSee('baris pertama untuk G1 dan G2')->assertSee('Dirender dengan javatext');
        // Asal korpus.
        $page->assertSee('1.158.012')->assertSee('95,82% kandidat')->assertSee('CER ≤ 5%')->assertSee('−80.492')->assertSee('+5.280')
            ->assertSee('paling sedikit 100 kali')->assertSee('20–29')->assertDontSee('Penyesuaian lain');
        // Rantai checkpoint.
        $page->assertSee('melanjutkan bobot 1 run sebelumnya')->assertSee('fase6_ctrl')->assertSee('19.712')->assertSee('64.612')
            ->assertSee('6,9% bagian latih pernah dilihat')->assertSee('jarak antar suku kata p 0,5 (hingga 0,3 em)')
            ->assertSee('spasi dibuang p 0,5')->assertDontSee('rantai di atas tidak lengkap');
        // Daftar dataset.
        $page->assertSee('NusaAksara, bagian aksara Jawa')->assertSee('92 halaman')->assertSee('Lisensi: non-komersial')
            ->assertSee('Tidak boleh')->assertSee('menunggu verifikasi')->assertSee('calon latih dan validasi')->assertSee('0 terverifikasi')
            ->assertSee('gerbang G1, G2')->assertSee('https://huggingface.co/datasets/NusaAksara/NusaAksara');
        // Font: catatan yang dihitung ekspor dan catatan repo.
        $page->assertSee('CarakanJawa')->assertSee('Sekeluarga dengan font uji javatext: lebar 72 dari 73 aksara, angka, dan pada persis sama.')
            ->assertSee('Tidak punya U+A98E (o).')->assertSee('Spasinya nyaris tak tampak')->assertSee('aturan GSUB font ini tidak diterapkan')
            ->assertSee('ikut repo')->assertSee('Tidak dipakai karena meragukan:')->assertSee('title="MERAGUKAN: susun-3 bertabrakan"', false)
            ->assertSee('Tidak dipakai karena ditolak:')->assertSee('Istaka');
        // Batasan dan data pendukung.
        $page->assertSee('Font uji bukan font yang benar-benar baru')->assertSee('Font uji javatext sekeluarga dengan font latih CarakanJawa')
            ->assertSee('Data nyata tidak punya bagian validasi')->assertSee('Semua 745 baris nyata masuk bagian uji')
            ->assertSee('43 baris papan nama Wikimedia Commons masih menunggu verifikasi')
            ->assertSee('Aksara langka nyaris tidak terlihat saat training')->assertSee('44 dari 91 karakter aksara Jawa di charset ada di hanya 100 baris')
            ->assertSee('muncul di 1–12 baris, rata-rata 5,5')
            ->assertSee('92 karakter + blank = 93 kelas')->assertSee('order 5 · 300.000 baris')->assertSee('Uji buta VLM')
            ->assertSee('belum diimpor')->assertSee('belum ada label')
            ->assertDontSee('Kartu data tertinggal')->assertDontSee('jenis_baru');
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
            ->assertSee('rantai di atas tidak lengkap')
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
        // jsonb PostgreSQL mengurutkan ulang kunci objek; urutan abjad di sini (test, train, val) menirunya.
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
        $this->get('/dataset')->assertOk()
            ->assertSee('90 / 5 / 5')->assertDontSee('5 / 5 / 90')->assertDontSee('5 / 90 / 5')
            ->assertSee('Menurut jumlah baris 90,1 / 5,0 / 4,9%')
            ->assertSeeInOrder(['Korpus teks sintetis', 'latih', '931.958', 'validasi', '51.756', 'uji', '50.643', 'gerbang G1, G2'])
            ->assertSee('Latih 90,1%, Validasi 5,0%, Uji 4,9%')
            ->assertSee('Font uji bukan font yang benar-benar baru')->assertSee('muncul di 1–12 baris, rata-rata 5,5');
    }

    public function test_limits_disappear_when_the_card_no_longer_has_them(): void
    {
        $card = $this->card();
        $card['fonts'][1]['shared_with_test'] = ['font' => 'javatext.ttf', 'same' => 3, 'of' => 73, 'family' => false];
        $card['datasets'][2]['roles'] = ['train' => 30, 'val' => 13];
        $card['datasets'][2]['status'] = 'used';
        $card['rare'] = ['codepoints' => 0, 'javanese' => 91, 'charset' => 92, 'max_line_fraction' => 0.001, 'train_lines' => [],
            'pool_lines' => [], 'scheduled_lines' => [], 'never_scheduled' => 0];
        $this->import($card);

        $this->actingAs(User::factory()->create());
        $this->get('/dataset')->assertOk()->assertSee('Daftar dataset')
            ->assertDontSee('Batasan data')->assertDontSee('Font uji bukan font yang benar-benar baru')
            ->assertDontSee('Sekeluarga dengan font uji')->assertDontSee('menunggu verifikasi')
            ->assertSeeInOrder(['Papan nama Wikimedia Commons', 'latih', '30', 'validasi', '13']);
    }

    public function test_a_card_for_another_run_than_the_official_pipeline_is_flagged(): void
    {
        $this->import($this->card());
        $this->actingAs(User::factory()->create());
        $pipeline = Pipeline::create(['key' => 'crnn_fase7_track', 'label' => 'CRNN fase7_track', 'config' => 'crnn:fase7_track@1500 · greedy',
            'kind' => 'crnn', 'status' => 'done', 'sort' => 0, 'official' => true]);
        $this->get('/dataset')->assertOk()->assertDontSee('Kartu data tertinggal');

        $pipeline->update(['official' => false]);
        Pipeline::create(['key' => 'crnn_fase8_x', 'label' => 'CRNN fase8_x', 'config' => 'crnn:fase8_x@1500 · greedy',
            'kind' => 'crnn', 'status' => 'done', 'sort' => 1, 'official' => true]);
        $this->get('/dataset')->assertOk()->assertSee('Kartu data tertinggal')
            ->assertSee('dihitung untuk run fase7_track, sedangkan angka resmi web sekarang milik CRNN fase8_x');
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

        $rejects('bukan JSON yang sah', raw: '{"schema": 1,');
        $rejects('Skema kartu data "1" tidak didukung', raw: '{"schema": "1"}');
        $rejects('Skema kartu data 2 tidak didukung (butuh 1)', ['schema' => 2] + $this->card());
        $rejects('Skema kartu data null tidak didukung', raw: '{"corpus": {}}');
        $rejects('bukan kartu data: isinya harus objek JSON', raw: '"teks"');
        $card = $this->card();
        unset($card['usage']['train']['pool']);
        $rejects('kunci usage.train.pool tidak ada', $card);
        $card = $this->card();
        unset($card['rare']);
        $rejects('kunci rare.codepoints tidak ada', $card);
        $card = $this->card();
        $card['splits'] = array_reverse($card['splits']);
        $rejects('splits harus berisi train, val, test', $card);

        unlink($this->dir.'/datasets.json');
        $this->artisan('aksara:datasets', ['--path' => $this->dir])
            ->expectsOutputToContain('python scripts/export_datasets.py')->assertFailed();

        $this->assertSame(1, DatasetReport::count());
        $this->assertSame(1034357, DatasetReport::current()->payload['corpus']['lines']);
    }

    public function test_results_import_picks_up_the_card_next_to_the_manifest(): void
    {
        $fixture = base_path('tests/Fixtures/results');
        if (! is_dir($fixture)) {
            $this->markTestSkipped('Fixture hasil OCR tidak ikut repo (memuat potongan label NusaAksara).');
        }
        (new Filesystem)->copyDirectory($fixture, $this->dir);

        $this->artisan('aksara:import', ['--path' => $this->dir])
            ->expectsOutputToContain('Kartu data (datasets.json) tidak ada di folder itu')->assertSuccessful();
        $this->assertSame(0, DatasetReport::count());

        $this->write($this->card());
        $this->artisan('aksara:import', ['--path' => $this->dir])
            ->expectsOutputToContain('Kartu data diimpor: 3 dataset, 7 font')->assertSuccessful();
        $this->assertSame('fase7_track', DatasetReport::sole()->official_run);

        $this->write(['schema' => 9] + $this->card());
        $this->artisan('aksara:import', ['--path' => $this->dir])
            ->expectsOutputToContain('Skema kartu data 9 tidak didukung')->assertFailed();
        $this->assertSame('fase7_track', DatasetReport::sole()->official_run);
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
            ->assertSee(nfmt($card['usage']['train']['distinct_lines']))
            ->assertSee('Daftar dataset')->assertSee('Font per peran')->assertSee('Data pendukung');
    }
}
