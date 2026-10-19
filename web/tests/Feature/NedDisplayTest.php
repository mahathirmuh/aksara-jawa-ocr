<?php

namespace Tests\Feature;

use App\Livewire\Pages\Penjelajah;
use App\Livewire\Pages\Perbandingan;
use App\Models\Metric;
use App\Models\Prediction;
use App\Models\User;
use App\Services\ResultImporter;
use Illuminate\Filesystem\Filesystem;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Livewire\Livewire;
use RuntimeException;
use Tests\Concerns\ReadsPage;
use Tests\TestCase;

/**
 * NED (selalu 0-100%) sebagai angka utama di Perbandingan dan Penjelajah baris, CER (angka gerbang, bisa lewat 100%)
 * di sampingnya: permintaan user 2026-10-19. Folder hasil buatan berisi tiga baris contoh, tanpa data NusaAksara.
 */
class NedDisplayTest extends TestCase
{
    use ReadsPage;
    use RefreshDatabase;

    private string $dir;

    protected function setUp(): void
    {
        parent::setUp();
        $this->dir = sys_get_temp_dir().DIRECTORY_SEPARATOR.'aksara-ned-'.uniqid();
        mkdir($this->dir);
    }

    protected function tearDown(): void
    {
        (new Filesystem)->deleteDirectory($this->dir);
        parent::tearDown();
    }

    /**
     * Baris a: keluaran penuh sisipan (CER 400%, NED 80%); baris b: keluaran kosong (CER = NED = 100%); baris c: benar.
     * Menurut CER baris a paling buruk, menurut NED baris b: urutan Penjelajah menunjukkan ukuran mana yang dipakai.
     */
    private function writeResults(bool $withNed = true): void
    {
        $ned = fn (array $row, float $value) => $withNed ? $row + ['ned' => $value] : $row;
        file_put_contents($this->dir.'/manifest.json', json_encode([
            'schema' => 1, 'generated' => '2026-10-19T00:00:00', 'limited' => false, 'official' => 'crnn_resmi',
            'pipelines' => [
                ['key' => 'crnn_resmi', 'label' => 'CRNN resmi', 'config' => 'crnn:resmi@1 · greedy', 'kind' => 'crnn', 'status' => 'done', 'sort' => 0],
                ['key' => 'vlm_zeroshot', 'label' => 'VLM zero-shot', 'config' => 'vlm · zero-shot', 'kind' => 'vlm', 'status' => 'done', 'sort' => 1],
            ],
            'gates' => [], 'ablation' => [], 'confusion' => ['pipeline' => 'crnn_resmi', 'items' => []],
            'metrics' => [
                $ned(['scope' => 'nusaaksara_745', 'pipeline' => 'crnn_resmi', 'lines' => 3, 'cer' => 1.0438, 'cer_no_space' => 0.9146, 'exact' => 1 / 3], 0.6667),
                $ned(['scope' => 'blind_50', 'pipeline' => 'crnn_resmi', 'lines' => 2, 'cer' => 1.25, 'cer_no_space' => 1.25, 'exact' => 0.0], 0.8333),
                $ned(['scope' => 'blind_50', 'pipeline' => 'vlm_zeroshot', 'lines' => 2, 'cer' => 0.75, 'cer_no_space' => 0.75, 'exact' => 0.0], 0.6),
            ],
        ], JSON_UNESCAPED_UNICODE));
        $lines = [['a.png', 'ꦏ', ['uji buta']], ['b.png', 'ꦠꦤꦩ', ['uji buta']], ['c.png', 'ꦧꦝ', []]];
        file_put_contents($this->dir.'/lines.jsonl', implode('', array_map(fn ($l) => json_encode(['dataset' => 'contoh', 'id' => $l[0],
            'image' => 'contoh/'.$l[0], 'width' => 100, 'height' => 20, 'reference' => $l[1], 'condition' => 'contoh',
            'source_id' => 'hal_0', 'tags' => $l[2]], JSON_UNESCAPED_UNICODE)."\n", $lines)));
        $predictions = [
            [$ned(['line' => 'a.png', 'pipeline' => 'crnn_resmi', 'text' => 'ꦏꦭꦩꦤꦒ', 'cer' => 4.0], 0.8), 'ꦏꦭꦩꦤꦒ'],
            [$ned(['line' => 'b.png', 'pipeline' => 'crnn_resmi', 'text' => '', 'cer' => 1.0], 1.0), ''],
            [$ned(['line' => 'c.png', 'pipeline' => 'crnn_resmi', 'text' => 'ꦧꦝ', 'cer' => 0.0], 0.0), 'ꦧꦝ'],
            [$ned(['line' => 'a.png', 'pipeline' => 'vlm_zeroshot', 'text' => 'ꦏꦭ', 'cer' => 1.0], 0.5), 'ꦏꦭ'],
            [$ned(['line' => 'b.png', 'pipeline' => 'vlm_zeroshot', 'text' => 'ꦠ', 'cer' => 2 / 3], 2 / 3), 'ꦠ'],
        ];
        file_put_contents($this->dir.'/predictions.jsonl', implode('', array_map(fn ($p) => json_encode($p[0]
            + ['segments' => $p[1] === '' ? [] : [['t' => $p[1], 's' => 'ok']]], JSON_UNESCAPED_UNICODE)."\n", $predictions)));
    }

    private function import(): void
    {
        $this->artisan('aksara:import', ['--path' => $this->dir])->assertSuccessful();
        $this->actingAs(User::factory()->create());
    }

    public function test_comparison_page_leads_with_ned_and_keeps_cer_beside_it(): void
    {
        $this->writeResults();
        $this->import();
        $this->assertSame(0.6667, Metric::where(['scope' => 'nusaaksara_745', 'pipeline' => 'crnn_resmi'])->value('ned'));
        $this->assertSame(0.8, Prediction::where('pipeline', 'crnn_resmi')->orderByDesc('cer')->value('ned'));

        $html = $this->get('/perbandingan')->assertOk()->getContent();
        // Kolom: NED, CER, CER tanpa spasi, baris persis, catatan.
        $this->assertStringEndsWith('crnn:resmi@1 · greedy 66,67% 104,38% 91,46% 33,3% angka G3 resmi',
            $this->part($html, 'data-metric', 'crnn_resmi'));
        $this->assertSame('CRNN resmi 83,3% 125,0% 125,0%', $this->part($html, 'data-blind', 'crnn_resmi'));
        $this->assertStringContainsString('Batang = NED (0–100%)', $html);
        $this->assertSame(['labels' => ['CRNN resmi', 'VLM zero-shot'], 'values' => [0.8333, 0.6],
            'configs' => ['crnn:resmi@1 · greedy', 'vlm · zero-shot'], 'metric' => 'NED'],
            Livewire::test(Perbandingan::class)->viewData('blindChart'));
    }

    public function test_line_explorer_lists_sorts_and_badges_by_ned(): void
    {
        $this->writeResults();
        $this->import();

        $order = fn (string $sort) => Livewire::test(Penjelajah::class)->set('sort', $sort)->viewData('page')->pluck('external_id')->all();
        $this->assertSame(['b.png', 'a.png', 'c.png'], $order('fonts_desc'));  // menurut CER: a, b, c
        $this->assertSame(['c.png', 'a.png', 'b.png'], $order('fonts_asc'));
        $this->assertSame('NED resmi, terburuk dulu', Livewire::test(Penjelajah::class)->viewData('sorts')['fonts_desc']);

        $html = $this->get('/penjelajah?baris=a.png')->assertOk()->getContent();
        $this->assertSame('Angka = NED baris (0–100%): suntingan dibagi panjang terpanjang label/keluaran.',
            $this->part($html, 'data-list-metric', 'ned'));
        $this->assertStringContainsString('CRNN 80% · VLM 50%', $this->part($html, 'data-line', 'a.png'));
        $this->assertStringContainsString('CRNN 100% · VLM 67%', $this->part($html, 'data-line', 'b.png'));
        $this->assertStringContainsString('CRNN 0%', $this->part($html, 'data-line', 'c.png'));
        $this->assertStringContainsString('CRNN resmi NED 80% CER 400%', $this->part($html, 'data-pred', 'crnn_resmi'));
        $this->assertStringContainsString('VLM zero-shot NED 50% CER 100%', $this->part($html, 'data-pred', 'vlm_zeroshot'));
        // Lencana biru = NED terendah di baris ini (VLM), bukan CER terendah.
        $blue = fn (string $pipeline) => $this->xpath($html)->query("//*[@data-pred='{$pipeline}']//span[contains(@class, 'status-blue')]")->length;
        $this->assertSame([0, 1], [$blue('crnn_resmi'), $blue('vlm_zeroshot')]);
    }

    public function test_results_exported_before_ned_still_show_cer(): void
    {
        $this->writeResults(withNed: false);
        $this->import();
        $this->assertNull(Metric::where('scope', 'nusaaksara_745')->value('ned'));

        $html = $this->get('/perbandingan')->assertOk()->getContent();
        $this->assertStringEndsWith('crnn:resmi@1 · greedy – 104,38% 91,46% 33,3% angka G3 resmi', $this->part($html, 'data-metric', 'crnn_resmi'));
        $chart = Livewire::test(Perbandingan::class)->viewData('blindChart');
        $this->assertSame(['CER', [1.25, 0.75]], [$chart['metric'], $chart['values']]);

        $this->assertSame(['a.png', 'b.png', 'c.png'], Livewire::test(Penjelajah::class)->viewData('page')->pluck('external_id')->all());
        $html = $this->get('/penjelajah?baris=a.png')->assertOk()->getContent();
        $this->assertSame('Angka = CER baris.', $this->part($html, 'data-list-metric', 'cer'));
        $this->assertStringContainsString('CRNN 400% · VLM 100%', $this->part($html, 'data-line', 'a.png'));
        $this->assertStringContainsString('CRNN resmi CER 400%', $this->part($html, 'data-pred', 'crnn_resmi'));
        $this->assertStringNotContainsString('NED', $this->part($html, 'data-pred', 'crnn_resmi'));
    }

    /** @return array<string, array{0: string, 1: int, 2: mixed, 3: string}> */
    public static function brokenNed(): array
    {
        return [
            'metrik di atas 1' => ['manifest.json', 0, 1.2, 'NED metrik nusaaksara_745 crnn_resmi harus bilangan 0 sampai 1, bukan 1.2'],
            'prediksi negatif' => ['predictions.jsonl', 1, -0.1, 'NED prediksi crnn_resmi b.png harus bilangan 0 sampai 1, bukan -0.1'],
            'prediksi berupa teks' => ['predictions.jsonl', 0, '0.5', 'NED prediksi crnn_resmi a.png harus bilangan 0 sampai 1, bukan "0.5"'],
        ];
    }

    /** @dataProvider brokenNed */
    public function test_ned_outside_zero_to_one_is_rejected_and_the_old_results_kept(string $file, int $row, mixed $value, string $message): void
    {
        $this->writeResults();
        app(ResultImporter::class)->import($this->dir);
        $before = Prediction::orderBy('id')->pluck('ned')->all();

        $path = $this->dir.'/'.$file;
        if ($file === 'manifest.json') {
            $manifest = json_decode(file_get_contents($path), true);
            $manifest['metrics'][$row]['ned'] = $value;
            file_put_contents($path, json_encode($manifest, JSON_UNESCAPED_UNICODE));
        } else {
            $rows = array_map(fn ($l) => json_decode($l, true), array_filter(explode("\n", file_get_contents($path))));
            $rows[$row]['ned'] = $value;
            file_put_contents($path, implode('', array_map(fn ($r) => json_encode($r, JSON_UNESCAPED_UNICODE)."\n", $rows)));
        }
        $error = null;
        try {
            app(ResultImporter::class)->import($this->dir);
        } catch (RuntimeException $e) {
            $error = $e;
        }
        $this->assertNotNull($error, 'NED di luar 0-1 harus ditolak');
        $this->assertSame($message.'.', $error->getMessage());
        $this->assertSame($before, Prediction::orderBy('id')->pluck('ned')->all());
        $this->assertSame(0.6667, Metric::where(['scope' => 'nusaaksara_745', 'pipeline' => 'crnn_resmi'])->value('ned'));
    }
}
