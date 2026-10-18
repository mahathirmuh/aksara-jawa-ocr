<?php

namespace Tests\Feature;

use App\Models\MethodMatrix;
use App\Models\User;
use App\Services\MatrixImporter;
use Illuminate\Filesystem\Filesystem;
use Illuminate\Foundation\Testing\RefreshDatabase;
use RuntimeException;
use Tests\Concerns\ReadsPage;
use Tests\TestCase;

class MatrixPageTest extends TestCase
{
    use ReadsPage;
    use RefreshDatabase;

    private string $dir;

    protected function setUp(): void
    {
        parent::setUp();
        $this->dir = sys_get_temp_dir().DIRECTORY_SEPARATOR.'aksara-matrix-'.uniqid();
        mkdir($this->dir);
    }

    protected function tearDown(): void
    {
        (new Filesystem)->deleteDirectory($this->dir);
        parent::tearDown();
    }

    private function cell(float $cer, int $lines, ?float $delta = null, array $extra = []): array
    {
        // NED dibuat 0,9 x CER yang dipotong di 100%, selisihnya 0,8 x selisih CER: angka berbeda supaya kolomnya tidak tertukar.
        return $extra + ['cer' => $cer, 'ned' => round(min($cer, 1.0) * 0.9, 5), 'lines' => $lines, 'pages' => 7]
            + ($delta === null ? [] : ['delta' => $delta, 'ci95' => [$delta - 0.01, $delta + 0.01],
                'ned_delta' => $delta * 0.8, 'ned_ci95' => [$delta * 0.8 - 0.01, $delta * 0.8 + 0.01]]);
    }

    private function method(string $key, string $group, ?float $cetak, ?float $delta = null, array $extra = []): array
    {
        $full = ['cer_no_space' => 0.1, 'exact' => 0.25, 'collapsed' => 0.5, 'non_javanese' => 0.0];

        return $extra + ['key' => $key, 'label' => "Metode {$key}", 'group' => $group, 'offline' => true, 'trained' => $group === 'kita',
            'note' => "catatan {$key}", 'params' => 4590909, 'seconds_per_line' => 0.05,
            'cetak' => $cetak === null ? null : $this->cell($cetak, 308, $delta, $full),
            'tulisan_tangan' => $cetak === null ? null : $this->cell($cetak + 0.1, 437, $delta), 'semua' => null, 'partial' => null];
    }

    private function pair(string $a, string $group, float $delta): array
    {
        $c = ['cer_a' => 0.2, 'cer_b' => 0.2 - $delta, 'delta' => $delta, 'ci95_pages' => [$delta - 0.005, $delta + 0.005], 'lines' => 308, 'pages' => 72];

        return ['group' => $group, 'label' => "Komponen {$a}", 'a' => $a, 'b' => "{$a}_ctrl", 'status' => 'dipakai',
            'cetak' => $c, 'tulisan_tangan' => $c, 'semua' => $c];
    }

    private function matrix(array $override = []): array
    {
        return $override + [
            'schema' => 1, 'generated' => '2026-10-18 14:00:00', 'reference' => 'crnn_resmi', 'prompt' => 'Transcribe it.',
            'prompt_sha' => 'abc123', 'rules' => 'NFC; CER Levenshtein',
            'subsets' => ['cetak' => ['lines' => 308, 'pages' => 72], 'tulisan_tangan' => ['lines' => 437, 'pages' => 20],
                'semua' => ['lines' => 745, 'pages' => 92]],
            'methods' => [
                $this->method('crnn_resmi', 'kita', 0.1234),
                $this->method('gpt', 'vlm', 1.1197, 0.9963, ['offline' => false, 'output_tokens_per_line' => 87.0, 'usd_per_1000_lines' => 2.17]),
                $this->method('tess', 'umum', 1.0933, 0.9699),
                $this->method('lama', 'vlm', null, null, ['partial' => ['lines' => 50, 'cer' => 0.9099, 'reference_cer' => 0.2299, 'ned' => 0.8, 'reference_ned' => 0.2]]),
            ],
            'architecture' => [['key' => 'arch_tr', 'label' => 'CNN + Transformer + CTC', 'params' => 4571709, 'val_clean' => 0.004,
                'g1' => 0.006, 'g2' => 0.5, 'cetak' => ['cer' => 1.02, 'ned' => 0.9, 'lines' => 308, 'pages' => 72]]],
            'ablation' => [$this->pair('crnn_fonts', 'data', -0.2013), $this->pair('crnn_beam', 'baca', -0.0156), $this->pair('noise', 'rujukan', -0.0004)],
            'augment' => [['k' => 0, 'op' => '(kontrol)', 'clean' => 0.003, 'heavy' => 0.65, 'g3' => 0.75, 'g3_no_space' => 0.7],
                ['k' => 1, 'op' => 'rotate', 'clean' => 0.003, 'heavy' => 0.576, 'g3' => 0.8785, 'g3_no_space' => 0.78]],
        ];
    }

    private function write(array $matrix): void
    {
        file_put_contents($this->dir.DIRECTORY_SEPARATOR.'matrix.json', json_encode($matrix, JSON_UNESCAPED_UNICODE));
    }

    public function test_import_and_page_show_each_method_pair_and_architecture_row(): void
    {
        $this->write($this->matrix());
        $this->artisan('aksara:matrix', ['--path' => $this->dir])
            ->expectsOutputToContain('Matriks diimpor: 4 metode, 1 arsitektur, 3 pasangan ablasi, 2 langkah augmentasi.')
            ->assertSuccessful();

        $this->actingAs(User::factory()->create());
        $html = $this->get(route('matriks'))->assertOk()->getContent();

        $this->assertSame('Metode crnn_resmi catatan crnn_resmi Metode kita 11,11% pembanding 12,34% pembanding 22,34% – 25,0%',
            $this->part($html, 'data-method', 'crnn_resmi'));
        $this->assertSame('Metode gpt catatan gpt VLM zero-shot 90,00% +79,70 [+78,70; +80,70] 111,97% +99,63 [+98,63; +100,63] 121,97% – 25,0%',
            $this->part($html, 'data-method', 'gpt'));
        $this->assertStringContainsString('OCR umum 90,00% +77,59', $this->part($html, 'data-method', 'tess'));
        $this->assertSame('Metode lama catatan lama VLM zero-shot hanya 50 baris dibaca: NED 80,0%, CER 91,0% '
            .'(model resmi pada baris yang sama: NED 20,0%, CER 23,0%)', $this->part($html, 'data-method', 'lama'));
        $this->assertSame('Metode gpt tidak tidak 4.590.909 0,05 87 2,17', $this->part($html, 'data-practical', 'gpt'));
        $this->assertSame('Metode crnn_resmi ya ya 4.590.909 0,05 – –', $this->part($html, 'data-practical', 'crnn_resmi'));
        $this->assertSame('CNN + Transformer + CTC 4.571.709 0,40% 0,60% 50,00% 90,0% / 102,0%', $this->part($html, 'data-arch', 'arch_tr'));
        $this->assertSame('Komponen crnn_fonts fonts − fonts_ctrl −20,13 [−20,63; −19,63] −20,13 [−20,63; −19,63] −20,13 [−20,63; −19,63] dipakai',
            $this->part($html, 'data-pair', 'crnn_fonts'));
        $this->assertSame('1 rotate 57,6% 87,9% +12,85', $this->part($html, 'data-augment', '1'));
        $this->assertStringContainsString('abc123', $html);
        $this->assertSame(['labels' => ['Metode crnn_resmi', 'Metode gpt', 'Metode tess'], 'values' => [0.11106, 0.9, 0.9],
            'configs' => ['Metode kita', 'VLM zero-shot', 'OCR umum'], 'max' => 1.0],
            \App\Livewire\Pages\Matriks::viewData($this->matrix())['chart']);

        $highlighted = $this->xpath($html)->query("//tr[contains(@class, 'is-highlight')]/@data-method");
        $this->assertSame(['crnn_resmi'], array_map(fn ($n) => $n->value, iterator_to_array($highlighted)));
    }

    public function test_page_without_matrix_explains_how_to_make_one(): void
    {
        $this->actingAs(User::factory()->create());
        $this->get(route('matriks'))->assertOk()->assertSee('Matriks belum diimpor')->assertSee('aksara:matrix');
    }

    /** @return array<string, array{0: callable(array): array, 1: string}> */
    public static function brokenMatrices(): array
    {
        return [
            'skema lain' => [fn ($m) => ['schema' => 2] + $m, 'schema harus 1'],
            'tanpa metode' => [fn ($m) => ['methods' => []] + $m, 'methods kosong'],
            'kelompok asing' => [function ($m) {
                $m['methods'][1]['group'] = 'lain';

                return $m;
            }, 'methods.1.group'],
            'kunci ganda' => [function ($m) {
                $m['methods'][2]['key'] = 'gpt';

                return $m;
            }, 'berulang: gpt'],
            'pembanding hilang' => [fn ($m) => ['reference' => 'tidak_ada'] + $m, 'pembanding tidak_ada'],
            'selisih tanpa SK' => [function ($m) {
                unset($m['methods'][1]['cetak']['ci95']);

                return $m;
            }, 'methods.1.cetak'],
            'NED di atas 100%' => [function ($m) {
                $m['methods'][0]['cetak']['ned'] = 1.2;

                return $m;
            }, 'methods.0.cetak'],
            'selisih NED tanpa SK' => [function ($m) {
                unset($m['methods'][2]['cetak']['ned_ci95']);

                return $m;
            }, 'methods.2.cetak'],
            'CER negatif' => [function ($m) {
                $m['methods'][0]['cetak']['cer'] = -0.1;

                return $m;
            }, 'methods.0.cetak'],
            'tanpa angka sama sekali' => [function ($m) {
                $m['methods'][3]['partial'] = null;

                return $m;
            }, 'methods.3 tidak punya'],
            'SK terbalik' => [function ($m) {
                $m['ablation'][0]['cetak']['ci95_pages'] = [0.1, -0.1];

                return $m;
            }, 'ablation.0.cetak'],
            'augmentasi tanpa G3' => [function ($m) {
                unset($m['augment'][1]['g3']);

                return $m;
            }, 'augment.1'],
            'subset tanpa halaman' => [function ($m) {
                $m['subsets']['cetak']['pages'] = 0;

                return $m;
            }, 'subsets.cetak'],
        ];
    }

    /** @dataProvider brokenMatrices */
    public function test_broken_matrix_is_rejected_and_the_old_one_kept(callable $break, string $message): void
    {
        $this->write($this->matrix());
        app(MatrixImporter::class)->import($this->dir);
        $before = MethodMatrix::current()->payload;

        $this->write($break($this->matrix()));
        $error = null;
        try {
            app(MatrixImporter::class)->import($this->dir);
        } catch (RuntimeException $e) {
            $error = $e;
        }
        $this->assertNotNull($error, 'matriks rusak harus ditolak');
        $this->assertStringContainsString($message, $error->getMessage());
        $this->assertSame(1, MethodMatrix::count());
        $this->assertSame($before, MethodMatrix::current()->payload);
        $this->artisan('aksara:matrix', ['--path' => $this->dir])->assertFailed();
    }
}
