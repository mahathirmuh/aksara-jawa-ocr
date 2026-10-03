<?php

namespace Tests\Feature;

use App\Models\Line;
use App\Models\MachineTranslation;
use App\Models\TranslationRun;
use App\Models\User;
use App\Services\TranslationService;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Process\PendingProcess;
use Illuminate\Support\Facades\Process;
use Tests\TestCase;

class TranslationTest extends TestCase
{
    use RefreshDatabase;

    protected function setUp(): void
    {
        parent::setUp();
        $this->artisan('aksara:import', ['--path' => base_path('tests/Fixtures/results')])->assertSuccessful();
        $this->artisan('aksara:annotations', ['--path' => base_path('tests/Fixtures/annotations.json')])->assertSuccessful();
    }

    /** Pengganti tools/translate_batch.py: menulis keluaran seolah dari NLLB, tanpa memuat model. */
    private function fakePython(): void
    {
        Process::fake(function (PendingProcess $process) {
            [, , $in, $out, $summary] = $process->command;
            $rows = array_map(fn ($l) => json_decode($l, true), array_filter(file($in, FILE_IGNORE_NEW_LINES)));
            $sources = [];
            file_put_contents($out, implode('', array_map(function ($r) use (&$sources) {
                $sources[$r['source']] = ($sources[$r['source']] ?? 0) + 1;

                return json_encode($r + ['output' => 'terjemahan '.$r['external_id'], 'chrf' => 42.5], JSON_UNESCAPED_UNICODE)."\n";
            }, $rows)));
            file_put_contents($summary, json_encode(['model' => 'facebook/nllb-200-distilled-600M', 'seconds' => 1,
                'runs' => array_map(fn ($s, $n) => ['source' => $s, 'lines' => $n, 'chrf' => 38.2, 'bleu' => 9.1],
                    array_keys($sources), $sources)]));

            return Process::result('label: chrF 38.2');
        });
    }

    public function test_translate_command_prepares_inputs_and_imports_scores(): void
    {
        $this->fakePython();

        $this->artisan('aksara:translate', ['--source' => 'label,crnn_fonts_beam'])
            ->expectsOutputToContain('label: chrF 38.2')
            ->assertSuccessful();

        $this->assertSame(10, MachineTranslation::where('source', 'label')->count());
        $this->assertSame(10, MachineTranslation::where('source', 'crnn_fonts_beam')->count());
        $this->assertSame(2, TranslationRun::count());
        Process::assertRan(fn (PendingProcess $p) => str_ends_with($p->command[1], 'translate_batch.py'));

        // Masukan "label" adalah transliterasi manusia; masukan pipeline adalah Latin draf dari keluaran OCR.
        $line = Line::with('annotation')->orderBy('external_id')->first();
        $this->assertSame($line->annotation->transliteration,
            MachineTranslation::where('source', 'label')->where('external_id', $line->external_id)->value('input'));
    }

    public function test_translations_appear_on_pages(): void
    {
        $this->fakePython();
        $this->artisan('aksara:translate')->assertSuccessful();
        $this->actingAs(User::factory()->create());
        $line = Line::orderBy('external_id')->first();

        $this->get('/ringkasan')->assertOk()->assertSee('chrF 38,2 (NLLB, dari transliterasi manusia)');
        $this->get('/perbandingan')->assertOk()->assertSee('transliterasi manusia')->assertSee('chrF 38,2');
        $this->get('/penjelajah?baris='.$line->external_id.'&latin=1')->assertOk()
            ->assertSee('terjemahan '.$line->external_id)
            ->assertSee('arti dari keluaran ini (mesin)');
    }

    public function test_failed_python_is_reported(): void
    {
        Process::fake(fn () => Process::result(errorOutput: 'ModuleNotFoundError: sacrebleu', exitCode: 1));

        $this->artisan('aksara:translate', ['--source' => 'label'])
            ->expectsOutputToContain('ModuleNotFoundError: sacrebleu')
            ->assertFailed();
        $this->assertSame(0, MachineTranslation::count());
    }

    public function test_tests_never_write_translation_files_to_real_storage(): void
    {
        $real = storage_path('app/mt');
        $dir = app(TranslationService::class)->dir();
        // Diperiksa sebelum apa pun ditulis: bila gagal di sini, berkas asli belum tersentuh.
        $this->assertNotSame(realpath($real) ?: $real, realpath($dir) ?: $dir);
        $this->assertStringStartsWith(sys_get_temp_dir(), $dir);
        $before = $this->snapshot($real);

        $this->fakePython();
        $this->artisan('aksara:translate', ['--source' => 'label'])->assertSuccessful();
        $this->artisan('aksara:translate', ['--dump' => true])->assertSuccessful();
        $this->artisan('aksara:translate', ['--import-only' => true])->assertSuccessful();
        // NLLB gagal: runBatch sudah menulis input.jsonl dan menghapus output.jsonl (begitu output asli dulu hilang).
        Process::fake(fn () => Process::result(exitCode: 1));
        $this->artisan('aksara:translate', ['--source' => 'label'])->assertFailed();

        $this->assertSame($before, $this->snapshot($real), 'test tidak boleh mengubah storage/app/mt');
        $this->assertFileExists($dir.'/input.jsonl');
    }

    public function test_dump_rebuilds_files_that_import_only_restores(): void
    {
        $this->fakePython();
        $this->artisan('aksara:translate', ['--source' => 'label,crnn_fonts_beam'])->assertSuccessful();
        [$rowColumns, $runColumns] = [['dataset', 'external_id', 'source', 'input', 'output', 'chrf', 'model'],
            ['source', 'model', 'lines', 'chrf', 'bleu']];
        $rows = MachineTranslation::orderBy('id')->get($rowColumns)->toArray();
        $runs = TranslationRun::orderBy('id')->get($runColumns)->toArray();
        array_map('unlink', glob($this->mtDir.'/*'));   // berkas hilang, database utuh: keadaan yang dipulihkan --dump

        $this->artisan('aksara:translate', ['--dump' => true])
            ->expectsOutputToContain('20 terjemahan dari database')
            ->expectsOutputToContain('crnn_fonts_beam: chrF 38.2 · BLEU 9.1 (10 baris)')
            ->assertSuccessful();

        $out = array_values(array_filter(file($this->mtDir.'/output.jsonl', FILE_IGNORE_NEW_LINES), 'strlen'));
        $this->assertCount(MachineTranslation::count(), $out);
        $first = json_decode($out[0], true);
        $this->assertSame(['dataset', 'external_id', 'source', 'input', 'reference', 'output', 'chrf'], array_keys($first));
        $line = Line::with('annotation')->orderBy('external_id')->first();
        $this->assertSame([$line->external_id, 'label', $line->annotation->translation],
            [$first['external_id'], $first['source'], $first['reference']]);
        $summary = json_decode(file_get_contents($this->mtDir.'/summary.json'), true);
        $this->assertSame('facebook/nllb-200-distilled-600M', $summary['model']);
        $this->assertSame(['label', 'crnn_fonts_beam'], array_column($summary['runs'], 'source'));

        // Database dikosongkan lalu dibangun ulang dari berkas hasil --dump: isinya harus sama.
        MachineTranslation::query()->delete();
        TranslationRun::query()->delete();
        $this->artisan('aksara:translate', ['--import-only' => true])->assertSuccessful();
        $this->assertEquals($rows, MachineTranslation::orderBy('id')->get($rowColumns)->toArray());
        $this->assertEquals($runs, TranslationRun::orderBy('id')->get($runColumns)->toArray());
    }

    public function test_dump_refuses_empty_database_and_never_shrinks_output(): void
    {
        $this->artisan('aksara:translate', ['--dump' => true])
            ->expectsOutputToContain('tidak ada berkas yang ditulis')
            ->assertFailed();
        $this->assertFileDoesNotExist($this->mtDir.'/output.jsonl');

        $this->fakePython();
        $this->artisan('aksara:translate', ['--source' => 'label'])->assertSuccessful();
        $before = md5_file($this->mtDir.'/output.jsonl');
        // Database tinggal 4 dari 10 baris berkas: berkasnya salinan yang lebih lengkap.
        MachineTranslation::whereNotIn('id', MachineTranslation::orderBy('id')->limit(4)->pluck('id'))->delete();

        $this->artisan('aksara:translate', ['--dump' => true])
            ->expectsOutputToContain('lebih banyak dari database (4)')
            ->assertFailed();
        $this->assertSame($before, md5_file($this->mtDir.'/output.jsonl'));
        $this->artisan('aksara:translate', ['--dump' => true, '--import-only' => true])->assertFailed();
    }

    /** Nama berkas => md5 isi; kosong bila folder tidak ada. */
    private function snapshot(string $dir): array
    {
        return collect(glob($dir.'/*') ?: [])->mapWithKeys(fn ($f) => [basename($f) => md5_file($f)])->all();
    }
}
