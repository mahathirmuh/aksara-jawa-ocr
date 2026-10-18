<?php

namespace Tests\Feature;

use App\Models\AblationRun;
use App\Models\Confusion;
use App\Models\Gate;
use App\Models\Line;
use App\Models\LineAnnotation;
use App\Models\Metric;
use App\Models\Pipeline;
use App\Models\Prediction;
use App\Models\ResultImport;
use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\TestCase;

class AksaraImportTest extends TestCase
{
    use RefreshDatabase;

    private function fixture(): string
    {
        return base_path('tests/Fixtures/results');
    }

    public function test_import_loads_contract_files(): void
    {
        $this->artisan('aksara:import', ['--path' => $this->fixture()])->assertSuccessful();

        $this->assertSame(10, Line::count());
        $this->assertSame(45, Prediction::count());
        $this->assertSame(9, Pipeline::count());
        $this->assertSame(4, Gate::count());
        $this->assertSame(12, AblationRun::count());
        $this->assertGreaterThan(0, Confusion::count());
        $this->assertTrue(ResultImport::first()->limited);

        $pred = Prediction::where('pipeline', 'crnn_fonts')->first();
        $this->assertIsArray($pred->segments);
        $this->assertNotEmpty($pred->spans, 'CRNN greedy harus membawa kolom citra per suku kata');
        $this->assertSame(
            $pred->text,
            collect($pred->segments)->reject(fn ($s) => $s['s'] === 'del')->map(fn ($s) => $s['t'])->join(''),
            'teks prediksi harus bisa dibangun ulang dari segmen non-hapus',
        );
    }

    /** Salinan fixture dengan manifest yang diubah (mis. pipeline resmi lain). */
    private function fixtureWith(callable $change): string
    {
        $dir = sys_get_temp_dir().'/aksara-official-'.uniqid();
        mkdir($dir);
        foreach (['lines.jsonl', 'predictions.jsonl'] as $file) {
            copy($this->fixture().'/'.$file, $dir.'/'.$file);
        }
        $manifest = json_decode(file_get_contents($this->fixture().'/manifest.json'), true);
        file_put_contents($dir.'/manifest.json', json_encode($change($manifest), JSON_UNESCAPED_UNICODE));

        return $dir;
    }

    public function test_official_pipeline_comes_from_the_manifest(): void
    {
        // Ekspor sebelum kunci "official" ada: angka resmi = crnn_fonts.
        $this->artisan('aksara:import', ['--path' => $this->fixture()])->assertSuccessful();
        $this->assertSame('crnn_fonts', Pipeline::official()->key);
        $this->assertSame(['crnn_fonts'], Pipeline::where('official', true)->pluck('key')->all());

        // Ekspor yang menetapkan pipeline lain: tandanya pindah, tidak bertambah.
        $dir = $this->fixtureWith(function (array $manifest) {
            $manifest['official'] = 'crnn_core';
            $manifest['confusion']['pipeline'] = 'crnn_core';

            return $manifest;
        });
        $this->artisan('aksara:import', ['--path' => $dir])->assertSuccessful();
        $this->assertSame('crnn_core', Pipeline::official()->key);
        $this->assertSame(['crnn_core'], Pipeline::where('official', true)->pluck('key')->all());
    }

    /** Salinan fixture dengan rincian G3 tertentu. */
    private function fixtureWithParts(mixed $parts): string
    {
        return $this->fixtureWith(function (array $manifest) use ($parts) {
            foreach ($manifest['gates'] as $i => $gate) {
                if ($gate['code'] === 'G3') {
                    $manifest['gates'][$i]['parts'] = $parts;
                }
            }

            return $manifest;
        });
    }

    public function test_g3_parts_are_imported_beside_the_gate_and_shown_on_the_summary(): void
    {
        // Ekspor lama tanpa rincian: kolomnya kosong dan Ringkasan tidak menampilkan rincian.
        $this->artisan('aksara:import', ['--path' => $this->fixture()])->assertSuccessful();
        $this->assertNull(Gate::where('code', 'G3')->first()->parts);
        $this->actingAs(User::factory()->create());
        $this->get('/ringkasan')->assertOk()->assertDontSee('data-gate-parts', false);

        $this->artisan('aksara:import', ['--path' => $this->fixtureWithParts([
            ['key' => 'cetak', 'label' => 'baris cetak', 'value' => 0.1234, 'lines' => 308, 'pages' => 72, 'chars' => 9979],
            ['key' => 'tulisan_tangan', 'label' => 'tulisan tangan', 'value' => 0.2796, 'lines' => 437, 'pages' => 20, 'chars' => 10405],
        ])])->assertSuccessful();
        $gates = Gate::all()->keyBy('code');
        $this->assertSame([
            ['key' => 'cetak', 'label' => 'baris cetak', 'value' => 0.1234, 'lines' => 308, 'pages' => 72],
            ['key' => 'tulisan_tangan', 'label' => 'tulisan tangan', 'value' => 0.2796, 'lines' => 437, 'pages' => 20],
        ], $gates['G3']->parts);
        $this->assertNull($gates['G1']->parts);
        $this->assertNull($gates['G4']->parts);

        // Ringkasan: rincian hanya di ubin G3, dan angka G3 tetap angka gerbang (fixture 37,2%).
        $html = $this->get('/ringkasan')->assertOk()->assertSee('37,2%')->getContent();
        $this->assertSame(1, substr_count($html, 'data-gate-parts='));
        $dom = new \DOMDocument;
        @$dom->loadHTML('<?xml encoding="utf-8"?>'.$html);
        $xpath = new \DOMXPath($dom);
        $box = $xpath->query('//*[@data-gate-parts="G3"]')->item(0);
        $this->assertNotNull($box);
        $parts = [];
        foreach ($xpath->query('.//*[@data-gate-part]', $box) as $node) {
            $parts[$node->getAttribute('data-gate-part')] = trim($node->textContent);
        }
        $this->assertSame([
            'cetak' => 'baris cetak 12,3% (308 baris, 72 halaman)',
            'tulisan_tangan' => 'tulisan tangan 28,0% (437 baris, 20 halaman)',
        ], $parts);
        $this->assertStringContainsString('belum diperiksa manusia', $box->textContent);
        $this->assertStringContainsString('baris cetak 12,3% (308 baris, 72 halaman) · tulisan tangan', preg_replace('/\s+/', ' ', $box->textContent));
    }

    public function test_invalid_g3_parts_are_rejected_and_keep_the_old_results(): void
    {
        $this->artisan('aksara:import', ['--path' => $this->fixture()])->assertSuccessful();
        $good = ['key' => 'cetak', 'label' => 'baris cetak', 'value' => 0.12, 'lines' => 308, 'pages' => 72];
        foreach ([
            'bukan daftar' => ['cetak' => $good],
            'kosong' => [],
            'butir bukan objek' => ['cetak'],
            'tanpa kunci' => [array_diff_key($good, ['key' => 1])],
            'kunci kosong' => [['key' => ''] + $good],
            'tanpa label' => [array_diff_key($good, ['label' => 1])],
            'label kosong' => [['label' => ''] + $good],
            'nilai teks' => [['value' => '0.12'] + $good],  // teks berangka pun bukan bilangan JSON
            'nilai negatif' => [['value' => -0.1] + $good],
            'tanpa nilai' => [array_diff_key($good, ['value' => 1])],
            'baris nol' => [['lines' => 0] + $good],
            'baris pecahan' => [['lines' => 3.5] + $good],
            'halaman nol' => [['pages' => 0] + $good],
            'halaman teks' => [['pages' => '72'] + $good],
            'butir kedua rusak' => [$good, ['lines' => -1] + $good],
        ] as $case => $parts) {
            $this->artisan('aksara:import', ['--path' => $this->fixtureWithParts($parts)])
                ->expectsOutputToContain('Rincian gerbang G3 tidak sah')
                ->assertFailed();
            $this->assertNull(Gate::where('code', 'G3')->first()->parts, $case);
            $this->assertSame(4, Gate::count(), $case);
        }
        // Nilai 0 (bilangan bulat di JSON) sah; json_encode menyimpannya tanpa ",0", tampilannya tetap "0,0%".
        $this->artisan('aksara:import', ['--path' => $this->fixtureWithParts([['value' => 0] + $good])])->assertSuccessful();
        $this->assertEquals(0, Gate::where('code', 'G3')->first()->parts[0]['value']);
        $this->actingAs(User::factory()->create());
        $this->get('/ringkasan')->assertOk()->assertSee('baris cetak 0,0% (308 baris, 72 halaman)');
    }

    public function test_import_rejects_inconsistent_official_pipeline(): void
    {
        $unknown = $this->fixtureWith(fn (array $manifest) => ['official' => 'crnn_tidak_ada'] + $manifest);
        $this->artisan('aksara:import', ['--path' => $unknown])
            ->expectsOutputToContain('Pipeline resmi crnn_tidak_ada tidak ada')
            ->assertFailed();

        // Halaman Kesalahan aksara menampilkan tabel kesalahan atas nama pipeline resmi: keduanya harus sama.
        $mismatch = $this->fixtureWith(fn (array $manifest) => ['official' => 'crnn_core'] + $manifest);
        $this->artisan('aksara:import', ['--path' => $mismatch])
            ->expectsOutputToContain('bukan pipeline resmi crnn_core')
            ->assertFailed();
        $this->assertSame(0, Pipeline::count());
    }

    public function test_reimport_keeps_line_ids_and_human_annotations(): void
    {
        $this->artisan('aksara:import', ['--path' => $this->fixture()])->assertSuccessful();
        $this->artisan('aksara:annotations', ['--path' => base_path('tests/Fixtures/annotations.json')])->assertSuccessful();
        $ids = Line::orderBy('external_id')->pluck('id', 'external_id');

        $this->artisan('aksara:import', ['--path' => $this->fixture()])->assertSuccessful();

        $this->assertEquals($ids, Line::orderBy('external_id')->pluck('id', 'external_id'));
        $this->assertSame(45, Prediction::count());
        $this->assertNotNull(Line::first()->annotation?->translation);
        $this->assertSame(1, Metric::where('scope', 'translit_draft')->count());
    }

    public function test_annotations_score_draft_transliteration_against_humans(): void
    {
        $this->artisan('aksara:import', ['--path' => $this->fixture()])->assertSuccessful();
        $this->artisan('aksara:annotations', ['--path' => base_path('tests/Fixtures/annotations.json')])
            ->expectsOutputToContain('10 anotasi diimpor')
            ->assertSuccessful();

        $this->assertSame(10, LineAnnotation::whereNotNull('transliteration')->count());
        $metric = Metric::where('scope', 'translit_draft')->first();
        $this->assertSame(10, $metric->lines);
        $this->assertLessThan(0.5, $metric->cer, 'draf transliterasi harus jauh lebih dekat ke manusia daripada acak');
    }

    public function test_import_rejects_unknown_schema(): void
    {
        $dir = sys_get_temp_dir().'/aksara-schema-'.uniqid();
        mkdir($dir);
        file_put_contents($dir.'/manifest.json', json_encode(['schema' => 99]));

        $this->artisan('aksara:import', ['--path' => $dir])
            ->expectsOutputToContain('Skema manifest 99 tidak didukung')
            ->assertFailed();
        $this->assertSame(0, Line::count());
    }

    public function test_speech_level_labels_round_trip_through_json(): void
    {
        $this->artisan('aksara:import', ['--path' => $this->fixture()])->assertSuccessful();
        $user = User::factory()->create(['email' => 'rekan@example.test']);
        $line = Line::orderBy('external_id')->first();
        LineAnnotation::create(['dataset' => $line->dataset, 'external_id' => $line->external_id, 'speech_level' => 'krama',
            'labeled_by' => $user->id, 'labeled_at' => now(), 'source' => 'label manusia (web)']);
        $path = sys_get_temp_dir().'/labels-'.uniqid().'.json';

        $this->artisan('aksara:labels', ['action' => 'export', '--path' => $path])
            ->expectsOutputToContain('1 label diekspor')->assertSuccessful();
        LineAnnotation::query()->delete();
        $this->artisan('aksara:labels', ['action' => 'import', '--path' => $path])->assertSuccessful();

        $restored = LineAnnotation::where('external_id', $line->external_id)->first();
        $this->assertSame('krama', $restored->speech_level);
        $this->assertSame($user->id, $restored->labeled_by);
    }

    public function test_import_reports_missing_results(): void
    {
        $this->artisan('aksara:import', ['--path' => sys_get_temp_dir().'/tidak-ada-'.uniqid()])
            ->expectsOutputToContain('export_results.py')
            ->assertFailed();
    }
}
