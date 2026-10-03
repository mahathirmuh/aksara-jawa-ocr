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
