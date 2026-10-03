<?php

namespace Tests\Feature;

use App\Livewire\Pages\Penjelajah;
use App\Models\Line;
use App\Models\Metric;
use App\Models\Pipeline;
use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Livewire\Livewire;
use Tests\TestCase;

class AksaraPagesTest extends TestCase
{
    use RefreshDatabase;

    private const PAGES = ['/ringkasan', '/perbandingan', '/ablasi', '/penjelajah', '/kesalahan', '/demo'];

    private function importFixture(): void
    {
        $this->artisan('aksara:import', ['--path' => base_path('tests/Fixtures/results')])->assertSuccessful();
        $this->artisan('aksara:annotations', ['--path' => base_path('tests/Fixtures/annotations.json')])->assertSuccessful();
    }

    public function test_guests_are_redirected_to_login(): void
    {
        foreach (self::PAGES as $page) {
            $this->get($page)->assertRedirect('/login');
        }
    }

    public function test_pages_explain_how_to_import_when_empty(): void
    {
        $this->actingAs(User::factory()->create());
        foreach (['/ringkasan', '/perbandingan', '/ablasi', '/penjelajah', '/kesalahan'] as $page) {
            $this->get($page)->assertOk()->assertSee('aksara:import');
        }
    }

    public function test_pages_render_imported_results(): void
    {
        $this->importFixture();
        $this->actingAs(User::factory()->create());

        $this->get('/ringkasan')->assertOk()
            ->assertSee('G3')->assertSee('37,2%')->assertSee('target 8%')
            ->assertSee('Tingkat tutur')->assertSee('10/10 baris punya arti manusia')
            ->assertSee('CER draf')->assertSee('(greedy)');
        $this->get('/perbandingan')->assertOk()->assertSee('CRNN fase5_fonts')->assertSee('VLM zero-shot')->assertSee('Sahabat-AI')
            ->assertDontSee('satu run per kondisi');
        $this->get('/ablasi')->assertOk()->assertSee('tight')->assertSee('speckle');
        $this->get('/kesalahan')->assertOk()->assertSee('Tertukar')->assertSee('Hilang');

        $line = Line::orderBy('external_id')->first();
        $this->get('/penjelajah?baris='.$line->external_id)->assertOk()
            ->assertSee($line->reference)
            ->assertSee($line->annotation->translation)
            ->assertSee('leksikon (dari transliterasi manusia)')
            ->assertSee('data-x0', false);
    }

    public function test_fase6_runs_are_shown_as_comparators_not_official(): void
    {
        $this->importFixture();
        // Urutan seperti export_results.py: rare lalu ctrl, tepat setelah beam.
        foreach (['crnn_fase6_rare' => 0.30, 'crnn_fase6_ctrl' => 0.31] as $key => $cer) {
            Pipeline::create(['key' => $key, 'label' => 'CRNN '.substr($key, 5), 'config' => "crnn:{$key} · greedy",
                'kind' => 'crnn', 'status' => 'done', 'sort' => $key === 'crnn_fase6_rare' ? 4 : 5]);
            Metric::create(['scope' => 'nusaaksara_745', 'pipeline' => $key, 'lines' => 10, 'cer' => $cer,
                'cer_no_space' => $cer - 0.05, 'exact' => 0.0]);
        }
        $this->actingAs(User::factory()->create());

        $this->get('/perbandingan')->assertOk()
            ->assertDontSee('Checkpoint sama')
            ->assertSee('fase5_fonts + 1.500 langkah dengan sisipan aksara langka')
            ->assertSee('pembanding: langkah dan titik lanjut sama, tanpa aksara langka')
            ->assertSee('CRNN fase6_rare dan CRNN fase6_ctrl: satu run per kondisi');
        // Beam 29,5%, fase6 30–31%, fonts 34,9%: terlalu rapat untuk label di samping titik, jadi semua nilai
        // ada di daftar di bawah sumbu, urut CER.
        $this->get('/ringkasan')->assertOk()->assertSeeInOrder(['CER G3 per pipeline, terbaik dulu',
            'Beam + LM', '29,5%', 'CRNN fase6_rare', '30,0%', 'CRNN fase6_ctrl', '31,0%',
            'CRNN fase5_fonts', '34,9%', 'CRNN fase5_core', '54,4%', 'CRNN 4b', '93,3%']);

        // Hanya satu run fase6 yang diekspor (mis. fase6_ctrl gagal sebelum snapshot pertamanya): peringatan
        // hanya menyebut baris yang ada di tabel.
        Metric::where('pipeline', 'crnn_fase6_ctrl')->delete();
        $this->get('/perbandingan')->assertOk()
            ->assertSee('CRNN fase6_rare: satu run per kondisi')
            ->assertDontSee('fase6_ctrl');
    }

    public function test_explorer_filters_and_toggles_pipelines(): void
    {
        $this->importFixture();
        $this->actingAs(User::factory()->create());
        $blind = Line::whereJsonContains('tags', 'uji buta')->first();

        $row = fn (string $pipeline) => 'row-'.$blind->id.'-'.$pipeline;

        Livewire::test(Penjelajah::class)
            ->set('tag', 'uji buta')
            ->call('select', $blind->external_id)
            ->assertSeeHtml($row('vlm_zeroshot'))
            ->assertDontSeeHtml($row('crnn_4b'))
            ->call('togglePipe', 'crnn_4b')
            ->assertSeeHtml($row('crnn_4b'))
            ->call('togglePipe', 'vlm_zeroshot')
            ->assertDontSeeHtml($row('vlm_zeroshot'))
            ->assertDontSee('draf ·')
            ->set('latin', true)
            ->assertSee('draf ·');
    }

    public function test_human_speech_level_labels_feed_the_accuracy(): void
    {
        $this->importFixture();
        $this->actingAs(User::factory()->create(['name' => 'Rekan']));
        $line = Line::orderBy('external_id')->first();

        Livewire::test(Penjelajah::class)
            ->call('setSpeechLevel', $line->external_id, 'krama')
            ->assertSee('oleh Rekan');
        $this->assertSame('krama', $line->annotation()->first()->speech_level);
        $this->get('/ringkasan')->assertSee('1/10 baris berlabel manusia')->assertSee('pada 1 baris');

        // Anotasi NusaAksara diimpor ulang: label manusia tidak boleh hilang.
        $this->artisan('aksara:annotations', ['--path' => base_path('tests/Fixtures/annotations.json')])->assertSuccessful();
        $this->assertSame('krama', $line->annotation()->first()->speech_level);

        Livewire::test(Penjelajah::class)->call('setSpeechLevel', $line->external_id, null);
        $this->assertNull($line->annotation()->first()->speech_level);
        Livewire::test(Penjelajah::class)->call('setSpeechLevel', $line->external_id, 'bukan-tingkat')->assertStatus(422);
    }

    public function test_line_image_is_served_only_from_ocr_repo(): void
    {
        $this->importFixture();
        $this->actingAs(User::factory()->create());
        $line = Line::first();

        $this->get(route('line.image', $line))->assertOk();

        $line->update(['image_path' => '../../../../Windows/win.ini']);
        $this->get(route('line.image', $line))->assertNotFound();
    }
}
