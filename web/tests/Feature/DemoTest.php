<?php

namespace Tests\Feature;

use App\Livewire\Pages\Demo;
use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Http\UploadedFile;
use Illuminate\Support\Facades\Http;
use Livewire\Livewire;
use Tests\TestCase;

class DemoTest extends TestCase
{
    use RefreshDatabase;

    private function prediction(): array
    {
        return [
            'pipeline' => 'crnn_beam_lm',
            'config' => 'crnn:fase5_fonts · beam 16 · LM o5_n300000_ds0.5 · α 0.25 · β 1.0',
            'text' => 'ꦲꦏꦸꦧꦺꦴꦠꦼꦤ꧀',
            'candidates' => [
                ['source' => 'crnn_greedy', 'text' => 'ꦲꦏꦸꦧꦺꦴꦠꦼꦤ꧀', 'ctc_logp' => -0.8, 'lm_logp' => -12.5, 'score' => 3.1, 'chosen' => true],
            ],
            'syllables' => [['text' => 'ꦲ', 'x0' => 0.02, 'x1' => 0.1], ['text' => 'ꦏꦸ', 'x0' => 0.12, 'x1' => 0.3]],
            'image' => ['width' => 400, 'height' => 60],
            'elapsed_ms' => 42.0,
        ];
    }

    public function test_demo_reads_uploaded_line_through_service(): void
    {
        Http::fake([
            '*/health' => Http::response(['status' => 'ok', 'checkpoint' => 'out/checkpoints/fase5_fonts/last_snapshot.pt', 'lm' => 'data/charlm/x.pkl']),
            '*/predict' => Http::response($this->prediction()),
            '*/translate' => Http::response(['translation' => 'saya tidak', 'model' => 'nllb', 'elapsed_ms' => 900.0]),
        ]);
        $this->actingAs(User::factory()->create());

        Livewire::test(Demo::class)
            ->assertSee('Layanan model berjalan')
            ->set('photo', UploadedFile::fake()->image('baris.png', 400, 60))
            ->call('read')
            ->assertHasNoErrors()
            ->assertSee('ꦲꦏꦸꦧꦺꦴꦠꦼꦤ꧀')
            ->assertSee('akubotên')
            ->assertSee('dipilih')
            ->assertSee('data-x0', false)
            ->assertSee('saya tidak')
            ->assertSee('leksikon · keyakinan rendah (teks tanpa spasi)');

        Http::assertSent(fn ($request) => str_ends_with($request->url(), '/predict'));
    }

    public function test_demo_explains_when_service_is_down(): void
    {
        Http::fake(['*' => Http::failedConnection()]);
        $this->actingAs(User::factory()->create());

        Livewire::test(Demo::class)
            ->assertSee('Layanan model belum berjalan')
            ->set('photo', UploadedFile::fake()->image('baris.png', 400, 60))
            ->call('read')
            ->assertSee('uvicorn src.serve:app');
    }

    public function test_demo_requires_an_image(): void
    {
        Http::fake(['*' => Http::failedConnection()]);
        $this->actingAs(User::factory()->create());

        Livewire::test(Demo::class)
            ->set('photo', UploadedFile::fake()->create('catatan.txt', 1, 'text/plain'))
            ->call('read')
            ->assertHasErrors(['photo']);
        // mount() memang memeriksa /health; yang tidak boleh terjadi adalah mengirim berkas ke /predict.
        Http::assertNotSent(fn ($request) => str_ends_with($request->url(), '/predict'));
    }
}
