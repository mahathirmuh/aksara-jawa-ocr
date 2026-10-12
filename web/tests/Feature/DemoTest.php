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
            'config' => 'crnn:fase5_fonts · beam 16 · LM o5_n300000_ds0.5 · α 0.75 · β 2.0',
            'beam' => ['alpha' => 0.75, 'beta' => 2.0, 'width' => 16],
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
            '*/health' => Http::response(['status' => 'ok', 'checkpoint' => 'out/checkpoints/fase5_fonts/last_snapshot.pt', 'lm' => 'data/charlm/x.pkl',
                'beam' => ['alpha' => 0.5, 'beta' => 1.0, 'width' => 16]]),
            '*/predict' => Http::response($this->prediction()),
            '*/translate' => Http::response(['translation' => 'saya tidak', 'model' => 'nllb', 'elapsed_ms' => 900.0]),
        ]);
        $this->actingAs(User::factory()->create());

        // Bobot LM tidak tertulis di web: sebelum membaca, dari /health layanan; sesudah membaca, dari jawaban baca itu.
        Livewire::test(Demo::class)
            ->assertDontSee('LM karakter o5')
            ->set('pipeline', 'crnn_beam_lm')
            ->assertSee('LM karakter o5 · α 0,5 · β 1')
            ->set('photo', UploadedFile::fake()->image('baris.png', 400, 60))
            ->call('read')
            ->assertSee('LM karakter o5 · α 0,75 · β 2')
            ->assertSee('Skor = log P CTC + 0,75 × log P LM + 2 × panjang.')
            ->assertDontSee('0,25');

        Livewire::test(Demo::class)
            ->assertSee('Layanan model berjalan')
            // Nama checkpoint dari /health layanan, bukan tertanam di web; bawaannya jalur resmi (greedy).
            ->assertSet('pipeline', 'crnn_greedy')
            ->assertSee('CRNN fase5_fonts · greedy (jalur resmi)')
            ->assertSee('CRNN fase5_fonts · beam + LM')
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

    public function test_demo_does_not_invent_beam_weights_the_service_did_not_report(): void
    {
        // Layanan versi lama (tanpa kunci "beam"): rumus ditulis dengan huruf, bukan dengan bobot karangan web.
        Http::fake([
            '*/health' => Http::response(['status' => 'ok', 'checkpoint' => 'out/checkpoints/fase5_fonts/last_snapshot.pt']),
            '*/predict' => Http::response(array_diff_key($this->prediction(), ['beam' => true])),
            '*/translate' => Http::response(['translation' => 'saya tidak', 'model' => 'nllb', 'elapsed_ms' => 900.0]),
        ]);
        $this->actingAs(User::factory()->create());
        Livewire::test(Demo::class)
            ->set('pipeline', 'crnn_beam_lm')
            ->assertSee('LM karakter o5')->assertDontSee('LM karakter o5 ·')
            ->set('photo', UploadedFile::fake()->image('baris.png', 400, 60))
            ->call('read')
            ->assertSee('Skor = log P CTC + α × log P LM + β × panjang; bobot α dan β tertulis di konfigurasi layanan.')
            ->assertDontSee('× log P LM + 1');
        // Bobot yang tidak lengkap atau bukan angka dianggap tidak dilaporkan; sumber pertama yang sah menang.
        $this->assertNull(\App\Services\OcrService::beamWeights(null, []));
        $this->assertNull(\App\Services\OcrService::beamWeights(['beam' => ['alpha' => 'banyak', 'beta' => 1.0]]));
        $this->assertNull(\App\Services\OcrService::beamWeights(['beam' => ['alpha' => 0.5, 'beta' => 'sedikit']]));
        $this->assertNull(\App\Services\OcrService::beamWeights(['beam' => ['alpha' => 0.5]], ['beam' => ['beta' => 1.0]], ['beam' => 'x']));
        $this->assertSame(['alpha' => 0.75, 'beta' => 2.0], \App\Services\OcrService::beamWeights(
            ['beam' => ['alpha' => 0.5]], ['beam' => ['alpha' => 0.75, 'beta' => 2.0]], ['beam' => ['alpha' => 0.5, 'beta' => 1.0]]));
        $this->assertSame(['alpha' => 0.5, 'beta' => 1.0], \App\Services\OcrService::beamWeights(null, ['beam' => ['alpha' => '0.5', 'beta' => 1]]));
        $this->assertSame(['0,5', '1', '0,125', '0'], array_map(\App\Services\OcrService::weight(...), [0.5, 1.0, 0.125, 0.0]));
    }

    public function test_demo_explains_when_service_is_down(): void
    {
        Http::fake(['*' => Http::failedConnection()]);
        $this->actingAs(User::factory()->create());

        Livewire::test(Demo::class)
            ->assertSee('Layanan model belum berjalan')
            ->assertSee('CRNN (layanan belum berjalan) · greedy')
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
