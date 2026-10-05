<?php

namespace Tests\Feature;

use App\Livewire\Pages\Kamus;
use App\Livewire\Pages\Penjelajah;
use App\Livewire\Pages\Perbandingan;
use App\Models\Line;
use App\Models\Metric;
use App\Models\Pipeline;
use App\Models\Prediction;
use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Livewire\Livewire;
use Tests\TestCase;

class AksaraPagesTest extends TestCase
{
    use RefreshDatabase;

    private const PAGES = ['/ringkasan', '/perbandingan', '/ablasi', '/penjelajah', '/kesalahan', '/demo', '/kamus'];

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
            ->assertSee('CER draf')->assertSee('(greedy)')
            ->assertSee('titik penuh = angka G3 resmi (CRNN fase5_fonts, greedy)')
            ->assertSee('CRNN fase5_fonts · beam + LM');
        // Grafik uji buta: 5 batang di fixture, jadi tingginya yang paling kecil (16rem).
        $this->get('/perbandingan')->assertOk()->assertSee('CRNN fase5_fonts')->assertSee('VLM zero-shot')->assertSee('Sahabat-AI')
            ->assertDontSee('satu run per kondisi')
            ->assertSee('height: 16rem', false);
        $this->get('/ablasi')->assertOk()->assertSee('tight')->assertSee('speckle');
        $this->get('/kesalahan')->assertOk()->assertSee('Tertukar')->assertSee('Hilang')
            ->assertSee('CRNN fase5_fonts · greedy · 745 baris');

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

    /** Run lanjutan seperti hasil export_results.py: pipeline selesai dengan metrik 745 baris dan uji buta. */
    private function addFollowupRun(string $key, float $cer, int $sort, string $config): void
    {
        Pipeline::create(['key' => $key, 'label' => 'CRNN '.substr($key, 5), 'config' => $config, 'kind' => 'crnn',
            'status' => 'done', 'sort' => $sort]);
        foreach (['nusaaksara_745' => 10, 'blind_50' => 5] as $scope => $lines) {
            Metric::create(['scope' => $scope, 'pipeline' => $key, 'lines' => $lines, 'cer' => $cer,
                'cer_no_space' => $cer - 0.05, 'exact' => 0.0]);
        }
    }

    public function test_fase7_runs_are_shown_as_comparators_not_official(): void
    {
        $this->importFixture();
        // Urutan seperti export_results.py: fase6 (rare, ctrl) lalu fase7 (track, track_rare, ctrl), tepat setelah
        // beam (sort 3); pipeline sesudahnya bergeser. Config fase7 = teks yang dibangun ekspor dari checkpoint.
        Pipeline::where('sort', '>', 3)->increment('sort', 5);
        $this->addFollowupRun('crnn_fase6_rare', 0.30, 4, 'crnn:fase6_rare@1500 (lanjutan fase5_fonts@1298) · 10 font · '
            .'aug fase5 · + aksara langka (sisip 0,3 · adeg-adeg 0,15) · greedy');
        $this->addFollowupRun('crnn_fase6_ctrl', 0.31, 5, 'crnn:fase6_ctrl@1500 (lanjutan fase5_fonts@1298) · 10 font · '
            .'aug fase5 · tanpa aksara langka (pembanding) · greedy');
        $this->addFollowupRun('crnn_fase7_track', 0.27, 6, 'crnn:fase7_track@1500 (lanjutan fase6_ctrl@1500) · 10 font · '
            .'aug fase5 · jarak antar-aksara p 0,5 hingga 0,3 em · greedy');
        $this->addFollowupRun('crnn_fase7_track_rare', 0.28, 7, 'crnn:fase7_track_rare@1500 (lanjutan fase6_ctrl@1500) · '
            .'10 font · aug fase5 · jarak antar-aksara p 0,5 hingga 0,3 em · + aksara langka tersaring (sisip 0,3 · '
            .'adeg-adeg 0,15 · mirip >= 0,9 · tempel) · greedy');
        $this->addFollowupRun('crnn_fase7_ctrl', 0.32, 8, 'crnn:fase7_ctrl@1500 (lanjutan fase6_ctrl@1500) · 10 font · '
            .'aug fase5 · tanpa jarak tambahan (pembanding) · greedy');
        $this->actingAs(User::factory()->create());

        // Baris tabel 745 baris urut sort, tiap run fase7 dengan catatannya; angka resmi tetap fase5_fonts walaupun
        // run fase7 di sini lebih rendah. Grafik uji buta meninggi mengikuti 10 batangnya (5 fixture + 5 run).
        $this->get('/perbandingan')->assertOk()->assertSee('height: 20.5rem', false)->assertSeeInOrder(['745 baris nyata',
            'CRNN fase5_fonts', '34,92%', 'angka G3 resmi', 'Beam + LM',
            'CRNN fase6_rare', 'fase5_fonts + 1.500 langkah dengan sisipan aksara langka',
            'CRNN fase6_ctrl', 'pembanding: langkah dan titik lanjut sama, tanpa aksara langka',
            'CRNN fase7_track', 'jarak antar-aksara p 0,5 hingga 0,3 em', '27,00%',
            'fase6_ctrl + 1.500 langkah dengan jarak antar-aksara acak',
            'CRNN fase7_track_rare', 'mirip >= 0,9 · tempel', '28,00%',
            'sama dengan fase7_track + sisipan aksara langka yang disaring',
            'CRNN fase7_ctrl', 'tanpa jarak tambahan (pembanding)', '32,00%',
            'pembanding: langkah sama tanpa jarak tambahan',
            'CRNN fase6_rare, CRNN fase6_ctrl, CRNN fase7_track, CRNN fase7_track_rare dan CRNN fase7_ctrl: satu run per kondisi',
            'angka G3 resmi tetap CRNN fase5_fonts (greedy)']);
        // Sembilan titik pada sumbu; daftar di bawahnya urut CER. Kartu gerbang G3 dan tahap 1 tetap fase5_fonts.
        $this->get('/ringkasan')->assertOk()
            ->assertSee('37,2%')->assertSee('CER 34,9% (greedy)')
            ->assertSeeInOrder(['CER G3 per pipeline, terbaik dulu',
                'CRNN fase7_track', '27,0%', 'CRNN fase7_track_rare', '28,0%', 'Beam + LM', '29,5%',
                'CRNN fase6_rare', '30,0%', 'CRNN fase6_ctrl', '31,0%', 'CRNN fase7_ctrl', '32,0%',
                'CRNN fase5_fonts', '34,9%', 'CRNN fase5_core', '54,4%', 'CRNN 4b', '93,3%']);

        // Rantai fase7 baru menyelesaikan run pertama: ekspor melewati run tanpa last_snapshot.pt, jadi pipeline
        // dan metriknya tidak ada. Halaman hanya menyebut run yang ada.
        Pipeline::whereIn('key', ['crnn_fase7_track_rare', 'crnn_fase7_ctrl'])->delete();
        Metric::whereIn('pipeline', ['crnn_fase7_track_rare', 'crnn_fase7_ctrl'])->delete();
        $this->get('/perbandingan')->assertOk()
            ->assertSee('CRNN fase6_rare, CRNN fase6_ctrl dan CRNN fase7_track: satu run per kondisi')
            ->assertSee('fase6_ctrl + 1.500 langkah dengan jarak antar-aksara acak')
            ->assertDontSee('fase7_track_rare')->assertDontSee('fase7_ctrl')
            ->assertDontSee('pembanding: langkah sama tanpa jarak tambahan')
            ->assertSee('height: 17rem', false);
        $this->get('/ringkasan')->assertOk()
            ->assertSeeInOrder(['CER G3 per pipeline, terbaik dulu', 'CRNN fase7_track', '27,0%', 'Beam + LM', '29,5%'])
            ->assertDontSee('fase7_track_rare')->assertDontSee('fase7_ctrl');
    }

    public function test_official_pipeline_can_be_a_followup_run(): void
    {
        $this->importFixture();
        Pipeline::where('sort', '>', 3)->increment('sort', 5);
        $this->addFollowupRun('crnn_fase6_ctrl', 0.31, 5, 'crnn:fase6_ctrl@1500 · greedy');
        $this->addFollowupRun('crnn_fase7_track', 0.27, 6, 'crnn:fase7_track@1500 · greedy');
        $this->addFollowupRun('crnn_fase7_ctrl', 0.32, 8, 'crnn:fase7_ctrl@1500 · greedy');
        // Seperti impor manifest dengan "official": "crnn_fase7_track".
        Pipeline::query()->update(['official' => false]);
        Pipeline::where('key', 'crnn_fase7_track')->update(['official' => true]);
        foreach (Line::all() as $line) {
            Prediction::create(['line_id' => $line->id, 'pipeline' => 'crnn_fase7_track', 'text' => $line->reference,
                'cer' => 0.05, 'segments' => [['t' => $line->reference, 's' => 'ok']], 'spans' => null]);
        }
        $this->actingAs(User::factory()->create());

        // Tanda resmi pindah ke run fase7 (catatannya tetap), fase5_fonts menjadi angka resmi sebelumnya, dan
        // peringatan menyebut run lain sebagai pembanding angka resmi yang baru.
        $this->get('/perbandingan')->assertOk()->assertSeeInOrder(['745 baris nyata',
            'CRNN fase5_fonts', '34,92%', 'angka G3 resmi sebelumnya', 'Beam + LM',
            'CRNN fase6_ctrl', 'pembanding: langkah dan titik lanjut sama, tanpa aksara langka',
            'CRNN fase7_track', '27,00%', 'angka G3 resmi · fase6_ctrl + 1.500 langkah dengan jarak antar-aksara acak',
            'CRNN fase7_ctrl', 'pembanding: langkah sama tanpa jarak tambahan',
            'CRNN fase6_ctrl dan CRNN fase7_ctrl: satu run per kondisi',
            'Angka G3 resmi = CRNN fase7_track (greedy)'])
            ->assertDontSee('angka G3 resmi tetap');
        // Kartu tahap 1 memakai pipeline resmi; beam tidak disebut karena membaca checkpoint lain (fase5_fonts).
        $this->get('/ringkasan')->assertOk()
            ->assertSee('titik penuh = angka G3 resmi (CRNN fase7_track, greedy)')
            ->assertSee('CER 27,0% (greedy)')
            ->assertSee('CRNN fase7_track · target < 8%')
            ->assertDontSee('beam + LM 29,5% · target');
        $this->get('/kesalahan')->assertOk()->assertSee('CRNN fase7_track · greedy · 745 baris');
        // Penjelajah: urutan dan angka di daftar dari pipeline resmi, yang juga ikut tampil sejak halaman dibuka.
        Livewire::test(Penjelajah::class)
            ->assertSet('pipes', ['crnn_fase7_track', 'crnn_fonts', 'crnn_fonts_beam', 'vlm_zeroshot'])
            ->assertSee('CER fase7_track, terburuk dulu')
            ->assertSee('CRNN 5%')
            ->assertSeeHtml('row-'.Line::orderBy('external_id')->first()->id.'-crnn_fase7_track');
    }

    public function test_followup_runs_match_the_export_script(): void
    {
        // Web mengenali run lanjutan lewat kuncinya. Run yang ditambahkan ke ekspor tanpa ditambahkan di sini akan
        // tampil tanpa catatan, tanpa peringatan "satu run per kondisi", dan tanpa titik di Ringkasan.
        $script = dirname(base_path()).'/scripts/export_results.py';
        if (! is_file($script)) {
            $this->markTestSkipped('scripts/export_results.py tidak ada di samping web/.');
        }
        $source = str_replace("\r\n", "\n", file_get_contents($script));
        $this->assertSame(1, preg_match('/^OPTIONAL_CHECKPOINTS = \[\n(.*?)^\]/ms', $source, $block),
            'daftar OPTIONAL_CHECKPOINTS tidak ditemukan di scripts/export_results.py');
        preg_match_all('/"key": "([a-z0-9_]+)"/', $block[1], $keys);

        $this->assertNotEmpty($keys[1]);
        $this->assertSame($keys[1], array_keys(Perbandingan::FOLLOWUP_RUNS));
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

    public function test_dictionary_page_lists_script_language_model_and_lexicon(): void
    {
        $this->actingAs(User::factory()->create());

        // Tanpa hasil yang diimpor: kamus aksara dan leksikon tetap tampil, kamus koreksi menjelaskan cara impor.
        $this->get('/kamus')->assertOk()
            ->assertSee('Kamus aksara')->assertSee('wulu')->assertSee('U+A9B6')->assertSee('pada lingsa')
            ->assertSee('(93 kelas)')
            ->assertSee('Leksikon tingkat tutur')->assertSee('panjenengan')->assertSee('sampeyan')
            ->assertSee('Kamus koreksi OCR')->assertSee('aksara:import')
            ->assertDontSee('label uji ditulis tanpa spasi');

        $this->importFixture();
        $greedy = Metric::where(['scope' => 'nusaaksara_745', 'pipeline' => 'crnn_fonts'])->firstOrFail();
        $beam = Metric::where(['scope' => 'nusaaksara_745', 'pipeline' => 'crnn_fonts_beam'])->firstOrFail();
        $unspaced = Line::whereJsonContains('tags', 'label tanpa spasi')->count();
        // Satu baris tabel: data, greedy, beam + LM, selisih bertanda, lalu jumlah baris yang membaik/memburuk.
        $this->get('/kamus')->assertOk()
            ->assertSee($unspaced.' dari '.Line::count().' label uji ditulis tanpa spasi')
            ->assertSeeInOrder(['Baris cetak nyata (NusaAksara)', pct($greedy->cer, 2), pct($beam->cer, 2),
                pt($beam->cer - $greedy->cer), $beam->better.' baris membaik, '.$beam->worse.' memburuk'])
            ->assertSee('Dev sintetis (aug fase5)')->assertSee('Font held-out bersih')
            ->assertDontSee('aksara:import')
            // Di fixture pipeline resmi = checkpoint yang sama dengan pembanding greedy: tidak ada peringatan.
            ->assertDontSee('Kamus belum diukur di atas pipeline resmi');

        // Pipeline resmi lain: angka kamus masih milik checkpoint lama, dan halaman mengatakannya. Hanya satu pipeline
        // yang boleh bertanda resmi (dengan dua, urutan baris PostgreSQL dan SQLite memilih yang berbeda).
        Pipeline::query()->update(['official' => false]);
        Pipeline::where('key', 'crnn_core')->update(['official' => true]);
        $this->get('/kamus')->assertOk()
            ->assertSee('Kamus belum diukur di atas pipeline resmi CRNN fase5_core, dan bobotnya disetel pada CRNN fase5_fonts.');
    }

    public function test_dictionary_search_filters_and_breaks_down_script(): void
    {
        $this->actingAs(User::factory()->create());

        // Nama aksara: hanya ubin yang cocok; leksikon tidak punya kata "wulu".
        Livewire::test(Kamus::class)->set('q', 'wulu')
            ->assertSee('U+A9B6')->assertDontSee('U+A9C0')
            ->assertSee('Tidak ada yang cocok.');

        // Kata penanda: tersaring ke tingkatnya, dan tidak ada aksara bernama itu.
        Livewire::test(Kamus::class)->set('q', 'panjenengan')
            ->assertSee('panjenengan')->assertDontSee('sampeyan')
            ->assertSee('Tidak ada aksara yang cocok dengan pencarian ini.');

        // Aksara yang ditempel (ka + wulu): diurai per codepoint dan diberi bacaan Latin draf.
        Livewire::test(Kamus::class)->set('q', mb_chr(0xA98F).mb_chr(0xA9B6))
            ->assertSeeInOrder(['Uraian', 'Latin (draf aturan):', 'ki'])
            ->assertSee('U+A98F')->assertSee('U+A9B6')->assertDontSee('U+A9C0');

        // Kalimat Latin berspasi: tingkat tutur dari leksikon beserta kata buktinya.
        Livewire::test(Kamus::class)->set('q', 'kula badhe tindak')
            ->assertSee('Tingkat tutur menurut leksikon:')->assertSee('kula · krama')->assertSee('keyakinan cukup');
    }
}
