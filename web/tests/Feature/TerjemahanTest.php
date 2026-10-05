<?php

namespace Tests\Feature;

use App\Livewire\Pages\Terjemahan;
use App\Models\User;
use App\Support\TalingRestorer;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Http\Client\Request;
use Illuminate\Support\Facades\Http;
use Livewire\Livewire;
use Tests\TestCase;

/** Halaman Terjemahan: model terjemahan lewat layanan (dipalsukan di sini), alih aksara dan pemulihan tanda é di web. */
class TerjemahanTest extends TestCase
{
    use RefreshDatabase;

    private const HEALTH = ['status' => 'ok', 'model' => 'facebook/nllb-200-distilled-600M', 'directions' => ['jv-id', 'id-jv'], 'loaded' => true];

    /** Layanan palsu: /health hidup, /translate menjawab daftar terjemahan yang diberikan. */
    private function fakeService(array $translations): void
    {
        Http::fake([
            '*/health' => Http::response(self::HEALTH),
            '*/translate' => Http::response(['translations' => $translations, 'model' => self::HEALTH['model'], 'direction' => 'x', 'elapsed_ms' => 2100.0]),
        ]);
    }

    public function test_guests_are_redirected_and_the_page_is_in_the_sidebar(): void
    {
        Http::fake(['*' => Http::failedConnection()]);
        $this->get('/terjemahan')->assertRedirect('/login');

        $this->actingAs(User::factory()->create());
        $this->get('/ringkasan')->assertOk()->assertSee(route('terjemahan'), false)->assertSee('Terjemahan');
    }

    public function test_indonesian_is_translated_then_spelled_in_javanese_script(): void
    {
        $this->fakeService(['Apa kowe wis mangan?', 'Omahe gedhe.']);
        $this->actingAs(User::factory()->create());

        $page = Livewire::test(Terjemahan::class)
            ->assertSee('Layanan terjemahan berjalan')
            ->set('text', "Apakah kamu sudah makan? Rumahnya besar.")
            ->call('translate')
            ->assertHasNoErrors()
            // Tanda é dipulihkan pada teks Latin yang terlihat; aksara dihitung dari teks itu (taling, bukan pepet).
            ->assertSet('javanese', 'Apa kowé wis mangan? Omahé gedhé.')
            ->assertSet('restored', ['kowe' => 'kowé', 'Omahe' => 'Omahé', 'gedhe' => 'gedhé'])
            ->assertSee('ꦲꦥ ꦏꦺꦴꦮꦺ ꦮꦶꦱ꧀ ꦩꦔꦤ꧀? ꦲꦺꦴꦩꦲꦺ ꦒꦼꦝꦺ꧉')
            ->assertSee('3 kata dipulihkan')
            ->assertSee('2 kalimat')->assertSee('2,1 detik')
            ->assertSee('Arti per kata di Kamus');

        // Dua kalimat dikirim terpisah, dari bahasa Indonesia ke bahasa Jawa.
        Http::assertSent(fn (Request $request) => str_ends_with($request->url(), '/translate')
            && $request['texts'] === ['Apakah kamu sudah makan?', 'Rumahnya besar.'] && $request['source'] === 'id' && $request['target'] === 'jv');

        // Teks Latin boleh disunting: aksara mengikuti tanpa memanggil layanan lagi.
        $page->set('javanese', 'aku arep lunga')->assertSee('ꦲꦏꦸ ꦲꦉꦥ꧀ ꦭꦸꦔ');
        Http::assertSentCount(2);
    }

    public function test_pasted_script_is_romanised_then_translated_to_indonesian(): void
    {
        $this->fakeService(['Saya akan pergi ke pasar.']);
        $this->actingAs(User::factory()->create());

        Livewire::test(Terjemahan::class)
            ->set('direction', 'jv-id')
            ->set('text', 'ꦏꦸꦭ ꦧꦝꦺ ꦠꦶꦤ꧀ꦢꦏ꧀ ꦝꦠꦼꦁ ꦥꦼꦏꦼꦤ꧀꧉')
            ->assertSeeInOrder(['Latin (draf aturan)', 'kula badh'])
            ->call('translate')
            ->assertHasNoErrors()
            ->assertSet('indonesian', 'Saya akan pergi ke pasar.')
            ->assertSee('Saya akan pergi ke pasar.')
            ->assertSee('Tingkat tutur menurut leksikon:');

        // Yang dikirim ke model adalah bacaan Latin, bukan aksaranya, dan pepetnya tanpa tanda (model tidak mengenal "ê").
        Http::assertSent(fn (Request $request) => str_ends_with($request->url(), '/translate')
            && $request['texts'] === ['kula badhé tindak dhateng peken.'] && $request['source'] === 'jv' && $request['target'] === 'id');
    }

    public function test_latin_javanese_input_is_translated_as_typed(): void
    {
        $this->fakeService(['Saya makan nasi.', 'Dia tidur.']);
        $this->actingAs(User::factory()->create());

        Livewire::test(Terjemahan::class, ['direction' => 'jv-id'])
            ->set('direction', 'jv-id')
            ->set('text', "Aku mangan sega.\nDheweke turu.")
            ->assertDontSee('Latin (draf aturan)')
            ->call('translate')
            // Pemisah baris teks asal dipertahankan di hasil.
            ->assertSet('indonesian', "Saya makan nasi.\nDia tidur.");
        Http::assertSent(fn (Request $request) => str_ends_with($request->url(), '/translate') && $request['texts'] === ['Aku mangan sega.', 'Dheweke turu.']);
    }

    public function test_script_conversion_works_without_the_service_and_failures_are_explained(): void
    {
        Http::fake(['*' => Http::failedConnection()]);
        $this->actingAs(User::factory()->create());

        Livewire::test(Terjemahan::class)
            ->assertSee('Layanan terjemahan belum berjalan')->assertSee('translate_service:app')
            // Tanpa layanan: bahasa Jawa yang diketik sendiri tetap dialihaksarakan, dan tanda é bisa dipulihkan.
            ->set('javanese', 'kowe arep menyang endi')
            ->assertSee('ꦏꦺꦴꦮꦼ ꦲꦉꦥ꧀ ꦩꦼꦚꦁ ꦲꦼꦤ꧀ꦢꦶ')
            ->call('restoreTaling')
            ->assertSet('javanese', 'kowé arep menyang endi')
            ->assertSee('ꦏꦺꦴꦮꦺ ꦲꦉꦥ꧀ ꦩꦼꦚꦁ ꦲꦼꦤ꧀ꦢꦶ')
            // Menerjemahkan butuh layanan.
            ->set('text', 'Saya makan.')
            ->call('translate')
            ->assertSet('error', 'Layanan terjemahan belum berjalan.')
            ->assertSet('javanese', 'kowé arep menyang endi');
    }

    public function test_old_one_way_service_is_recognised(): void
    {
        $this->actingAs(User::factory()->create());
        // Layanan versi lama: tidak melaporkan arah dan menjawab bentuk satu teks.
        Http::fake([
            '*/health' => Http::response(['status' => 'ok', 'model' => 'nllb', 'loaded' => true]),
            '*/translate' => Http::response(['translation' => 'x', 'model' => 'nllb', 'elapsed_ms' => 1.0]),
        ]);

        Livewire::test(Terjemahan::class)
            ->assertSee('Layanan terjemahan versi lama')
            ->set('text', 'Saya makan.')->call('translate')
            ->assertSee('versi lama (hanya Jawa → Indonesia)')
            ->assertSet('javanese', '');
    }

    public function test_service_errors_and_a_loading_model_are_reported(): void
    {
        $this->actingAs(User::factory()->create());
        Http::fake([
            '*/health' => Http::response(['loaded' => false] + self::HEALTH),
            '*/translate' => Http::response(['detail' => 'paling banyak 16 teks per permintaan'], 422),
        ]);

        Livewire::test(Terjemahan::class)
            ->assertSee('Model sedang dimuat')
            ->set('text', 'Saya makan.')->call('translate')
            ->assertSee('Layanan terjemahan menolak permintaan: paling banyak 16 teks per permintaan');
    }

    public function test_input_limits_and_direction_swap(): void
    {
        $this->fakeService(['Aku mangan.']);
        $this->actingAs(User::factory()->create());

        Livewire::test(Terjemahan::class)
            ->call('translate')->assertHasErrors(['text' => 'required'])
            ->set('text', str_repeat('a', Terjemahan::MAX_CHARS + 1))->call('translate')->assertHasErrors(['text' => 'max'])
            ->set('text', implode(' ', array_fill(0, Terjemahan::MAX_SENTENCES + 1, 'Saya makan.')))->call('translate')
            ->assertSee('Teks berisi 13 kalimat; paling banyak 12 sekali jalan');
        // Ketiganya ditolak sebelum layanan dipanggil (yang terkirim hanya pemeriksaan /health saat halaman dibuka).
        Http::assertNotSent(fn (Request $request) => str_ends_with($request->url(), '/translate'));

        // Tukar arah: aksara hasil menjadi masukan arah sebaliknya, hasil lama dikosongkan.
        Livewire::test(Terjemahan::class)
            ->set('text', 'Saya makan.')->call('translate')
            ->assertSet('javanese', 'Aku mangan.')
            ->call('swap')
            ->assertSet('direction', 'jv-id')->assertSet('text', 'ꦲꦏꦸ ꦩꦔꦤ꧀꧉')->assertSet('javanese', '')->assertSet('meta', null)
            ->set('direction', 'bukan-arah')->assertSet('direction', 'id-jv');

        // ?q= dan ?arah= mengisi halaman dari tautan: aksara di ?q= langsung dilatinkan (isi kotaknya sendiri dikirim Livewire).
        $this->get('/terjemahan?arah=jv-id&q='.rawurlencode('ꦲꦏꦸ ꦩꦔꦤ꧀'))->assertOk()
            ->assertSeeInOrder(['Latin (draf aturan)', 'aku mangan']);
        // ?jawa= mengisi kotak bahasa Jawa Latin: aksaranya langsung tampil, tanpa layanan.
        $this->get('/terjemahan?jawa='.rawurlencode('aku mangan sega'))->assertOk()->assertSee('ꦲꦏꦸ ꦩꦔꦤ꧀ ꦱꦼꦒ');
    }

    public function test_sentences_are_split_per_line_and_per_sentence(): void
    {
        $this->assertSame(
            [['text' => 'Satu.', 'break' => false], ['text' => 'Dua?', 'break' => false], ['text' => 'Tiga', 'break' => true], ['text' => 'Empat!', 'break' => true]],
            Terjemahan::sentences("Satu. Dua?  Tiga\n\n  Empat!  "),
        );
        $this->assertSame([], Terjemahan::sentences("  \n "));
    }

    public function test_taling_lexicon_restores_only_unmarked_known_words(): void
    {
        $restored = TalingRestorer::restore('Kowe lan aku gedhe, kabeh seneng. Désa kae gedhé.');
        // Kata bertanda tidak disentuh, kata pepet (seneng) tetap, huruf besar asal dipertahankan.
        $this->assertSame('Kowé lan aku gedhé, kabèh seneng. Désa kaé gedhé.', $restored['text']);
        $this->assertSame(['Kowe' => 'Kowé', 'gedhe' => 'gedhé', 'kabeh' => 'kabèh', 'kae' => 'kaé'], $restored['changed']);

        $meta = TalingRestorer::meta();
        $this->assertGreaterThan(8000, $meta['words']);
        $this->assertGreaterThan($meta['held_out']['accuracy_without'] + 0.3, $meta['held_out']['accuracy_with']);

        // Tanpa berkas leksikon: tidak ada yang diubah, halaman tetap jalan.
        config(['aksara.taling_lexicon' => sys_get_temp_dir().'/tidak-ada-'.uniqid().'.json']);
        $this->assertSame(['text' => 'kowe gedhe', 'changed' => []], TalingRestorer::restore('kowe gedhe'));
        $this->assertNull(TalingRestorer::meta());
    }
}
