<?php

namespace Tests\Feature;

use App\Livewire\Pages\Kamus;
use App\Models\DictionaryEntry;
use App\Models\DictionarySource;
use App\Models\User;
use App\Support\DictionarySearch;
use Illuminate\Filesystem\Filesystem;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Livewire\Livewire;
use Tests\TestCase;

/** Kamus kata (bahasa Jawa dan bahasa Indonesia) di halaman Kamus: impor berkas, pencarian, dan tampilannya. */
class DictionaryTest extends TestCase
{
    use RefreshDatabase;

    private const SOURCES = [
        'jv' => [['key' => 'contoh', 'name' => 'Kamus Contoh Jawa', 'license' => 'CC0', 'url' => 'https://example.org/jv', 'retrieved' => '2026-10-05']],
        'id' => [['key' => 'contoh', 'name' => 'Kamus Contoh Indonesia', 'license' => 'CC BY-SA 4.0', 'url' => 'https://example.org/id']],
    ];

    private string $dir;

    protected function setUp(): void
    {
        parent::setUp();
        $this->dir = sys_get_temp_dir().DIRECTORY_SEPARATOR.'aksara-kamus-'.uniqid();
        mkdir($this->dir);
    }

    protected function tearDown(): void
    {
        (new Filesystem)->deleteDirectory($this->dir);
        parent::tearDown();
    }

    private static function jv(string $word, array $glosses, array $extra = []): array
    {
        return ['word' => $word, 'glosses' => $glosses, 'gloss_lang' => 'id', 'source' => 'contoh'] + $extra;
    }

    /** Tulis sources.json dan berkas entri; $lines boleh berisi string mentah untuk menguji baris rusak. */
    private function write(array $files, bool $gzip = false, ?array $sources = null): void
    {
        file_put_contents($this->dir.'/sources.json', json_encode($sources ?? self::SOURCES, JSON_UNESCAPED_UNICODE));
        foreach ($files as $dictionary => $lines) {
            $text = implode("\n", array_map(fn ($line) => is_string($line) ? $line : json_encode($line, JSON_UNESCAPED_UNICODE), $lines))."\n";
            @unlink("{$this->dir}/{$dictionary}.jsonl");
            @unlink("{$this->dir}/{$dictionary}.jsonl.gz");
            file_put_contents("{$this->dir}/{$dictionary}.jsonl".($gzip ? '.gz' : ''), $gzip ? gzencode($text) : $text);
        }
    }

    private function seedDictionaries(): void
    {
        $this->write([
            'jv' => [
                self::jv('omah', ['rumah', 'tempat tinggal'], ['pos' => 'nomina', 'aksara' => 'ꦲꦺꦴꦩꦃ', 'register' => 'ngoko', 'url' => 'https://example.org/jv/omah',
                    'note' => 'ngoko omah · krama griya', 'examples' => ['Omahé gedhé. — Rumahnya besar.', str_repeat('x', 301), 'dua', 'tiga', 'empat']]),
                self::jv('omah-omah', ['berumah tangga'], ['pos' => 'verba']),
                self::jv('pomahan', ['pekarangan']),
                self::jv('griya', ['rumah'], ['pos' => 'nomina', 'aksara' => 'ꦒꦿꦶꦪ', 'register' => 'krama']),
                self::jv('gêdhé', ['besar'], ['pos' => 'adjektiva']),
                self::jv('kula', ['saya'], ['pos' => 'pronomina', 'register' => 'krama']),
                self::jv('tindak', ['pergi', 'berjalan'], ['register' => 'krama inggil']),
                self::jv('warung', ['kedai rumahan']),
                ['word' => 'toya', 'glosses' => ['water'], 'gloss_lang' => 'en', 'source' => 'contoh', 'aksara' => 'ꦠꦺꦴꦪ'],
            ],
            'id' => [
                ['word' => 'rumah', 'glosses' => ['bangunan untuk tempat tinggal', 'bangunan pada umumnya'], 'gloss_lang' => 'id', 'source' => 'contoh', 'pos' => 'nomina'],
                ['word' => 'rumah sakit', 'glosses' => ['gedung tempat merawat orang sakit'], 'gloss_lang' => 'id', 'source' => 'contoh', 'pos' => 'nomina'],
                ['word' => 'sakit', 'glosses' => ['berasa tidak nyaman di tubuh'], 'gloss_lang' => 'id', 'source' => 'contoh', 'pos' => 'adjektiva'],
            ],
        ]);
        $this->artisan('aksara:dictionary', ['--path' => $this->dir])->assertSuccessful();
    }

    public function test_import_fills_both_dictionaries_and_replaces_old_entries(): void
    {
        $this->seedDictionaries();

        $this->assertSame(9, DictionaryEntry::where('dictionary', 'jv')->count());
        $this->assertSame(3, DictionaryEntry::where('dictionary', 'id')->count());
        $big = DictionaryEntry::where('word', 'gêdhé')->firstOrFail();
        $this->assertSame('gedhe', $big->lookup);
        $this->assertSame(['besar'], $big->glosses);
        $this->assertSame(' rumah tempat tinggal ', DictionaryEntry::where('word', 'omah')->value('gloss_text'));
        // Contoh: yang lebih dari 300 karakter dibuang, paling banyak tiga disimpan; entri tanpa contoh = null.
        $this->assertSame(['Omahé gedhé. — Rumahnya besar.', 'dua', 'tiga'], DictionaryEntry::where('word', 'omah')->firstOrFail()->examples);
        $this->assertNull($big->examples);
        $source = DictionarySource::where('dictionary', 'jv')->firstOrFail();
        $this->assertSame(['Kamus Contoh Jawa', 'CC0', 9, '2026-10-05'], [$source->name, $source->license, $source->entries, $source->retrieved]);

        // Impor ulang hanya kamus Indonesia, dari berkas .gz: kamus itu diganti seutuhnya, kamus Jawa tidak disentuh.
        $this->write(['id' => [['word' => 'buku', 'glosses' => ['lembar kertas berjilid'], 'gloss_lang' => 'id', 'source' => 'contoh']]], gzip: true);
        $this->artisan('aksara:dictionary', ['--path' => $this->dir, '--only' => 'id'])
            ->expectsOutputToContain('Kamus bahasa Indonesia: 1 entri')->assertSuccessful();
        $this->assertSame(['buku'], DictionaryEntry::where('dictionary', 'id')->pluck('word')->all());
        $this->assertSame(1, DictionarySource::where('dictionary', 'id')->value('entries'));
        $this->assertSame(9, DictionaryEntry::where('dictionary', 'jv')->count());
    }

    public function test_import_refuses_broken_files_and_keeps_existing_entries(): void
    {
        $this->seedDictionaries();

        $broken = [
            'tanpa arti' => [self::jv('sega', [])],
            'sumber tak dikenal' => [['word' => 'sega', 'glosses' => ['nasi'], 'gloss_lang' => 'id', 'source' => 'entah']],
            'bahasa arti salah' => [['word' => 'sega', 'glosses' => ['nasi'], 'gloss_lang' => 'xx', 'source' => 'contoh']],
            'bukan JSON' => [self::jv('sega', ['nasi']), '{"word": "rusak"'],
        ];
        foreach ($broken as $case => $lines) {
            $this->write(['jv' => $lines]);
            $this->artisan('aksara:dictionary', ['--path' => $this->dir, '--only' => 'jv'])
                ->expectsOutputToContain('jv.jsonl baris '.count($lines))->assertFailed();
            // Satu transaksi per kamus: isi lama tetap ada.
            $this->assertSame(9, DictionaryEntry::where('dictionary', 'jv')->count(), $case);
            $this->assertSame(9, DictionarySource::where('dictionary', 'jv')->value('entries'), $case);
        }

        // Tautan sumber juga menjadi href.
        $bad = self::SOURCES;
        $bad['jv'][0]['url'] = 'javascript:alert(1)';
        $this->write(['jv' => [self::jv('sega', ['nasi'])]], sources: $bad);
        $this->artisan('aksara:dictionary', ['--path' => $this->dir, '--only' => 'jv'])->expectsOutputToContain('harus berawalan http')->assertFailed();
        $this->assertSame(9, DictionaryEntry::where('dictionary', 'jv')->count());

        (new Filesystem)->cleanDirectory($this->dir);
        $this->artisan('aksara:dictionary', ['--path' => $this->dir])->expectsOutputToContain('tools/dictionaries.py')->assertFailed();
        $this->artisan('aksara:dictionary', ['--path' => $this->dir, '--only' => 'en'])->expectsOutputToContain('--only harus jv atau id')->assertFailed();
    }

    public function test_page_explains_import_until_dictionaries_exist_then_credits_sources(): void
    {
        $this->actingAs(User::factory()->create());

        $this->get('/kamus')->assertOk()
            ->assertSee('Kamus bahasa Jawa belum diimpor')->assertSee('Kamus bahasa Indonesia belum diimpor')
            ->assertSee('aksara:dictionary')->assertSee('Kamus aksara');

        $this->seedDictionaries();
        $this->get('/kamus')->assertOk()
            ->assertDontSee('belum diimpor')
            ->assertSee('9 entri bahasa Jawa')->assertSee('3 entri bahasa Indonesia')
            // Kata contoh: hanya yang ada di data ("banyu" tidak ada di kamus contoh).
            ->assertSee("\$set('q', 'omah')", false)->assertSee("\$set('q', 'rumah')", false)->assertDontSee("\$set('q', 'banyu')", false)
            ->assertSeeInOrder(['Sumber:', 'Kamus Contoh Jawa', 'CC0; 9 entri; diambil 2026-10-05'])
            ->assertSeeInOrder(['Sumber:', 'Kamus Contoh Indonesia', 'CC BY-SA 4.0; 3 entri'])
            ->assertSee('https://example.org/jv', false);
    }

    /** Teks dua bagian kamus kata (tanpa tag HTML) setelah mencari $q. */
    private function sections(string $q): string
    {
        $html = Livewire::test(Kamus::class)->set('q', $q)->html();
        $from = strpos($html, 'id="kamus-jv"');
        $to = strpos($html, 'id="kamus-aksara"');
        $this->assertTrue($from !== false && $to !== false && $from < $to, 'bagian kamus kata harus ada sebelum kamus aksara');

        return html_entity_decode(trim(preg_replace('/\s+/u', ' ', strip_tags(substr($html, $from, $to - $from)))), ENT_QUOTES);
    }

    /** assertSeeInOrder milik Livewire membandingkan dengan JSON balasan, tempat huruf non-ASCII ter-escape. */
    private function assertOrder(array $values, string $text): void
    {
        $at = 0;
        foreach ($values as $value) {
            $found = mb_strpos($text, $value, $at);
            $this->assertNotFalse($found, "'{$value}' tidak ada sesudah karakter ke-{$at} di: {$text}");
            $at = $found + mb_strlen($value);
        }
    }

    public function test_search_ranks_word_matches_and_finds_words_by_their_meaning(): void
    {
        $this->actingAs(User::factory()->create());
        $this->seedDictionaries();

        // Kata: persis sama, lalu berawalan sama, lalu memuat. Ejaan aksara, kelas kata, dan ragam ikut tampil.
        $text = $this->sections('omah');
        $this->assertOrder(['3 entri cocok dengan katanya', 'omah', 'ꦲꦺꦴꦩꦃ', 'nomina', 'ngoko', 'rumah', 'tempat tinggal', 'Omahé gedhé. — Rumahnya besar.', 'dua',
            'ngoko omah · krama griya', 'Kamus Contoh Jawa', 'omah-omah', 'pomahan',
            'Kamus bahasa Indonesia', 'Tidak ada entri yang cocok dengan pencarian ini.'], $text);
        // Entri griya (dikenali dari ejaan aksaranya) tidak ikut: "griya" hanya disebut di catatan ragam omah.
        $this->assertStringNotContainsString('ꦒꦿꦶꦪ', $text);
        Livewire::test(Kamus::class)->set('q', 'omah')->assertSee('https://example.org/jv/omah', false);

        // Arti: "rumah" tidak ada sebagai kata Jawa, tetapi ada di arti griya dan omah (arti terpendek dulu);
        // "rumahan" di arti warung bukan kata yang sama.
        $text = $this->sections('rumah');
        $this->assertOrder(['Kamus bahasa Jawa', 'Kata lain yang artinya memuat kata ini · 2', 'griya', 'krama', 'omah', 'ngoko',
            'Kamus bahasa Indonesia', '2 entri cocok dengan katanya', 'rumah', 'bangunan untuk tempat tinggal', 'rumah sakit'], $text);
        $this->assertStringNotContainsString('warung', $text);

        // Diakritik dan huruf besar diabaikan; arti berbahasa selain Indonesia diberi tanda.
        $this->assertOrder(['gêdhé', 'adjektiva', 'besar'], $this->sections('GEDHE'));
        $this->assertOrder(['toya', 'ꦠꦺꦴꦪ', 'arti berbahasa Inggris', 'water'], $this->sections('toya'));

        // Tidak ada yang cocok di kedua kamus.
        $this->assertSame(2, substr_count($this->sections('zzz'), 'Tidak ada entri yang cocok dengan pencarian ini.'));
    }

    public function test_pasted_script_is_looked_up_by_its_latin_draft_and_spelling(): void
    {
        $this->actingAs(User::factory()->create());
        $this->seedDictionaries();

        // ꦒꦿꦶꦪ -> "griya" (draf aturan) -> entri griya.
        Livewire::test(Kamus::class)->set('q', 'ꦒꦿꦶꦪ')->assertSeeInOrder(['Latin (draf aturan):', 'griya']);
        $this->assertOrder(['Kamus bahasa Jawa', '1 entri cocok dengan katanya', 'griya', 'ꦒꦿꦶꦪ', 'nomina', 'krama', 'rumah'], $this->sections('ꦒꦿꦶꦪ'));

        // Ejaan aksara dicocokkan langsung walau bacaan Latinnya tidak dipakai, dan hanya di kamus Jawa.
        $found = DictionarySearch::search('jv', '', 10, 'ꦠꦺꦴꦪ');
        $this->assertSame(['toya'], $found['matches']->pluck('word')->all());
        $this->assertSame([], DictionarySearch::search('id', '', 10, 'ꦠꦺꦴꦪ')['matches']->all());
        $this->assertSame([], DictionarySearch::search('jv', '', 10)['matches']->all());
    }

    public function test_several_words_are_glossed_one_by_one(): void
    {
        $this->actingAs(User::factory()->create());
        $this->seedDictionaries();

        $text = $this->sections('kula badhé tindak');
        $this->assertOrder(['Kamus bahasa Jawa', 'Arti per kata', 'kula', 'pronomina', 'krama', 'saya', 'badhé', 'tidak ada di kamus ini',
            'tindak', 'krama inggil', 'pergi; berjalan', 'Kamus bahasa Indonesia', 'Arti per kata', 'kula', 'tidak ada di kamus ini'], $text);
        Livewire::test(Kamus::class)->set('q', 'kula badhé tindak')->assertSee('Tingkat tutur menurut leksikon:');

        // Frasa yang utuhnya ada di kamus ditampilkan dulu, lalu kata per kata.
        $this->assertOrder(['Kamus bahasa Indonesia', 'rumah sakit', 'gedung tempat merawat orang sakit', 'Arti per kata',
            'Rumah', 'bangunan untuk tempat tinggal; bangunan pada umumnya', 'Sakit', 'berasa tidak nyaman di tubuh'], $this->sections('Rumah Sakit'));

        $this->assertSame(['kula', 'badhé', 'tindak'], DictionarySearch::words('  kula, badhé — tindak; KULA '));
        $this->assertCount(DictionarySearch::MAX_WORDS, DictionarySearch::words(implode(' ', range('a', 'z'))));
    }

    public function test_pattern_characters_and_markup_in_data_are_harmless(): void
    {
        $this->actingAs(User::factory()->create());
        $this->write(['jv' => [
            self::jv('sega', ['nasi <script>alert(1)</script>'], ['url' => 'javascript:alert(2)']),
            self::jv('50%', ['separuh'], ['url' => 'https://example.org/jv/50']),
        ]]);
        $this->artisan('aksara:dictionary', ['--path' => $this->dir])->assertSuccessful();
        // Tautan entri menjadi href: skema selain http(s) dibuang saat impor.
        $this->assertNull(DictionaryEntry::where('word', 'sega')->value('url'));
        $this->assertSame('https://example.org/jv/50', DictionaryEntry::where('word', '50%')->value('url'));

        // % dan _ bukan pola: tidak mengembalikan semua entri.
        foreach (['%', '_', '%%%', 's_ga'] as $pattern) {
            Livewire::test(Kamus::class)->set('q', $pattern)->assertDontSee('nasi');
        }
        Livewire::test(Kamus::class)->set('q', 'sega')
            ->assertSee('nasi')->assertDontSee('<script>alert(1)</script>', false)->assertSee('&lt;script&gt;alert(1)&lt;/script&gt;', false);
    }

    public function test_more_raises_the_number_of_shown_entries(): void
    {
        $this->actingAs(User::factory()->create());
        $this->write(['jv' => array_map(fn (int $i) => self::jv(sprintf('kata%02d', $i), ["arti {$i}"]), range(1, 45))]);
        $this->artisan('aksara:dictionary', ['--path' => $this->dir])->assertSuccessful();

        Livewire::test(Kamus::class)->set('q', 'kata')
            ->assertSee('45 entri cocok dengan katanya')->assertSee('20 ditampilkan')->assertSee('kata20')->assertDontSee('kata21')
            ->call('more', 'jv')->assertSee('40 ditampilkan')->assertSee('kata40')->assertDontSee('kata41')
            ->call('more', 'bukan-kamus')->assertSee('40 ditampilkan')
            ->call('more', 'jv')->assertSee('kata45')->assertDontSee('Tampilkan lebih banyak')
            // Kueri baru mulai lagi dari halaman pertama.
            ->set('q', 'kata0')->set('q', 'kata')->assertSee('20 ditampilkan');
    }
}
