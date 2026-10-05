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

    /** Kasus tepi pencarian: frasa berkata ulang, imbuhan, kata bertanda hubung atau berapostrof, ejaan aksara. */
    private function seedEdgeCases(): void
    {
        $en = fn (string $word, array $glosses, array $extra = []) => ['word' => $word, 'glosses' => $glosses, 'gloss_lang' => 'en', 'source' => 'contoh'] + $extra;
        $this->write([
            'jv' => [
                // Draf aturan membaca ꦧꦤ꧀ꦣ sebagai "bandha"; entri ber-ejaan itu sendiri ditulis "banḍa".
                self::jv('banḍa', ['ejaan lain bandha'], ['aksara' => 'ꦧꦤ꧀ꦣ']),
                self::jv('bandha', ['harta']),
                // Draf aturan membaca ꦱꦶꦤꦲꦸ sebagai "sinahu": hanya ejaan aksara yang mempertemukannya dengan "sinau".
                self::jv('sinau', ['belajar'], ['aksara' => 'ꦱꦶꦤꦲꦸ']),
                $en('toya', ['water']),
                self::jv('toya', ['air']),
                self::jv('ana', ['ada']),
            ],
            'id' => [
                $en('mau', ['to want']), $en('tak', ['not']), $en('mau tak mau', ['willy-nilly']),
                $en('-an', ['suffix forming nouns']), $en('ke-', ['prefix']), $en('ke- -an', ['circumfix forming an abstract noun']),
                $en('menantu', ['son-in-law', 'daughter-in-law']), $en("Al-Qur'an", ['the holy book of Islam']),
                $en('rumah', ['house'], ['url' => 'https://example.org/id/rumah']),
                $en('anak-anak', ['children']), $en('anak tiri', ['stepchild']),
                $en('apel', ['an apple']), $en('kupu-kupu', ['butterfly']),
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

    public function test_phrases_affixes_and_quoted_words_are_found_as_typed(): void
    {
        $this->actingAs(User::factory()->create());
        $this->seedEdgeCases();
        $words = fn (string $term, string $list = 'matches') => DictionarySearch::search('id', $term, 20)[$list]->pluck('word')->all();

        // Frasa berkata ulang dicari utuh seperti diketik; sesudahnya kata per kata, tanpa pengulangan.
        $this->assertOrder(['Kamus bahasa Indonesia', 'mau tak mau', 'willy-nilly', 'Arti per kata', 'mau', 'to want', 'tak', 'not'], $this->sections('mau tak mau'));

        // Imbuhan: tanda hubung di depan ikut dicari. "an" di arti berbahasa Inggris bukan padanannya, jadi tanpa pencarian balik.
        // (arti "apel" memuat kata "an": tanpa syarat panjang itu apel muncul di daftar "artinya memuat kata ini".)
        $suffix = DictionarySearch::search('id', '-an', 20);
        $this->assertSame(['-an', 'ke- -an', 'anak-anak'], $suffix['matches']->pluck('word')->all());
        $this->assertSame(0, $suffix['reverse_total']);
        $text = $this->sections('-an');
        $this->assertOrder(['Kamus bahasa Indonesia', '3 entri cocok dengan katanya', '-an', 'suffix forming nouns'], $text);
        $this->assertStringNotContainsString('apel', $text);
        $this->assertStringNotContainsString('artinya memuat kata ini', $text);
        $this->assertOrder(['Kamus bahasa Indonesia', 'ke- -an', 'circumfix forming an abstract noun', 'Arti per kata', 'ke-', 'prefix', '-an', 'suffix forming nouns'],
            $this->sections('ke- -an'));
        $this->assertSame(['ke-', '-an'], DictionarySearch::words('ke- -an'));
        // Kata biasa yang kebetulan didahului tanda hubung tetap diartikan dan tetap ditemukan.
        $this->assertSame(['rumah'], DictionarySearch::gloss('id', ['-rumah'])[0]['entries']->pluck('word')->all());
        $this->assertSame(['rumah'], $words('-rumah'));
        // ...tetapi hanya bila kata itu ada: kamus tanpa imbuhan "-an" tidak lalu menampilkan semua kata berawalan "an".
        $this->assertSame(0, DictionarySearch::search('jv', '-an', 20)['total']);
        // Satu kata tetap satu kata: kata ulang yang diketik berspasi dan tanda hubung lepas dicari sebagai katanya,
        // dan kata itu juga yang ditautkan ke KBBI.
        $this->assertOrder(['Kamus bahasa Indonesia', '1 entri cocok dengan katanya', 'kupu-kupu', 'butterfly'], $this->sections('kupu kupu'));
        // Entri bertanda hubungnya didahulukan (menurut urutan biasa "anak tiri" di depan "anak-anak").
        $this->assertOrder(['Kamus bahasa Indonesia', '2 entri cocok dengan katanya', 'anak-anak', 'anak tiri'], $this->sections('anak anak'));
        Livewire::test(Kamus::class)->set('q', '- rumah')
            ->assertSee('href="https://example.org/id/rumah"', false)->assertSee('href="https://kbbi.kemendikdasmen.go.id/entri/rumah"', false);

        // Tanda kutip yang membungkus kata dibuang; apostrof yang bagian dari kata tidak.
        $this->assertSame(['rumah'], $words("'rumah'"));
        $this->assertSame(['rumah'], $words('‘rumah’'));
        $this->assertSame(["Al-Qur'an"], $words('al-qur’an'));
        $this->assertSame(["Al-Qur'an"], $words("'Al-Qur'an'"));
        $this->assertSame(['omah', 'lan', 'griya'], DictionarySearch::words("'omah' lan 'griya'"));
        Livewire::test(Kamus::class)->set('q', "'rumah'")
            ->assertSee('href="https://kbbi.kemendikdasmen.go.id/entri/rumah"', false)
            // Arti berbahasa Inggris di halaman berbahasa Indonesia dinyatakan bahasanya.
            ->assertSee('lang="en"', false);

        // Pencarian balik mengenai kata bertanda hubung di arti.
        $this->assertSame(['menantu'], $words('son-in-law', 'reverse'));

        // Kata sama panjang diurutkan menurut byte (spasi sebelum tanda hubung), sama di PostgreSQL dan SQLite. SQLite
        // memang mengurutkan begitu; assertion ini baru menguji COLLATE "C" saat dijalankan lewat phpunit.pgsql.xml.
        $this->assertSame(['anak tiri', 'anak-anak'], $words('anak'));
        // Arti berbahasa Indonesia didahulukan walau diimpor belakangan.
        $this->assertSame(['id', 'en'], DictionarySearch::search('jv', 'toya', 20)['matches']->pluck('gloss_lang')->all());

        // Tabel arti per kata bertaut ke halaman asal entrinya.
        Livewire::test(Kamus::class)->set('q', 'rumah mau')
            ->assertSee('href="https://example.org/id/rumah"', false)->assertSee('Sumber entri rumah: Kamus Contoh Indonesia', false);
    }

    public function test_entry_found_by_its_script_spelling_is_listed_once(): void
    {
        $this->actingAs(User::factory()->create());
        $this->seedEdgeCases();

        // Entri banḍa cocok lewat ejaan aksaranya, entri bandha lewat katanya. Arti banḍa juga menyebut "bandha",
        // tetapi entri itu tidak diulang di daftar "artinya memuat kata ini", dan jumlahnya tidak dihitung dua kali.
        $found = DictionarySearch::search('jv', 'bandha', 20, 'ꦧꦤ꧀ꦣ');
        $this->assertSame(['banḍa', 'bandha'], $found['matches']->pluck('word')->all());
        $this->assertSame([2, 0, 0], [$found['total'], $found['reverse_total'], $found['reverse']->count()]);
        // Tanpa ejaan aksara, banḍa hanya ditemukan dari artinya.
        $plain = DictionarySearch::search('jv', 'bandha', 20);
        $this->assertSame([['bandha'], ['banḍa']], [$plain['matches']->pluck('word')->all(), $plain['reverse']->pluck('word')->all()]);

        // Di halaman: aksara yang ditempel menemukan entrinya walau bacaan Latin drafnya ("sinahu") bukan katanya,
        // juga bila ditempel bersama tanda kutip atau pada lingsa.
        foreach (['ꦱꦶꦤꦲꦸ', '‘ꦱꦶꦤꦲꦸ’', 'ꦱꦶꦤꦲꦸ꧈'] as $pasted) {
            $this->assertOrder(['Kamus bahasa Jawa', '1 entri cocok dengan katanya', 'sinau', 'ꦱꦶꦤꦲꦸ', 'belajar'], $this->sections($pasted));
        }
    }

    public function test_query_cleaning_keeps_non_ascii_text_intact_and_stays_fast(): void
    {
        $this->actingAs(User::factory()->create());
        $this->seedDictionaries();

        // Tepi teks dipangkas per karakter, bukan per byte: huruf dan tanda non-ASCII di tepi tidak rusak.
        $this->assertSame('— kula badhé tindak', DictionarySearch::unquote('— kula badhé tindak'));
        $this->assertSame('śrī', DictionarySearch::unquote('śrī'));
        $this->assertSame('rumah', DictionarySearch::unquote("\u{00A0}“rumah”\u{00A0}"));
        $this->assertSame(['śrī', 'nabī'], DictionarySearch::words('śrī nabī'));
        $this->assertSame(['kula', 'badhé', 'tindak'], DictionarySearch::words('— kula badhé tindak 😀'));
        $this->assertOrder(['Kamus bahasa Jawa', 'Arti per kata', 'kula', 'saya', 'badhé', 'tidak ada di kamus ini', 'tindak', 'pergi; berjalan'],
            $this->sections('— kula badhé tindak'));

        // Ratusan ribu tanda kutip tidak membuat pola berjalan kuadratik.
        $start = microtime(true);
        DictionarySearch::unquote(str_repeat("'", 200000));
        DictionarySearch::unquote("'".str_repeat("a'b ", 50000)."x'");
        $this->assertLessThan(2.0, microtime(true) - $start);

        // Kueri dipotong: kata sesudah batas panjang tidak ikut dicari.
        Livewire::test(Kamus::class)->set('q', 'rumah '.str_repeat('x', Kamus::MAX_QUERY))->assertSee('bangunan untuk tempat tinggal');
        Livewire::test(Kamus::class)->set('q', str_repeat('x', Kamus::MAX_QUERY).' rumah')->assertDontSee('bangunan untuk tempat tinggal');
    }

    public function test_code_points_and_marker_words_go_to_their_own_dictionaries(): void
    {
        $this->actingAs(User::factory()->create());
        $this->seedDictionaries();

        // Kode titik Unicode milik kamus aksara: kamus kata tidak mengartikan "U" dan "A9B6" satu per satu.
        $text = $this->sections('U+A9B6');
        $this->assertStringNotContainsString('Arti per kata', $text);
        $this->assertSame(2, substr_count($text, 'Tidak ada entri yang cocok dengan pencarian ini.'));
        // (kata "wulu" juga ada di placeholder kotak cari, jadi yang diperiksa ubin aksaranya.)
        Livewire::test(Kamus::class)->set('q', 'U+A9B6')->assertSee('wire:key="ak-U+A9B6"', false)->assertDontSee('KBBI Daring');
        Livewire::test(Kamus::class)->set('q', 'U+A98F')->assertSee('wire:key="ak-U+A98F"', false)->assertDontSee('wire:key="ak-U+A9B6"', false);

        // Leksikon tingkat tutur menampilkan penanda yang ditemukan di teks, sama dengan uraian: beberapa kata, kata
        // bertanda hubung, dan aksara yang ditempel tanpa spasi.
        $markers = function (string $q): array {
            $html = Livewire::test(Kamus::class)->set('q', $q)->html();
            $this->assertSame(1, preg_match('/Leksikon tingkat tutur\s*<span class="chip-count num">(\d+)<\/span>/u', $html, $count));

            return [(int) $count[1], strip_tags(substr($html, strpos($html, 'id="leksikon-tutur"')))];
        };
        [$count, $lexicon] = $markers('kula badhé tindak');
        $this->assertSame(3, $count);
        $this->assertOrder(['kula', 'badhe', 'tindak'], $lexicon);
        $this->assertStringNotContainsString('sampun', $lexicon);
        [$count, $lexicon] = $markers('sampun-sampun kula tindak');
        $this->assertSame(3, $count);
        $this->assertOrder(['kula', 'sampun', 'tindak'], $lexicon);
        // Tanpa spasi hanya penanda sepanjang lima huruf atau lebih yang dikenali (aturan tahap 4): badhe, tindak.
        [$count, $lexicon] = $markers('ꦏꦸꦭꦧꦝꦺꦠꦶꦤ꧀ꦢꦏ꧀');
        $this->assertSame(2, $count);
        $this->assertOrder(['badhe', 'tindak'], $lexicon);
        // Satu kata tetap dicari sebagai bagian kata penanda.
        $this->assertSame(1, $markers('omah')[0]);
    }

    public function test_shipped_dictionary_files_import_cleanly_and_keep_the_builder_rules(): void
    {
        // Berkas yang ikut repo harus lolos validasi pengimpor: bangun ulang data yang merusaknya ketahuan di sini,
        // bukan saat impor ke database situs. Sekaligus impor yang jauh lebih besar dari satu potongan sisipan.
        $this->artisan('aksara:dictionary')->assertSuccessful();

        $counts = DictionaryEntry::selectRaw('dictionary, count(*) as n')->groupBy('dictionary')->pluck('n', 'dictionary');
        $this->assertGreaterThan(5000, $counts['jv']);
        $this->assertGreaterThan(30000, $counts['id']);
        $this->assertSame((int) $counts['jv'], (int) DictionarySource::where('dictionary', 'jv')->sum('entries'));
        // Yang dijanjikan dokumen: tiap entri bertaut ke halaman asalnya; kamus Jawa dari dua sumber (Wiktionary bahasa
        // Inggris dan bagian bahasa Jawa Wikikamus), kamus Indonesia dari satu; tidak ada sumber lain.
        $this->assertSame(0, DictionaryEntry::whereNull('url')->count());
        $this->assertSame(['enwiktionary', 'idwiktionary'], DictionaryEntry::where('dictionary', 'jv')->distinct()->orderBy('source')->pluck('source')->all());
        $this->assertSame(['enwiktionary'], DictionaryEntry::where('dictionary', 'id')->distinct()->pluck('source')->all());
        // Ejaan aksara hanya berisi aksara Jawa (plus spasi dan tanda hubung antar-kata).
        $this->assertSame([], DictionaryEntry::whereNotNull('aksara')->pluck('aksara')
            ->reject(fn (string $aksara) => preg_match('/^[\x{A980}-\x{A9DF}\x{200C} -]+$/u', $aksara) === 1)->take(5)->values()->all());

        // Aturan pembangun data (web/tools/dictionaries.py) yang lahir dari tinjauan, diperiksa pada berkas jadinya.
        // Sesudah `dictionaries.py --refresh` isi sumber bisa berubah: cocokkan dengan halaman asalnya sebelum
        // mengubah harapan di bawah.
        $jv = fn (string $word) => DictionaryEntry::where('dictionary', 'jv')->where('source', 'enwiktionary')->where('word', $word)->orderBy('id')->get();
        $wiki = fn (string $word) => DictionaryEntry::where('dictionary', 'jv')->where('source', 'idwiktionary')->where('word', $word)->orderBy('id')->get();
        $id = fn (string $word) => DictionaryEntry::where('dictionary', 'id')->where('word', $word)->where('pos', 'nomina')->firstOrFail();
        // Satu halaman beraksara memuat dua kata (ꦲꦗꦶ = aji dan haji): romanisasi dibaca per entri.
        $this->assertSame([2, ['ꦲꦗꦶ']], [$jv('aji')->count(), $jv('aji')->pluck('aksara')->unique()->values()->all()]);
        $this->assertSame(['ꦲꦗꦶ'], $jv('haji')->pluck('aksara')->all());
        // Ragam dari kepala entri; krama-ngoko menang atas ngoko; kata "krama" sendiri tidak berlabel krama.
        $this->assertSame(['krama'], $jv('ageng')->pluck('register')->unique()->values()->all());
        $this->assertSame(['krama-ngoko'], $jv('garing')->pluck('register')->unique()->values()->all());
        $this->assertSame([null], $jv('krama')->pluck('register')->unique()->values()->all());
        $this->assertSame(['kawi'], $jv('jaladri')->pluck('register')->all());          // kepala "jaladri (kawi jaladri)"
        $this->assertSame(['krama-ngoko'], $jv('tungguk')->pluck('register')->all());   // semua arti berlabel "(ngoko, krama)"
        // Catatan hanya memuat padanan di ragam lain, dengan tanda aksara yang sudah dirapatkan (ꦢꦶꦤ꧀ꦠꦼꦤ꧀ -> dinten),
        // romanisasi bila sumbernya memberi, dan tanpa kata itu sendiri atau ejaan lamanya.
        $this->assertSame(['ngoko', 'krama dinten'], [$jv('dina')->first()->register, $jv('dina')->first()->note]);
        $this->assertSame(['ngoko', 'krama alit'], [$jv('cilik')->first()->register, $jv('cilik')->first()->note]);
        $this->assertSame('krama latu', $jv('geni')->first()->note);
        $this->assertSame(['krama-ngoko', 'ngoko alus rama kuwalon · krama alus rama kuwalon'],
            [$jv('bapak kuwalon')->first()->register, $jv('bapak kuwalon')->first()->note]);
        $this->assertSame(0, DictionaryEntry::where('note', 'like', '%꧈%')->orWhere('note', 'like', '%lugu %')->count());
        // Ejaan aksara: varian ejaan yang hanya beda diakritik mewarisinya; ejaan yang dinyatakan halaman kata itu
        // dipakai bila memang terbaca sebagai kata itu (enèm = ꦲꦼꦤꦺꦩ꧀, bukan ꦲꦼꦤꦼꦩ꧀ milik "enem"); ejaan milik kata
        // lain yang salah tempel di sumbernya ditolak (ngétan <- ꦮꦺꦠꦤ꧀ "wétan", émah <- ꦲꦺꦴꦩꦃ "omah").
        $this->assertSame('ꦝꦺꦮꦺꦏꦺ', $jv('dheweke')->first()->aksara);
        $this->assertSame('ꦲꦼꦤꦺꦩ꧀', $jv('enèm')->first()->aksara);
        $this->assertSame(['ꦥꦶꦱꦁ'], $jv('pisang')->pluck('aksara')->unique()->values()->all());
        $this->assertSame([null], $jv('ngétan')->pluck('aksara')->all());
        $this->assertSame([null], $jv('émah')->pluck('aksara')->all());
        // Varian ejaan dan judul penunjuk romanisasi tetap bisa dicari, dengan aksara yang disebut sumbernya.
        $this->assertSame([['nonstandard spelling of apa, romanization of ꦲꦥ'], 'ꦲꦥ'], [$jv('opo')->first()->glosses, $jv('opo')->first()->aksara]);
        $this->assertSame([['romanization of ꦕꦤ꧀ꦝꦶ (candhi)'], 'ꦕꦤ꧀ꦝꦶ'], [$jv('candi')->first()->glosses, $jv('candi')->first()->aksara]);
        $this->assertSame([2, 0], [$jv('jancuk')->count(), $jv('janycuk')->count()]);
        // Wikikamus: arti berbahasa Indonesia, contoh kalimat, sinonim; tanpa ragam dan ejaan aksara (tidak andal di sana).
        $banyu = $wiki('banyu')->first();
        $this->assertSame([['air'], 'id', ['Banyu degan seger rasané. — Air kelapa segar rasanya.'], 'sinonim: toya (krama)'],
            [$banyu->glosses, $banyu->gloss_lang, $banyu->examples, $banyu->note]);
        $this->assertSame(['(krama) lima'], $wiki('gangsal')->first()->glosses);
        $this->assertSame(0, DictionaryEntry::where('source', 'idwiktionary')->where(fn ($q) => $q->whereNotNull('aksara')->orWhereNotNull('register'))->count());
        $this->assertSame(0, DictionaryEntry::where('source', 'idwiktionary')->where('gloss_lang', '<>', 'id')->count());
        // Arti Indonesia didahulukan, dan kata Indonesia menemukan padanan Jawanya lewat pencarian balik.
        $this->assertSame('idwiktionary', DictionarySearch::search('jv', 'omah', 20)['matches']->first()->source);
        $this->assertContains('omah', DictionarySearch::search('jv', 'rumah', 200)['reverse']->pluck('word')->all());
        // Arti bersarang: induk pendek dirangkai dengan anaknya, induk panjang ditulis sekali lalu anaknya menyusul.
        $this->assertContains('house: abode', $id('rumah')->glosses);
        $this->assertStringStartsWith('(zoology) A biting midge', $id('agas')->glosses[0]);
        $this->assertStringStartsWith('(by extension) A sandfly', $id('agas')->glosses[1]);
        $this->assertContains('nucleus (chemistry, nuclear physics)', $id('nukleus')->glosses);     // anak yang hanya label
        $this->assertSame(0, DictionaryEntry::whereRaw('cast(glosses as text) like ?', ['%{{%'])->count());   // sisa templat sumber
        // Tanda baca penutup arti dibuang (di JSON, arti yang berakhir titik koma tampak sebagai ;").
        foreach ([';', ':'] as $mark) {
            $this->assertSame(0, DictionaryEntry::whereRaw('cast(glosses as text) like ?', ['%'.$mark.'"%'])->count(), "arti berakhir '{$mark}'");
        }
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
            ->assertSee('Tampilkan lebih banyak')
            ->call('more', 'jv')->assertSee('40 ditampilkan')->assertSee('kata40')->assertDontSee('kata41')
            ->call('more', 'bukan-kamus')->assertSee('40 ditampilkan')
            ->call('more', 'jv')->assertSee('kata45')->assertDontSee('Tampilkan lebih banyak')
            // Kueri baru mulai lagi dari halaman pertama.
            ->set('q', 'kata0')->set('q', 'kata')->assertSee('20 ditampilkan');
    }
}
