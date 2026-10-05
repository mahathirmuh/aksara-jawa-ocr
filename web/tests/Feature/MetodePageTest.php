<?php

namespace Tests\Feature;

use App\Livewire\Pages\Metode;
use App\Models\LineAnnotation;
use App\Models\MethodReport;
use App\Models\Metric;
use App\Models\Pipeline;
use App\Models\ResultImport;
use App\Models\TranslationRun;
use App\Models\User;
use App\Services\MethodImporter;
use App\Support\AksaraWriter;
use App\Support\SpeechLevel;
use App\Support\TalingRestorer;
use Illuminate\Filesystem\Filesystem;
use Illuminate\Foundation\Testing\RefreshDatabase;
use RuntimeException;
use Tests\Concerns\ReadsPage;
use Tests\TestCase;

class MetodePageTest extends TestCase
{
    use ReadsPage;
    use RefreshDatabase;

    private string $dir;

    protected function setUp(): void
    {
        parent::setUp();
        $this->dir = sys_get_temp_dir().DIRECTORY_SEPARATOR.'aksara-methods-'.uniqid();
        mkdir($this->dir);
    }

    protected function tearDown(): void
    {
        (new Filesystem)->deleteDirectory($this->dir);
        parent::tearDown();
    }

    /** Satu butir metode, bentuknya seperti keluaran scripts/export_methods.py. */
    private function entry(string $key, string $name, string $kind, string $status, string $summary, array $settings = [],
        ?string $evidence = null, ?string $source = null, array $files = []): array
    {
        return ['key' => $key, 'name' => $name, 'kind' => $kind, 'status' => $status, 'summary' => $summary,
            'settings' => $settings, 'evidence' => $evidence, 'evidence_source' => $evidence ? $source : null, 'files' => $files];
    }

    /** Kartu metode kecil dengan angka model sebenarnya (2026-10-05). */
    private function card(): array
    {
        return [
            'schema' => 1, 'generated' => '2026-10-05T23:40:00', 'source' => 'metopenv5',
            'official' => ['run' => 'fase7_track', 'pipeline' => 'crnn_fase7_track',
                'checkpoint' => 'out/checkpoints/fase7_track/last_snapshot.pt', 'step' => 1500],
            'results_generated' => '2026-10-04T15:46:51',
            'model' => ['classes' => 93, 'channels' => [32, 64, 128, 256], 'hidden' => 256, 'height' => 96, 'feature_height' => 3,
                'width_stride' => 4, 'conv_layers' => 7, 'kernel' => 3, 'lstm_layers' => 2, 'bidirectional' => true,
                'parameters' => ['total' => 4590909, 'cnn' => 1716704, 'proj' => 196864, 'rnn' => 2629632, 'head' => 47709]],
            'groups' => [
                ['key' => 'ocr', 'title' => 'Pembaca aksara', 'intro' => 'Satu model membaca seluruh baris sekaligus.', 'methods' => [
                    $this->entry('cnn', 'CNN (jaringan konvolusi)', 'model', 'official', 'Mengubah piksel citra baris menjadi ciri visual.',
                        [['Lapis konvolusi', '7 (3×3, BatchNorm, ReLU)'], ['Kanal', '32, 64, 128, 256']], files: ['src/model.py']),
                    $this->entry('ctc', 'CTC (Connectionist Temporal Classification)', 'training', 'official',
                        'Fungsi rugi yang melatih model hanya dari teks satu baris.', [['Kelas kosong', 'indeks 0']]),
                    $this->entry('visual_order', 'Tokenizer urutan visual', 'rule', 'official', 'Model dilatih pada urutan seperti yang terlihat.',
                        [['Tanda yang dipindah ke depan', 'taling (U+A9BA), dirga mure (U+A9BB)']],
                        'Bolak-balik benar pada 1.034.357 baris korpus (100%, gerbang G4).', 'data/tokenizer.json', ['src/tokenizer.py']),
                ]],
                ['key' => 'data', 'title' => 'Data latih sintetis', 'intro' => 'Model tidak pernah dilatih pada foto.', 'methods' => [
                    $this->entry('tracking', 'Jarak antar suku kata acak', 'data', 'official', 'Sebagian baris dirender dengan jarak tambahan.',
                        [['Peluang', '0,5'], ['Jarak tambahan', '0 sampai 0,3 em']],
                        'Terhadap run kontrol fase7_ctrl: G3 30,75% → 21,46% (−9,29 poin; selang kepercayaan 95% per halaman −11,92 sampai −6,83).',
                        'out/compare/crnn_fase7_track_vs_crnn_fase7_ctrl.json', ['src/render.py', 'src/dataset.py']),
                    $this->entry('rare', 'Sisipan aksara langka', 'data', 'tested', 'Menyisipkan satu aksara langka ke teks sebelum dirender.',
                        [], 'Hanya 40,6% dari keluarannya benar. Karena itu tidak dipakai di run resmi.', 'out/compare/x.json'),
                ]],
                ['key' => 'decoding', 'title' => 'Koreksi sesudah baca', 'intro' => 'Cara baca alternatif.', 'methods' => [
                    $this->entry('beam_lm', 'Beam search dengan model bahasa karakter', 'statistic', 'available',
                        'Beberapa kandidat teks dipertahankan dan dinilai.', [['Lebar beam', '16']],
                        'Pada checkpoint fase5_fonts: G3 37,25% → 33,83%. Belum diuji pada checkpoint resmi.', 'out/results/manifest.json'),
                ]],
                ['key' => 'evaluation', 'title' => 'Evaluasi dan statistik', 'intro' => 'Angka dilaporkan apa adanya.', 'methods' => [
                    $this->entry('gates', 'Gerbang dengan target tetap', 'evaluation', 'used', 'Empat ukuran dengan target yang ditetapkan lebih dulu.',
                        [['G3, cetakan nyata', 'CER < 8% pada 745 baris']]),
                    $this->entry('vlm_blind', 'Uji buta terhadap model bahasa-visual', 'evaluation', 'comparator',
                        'Sebuah model bahasa-visual umum membaca baris nyata.', [['Baris', '50']], 'CER 90,99% melawan 38,43%.', 'out/results/manifest.json'),
                ]],
            ],
            'planned' => [['key' => 'vlm_finetune', 'label' => 'VLM fine-tune', 'config' => 'vlm:GraniteDocling-258M · LoRA'],
                ['key' => 'gpt', 'label' => 'GPT korektor', 'config' => 'gpt:teks saja']],
        ];
    }

    private function write(array $card): void
    {
        file_put_contents($this->dir.'/methods.json', json_encode($card, JSON_UNESCAPED_UNICODE));
    }

    private function import(array $card): void
    {
        $this->write($card);
        $this->artisan('aksara:methods', ['--path' => $this->dir])->assertSuccessful();
    }

    /**
     * Satu butir metode seperti terbaca di halaman: nama, jenis, status dan kelas pilnya, penjelasan, bukti, sumber
     * bukti, lalu pengaturan sebagai pasangan [label, nilai] (berkas kode = baris "Kode").
     */
    private function shown(string $html, string $key): array
    {
        $xpath = $this->xpath($html);
        $rows = $xpath->query("//article[@data-method='{$key}']");
        $this->assertSame(1, $rows->length, "butir {$key}");
        $row = $rows->item(0);
        $has = fn (string $class) => "contains(concat(' ', normalize-space(@class), ' '), ' {$class} ')";
        $text = fn (string $query) => $this->squash($xpath->query($query, $row)->item(0));
        $pill = $xpath->query(".//span[{$has('status')}]", $row)->item(0);
        preg_match('/^Terukur\. (.*?)(?: Sumber: (.+))?$/u', (string) $text(".//p[{$has('method-evidence')}]"), $evidence);
        $settings = [];
        foreach ($xpath->query(".//dl[{$has('method-settings')}]/div", $row) as $pair) {
            $settings[] = [$this->squash($xpath->query('./dt', $pair)->item(0)), $this->squash($xpath->query('./dd', $pair)->item(0))];
        }

        return [
            'name' => $text(".//h3[{$has('method-name')}]"),
            'kind' => $text(".//span[{$has('badge')}]"),
            'status' => $this->squash($pill),
            'pill' => implode(' ', array_diff(preg_split('/\s+/', trim($pill->getAttribute('class'))), ['status'])),
            'summary' => $text(".//p[{$has('method-summary')}]"),
            'evidence' => $evidence[1] ?? null,
            'source' => $evidence[2] ?? null,
            'settings' => $settings,
        ];
    }

    /** Butir kartu seperti seharusnya terbaca di halaman, dengan label jenis dan status yang diharapkan. */
    private function expected(array $method, string $kind, string $status, string $pill): array
    {
        return [
            'name' => $method['name'], 'kind' => $kind, 'status' => $status, 'pill' => $pill, 'summary' => $this->squash($method['summary']),
            'evidence' => $method['evidence'] !== null ? $this->squash($method['evidence']) : null,
            'source' => $method['evidence'] !== null ? $method['evidence_source'] : null,
            'settings' => array_merge(array_map(fn ($pair) => [$pair[0], $this->squash($pair[1])], $method['settings']),
                $method['files'] ? [['Kode', implode(' ', $method['files'])]] : []),
        ];
    }

    /** Kunci butir tiap kelompok, dalam urutan tampil. */
    private function groupKeys(string $html): array
    {
        $xpath = $this->xpath($html);
        $groups = [];
        foreach ($xpath->query('//section[@data-group]') as $section) {
            $groups[$section->getAttribute('data-group')] = array_map(
                fn ($row) => $row->getAttribute('data-method'), iterator_to_array($xpath->query('.//article[@data-method]', $section)));
        }

        return $groups;
    }

    /** Peringatan "tidak sejalan" yang tampil, satu teks per kotak. */
    private function staleNotes(string $html): array
    {
        return array_map(fn ($node) => $this->squash($node), iterator_to_array($this->xpath($html)->query('//*[@data-stale]')));
    }

    public function test_guests_are_redirected_and_an_empty_page_explains_the_export(): void
    {
        $this->get('/metode')->assertRedirect('/login');

        $this->actingAs(User::factory()->create());
        $page = $this->get('/metode')->assertOk()
            ->assertSee('Kartu metode belum diimpor')->assertSee('scripts/export_methods.py')->assertSee('php artisan aksara:methods')
            ->assertSee('href="'.route('metode').'"', false)   // menu sidebar
            ->assertDontSee('Alur dari citra ke arti');
        $this->assertSame([], $this->groupKeys($page->getContent()));
    }

    public function test_command_imports_the_card_and_the_page_lists_every_method(): void
    {
        $card = $this->card();
        $this->write($card);
        $this->artisan('aksara:methods', ['--path' => $this->dir])
            ->expectsOutputToContain('Kartu metode diimpor: 8 butir di 4 kelompok, 2 direncanakan, model 4.590.909 parameter, run fase7_track.')
            ->assertSuccessful();
        $report = MethodReport::sole();
        $this->assertSame(['fase7_track', '2026-10-05T23:40:00', 1], [$report->official_run, $report->generated_at, $report->schema]);

        $this->actingAs(User::factory()->create());
        $page = $this->get('/metode')->assertOk()->assertSee('Kartu metode diekspor 2026-10-05T23:40:00 · run resmi fase7_track');
        $html = $page->getContent();
        $xpath = $this->xpath($html);

        // Empat ubin. Parameter CNN = konvolusi + proyeksi (1.716.704 + 196.864 = 1,9 juta). Butir: 8 dari kartu + 6
        // tahap web (leksikon é ikut repo, jadi butirnya selalu ada).
        $this->assertSame('Model pembaca CRNN + CTC CNN 7 lapis → BiLSTM 2 lapis → 93 kelas, dilatih dari nol', $this->tile($html, 'Model pembaca'));
        $this->assertSame('Parameter 4.590.909 CNN 1,9 juta · BiLSTM 2,6 juta · keluaran 47,7 ribu', $this->tile($html, 'Parameter'));
        $this->assertSame('Butir metode 14 di halaman ini 2 model deep learning · 1 teknik pelatihan · 2 teknik data · 2 statistik · '
            .'4 aturan atau algoritme · 3 evaluasi', $this->tile($html, 'Butir metode'));
        $this->assertSame('Model yang dipakai 3 model CRNN (dilatih dari nol) · model bahasa n-gram karakter (statistik) · '
            .'NLLB-200 (pralatih, tidak dilatih ulang)', $this->tile($html, 'Model yang dipakai'));

        // Alur: urutan langkah, rinciannya, dan langkah mana yang memakai model hasil belajar.
        $steps = [];
        foreach ($xpath->query("//ol[contains(@class, 'flow')]/li") as $step) {
            $learned = str_contains($step->getAttribute('class'), 'is-learned');
            $this->assertSame($learned, str_contains($step->textContent, '(machine learning)'), $step->getAttribute('data-step'));
            $steps[] = [$step->getAttribute('data-step'), $this->squash($xpath->query("./span[contains(@class, 'flow-detail')]", $step)->item(0)), $learned];
        }
        $this->assertSame([
            ['Citra baris', 'tinggi 96 piksel', false], ['CNN', '7 lapis konvolusi', true], ['BiLSTM', '2 lapis, dua arah', true],
            ['CTC', '93 kelas per kolom', true], ['Teks aksara', 'urutan visual ke Unicode', false], ['Latin', 'aturan tetap', false],
            ['Arti', 'NLLB-200', true], ['Tingkat tutur', 'leksikon penanda', false],
        ], $steps);

        // Kelompok dan butirnya, dalam urutan kartu; kelompok tahap web paling akhir.
        $this->assertSame([
            'ocr' => ['cnn', 'ctc', 'visual_order'], 'data' => ['tracking', 'rare'], 'decoding' => ['beam_lm'],
            'evaluation' => ['gates', 'vlm_blind'], 'web' => ['translit', 'nllb', 'mt_scores', 'speech', 'aksara_writer', 'taling'],
        ], $this->groupKeys($html));
        foreach (['ocr' => 'Pembaca aksara Satu model membaca seluruh baris sekaligus.', 'decoding' => 'Koreksi sesudah baca Cara baca alternatif.'] as $key => $head) {
            $this->assertStringStartsWith($head, $this->part($html, 'data-group', $key));
        }

        // Tiap butir kartu tampil utuh di barisnya sendiri: jenis dan status dengan labelnya, pengaturan, bukti, sumber, berkas.
        $labels = [
            'cnn' => ['model deep learning', 'dipakai model resmi', 'status-blue'],
            'ctc' => ['teknik pelatihan', 'dipakai model resmi', 'status-blue'],
            'visual_order' => ['aturan atau algoritme', 'dipakai model resmi', 'status-blue'],
            'tracking' => ['teknik data', 'dipakai model resmi', 'status-blue'],
            'rare' => ['teknik data', 'diuji, tidak dipakai', 'status-yellow'],
            'beam_lm' => ['statistik', 'tersedia, bukan jalur resmi', 'status-gray'],
            'gates' => ['evaluasi', 'dipakai', 'status-blue'],
            'vlm_blind' => ['evaluasi', 'pembanding', 'status-gray'],
        ];
        foreach ($card['groups'] as $group) {
            foreach ($group['methods'] as $method) {
                $this->assertSame($this->expected($method, ...$labels[$method['key']]), $this->shown($html, $method['key']), $method['key']);
            }
        }
        // Contoh yang ditulis lengkap, supaya pembanding di atas tidak hanya membandingkan dua hasil hitung.
        $this->assertSame([
            'name' => 'Jarak antar suku kata acak', 'kind' => 'teknik data', 'status' => 'dipakai model resmi', 'pill' => 'status-blue',
            'summary' => 'Sebagian baris dirender dengan jarak tambahan.',
            'evidence' => 'Terhadap run kontrol fase7_ctrl: G3 30,75% → 21,46% (−9,29 poin; selang kepercayaan 95% per halaman −11,92 sampai −6,83).',
            'source' => 'out/compare/crnn_fase7_track_vs_crnn_fase7_ctrl.json',
            'settings' => [['Peluang', '0,5'], ['Jarak tambahan', '0 sampai 0,3 em'], ['Kode', 'src/render.py src/dataset.py']],
        ], $this->shown($html, 'tracking'));
        $this->assertSame(['evidence' => null, 'source' => null, 'settings' => [['Kelas kosong', 'indeks 0']]],
            array_intersect_key($this->shown($html, 'ctc'), ['evidence' => 1, 'source' => 1, 'settings' => 1]));
        // Jalur berkas boleh patah sesudah garis miring, tanpa mengubah teksnya.
        $page->assertSee('out/<wbr>compare/<wbr>crnn_fase7_track_vs_crnn_fase7_ctrl.json', false)->assertSee('src/<wbr>render.py', false);

        // Tahap web tanpa data: butirnya tetap ada, dan buktinya menyebut apa yang belum ada.
        [$same, $of] = AksaraWriter::DICTIONARY_AGREEMENT;
        $translit = $this->shown($html, 'translit');
        $this->assertSame(['Alih aksara ke Latin', 'aturan atau algoritme', 'dipakai', 'status-blue', null, [['Kode', 'web/app/Support/Transliterator.php']]],
            [$translit['name'], $translit['kind'], $translit['status'], $translit['pill'], $translit['evidence'], $translit['settings']]);
        $nllb = $this->shown($html, 'nllb');
        $this->assertSame(['NLLB-200 (model terjemahan pralatih)', 'model deep learning', 'dipakai', null], [$nllb['name'], $nllb['kind'], $nllb['status'], $nllb['evidence']]);
        $this->assertSame([['Model', 'facebook/nllb-200-distilled-600M'], ['Lisensi', 'CC-BY-NC 4.0, hanya non-komersial'],
            ['Kode', 'web/tools/nllb.py web/tools/translate_batch.py']], $nllb['settings']);
        $speech = $this->shown($html, 'speech');
        $this->assertSame(['Belum dinilai: belum ada label manusia.', 'label di Penjelajah baris'], [$speech['evidence'], $speech['source']]);
        $this->assertSame(['Kata penanda', collect(SpeechLevel::lexicon())->map(fn ($words, $level) => $level.' '.nfmt(count($words)))->join(' · ')],
            $speech['settings'][0]);
        $writer = $this->shown($html, 'aksara_writer');
        $this->assertSame(['Sama dengan ejaan aksara kamus pada '.nfmt($same).' dari '.nfmt($of).' lema ('.pct($same / $of).').', 'kamus kata bahasa Jawa'],
            [$writer['evidence'], $writer['source']]);
        $taling = $this->shown($html, 'taling');
        $this->assertSame(['statistik', ['Kata di leksikon', nfmt(TalingRestorer::meta()['words'])]], [$taling['kind'], $taling['settings'][0]]);
        $this->assertSame(['chrF dan BLEU', 'evaluasi', null], [$this->shown($html, 'mt_scores')['name'], $this->shown($html, 'mt_scores')['kind'], $this->shown($html, 'mt_scores')['evidence']]);

        // Rencana: tiap pipeline di barisnya sendiri.
        $plans = [];
        foreach ($xpath->query('//tr[@data-plan]') as $row) {
            $plans[$row->getAttribute('data-plan')] = array_map(fn ($cell) => $this->squash($cell), iterator_to_array($xpath->query('./td', $row)));
        }
        $this->assertSame(['vlm_finetune' => ['VLM fine-tune', 'vlm:GraniteDocling-258M · LoRA', 'direncanakan'],
            'gpt' => ['GPT korektor', 'gpt:teks saja', 'direncanakan']], $plans);
        $this->assertSame([], $this->staleNotes($html));
    }

    public function test_web_stage_methods_show_numbers_from_web_tables(): void
    {
        $this->import($this->card());
        Metric::create(['scope' => 'translit_draft', 'pipeline' => null, 'lines' => 745, 'cer' => 0.0713]);
        Pipeline::create(['key' => 'crnn_fonts_beam', 'label' => 'Beam + LM', 'config' => 'x', 'kind' => 'beam', 'status' => 'done', 'sort' => 0]);
        TranslationRun::create(['source' => 'label', 'model' => 'facebook/nllb-200-distilled-600M', 'lines' => 745, 'chrf' => 36.2, 'bleu' => 10.53]);
        TranslationRun::create(['source' => 'crnn_fonts_beam', 'model' => 'facebook/nllb-200-distilled-600M', 'lines' => 745, 'chrf' => 19.37, 'bleu' => 0.96]);
        // Leksikon menandai "ora" dan "aku" ngoko, "boten" dan "kula" krama: dua baris berlabel, satu tebakan benar.
        LineAnnotation::create(['dataset' => 'nusaaksara', 'external_id' => 'a.png', 'transliteration' => 'aku ora lunga', 'speech_level' => 'ngoko', 'source' => 'uji']);
        LineAnnotation::create(['dataset' => 'nusaaksara', 'external_id' => 'b.png', 'transliteration' => 'aku ora lunga', 'speech_level' => 'krama', 'source' => 'uji']);

        $this->actingAs(User::factory()->create());
        $html = $this->get('/metode')->assertOk()->getContent();
        $translit = $this->shown($html, 'translit');
        $this->assertSame(['CER 7,1% terhadap alih aksara manusia pada 745 baris.', 'tabel metrik web'], [$translit['evidence'], $translit['source']]);
        $nllb = $this->shown($html, 'nllb');
        $this->assertSame(['Dari alih aksara manusia: chrF 36,2 / BLEU 10,5 pada 745 baris. Dari keluaran OCR (Beam + LM): chrF 19,4 / BLEU 1,0.',
            'tabel terjemahan web'], [$nllb['evidence'], $nllb['source']]);
        $speech = $this->shown($html, 'speech');
        $this->assertSame(['Akurasi 50,0% pada 2 baris berlabel manusia.', 'label di Penjelajah baris'], [$speech['evidence'], $speech['source']]);
        // Angka tahap web hanya ada di barisnya sendiri.
        $this->assertNull($this->shown($html, 'mt_scores')['evidence']);
        $this->assertStringNotContainsString('chrF 36,2', $this->part($html, 'data-method', 'translit'));

        // Terjemahan dari keluaran OCR dibuat dari pipeline yang bukan pipeline resmi: disebut, supaya angkanya tidak
        // dibaca sebagai milik model resmi.
        $official = Pipeline::create(['key' => 'crnn_fase7_track', 'label' => 'CRNN fase7_track', 'config' => 'x', 'kind' => 'crnn',
            'status' => 'done', 'sort' => 1, 'official' => true]);
        $this->assertStringContainsString('Dari keluaran OCR (Beam + LM, bukan pipeline resmi): chrF 19,4 / BLEU 1,0.',
            $this->shown($this->get('/metode')->getContent(), 'nllb')['evidence']);
        $official->update(['official' => false]);
        Pipeline::where('key', 'crnn_fonts_beam')->update(['official' => true]);
        $this->assertStringContainsString('Dari keluaran OCR (Beam + LM): chrF 19,4 / BLEU 1,0.',
            $this->shown($this->get('/metode')->getContent(), 'nllb')['evidence']);

        // Hanya terjemahan dari label manusia: kalimat keluaran OCR tidak ada.
        TranslationRun::where('source', 'crnn_fonts_beam')->delete();
        $this->assertSame('Dari alih aksara manusia: chrF 36,2 / BLEU 10,5 pada 745 baris.', $this->shown($this->get('/metode')->getContent(), 'nllb')['evidence']);
    }

    public function test_a_card_that_no_longer_matches_the_imported_results_is_flagged(): void
    {
        $this->import($this->card());
        $this->actingAs(User::factory()->create());
        $notes = fn () => $this->staleNotes($this->get('/metode')->assertOk()->getContent());
        $advice = ' Samakan keduanya dengan mengulang ekspor dan impor yang tertinggal: scripts/export_results.py dengan php artisan aksara:import, '
            .'atau scripts/export_methods.py dengan php artisan aksara:methods.';
        // Belum ada hasil yang diimpor: tidak ada yang bisa dibandingkan.
        $this->assertSame([], $notes());

        Pipeline::create(['key' => 'crnn_fase7_track', 'label' => 'CRNN fase7_track', 'config' => 'x', 'kind' => 'crnn', 'status' => 'done',
            'sort' => 0, 'official' => true]);
        ResultImport::create(['schema' => 1, 'generated_at' => '2026-10-04T15:46:51', 'source_path' => 'x', 'limited' => false, 'counts' => []]);
        $this->assertSame([], $notes());

        // Hasil diekspor ulang sesudah kartu dibuat: bukti di kartu dikutip dari ekspor yang lama.
        ResultImport::create(['schema' => 1, 'generated_at' => '2026-10-09T08:00:00', 'source_path' => 'x', 'limited' => false, 'counts' => []]);
        $dates = 'Kartu metode dan hasil tidak sejalan Bukti di halaman ini dibaca dari ekspor hasil 2026-10-04T15:46:51, sedangkan hasil yang '
            .'diimpor web berasal dari ekspor 2026-10-09T08:00:00.'.$advice;
        $this->assertSame([$dates], $notes());

        // Pipeline resmi web berganti: kartu ini dibuat untuk run lain.
        Pipeline::where('key', 'crnn_fase7_track')->update(['official' => false]);
        Pipeline::create(['key' => 'crnn_fase8_x', 'label' => 'CRNN fase8_x', 'config' => 'x', 'kind' => 'crnn', 'status' => 'done', 'sort' => 1, 'official' => true]);
        $this->assertSame(['Kartu metode dan hasil tidak sejalan Kartu metode ini dibuat untuk run fase7_track, sedangkan angka resmi web sekarang '
            .'milik CRNN fase8_x.'.$advice, $dates], $notes());

        // Kartu untuk run yang tidak dikenal ekspor hasil (kunci pipeline kosong) juga tidak cocok dengan pipeline resmi.
        Pipeline::where('key', 'crnn_fase8_x')->delete();
        Pipeline::where('key', 'crnn_fase7_track')->update(['official' => true]);
        ResultImport::where('generated_at', '2026-10-09T08:00:00')->delete();
        $this->assertSame([], $notes());
        $card = $this->card();
        $card['official']['pipeline'] = null;
        $this->import($card);
        $this->assertCount(1, $notes());
        $this->assertStringContainsString('sedangkan angka resmi web sekarang milik CRNN fase7_track.', $notes()[0]);
    }

    public function test_importer_rejects_cards_it_cannot_show_and_keeps_the_previous_one(): void
    {
        $this->import($this->card());
        $importer = app(MethodImporter::class);
        $rejects = function (string $expected, ?array $card = null, ?string $raw = null) use ($importer) {
            file_put_contents($this->dir.'/methods.json', $raw ?? json_encode($card, JSON_UNESCAPED_UNICODE));
            try {
                $importer->import($this->dir);
                $this->fail("Kartu metode seharusnya ditolak: {$expected}");
            } catch (RuntimeException $e) {
                $this->assertStringContainsString($expected, $e->getMessage());
            }
        };

        $rejects('bukan JSON yang sah', raw: '{"schema": 1,');
        $rejects('bukan kartu metode: isinya harus objek JSON', raw: '[1, 2]');
        $rejects('Skema kartu metode 2 tidak didukung (butuh 1)', ['schema' => 2] + $this->card());
        $card = $this->card();
        unset($card['model']['parameters']['rnn']);
        $rejects('kunci model.parameters.rnn tidak ada', $card);
        $card = $this->card();
        unset($card['groups'][1]['methods'][0]['summary']);
        $rejects('kunci groups.*.methods.*.summary tidak ada', $card);
        $card = $this->card();
        $card['groups'][0]['methods'][0]['settings'][] = ['Hanya label'];
        $rejects('pengaturan butir cnn harus pasangan [label, nilai] berupa teks', $card);
        $card = $this->card();
        $card['groups'][0]['methods'][0]['settings'][0][1] = 7;
        $rejects('pengaturan butir cnn harus pasangan [label, nilai] berupa teks', $card);
        $card = $this->card();
        $card['model']['classes'] = 'sembilan puluh tiga';
        $rejects('kunci model.classes harus angka', $card);
        $card = $this->card();
        $card['groups'] = 'bukan daftar';
        $rejects('kunci groups harus daftar', $card);
        $card = $this->card();
        $card['groups'][0]['methods'][2]['evidence'] = ['bukan', 'teks'];
        $rejects('evidence butir visual_order harus teks', $card);
        $card = $this->card();
        $card['official']['run'] = str_repeat('x', 300);
        $rejects('kunci official.run lebih panjang dari 255 karakter', $card);
        $card = $this->card();
        $card['planned'][0]['config'] = ['bukan', 'teks'];
        $rejects('kunci planned.*.config harus teks', $card);
        $card = $this->card();
        $card['groups'][0]['methods'][0]['name'] = ['bukan', 'teks'];
        $rejects('kunci groups.*.methods.*.name harus teks', $card);
        $card = $this->card();
        unset($card['results_generated']);
        $rejects('kunci results_generated tidak ada', $card);
        $card = $this->card();
        $card['official']['pipeline'] = ['bukan', 'teks'];
        $rejects('kunci official.pipeline harus teks', $card);
        // Kunci yang tidak terdaftar di bentuk kartu tetapi dibaca halaman: tertangkap uji tampil (halaman dirender
        // sekali dengan kartu calon), bukan menjadi galat saat halaman dibuka.
        $card = $this->card();
        $card['planned'][0]['key'] = ['bukan', 'teks'];
        $rejects('Kartu metode tidak bisa ditampilkan halamannya', $card);

        unlink($this->dir.'/methods.json');
        $this->artisan('aksara:methods', ['--path' => $this->dir])
            ->expectsOutputToContain('python scripts/export_methods.py')->assertFailed();

        $this->assertSame(1, MethodReport::count());
        $this->assertSame(4590909, MethodReport::current()->payload['model']['parameters']['total']);
        $this->actingAs(User::factory()->create());
        $this->assertSame('CNN (jaringan konvolusi)', $this->shown($this->get('/metode')->assertOk()->getContent(), 'cnn')['name']);
    }

    public function test_results_import_also_imports_the_method_card(): void
    {
        // Folder hasil OCR terkecil yang diterima aksara:import (satu pipeline, satu baris buatan) + kartu metode.
        file_put_contents($this->dir.'/manifest.json', json_encode([
            'schema' => 1, 'generated' => '2026-10-04T15:46:51', 'limited' => false, 'official' => 'crnn_fase7_track',
            'pipelines' => [['key' => 'crnn_fase7_track', 'label' => 'CRNN fase7_track', 'config' => 'crnn:fase7_track@1500 · greedy',
                'kind' => 'crnn', 'status' => 'done', 'sort' => 0]],
            'gates' => [], 'metrics' => [], 'ablation' => [], 'confusion' => ['pipeline' => 'crnn_fase7_track', 'items' => []],
        ], JSON_UNESCAPED_UNICODE));
        file_put_contents($this->dir.'/lines.jsonl', json_encode(['dataset' => 'contoh', 'id' => 'a.png', 'image' => 'contoh/a.png', 'width' => 100,
            'height' => 20, 'reference' => 'ꦏ', 'condition' => 'contoh', 'source_id' => 'hal_0', 'tags' => []], JSON_UNESCAPED_UNICODE)."\n");
        file_put_contents($this->dir.'/predictions.jsonl', json_encode(['line' => 'a.png', 'pipeline' => 'crnn_fase7_track', 'text' => 'ꦏ', 'cer' => 0.0,
            'segments' => [['t' => 'ꦏ', 's' => 'eq']]], JSON_UNESCAPED_UNICODE)."\n");
        $this->write($this->card());

        $this->artisan('aksara:import', ['--path' => $this->dir])
            ->expectsOutputToContain('Kartu metode diimpor: 8 butir di 4 kelompok')->assertSuccessful();
        $this->assertSame('fase7_track', MethodReport::sole()->official_run);
        // Kartu dan hasil yang diimpor berasal dari ekspor hasil yang sama dan run yang sama: tidak ada peringatan.
        $this->actingAs(User::factory()->create());
        $html = $this->get('/metode')->assertOk()->getContent();
        $this->assertSame([], $this->staleNotes($html));
        $this->assertSame(['cnn', 'ctc', 'visual_order'], $this->groupKeys($html)['ocr']);

        // Kartu yang ditolak tidak menggagalkan impor hasil OCR, dan kartu lama dipertahankan.
        $this->write(['schema' => 5] + $this->card());
        $this->artisan('aksara:import', ['--path' => $this->dir])
            ->expectsOutputToContain('PERINGATAN: hasil OCR sudah diimpor, tetapi kartu halaman Metode ditolak')->assertSuccessful();
        $this->assertSame(1, MethodReport::count());
        $this->artisan('aksara:methods', ['--path' => $this->dir])->expectsOutputToContain('Skema kartu metode 5 tidak didukung')->assertFailed();
    }

    public function test_unknown_kind_and_status_from_a_newer_exporter_still_render(): void
    {
        $card = $this->card();
        $card['groups'][0]['methods'][0]['kind'] = 'jenis_baru';
        $card['groups'][0]['methods'][0]['status'] = 'status_baru';
        $card['planned'] = [];
        $this->import($card);

        $this->actingAs(User::factory()->create());
        $page = $this->get('/metode')->assertOk()->assertDontSee('Direncanakan, belum dikerjakan');
        $shown = $this->shown($page->getContent(), 'cnn');
        $this->assertSame(['jenis_baru', 'status_baru', 'status-gray'], [$shown['kind'], $shown['status'], $shown['pill']]);
        $this->assertSame(['official', 'used', 'available', 'tested', 'comparator'], array_keys(Metode::STATUSES));
        // Jenis yang tidak dikenal tidak ikut dihitung di ubin, tetapi butirnya tetap terhitung.
        $this->assertSame('Butir metode 14 di halaman ini 1 model deep learning · 1 teknik pelatihan · 2 teknik data · 2 statistik · '
            .'4 aturan atau algoritme · 3 evaluasi', $this->tile($page->getContent(), 'Butir metode'));
    }

    public function test_tiles_and_text_follow_the_card(): void
    {
        // Kartu tanpa beam search: ubin model tidak menyebut model bahasa n-gram.
        $card = $this->card();
        $card['groups'] = array_values(array_filter($card['groups'], fn ($group) => $group['key'] !== 'decoding'));
        // Teks dari kartu selalu di-escape, juga jalur berkas dan sumber bukti yang diberi titik patah.
        $card['groups'][0]['methods'][0]['files'] = ['<b>src</b>/model.py'];
        $card['groups'][0]['methods'][2]['evidence_source'] = '<i>data</i>/tokenizer.json';
        $card['groups'][0]['methods'][2]['summary'] = 'Urutan <script>alert(1)</script> visual.';
        $card['model']['conv_layers'] = 5;
        $card['model']['parameters'] = ['total' => 950, 'cnn' => 300, 'proj' => 200, 'rnn' => 400, 'head' => 50];
        $this->import($card);

        $this->actingAs(User::factory()->create());
        $html = $this->get('/metode')->assertOk()->getContent();
        $this->assertSame('Model yang dipakai 2 model CRNN (dilatih dari nol) · NLLB-200 (pralatih, tidak dilatih ulang)', $this->tile($html, 'Model yang dipakai'));
        $this->assertSame('Model pembaca CRNN + CTC CNN 5 lapis → BiLSTM 2 lapis → 93 kelas, dilatih dari nol', $this->tile($html, 'Model pembaca'));
        // Bilangan kecil ditulis apa adanya: CNN 300 + proyeksi 200 = 500.
        $this->assertSame('Parameter 950 CNN 500 · BiLSTM 400 · keluaran 50', $this->tile($html, 'Parameter'));
        $this->assertSame(['ocr', 'data', 'evaluation', 'web'], array_keys($this->groupKeys($html)));

        $xpath = $this->xpath($html);
        $this->assertSame(0, $xpath->query('//article[@data-method]//b | //article[@data-method]//i | //article[@data-method]//script')->length);
        $this->assertSame([['Lapis konvolusi', '7 (3×3, BatchNorm, ReLU)'], ['Kanal', '32, 64, 128, 256'], ['Kode', '<b>src</b>/model.py']],
            $this->shown($html, 'cnn')['settings']);
        $order = $this->shown($html, 'visual_order');
        $this->assertSame(['Urutan <script>alert(1)</script> visual.', '<i>data</i>/tokenizer.json'], [$order['summary'], $order['source']]);
    }

    public function test_the_card_exported_on_this_machine_imports_and_renders(): void
    {
        $dir = config('aksara.ocr_repo').DIRECTORY_SEPARATOR.config('aksara.results_dir');
        if (! is_file(MethodImporter::path($dir))) {
            $this->markTestSkipped('out/results/methods.json belum diekspor di mesin ini.');
        }
        $this->artisan('aksara:methods', ['--path' => $dir])->assertSuccessful();
        $card = MethodReport::sole()->payload;

        $this->actingAs(User::factory()->create());
        $html = $this->get('/metode')->assertOk()->assertSee(nfmt($card['model']['parameters']['total']))->assertSee($card['official']['run'])->getContent();
        // Halaman menampilkan kartu apa adanya: tiap butir di kelompoknya, dengan nama, penjelasan, pengaturan, bukti, dan sumbernya.
        $groups = $this->groupKeys($html);
        foreach ($card['groups'] as $group) {
            $this->assertSame(array_column($group['methods'], 'key'), $groups[$group['key']], $group['key']);
            foreach ($group['methods'] as $method) {
                [$status, $pill] = Metode::STATUSES[$method['status']];
                $this->assertSame($this->expected($method, Metode::KINDS[$method['kind']], $status, $pill), $this->shown($html, $method['key']), $method['key']);
            }
        }
        $this->assertSame(['ocr', 'data', 'training', 'decoding', 'evaluation', 'web'], array_keys($groups));
        $this->assertSame(count($card['planned']), $this->xpath($html)->query('//tr[@data-plan]')->length);
    }
}
