<?php

namespace Tests\Feature;

use App\Livewire\Pages\Metode;
use App\Models\DatasetReport;
use App\Models\LineAnnotation;
use App\Models\MethodReport;
use App\Models\Metric;
use App\Models\Pipeline;
use App\Models\ResultImport;
use App\Models\TranslationRun;
use App\Models\User;
use App\Services\CardStorageException;
use App\Services\MethodImporter;
use App\Support\AksaraWriter;
use App\Support\SpeechLevel;
use App\Support\TalingRestorer;
use Illuminate\Filesystem\Filesystem;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Support\Facades\Event;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\View;
use LogicException;
use RuntimeException;
use Tests\Concerns\ReadsPage;
use Tests\Concerns\RejectsCards;
use Tests\TestCase;

class MetodePageTest extends TestCase
{
    use ReadsPage;
    use RefreshDatabase;
    use RejectsCards;

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
        ?string $evidence = null, ?string $source = null, array $files = [], ?string $note = null): array
    {
        return ['key' => $key, 'name' => $name, 'kind' => $kind, 'status' => $status, 'summary' => $summary,
            'settings' => $settings, 'evidence' => $evidence, 'evidence_source' => $evidence ? $source : null, 'note' => $note, 'files' => $files];
    }

    /** Kartu metode kecil dengan angka model sebenarnya (2026-10-05). */
    private function card(): array
    {
        return [
            'schema' => 1, 'generated' => '2026-10-05T23:40:00', 'source' => 'metopenv5',
            'official' => ['run' => 'fase7_track', 'pipeline' => 'crnn_fase7_track',
                'checkpoint' => 'out/checkpoints/fase7_track/last_snapshot.pt', 'step' => 1500],
            'results_generated' => '2026-10-04T15:46:51', 'chain_complete' => true,
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
                        'Gerbang G4, 100%: bolak-balik diuji pada 1.034.357 / 1.034.357 baris korpus (Fase 1).', 'tests/test_tokenizer.py', ['src/tokenizer.py']),
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
                    // Butir dengan bukti DAN catatan batasan, dan butir dengan catatan saja.
                    $this->entry('gates', 'Gerbang dengan target tetap', 'evaluation', 'used', 'Empat ukuran dengan target yang ditetapkan lebih dulu.',
                        [['G3, cetakan nyata', 'CER < 8% pada 745 baris']], 'G1 0,27% · G2 1,20% · G3 21,46% (belum tercapai) · G4 100%.',
                        'out/results/manifest.json', note: 'Font uji javatext sekeluarga dengan font latih CarakanJawa, jadi G1 dan G2 mengukur '
                            .'generalisasi di dalam keluarga huruf itu.'),
                    $this->entry('ablation', 'Ablasi kumulatif', 'evaluation', 'used', 'Operasi augmentasi ditambahkan satu per satu.',
                        [['Run', '12']], note: 'Tiap run dilatih satu kali.'),
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
     * bukti, catatan, lalu pengaturan sebagai pasangan [label, nilai] (berkas kode = baris "Kode").
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
        preg_match('/^Catatan\. (.*)$/u', (string) $text(".//p[{$has('method-note')}]"), $note);
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
            'note' => $note[1] ?? null,
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
            'note' => ($method['note'] ?? null) !== null ? $this->squash($method['note']) : null,
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

    /**
     * Langkah alur seperti tampil: [nama, rincian, bertanda ML]. Tanda yang terlihat ("ML"), kelas warnanya, dan teks
     * untuk pembaca layar harus selalu sejalan, dan teks pembaca layar itu harus tersembunyi dari tampilan.
     */
    private function flowSteps(string $html, string $query): array
    {
        $xpath = $this->xpath($html);
        $steps = [];
        foreach ($xpath->query($query) as $step) {
            $learned = str_contains($step->getAttribute('class'), 'is-learned');
            $tag = $xpath->query(".//span[contains(@class, 'flow-tag')]", $step);
            $hidden = $xpath->query(".//span[normalize-space()='(machine learning)']", $step);
            $this->assertSame($learned ? 1 : 0, $tag->length, $step->getAttribute('data-step'));
            $this->assertSame($learned ? 1 : 0, $hidden->length, $step->getAttribute('data-step'));
            if ($learned) {
                $this->assertSame('ML', $this->squash($tag->item(0)));
                $this->assertSame('true', $tag->item(0)->getAttribute('aria-hidden'));
                $this->assertSame('sr-only', $hidden->item(0)->getAttribute('class'));
            }
            $steps[] = [$step->getAttribute('data-step'), $this->squash($xpath->query("./span[contains(@class, 'flow-detail')]", $step)->item(0)), $learned];
        }

        return $steps;
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
            ->expectsOutputToContain('Kartu metode diimpor: 9 butir di 4 kelompok, 2 direncanakan, model 4.590.909 parameter, run fase7_track.')
            ->assertSuccessful();
        $report = MethodReport::sole();
        $this->assertSame(['fase7_track', '2026-10-05T23:40:00', 1], [$report->official_run, $report->generated_at, $report->schema]);

        $this->actingAs(User::factory()->create());
        $page = $this->get('/metode')->assertOk()->assertSee('Kartu metode diekspor 2026-10-05T23:40:00 · run resmi fase7_track');
        $html = $page->getContent();
        $xpath = $this->xpath($html);

        // Empat ubin. Parameter CNN = konvolusi + proyeksi (1.716.704 + 196.864 = 1,9 juta). Butir dihitung per kelompok:
        // 9 dari kartu + 6 tahap web (leksikon é ikut repo, jadi butirnya selalu ada). Model hasil belajar yang DIPAKAI
        // dua: pembaca aksara dan model terjemahan; model bahasa n-gram hanya tersedia, jadi disebut terpisah.
        $this->assertSame('Model pembaca CRNN + CTC CNN 7 lapis → BiLSTM 2 lapis → 93 kelas, dilatih dari nol', $this->tile($html, 'Model pembaca'));
        $this->assertSame('Parameter 4.590.909 CNN 1,9 juta · BiLSTM 2,6 juta · keluaran 47,7 ribu', $this->tile($html, 'Parameter'));
        $this->assertSame('Butir metode 15 di halaman ini Pembaca aksara 3 · Data latih sintetis 2 · Koreksi sesudah baca 1 · '
            .'Evaluasi dan statistik 3 · Tahap lanjutan alur 6', $this->tile($html, 'Butir metode'));
        $this->assertSame('Model hasil belajar 2 dipakai CRNN (dilatih dari nol) · NLLB-200 (pralatih, tidak dilatih ulang) · '
            .'tersedia, bukan jalur resmi: model bahasa n-gram karakter (statistik)', $this->tile($html, 'Model hasil belajar'));

        // Alur: enam langkah berurutan, lalu dua cabang yang sama-sama berangkat dari teks Latin.
        $this->assertSame([
            ['Citra baris', 'tinggi 96 piksel', false], ['CNN', '7 lapis konvolusi', true], ['BiLSTM', '2 lapis, dua arah', true],
            ['CTC', '93 kelas per kolom', true], ['Teks aksara', 'urutan visual ke Unicode', false], ['Latin', 'aturan tetap', false],
        ], $this->flowSteps($html, "//ol[contains(@class, 'flow')]/li[@data-step]"));
        $this->assertSame([['Arti', 'NLLB-200', true], ['Tingkat tutur', 'leksikon penanda', false]],
            $this->flowSteps($html, '//li[@data-branches]//li[@data-step]'));
        $this->assertSame('Urutan langkah: Citra baris, CNN, BiLSTM, CTC, Teks aksara, Latin, lalu dari teks Latin: Arti dan Tingkat tutur',
            $xpath->evaluate("string(//ol[contains(@class, 'flow')]/@aria-label)"));
        $page->assertSee('Kotak bertanda ML memakai model hasil belajar (machine learning). Kotak lain memakai aturan tetap atau leksikon.')
            ->assertSee('Arti dan tingkat tutur sama-sama dihitung dari teks Latin.');

        // Kelompok dan butirnya, dalam urutan kartu; kelompok tahap web paling akhir; tiap kelompok ber-id sendiri.
        $this->assertSame([
            'ocr' => ['cnn', 'ctc', 'visual_order'], 'data' => ['tracking', 'rare'], 'decoding' => ['beam_lm'],
            'evaluation' => ['gates', 'ablation', 'vlm_blind'], 'web' => ['translit', 'nllb', 'mt_scores', 'speech', 'aksara_writer', 'taling'],
        ], $this->groupKeys($html));
        $this->assertSame(['metode-ocr', 'metode-data', 'metode-decoding', 'metode-evaluation', 'metode-web'],
            array_map(fn ($node) => $node->getAttribute('id'), iterator_to_array($xpath->query('//section[@data-group]'))));
        foreach (['ocr' => 'Pembaca aksara Satu model membaca seluruh baris sekaligus.', 'decoding' => 'Koreksi sesudah baca Cara baca alternatif.'] as $key => $head) {
            $this->assertStringStartsWith($head, $this->part($html, 'data-group', $key));
        }

        // Tiap butir kartu tampil utuh di barisnya sendiri: jenis dan status dengan labelnya, pengaturan, bukti, sumber,
        // catatan, berkas.
        $labels = [
            'cnn' => ['model deep learning', 'dipakai model resmi', 'status-blue'],
            'ctc' => ['teknik pelatihan', 'dipakai model resmi', 'status-blue'],
            'visual_order' => ['aturan atau algoritme', 'dipakai model resmi', 'status-blue'],
            'tracking' => ['teknik data', 'dipakai model resmi', 'status-blue'],
            'rare' => ['teknik data', 'diuji, tidak dipakai', 'status-yellow'],
            'beam_lm' => ['statistik', 'tersedia, bukan jalur resmi', 'status-gray'],
            'gates' => ['evaluasi', 'dipakai', 'status-blue'],
            'ablation' => ['evaluasi', 'dipakai', 'status-blue'],
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
            'source' => 'out/compare/crnn_fase7_track_vs_crnn_fase7_ctrl.json', 'note' => null,
            'settings' => [['Peluang', '0,5'], ['Jarak tambahan', '0 sampai 0,3 em'], ['Kode', 'src/render.py src/dataset.py']],
        ], $this->shown($html, 'tracking'));
        $this->assertSame(['evidence' => null, 'source' => null, 'note' => null, 'settings' => [['Kelas kosong', 'indeks 0']]],
            array_intersect_key($this->shown($html, 'ctc'), ['evidence' => 1, 'source' => 1, 'note' => 1, 'settings' => 1]));
        // Catatan batasan tampil di kotaknya sendiri, terpisah dari bukti terukur: butir dengan keduanya, dan butir
        // dengan catatan saja (tanpa kotak "Terukur").
        $gates = $this->shown($html, 'gates');
        $this->assertSame(['G1 0,27% · G2 1,20% · G3 21,46% (belum tercapai) · G4 100%.', 'out/results/manifest.json',
            'Font uji javatext sekeluarga dengan font latih CarakanJawa, jadi G1 dan G2 mengukur generalisasi di dalam keluarga huruf itu.'],
            [$gates['evidence'], $gates['source'], $gates['note']]);
        $this->assertSame([null, 'Tiap run dilatih satu kali.'], [$this->shown($html, 'ablation')['evidence'], $this->shown($html, 'ablation')['note']]);
        $this->assertSame(0, $xpath->query("//article[@data-method='ablation']//p[contains(@class, 'method-evidence')]")->length);
        // Jalur berkas boleh patah sesudah garis miring, tanpa mengubah teksnya.
        $page->assertSee('out/<wbr>compare/<wbr>crnn_fase7_track_vs_crnn_fase7_ctrl.json', false)->assertSee('src/<wbr>render.py', false);

        // Tahap web tanpa data: butirnya tetap ada. Yang belum diukur tidak diberi kotak "Terukur": tingkat tutur tanpa
        // label manusia mendapat catatan, bukan bukti.
        [$same, $of] = AksaraWriter::DICTIONARY_AGREEMENT;
        $translit = $this->shown($html, 'translit');
        $this->assertSame(['Alih aksara ke Latin', 'aturan atau algoritme', 'dipakai', 'status-blue', null, null, [['Kode', 'web/app/Support/Transliterator.php']]],
            [$translit['name'], $translit['kind'], $translit['status'], $translit['pill'], $translit['evidence'], $translit['note'], $translit['settings']]);
        $nllb = $this->shown($html, 'nllb');
        $this->assertSame(['NLLB-200 (model terjemahan pralatih)', 'model deep learning', 'dipakai', null], [$nllb['name'], $nllb['kind'], $nllb['status'], $nllb['evidence']]);
        $this->assertSame([['Model', 'facebook/nllb-200-distilled-600M'], ['Lisensi', 'CC-BY-NC 4.0, hanya non-komersial'],
            ['Kode', 'web/tools/nllb.py web/tools/translate_batch.py']], $nllb['settings']);
        $speech = $this->shown($html, 'speech');
        $this->assertSame([null, null, 'Belum dinilai: belum ada label manusia.'], [$speech['evidence'], $speech['source'], $speech['note']]);
        $this->assertStringEndsWith('Kata buktinya ditampilkan di Penjelajah baris dan Kamus.', $speech['summary']);
        $this->assertStringContainsString('masing-masing paling sedikit dua kata dan seperempat dari semua penanda', $speech['summary']);
        $this->assertSame(['Kata penanda', collect(SpeechLevel::lexicon())->map(fn ($words, $level) => $level.' '.nfmt(count($words)))->join(' · ')],
            $speech['settings'][0]);
        $writer = $this->shown($html, 'aksara_writer');
        $this->assertSame(['Sama dengan ejaan aksara kamus pada '.nfmt($same).' dari '.nfmt($of).' lema berejaan baku ('.pct($same / $of).').', 'kamus kata bahasa Jawa'],
            [$writer['evidence'], $writer['source']]);
        // Pemulih tanda é: angka dari berkas leksikon yang ikut repo (jv-taling.json), tiap angka dari kuncinya sendiri.
        $meta = TalingRestorer::meta();
        $taling = $this->shown($html, 'taling');
        $this->assertSame('statistik', $taling['kind']);
        $this->assertSame([
            ['Kata di leksikon', nfmt($meta['words'])],
            ['Asal kata', nfmt($meta['from_wikipedia']).' dari Wikipedia bahasa Jawa, '.nfmt($meta['from_dictionary']).' dari lema kamus'],
            ['Syarat masuk', $meta['rule']],
            ['Kode', 'web/app/Support/TalingRestorer.php web/tools/taling_lexicon.py'],
        ], $taling['settings']);
        $this->assertSame(['Kata ber-e yang ejaannya benar naik dari '.pct($meta['held_out']['accuracy_without'], 0).' ke '
            .pct($meta['held_out']['accuracy_with'], 0).' pada '.nfmt($meta['held_out']['articles']).' artikel yang ditahan.', 'leksikon jv-taling.json'],
            [$taling['evidence'], $taling['source']]);
        $this->assertTrue($meta['held_out']['accuracy_without'] < $meta['held_out']['accuracy_with']
            && $meta['held_out']['articles'] !== $meta['held_out']['words'] && $meta['from_wikipedia'] !== $meta['from_dictionary']);
        $scores = $this->shown($html, 'mt_scores');
        $this->assertSame(['chrF dan BLEU', 'evaluasi', null, null], [$scores['name'], $scores['kind'], $scores['evidence'], $scores['note']]);

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
        // Dua run untuk tiap sumber: yang TERBARU yang ditampilkan (run lama memakai model dan angka lain).
        TranslationRun::create(['source' => 'label', 'model' => 'model-lama', 'lines' => 10, 'chrf' => 1.0, 'bleu' => 1.0]);
        TranslationRun::create(['source' => 'crnn_fonts_beam', 'model' => 'model-lama', 'lines' => 10, 'chrf' => 2.0, 'bleu' => 2.0]);
        TranslationRun::create(['source' => 'label', 'model' => 'facebook/nllb-200-distilled-1.3B', 'lines' => 745, 'chrf' => 36.2, 'bleu' => 10.53]);
        TranslationRun::create(['source' => 'crnn_fonts_beam', 'model' => 'facebook/nllb-200-distilled-1.3B', 'lines' => 740, 'chrf' => 19.37, 'bleu' => 0.96]);
        // Leksikon menandai "ora" dan "aku" ngoko, "boten" dan "kula" krama: dua baris berlabel, satu tebakan benar.
        // Label tanpa alih aksara tidak bisa dinilai dan tidak ikut dihitung.
        LineAnnotation::create(['dataset' => 'nusaaksara', 'external_id' => 'a.png', 'transliteration' => 'aku ora lunga', 'speech_level' => 'ngoko', 'source' => 'uji']);
        LineAnnotation::create(['dataset' => 'nusaaksara', 'external_id' => 'b.png', 'transliteration' => 'aku ora lunga', 'speech_level' => 'krama', 'source' => 'uji']);
        LineAnnotation::create(['dataset' => 'nusaaksara', 'external_id' => 'c.png', 'transliteration' => null, 'speech_level' => 'ngoko', 'source' => 'uji']);

        $this->actingAs(User::factory()->create());
        $html = $this->get('/metode')->assertOk()->getContent();
        $translit = $this->shown($html, 'translit');
        // Angka ini hanya membandingkan huruf, dan masukannya label aksara: keduanya disebut.
        $this->assertSame(['CER 7,1% terhadap alih aksara manusia pada 745 baris, huruf saja (spasi, tanda baca, dan huruf besar tidak dibandingkan). '
            .'Masukannya label aksara, bukan keluaran OCR.', 'tabel metrik web'], [$translit['evidence'], $translit['source']]);
        $nllb = $this->shown($html, 'nllb');
        $human = 'Dari alih aksara manusia: chrF 36,2 / BLEU 10,5 pada 745 baris.';
        $this->assertSame([$human.' Dari keluaran OCR (Beam + LM): chrF 19,4 / BLEU 1,0 pada 740 baris.', 'tabel terjemahan web'],
            [$nllb['evidence'], $nllb['source']]);
        $this->assertSame(['Model', 'facebook/nllb-200-distilled-1.3B'], $nllb['settings'][0]);
        $speech = $this->shown($html, 'speech');
        $this->assertSame(['Akurasi 50,0% pada 2 baris berlabel manusia.', 'label di Penjelajah baris', null], [$speech['evidence'], $speech['source'], $speech['note']]);
        // Angka tahap web hanya ada di barisnya sendiri.
        $this->assertNull($this->shown($html, 'mt_scores')['evidence']);
        $this->assertStringNotContainsString('chrF 36,2', $this->part($html, 'data-method', 'translit'));

        // Terjemahan dari keluaran OCR dibuat dari pipeline yang bukan pipeline resmi: disebut, supaya angkanya tidak
        // dibaca sebagai milik model resmi.
        $official = Pipeline::create(['key' => 'crnn_fase7_track', 'label' => 'CRNN fase7_track', 'config' => 'x', 'kind' => 'crnn',
            'status' => 'done', 'sort' => 1, 'official' => true]);
        $evidence = fn () => $this->shown($this->get('/metode')->getContent(), 'nllb')['evidence'];
        $this->assertSame($human.' Dari keluaran OCR (Beam + LM, bukan pipeline resmi): chrF 19,4 / BLEU 1,0 pada 740 baris.', $evidence());
        // Begitu ada terjemahan atas keluaran pipeline resmi, itulah yang dikutip (aturan yang sama dengan Ringkasan),
        // walau run pipeline lain lebih baru.
        TranslationRun::create(['source' => 'crnn_fase7_track', 'model' => 'facebook/nllb-200-distilled-1.3B', 'lines' => 745, 'chrf' => 25.04, 'bleu' => 3.0]);
        // Urutan pilihannya: keluaran pipeline resmi; kalau belum ada, keluaran pipeline resmi yang dibaca beam + LM
        // (walau run pipeline lain lebih baru); kalau belum ada juga, run terbaru yang bukan dari label manusia.
        $this->assertSame('crnn_fonts_beam', TranslationRun::forOcr('crnn_fonts')->source);
        $this->assertSame('crnn_fase7_track', TranslationRun::forOcr('crnn_lain')->source);
        TranslationRun::create(['source' => 'crnn_fonts_beam', 'model' => 'facebook/nllb-200-distilled-1.3B', 'lines' => 740, 'chrf' => 19.37, 'bleu' => 0.96]);
        $this->assertSame($human.' Dari keluaran OCR (CRNN fase7_track): chrF 25,0 / BLEU 3,0 pada 745 baris.', $evidence());
        $this->assertSame('crnn_fase7_track', TranslationRun::forOcr('crnn_fase7_track')->source);
        $this->assertSame('crnn_fonts_beam', TranslationRun::forOcr('crnn_lain')->source);
        $this->assertNull(TranslationRun::forOcr(null, collect()));
        $this->assertNull(TranslationRun::forOcr('crnn_fase7_track', TranslationRun::latestBySource()->only('label')));
        // Bacaan greedy pipeline resmi didahulukan dari bacaan beam + LM-nya, walau yang beam lebih baru.
        $beam = TranslationRun::create(['source' => 'crnn_fase7_track_beam', 'model' => 'm', 'lines' => 1, 'chrf' => 1.0, 'bleu' => 1.0]);
        $this->assertSame('crnn_fase7_track', TranslationRun::forOcr('crnn_fase7_track')->source);
        $beam->delete();
        TranslationRun::where('source', 'crnn_fase7_track')->delete();
        $official->update(['official' => false]);
        Pipeline::where('key', 'crnn_fonts_beam')->update(['official' => true]);
        $this->assertSame($human.' Dari keluaran OCR (Beam + LM): chrF 19,4 / BLEU 1,0 pada 740 baris.', $evidence());

        // Hanya terjemahan dari label manusia: kalimat keluaran OCR tidak ada. Hanya dari keluaran OCR: itu saja yang ada.
        TranslationRun::where('source', 'crnn_fonts_beam')->delete();
        $this->assertSame($human, $evidence());
        TranslationRun::query()->delete();
        TranslationRun::create(['source' => 'crnn_fonts_beam', 'model' => 'model-ocr', 'lines' => 5, 'chrf' => 19.37, 'bleu' => 0.96]);
        $this->assertSame('Dari keluaran OCR (Beam + LM): chrF 19,4 / BLEU 1,0 pada 5 baris.', $evidence());
        $this->assertSame(['Model', 'model-ocr'], $this->shown($this->get('/metode')->getContent(), 'nllb')['settings'][0]);
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
        $rejects = fn (string $expected, ?array $card = null, ?string $raw = null) => $this->assertCardRejected($importer, MethodReport::class, $this->dir, $expected, $card, $raw);
        $changed = function (callable $change): array {
            $card = $this->card();
            $change($card);

            return $card;
        };

        $rejects('bukan JSON yang sah', raw: '{"schema": 1,');
        $rejects('bukan kartu metode: isinya harus objek JSON', raw: '[1, 2]');
        $rejects('Skema kartu metode 2 tidak didukung (butuh 1)', ['schema' => 2] + $this->card());
        $rejects('Skema kartu metode "1" tidak didukung (butuh 1)', ['schema' => '1'] + $this->card());
        $rejects('kunci model.parameters.rnn tidak ada', $changed(function (&$c) {
            unset($c['model']['parameters']['rnn']);
        }));
        $rejects('kunci groups.*.methods.*.summary tidak ada', $changed(function (&$c) {
            unset($c['groups'][1]['methods'][0]['summary']);
        }));
        // Pengaturan = pasangan [label, nilai], dua-duanya teks.
        $rejects('pengaturan butir cnn harus pasangan [label, nilai] berupa teks', $changed(function (&$c) {
            $c['groups'][0]['methods'][0]['settings'][] = ['Hanya label'];
        }));
        $rejects('pengaturan butir cnn harus pasangan [label, nilai] berupa teks', $changed(function (&$c) {
            $c['groups'][0]['methods'][0]['settings'][] = ['Label', 'nilai', 'lebih'];
        }));
        $rejects('pengaturan butir cnn harus pasangan [label, nilai] berupa teks', $changed(function (&$c) {
            $c['groups'][0]['methods'][0]['settings'][0][1] = 7;
        }));
        $rejects('pengaturan butir cnn harus pasangan [label, nilai] berupa teks', $changed(function (&$c) {
            $c['groups'][0]['methods'][0]['settings'][0][0] = 7;
        }));
        // Tipe: angka dalam tanda kutip bukan angka, bilangan negatif bukan jumlah, objek bukan daftar, angka bukan teks.
        $rejects('kunci model.classes harus bilangan yang tidak negatif', $changed(function (&$c) {
            $c['model']['classes'] = 'sembilan puluh tiga';
        }));
        $rejects('kunci model.classes harus bilangan yang tidak negatif', $changed(function (&$c) {
            $c['model']['classes'] = '93';
        }));
        $rejects('kunci model.conv_layers harus bilangan yang tidak negatif', $changed(function (&$c) {
            $c['model']['conv_layers'] = -1;
        }));
        $rejects('kunci model.bidirectional harus benar/salah', $changed(function (&$c) {
            $c['model']['bidirectional'] = 'ya';
        }));
        $rejects('kunci groups harus daftar', $changed(function (&$c) {
            $c['groups'] = 'bukan daftar';
        }));
        $rejects('kunci groups harus daftar', $changed(function (&$c) {
            $c['groups'] = ['ocr' => $c['groups'][0]];
        }));
        $rejects('kunci groups.*.methods.*.name harus teks yang tidak kosong', $changed(function (&$c) {
            $c['groups'][0]['methods'][0]['name'] = ['bukan', 'teks'];
        }));
        $rejects('kunci groups.*.methods.*.name harus teks yang tidak kosong', $changed(function (&$c) {
            $c['groups'][0]['methods'][0]['name'] = 7;
        }));
        $rejects('kunci groups.*.title harus teks yang tidak kosong', $changed(function (&$c) {
            $c['groups'][0]['title'] = '  ';
        }));
        $rejects('kunci planned.*.config harus teks', $changed(function (&$c) {
            $c['planned'][0]['config'] = ['bukan', 'teks'];
        }));
        // Bukti, sumbernya, dan catatan: pilihan, tetapi harus teks bila ada.
        $rejects('kunci groups.*.methods.*.evidence harus teks', $changed(function (&$c) {
            $c['groups'][0]['methods'][2]['evidence'] = ['bukan', 'teks'];
        }));
        $rejects('kunci groups.*.methods.*.evidence_source harus teks', $changed(function (&$c) {
            $c['groups'][0]['methods'][2]['evidence_source'] = 7;
        }));
        $rejects('kunci groups.*.methods.*.note harus teks', $changed(function (&$c) {
            $c['groups'][3]['methods'][0]['note'] = ['bukan', 'teks'];
        }));
        $rejects('kunci results_generated tidak ada', $changed(function (&$c) {
            unset($c['results_generated']);
        }));
        $rejects('kunci official.pipeline harus teks', $changed(function (&$c) {
            $c['official']['pipeline'] = ['bukan', 'teks'];
        }));
        $rejects('kunci official.run lebih panjang dari 255 karakter', $changed(function (&$c) {
            $c['official']['run'] = str_repeat('x', 300);
        }));
        $rejects('kunci generated lebih panjang dari 255 karakter', $changed(function (&$c) {
            $c['generated'] = str_repeat('2', 300);
        }));
        $rejects('kunci generated harus teks yang tidak kosong', $changed(function (&$c) {
            $c['generated'] = '';
        }));
        // Kunci kelompok dan kunci butir dipakai sebagai id dan penanda di halaman: tidak boleh berulang, dan "web"
        // milik kelompok buatan halaman.
        $rejects('kunci kelompok berulang', $changed(function (&$c) {
            $c['groups'][] = $c['groups'][0];
        }));
        $rejects('kunci kelompok berulang, atau memakai kunci "web" milik halaman', $changed(function (&$c) {
            $c['groups'][0]['key'] = 'web';
        }));
        $rejects('kunci butir cnn berulang', $changed(function (&$c) {
            $c['groups'][1]['methods'][0]['key'] = 'cnn';
        }));
        $rejects('kunci groups.*.methods.*.settings harus daftar', $changed(function (&$c) {
            $c['groups'][0]['methods'][0]['settings'] = [['Label', 'nilai'], 'kunci' => ['Label', 'nilai']];
        }));
        $rejects('kunci groups.*.methods.*.files harus daftar', $changed(function (&$c) {
            $c['groups'][0]['methods'][0]['files'] = ['src/model.py', 7 => 'x'];
        }));
        $rejects('kunci groups.*.methods.*.files.* harus teks', $changed(function (&$c) {
            $c['groups'][0]['methods'][0]['files'] = ['src/model.py', ['bukan', 'teks']];
        }));

        // Kartu tanpa kunci pilihan (catatan, kunci rencana, arah LSTM) tetap diterima: memang boleh tidak ada.
        $bare = $this->card();
        unset($bare['model']['bidirectional'], $bare['planned'][0]['key'], $bare['chain_complete']);
        foreach ($bare['groups'] as &$group) {
            foreach ($group['methods'] as &$method) {
                unset($method['note'], $method['evidence_source']);
            }
        }
        unset($group, $method);
        $this->import($bare);
        $this->import($this->card());

        unlink($this->dir.'/methods.json');
        $this->artisan('aksara:methods', ['--path' => $this->dir])
            ->expectsOutputToContain('python scripts/export_methods.py')->assertFailed();

        $this->assertSame(1, MethodReport::count());
        $this->assertSame(4590909, MethodReport::current()->payload['model']['parameters']['total']);
        $this->actingAs(User::factory()->create());
        $this->assertSame('CNN (jaringan konvolusi)', $this->shown($this->get('/metode')->assertOk()->getContent(), 'cnn')['name']);
    }

    public function test_each_card_has_its_own_schema_number_and_default_folder(): void
    {
        // Nomor skema kartu metode dibaca dari konfigurasinya sendiri, bukan dari kartu data.
        $this->write($this->card());
        config(['aksara.dataset_schema' => 9]);
        $this->artisan('aksara:methods', ['--path' => $this->dir])->assertSuccessful();
        config(['aksara.dataset_schema' => 1, 'aksara.method_schema' => 2]);
        $this->artisan('aksara:methods', ['--path' => $this->dir])
            ->expectsOutputToContain('Skema kartu metode 1 tidak didukung (butuh 2)')->assertFailed();
        config(['aksara.method_schema' => 1]);

        // Tanpa --path, kartu dibaca dari <repo OCR>/<folder hasil> menurut konfigurasi.
        MethodReport::query()->delete();
        mkdir($this->dir.'/hasil-uji');
        file_put_contents($this->dir.'/hasil-uji/methods.json', json_encode($this->card(), JSON_UNESCAPED_UNICODE));
        config(['aksara.ocr_repo' => $this->dir, 'aksara.results_dir' => 'hasil-uji']);
        $this->artisan('aksara:methods')->expectsOutputToContain('Kartu metode diimpor: 9 butir di 4 kelompok')->assertSuccessful();
        $this->assertSame($this->dir.DIRECTORY_SEPARATOR.'hasil-uji', MethodReport::sole()->source_path);
    }

    /** Folder hasil OCR terkecil yang diterima aksara:import (satu pipeline, satu baris buatan). */
    private function writeResults(string $generated = '2026-10-04T15:46:51'): void
    {
        file_put_contents($this->dir.'/manifest.json', json_encode([
            'schema' => 1, 'generated' => $generated, 'limited' => false, 'official' => 'crnn_fase7_track',
            'pipelines' => [['key' => 'crnn_fase7_track', 'label' => 'CRNN fase7_track', 'config' => 'crnn:fase7_track@1500 · greedy',
                'kind' => 'crnn', 'status' => 'done', 'sort' => 0]],
            'gates' => [], 'metrics' => [], 'ablation' => [], 'confusion' => ['pipeline' => 'crnn_fase7_track', 'items' => []],
        ], JSON_UNESCAPED_UNICODE));
        file_put_contents($this->dir.'/lines.jsonl', json_encode(['dataset' => 'contoh', 'id' => 'a.png', 'image' => 'contoh/a.png', 'width' => 100,
            'height' => 20, 'reference' => 'ꦏ', 'condition' => 'contoh', 'source_id' => 'hal_0', 'tags' => []], JSON_UNESCAPED_UNICODE)."\n");
        file_put_contents($this->dir.'/predictions.jsonl', json_encode(['line' => 'a.png', 'pipeline' => 'crnn_fase7_track', 'text' => 'ꦏ', 'cer' => 0.0,
            'segments' => [['t' => 'ꦏ', 's' => 'eq']]], JSON_UNESCAPED_UNICODE)."\n");
    }

    public function test_results_import_also_imports_the_method_card(): void
    {
        $this->writeResults();
        $this->write($this->card());

        $this->artisan('aksara:import', ['--path' => $this->dir])
            ->expectsOutputToContain('Kartu metode diimpor: 9 butir di 4 kelompok')
            ->doesntExpectOutputToContain('PERINGATAN')->assertSuccessful();
        $this->assertSame('fase7_track', MethodReport::sole()->official_run);
        // Kartu dan hasil yang diimpor berasal dari ekspor hasil yang sama dan run yang sama: tidak ada peringatan.
        $this->actingAs(User::factory()->create());
        $html = $this->get('/metode')->assertOk()->getContent();
        $this->assertSame([], $this->staleNotes($html));
        $this->assertSame(['cnn', 'ctc', 'visual_order'], $this->groupKeys($html)['ocr']);

        // Kartu yang ditolak tidak menggagalkan impor hasil OCR, dan kartu lama dipertahankan. Peringatannya menyebut
        // alasan, dan perintah untuk kartu ITU (bukan kartu data).
        $this->write(['schema' => 5] + $this->card());
        $this->artisan('aksara:import', ['--path' => $this->dir])
            ->expectsOutputToContain('PERINGATAN: hasil OCR sudah diimpor, tetapi kartu halaman Metode ditolak. Skema kartu metode 5 tidak didukung '
                .'(butuh 1). Perbaiki lalu jalankan: php artisan aksara:methods')
            ->expectsOutputToContain('Kartu halaman Dataset (datasets.json) tidak ada di folder itu; halaman Dataset tidak diubah. Buat dengan: '
                .'python scripts/export_datasets.py')
            ->assertSuccessful();
        $this->assertSame(1, MethodReport::count());
        $this->artisan('aksara:methods', ['--path' => $this->dir])->expectsOutputToContain('Skema kartu metode 5 tidak didukung')->assertFailed();

        // Kartu data yang ditolak tidak menghentikan impor kartu metode di sebelahnya.
        MethodReport::query()->delete();
        $this->write($this->card());
        // (Satu baris keluaran hanya bisa dicocokkan dengan SATU harapan, jadi peringatannya dinyatakan utuh.)
        file_put_contents($this->dir.'/datasets.json', '{"schema": 7}');
        $this->artisan('aksara:import', ['--path' => $this->dir])
            ->expectsOutputToContain('PERINGATAN: hasil OCR sudah diimpor, tetapi kartu halaman Dataset ditolak. Skema kartu data 7 tidak didukung '
                .'(butuh 1). Perbaiki lalu jalankan: php artisan aksara:datasets')
            ->expectsOutputToContain('Kartu metode diimpor: 9 butir di 4 kelompok')->assertSuccessful();
        $this->assertSame(1, MethodReport::count());
        $this->assertSame(0, DatasetReport::count());
        unlink($this->dir.'/datasets.json');
    }

    public function test_import_commands_say_at_once_when_a_card_does_not_match_the_results(): void
    {
        // Hasil OCR diekspor ulang, tetapi kartu metode di folder itu masih mengutip ekspor yang lama: perintah impor
        // mengatakannya saat itu juga, dengan skrip dan perintah yang harus dijalankan ulang.
        $this->writeResults('2026-10-09T08:00:00');
        $this->write($this->card());
        $warning = 'PERINGATAN: kartu halaman Metode tidak sejalan dengan hasil OCR yang diimpor. Bukti di halaman ini dibaca dari ekspor hasil '
            .'2026-10-04T15:46:51, sedangkan hasil yang diimpor web berasal dari ekspor 2026-10-09T08:00:00. Jalankan ulang: python '
            .'scripts/export_methods.py, lalu php artisan aksara:methods';
        $this->artisan('aksara:import', ['--path' => $this->dir])
            ->expectsOutputToContain('Kartu metode diimpor: 9 butir di 4 kelompok')->expectsOutputToContain($warning)->assertSuccessful();
        $this->artisan('aksara:methods', ['--path' => $this->dir])->expectsOutputToContain($warning)->assertSuccessful();

        // Kartu untuk run lain daripada pipeline resmi hasil yang diimpor.
        $other = $this->card();
        $other['official'] = ['run' => 'fase8_x', 'pipeline' => 'crnn_fase8_x'] + $other['official'];
        $other['results_generated'] = '2026-10-09T08:00:00';
        $this->write($other);
        $this->artisan('aksara:methods', ['--path' => $this->dir])
            ->expectsOutputToContain('Kartu metode ini dibuat untuk run fase8_x, sedangkan angka resmi web sekarang milik CRNN fase7_track.')
            ->assertSuccessful();
        // Dua hal sekaligus (run lain DAN ekspor hasil lain): dua-duanya disebut, masing-masing di barisnya sendiri.
        $other['results_generated'] = '2026-10-04T15:46:51';
        $this->write($other);
        $head = 'PERINGATAN: kartu halaman Metode tidak sejalan dengan hasil OCR yang diimpor. ';
        $this->artisan('aksara:methods', ['--path' => $this->dir])
            ->expectsOutputToContain($head.'Kartu metode ini dibuat untuk run fase8_x')
            ->expectsOutputToContain($head.'Bukti di halaman ini dibaca dari ekspor hasil 2026-10-04T15:46:51')
            ->assertSuccessful();

        // Kartu yang sejalan: tidak ada peringatan.
        $this->writeResults();
        $this->write($this->card());
        $this->artisan('aksara:import', ['--path' => $this->dir])->doesntExpectOutputToContain('PERINGATAN')->assertSuccessful();
    }

    public function test_a_card_whose_page_cannot_be_rendered_is_refused(): void
    {
        // Jaring terakhir: sebelum disimpan, halaman dirender sekali dengan kartu calon. Apa pun yang gagal di situ
        // (di sini: galat buatan saat tampilan disusun) membatalkan impor, dengan pesan yang menyebut sebabnya dan
        // apa yang terjadi pada kartu yang tersimpan.
        $importer = app(MethodImporter::class);
        $break = fn () => View::composer('livewire.pages.metode', function () {
            throw new LogicException('halaman rusak');
        });
        $tail = '. Penyebabnya kartu itu, atau berkas dan tabel web lain yang dibaca halamannya.';

        $break();
        $this->assertCardRejected($importer, MethodReport::class, $this->dir,
            'Kartu metode tidak bisa ditampilkan halamannya (halaman rusak); tidak ada kartu yang disimpan'.$tail, $this->card());
        $this->assertSame(0, MethodReport::count());

        Event::forget('composing: livewire.pages.metode');
        $this->import($this->card());
        $break();
        $newer = ['generated' => '2026-10-09T10:00:00'] + $this->card();
        $this->assertCardRejected($importer, MethodReport::class, $this->dir,
            'Kartu metode tidak bisa ditampilkan halamannya (halaman rusak); kartu sebelumnya dipertahankan'.$tail, $newer);
        $this->write($newer);
        $this->artisan('aksara:methods', ['--path' => $this->dir])
            ->expectsOutputToContain('Kartu metode tidak bisa ditampilkan halamannya (halaman rusak); kartu sebelumnya dipertahankan')->assertFailed();
        $this->assertSame('2026-10-05T23:40:00', MethodReport::sole()->generated_at);

        Event::forget('composing: livewire.pages.metode');
        $this->import($newer);
        $this->assertSame('2026-10-09T10:00:00', MethodReport::sole()->generated_at);
    }

    public function test_database_problems_are_not_reported_as_a_bad_card(): void
    {
        $this->writeResults();
        $this->write($this->card());
        $importer = app(MethodImporter::class);

        // Tabel lain yang dibaca halaman belum dimigrasi: uji tampil gagal karena database, bukan karena kartunya.
        // Pesannya tidak menyalahkan kartu, dan kartu yang tersimpan tidak tersentuh.
        $this->import($this->card());
        Schema::drop('translation_runs');
        $this->assertCardRejected($importer, MethodReport::class, $this->dir,
            'Kartu metode tidak diperiksa: halamannya tidak bisa dirender karena keadaan database web', $this->card(), exception: CardStorageException::class);
        $this->assertCardRejected($importer, MethodReport::class, $this->dir,
            'kartu sebelumnya dipertahankan. Jalankan dulu: php artisan migrate', $this->card(), exception: CardStorageException::class);

        // Tabel kartu metode sendiri belum dimigrasi. Perintah kartunya gagal dengan petunjuk migrate, dan aksara:import
        // (yang hasil OCR-nya sudah tersimpan) juga keluar dengan kode gagal, bukan sekadar memperingatkan.
        Schema::drop('method_reports');
        $caught = null;
        try {
            $importer->import($this->dir);
        } catch (RuntimeException $e) {
            $caught = $e;
        }
        $this->assertInstanceOf(CardStorageException::class, $caught);
        $this->assertStringContainsString('Kartu metode tidak bisa disimpan karena keadaan database web', $caught->getMessage());
        $this->assertStringEndsWith('Jalankan dulu: php artisan migrate', $caught->getMessage());
        $this->artisan('aksara:methods', ['--path' => $this->dir])
            ->expectsOutputToContain('Kartu metode tidak bisa disimpan karena keadaan database web')->assertFailed();
        $this->artisan('aksara:import', ['--path' => $this->dir])
            ->expectsOutputToContain('Kartu halaman Metode tidak diimpor. Kartu metode tidak bisa disimpan karena keadaan database web')
            ->doesntExpectOutputToContain('PERINGATAN')
            ->assertFailed();
        $this->assertSame('crnn_fase7_track', Pipeline::official()->key);  // hasil OCR tetap diimpor
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
        // Status yang tidak dikenal bukan "dipakai": pembaca aksara itu tidak dihitung sebagai model yang dipakai.
        $this->assertSame('Model hasil belajar 1 dipakai NLLB-200 (pralatih, tidak dilatih ulang) · tersedia, bukan jalur resmi: '
            .'CRNN (dilatih dari nol), model bahasa n-gram karakter (statistik)', $this->tile($page->getContent(), 'Model hasil belajar'));
    }

    public function test_tiles_and_text_follow_the_card(): void
    {
        // Kartu tanpa beam search dan tanpa butir CNN, dengan LSTM satu arah: ubin dan alur mengikutinya.
        $card = $this->card();
        $card['groups'] = array_values(array_filter($card['groups'], fn ($group) => $group['key'] !== 'decoding'));
        array_shift($card['groups'][0]['methods']);
        // Teks dari kartu selalu di-escape: nama, penjelasan, pengaturan, bukti, catatan, juga jalur berkas dan sumber
        // bukti yang diberi titik patah.
        $card['groups'][0]['methods'][0]['name'] = 'CTC <b>tebal</b>';
        $card['groups'][0]['methods'][0]['settings'] = [['Kelas <i>kosong</i>', 'indeks <u>0</u>']];
        $card['groups'][0]['methods'][0]['files'] = ['<b>src</b>/model.py'];
        $card['groups'][0]['methods'][1]['evidence'] = 'Gerbang <b>G4</b>, 100%.';
        $card['groups'][0]['methods'][1]['evidence_source'] = '<i>data</i>/tokenizer.json';
        $card['groups'][0]['methods'][1]['summary'] = 'Urutan <script>alert(1)</script> visual.';
        $card['groups'][2]['methods'][0]['note'] = 'Catatan <u>berbahaya</u>.';
        $card['model']['conv_layers'] = 5;
        $card['model']['bidirectional'] = false;
        $card['model']['parameters'] = ['total' => 950, 'cnn' => 300, 'proj' => 200, 'rnn' => 400, 'head' => 50];
        $this->import($card);

        $this->actingAs(User::factory()->create());
        $html = $this->get('/metode')->assertOk()->getContent();
        $this->assertSame('Model hasil belajar 1 dipakai NLLB-200 (pralatih, tidak dilatih ulang)', $this->tile($html, 'Model hasil belajar'));
        $this->assertSame('Model pembaca CRNN + CTC CNN 5 lapis → LSTM 2 lapis → 93 kelas, dilatih dari nol', $this->tile($html, 'Model pembaca'));
        // Bilangan kecil ditulis apa adanya: CNN 300 + proyeksi 200 = 500.
        $this->assertSame('Parameter 950 CNN 500 · LSTM 400 · keluaran 50', $this->tile($html, 'Parameter'));
        $this->assertSame('Butir metode 13 di halaman ini Pembaca aksara 2 · Data latih sintetis 2 · Evaluasi dan statistik 3 · Tahap lanjutan alur 6',
            $this->tile($html, 'Butir metode'));
        $this->assertSame(['ocr', 'data', 'evaluation', 'web'], array_keys($this->groupKeys($html)));
        $this->assertSame(['LSTM', '2 lapis', true], $this->flowSteps($html, "//ol[contains(@class, 'flow')]/li[@data-step]")[2]);

        $xpath = $this->xpath($html);
        $this->assertSame(0, $xpath->query('//article[@data-method]//b | //article[@data-method]//i | //article[@data-method]//u | //article[@data-method]//script')->length);
        $ctc = $this->shown($html, 'ctc');
        $this->assertSame(['CTC <b>tebal</b>', [['Kelas <i>kosong</i>', 'indeks <u>0</u>'], ['Kode', '<b>src</b>/model.py']]], [$ctc['name'], $ctc['settings']]);
        $order = $this->shown($html, 'visual_order');
        $this->assertSame(['Urutan <script>alert(1)</script> visual.', 'Gerbang <b>G4</b>, 100%.', '<i>data</i>/tokenizer.json'],
            [$order['summary'], $order['evidence'], $order['source']]);
        $this->assertSame('Catatan <u>berbahaya</u>.', $this->shown($html, 'gates')['note']);
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
        // Halaman menampilkan kartu apa adanya: tiap butir di kelompoknya, dengan nama, penjelasan, pengaturan, bukti,
        // sumber, dan catatannya.
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
