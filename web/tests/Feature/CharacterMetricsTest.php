<?php

namespace Tests\Feature;

use App\Models\ClassMetric;
use App\Models\Metric;
use App\Models\User;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Tests\Concerns\ReadsPage;
use Tests\TestCase;

/**
 * Recall, presisi, F1, akurasi karakter (kolom baru tabel metrics) dan tabel per aksara (class_metrics) dari manifest
 * skema 1. Kontrak datanya dibuat di sini dari teks sintetis, bukan dari fixture NusaAksara, supaya test ini jalan di
 * mana pun.
 */
class CharacterMetricsTest extends TestCase
{
    use ReadsPage, RefreshDatabase;

    private const KA = 'ꦏ';

    private const NA = 'ꦤ';

    private const TA = 'ꦠ';

    /** Kontrak data dua pipeline pada dua baris; $withChars = false meniru ekspor sebelum metrik karakter ada. */
    private function contract(bool $withChars = true): string
    {
        $dir = sys_get_temp_dir().'/aksara-chars-'.uniqid();
        mkdir($dir);
        $lines = [
            ['dataset' => 'uji', 'id' => 'b1.png', 'image' => 'data/uji/b1.png', 'width' => 200, 'height' => 50,
                'reference' => self::KA.self::NA.self::TA, 'condition' => 'bersih', 'source_id' => 'h1', 'tags' => []],
            ['dataset' => 'uji', 'id' => 'b2.png', 'image' => 'data/uji/b2.png', 'width' => 200, 'height' => 50,
                'reference' => self::KA.self::NA, 'condition' => 'bersih', 'source_id' => 'h2', 'tags' => []],
        ];
        $predictions = [];
        foreach (['crnn_a' => [self::KA.self::NA.self::TA, self::KA], 'crnn_b' => [self::KA.self::TA.self::TA, self::KA.self::NA]] as $key => $texts) {
            foreach ($texts as $i => $text) {
                $predictions[] = ['line' => $lines[$i]['id'], 'pipeline' => $key, 'text' => $text, 'cer' => 0.1,
                    'segments' => [['t' => $text, 's' => 'ok']], 'spans' => null];
            }
        }
        $metric = fn (string $key, float $cer, array $chars) => ['scope' => 'nusaaksara_745', 'pipeline' => $key,
            'lines' => 2, 'cer' => $cer, 'cer_no_space' => $cer, 'exact' => 0.5] + ($withChars ? $chars : []);
        $manifest = [
            'schema' => 1, 'generated' => '2026-10-09T10:00:00', 'source' => 'test', 'limited' => false,
            'pipelines' => [
                ['key' => 'crnn_a', 'label' => 'CRNN A', 'kind' => 'crnn', 'status' => 'done', 'config' => 'crnn:a · greedy', 'sort' => 0],
                ['key' => 'crnn_b', 'label' => 'CRNN B', 'kind' => 'crnn', 'status' => 'done', 'config' => 'crnn:b · greedy', 'sort' => 1],
            ],
            'official' => 'crnn_a',
            'gates' => [
                ['code' => 'G1', 'name' => 'Sintetis bersih', 'value' => 0.003, 'target' => 0.02, 'passed' => true, 'basis' => 'uji', 'source' => 'uji'],
                ['code' => 'G2', 'name' => 'Sintetis berat', 'value' => 0.012, 'target' => 0.05, 'passed' => true, 'basis' => 'uji', 'source' => 'uji'],
                ['code' => 'G3', 'name' => 'Nyata', 'value' => 0.2, 'target' => 0.08, 'passed' => false, 'basis' => 'uji', 'source' => 'uji'],
                ['code' => 'G4', 'name' => 'Round-trip', 'value' => 1.0, 'target' => 1.0, 'passed' => true, 'basis' => 'uji', 'source' => 'uji'],
            ],
            'metrics' => [
                // crnn_a: baris 1 persis (3 cocok), baris 2 "ꦏꦤ" -> "ꦏ" (1 cocok, 1 hilang): M 4, D 1 -> recall 0,8, presisi 1.
                $metric('crnn_a', 0.2, ['precision' => 1.0, 'recall' => 0.8, 'f1' => 2 * 0.8 / 1.8, 'char_accuracy' => 0.8]),
                // crnn_b: baris 1 ꦤ -> ꦠ (2 cocok, 1 tertukar), baris 2 persis: M 4, S 1 -> 0,8 / 0,8.
                $metric('crnn_b', 0.2, ['precision' => 0.8, 'recall' => 0.8, 'f1' => 0.8, 'char_accuracy' => 0.8]),
                ['scope' => 'blind_50', 'pipeline' => 'crnn_a', 'lines' => 1, 'cer' => 0.0, 'cer_no_space' => 0.0, 'exact' => 1.0]
                    + ($withChars ? ['precision' => 1.0, 'recall' => 1.0, 'f1' => 1.0, 'char_accuracy' => 1.0] : []),
            ],
            'ablation' => [],
            'confusion' => ['pipeline' => 'crnn_a', 'scope' => 'nusaaksara_745', 'totals' => ['sub' => 0, 'del' => 1, 'ins' => 0],
                'items' => [['kind' => 'del', 'ref' => self::NA, 'hyp' => '', 'count' => 1]]],
        ];
        if ($withChars) {
            $row = fn (string $pipeline, string $char, int $ref, int $hyp, int $tp, int $fn, int $fp) => [
                'pipeline' => $pipeline, 'char' => $char, 'code' => sprintf('U+%04X', mb_ord($char)), 'name' => 'uji',
                'ref' => $ref, 'hyp' => $hyp, 'tp' => $tp, 'fn' => $fn, 'fp' => $fp,
                'precision' => $hyp ? $tp / $hyp : null, 'recall' => $ref ? $tp / $ref : null,
                'f1' => $ref && $hyp && $tp ? 2 * ($tp / $hyp) * ($tp / $ref) / ($tp / $hyp + $tp / $ref) : ($ref && $hyp ? 0.0 : null),
            ];
            $manifest['class_metrics'] = ['scope' => 'nusaaksara_745', 'items' => [
                $row('crnn_a', self::KA, 2, 2, 2, 0, 0), $row('crnn_a', self::NA, 2, 1, 1, 1, 0), $row('crnn_a', self::TA, 1, 1, 1, 0, 0),
                $row('crnn_b', self::KA, 2, 2, 2, 0, 0), $row('crnn_b', self::NA, 2, 1, 1, 1, 0), $row('crnn_b', self::TA, 1, 2, 1, 0, 1),
            ]];
        }
        file_put_contents($dir.'/manifest.json', json_encode($manifest, JSON_UNESCAPED_UNICODE));
        file_put_contents($dir.'/lines.jsonl', implode('', array_map(fn ($l) => json_encode($l, JSON_UNESCAPED_UNICODE)."\n", $lines)));
        file_put_contents($dir.'/predictions.jsonl', implode('', array_map(fn ($p) => json_encode($p, JSON_UNESCAPED_UNICODE)."\n", $predictions)));

        return $dir;
    }

    public function test_import_stores_character_metrics_and_pages_show_them(): void
    {
        $this->artisan('aksara:import', ['--path' => $this->contract()])->assertSuccessful();

        $a = Metric::where('scope', 'nusaaksara_745')->where('pipeline', 'crnn_a')->first();
        $this->assertSame([1.0, 0.8, 0.8], [$a->precision, $a->recall, $a->char_accuracy]);
        $this->assertEqualsWithDelta(0.8889, $a->f1, 0.0001);
        $this->assertSame(6, ClassMetric::count());
        $na = ClassMetric::where('pipeline', 'crnn_a')->where('char', self::NA)->first();
        $this->assertSame(['U+A9A4', 2, 1, 1, 1, 0], [$na->code, $na->ref, $na->hyp, $na->tp, $na->fn, $na->fp]);
        $this->assertSame(0.5, $na->recall);

        $this->actingAs(User::factory()->create());
        $html = $this->get('/perbandingan')->assertOk()->getContent();
        $rowA = $this->part($html, 'data-row', 'crnn_a');
        $this->assertStringContainsString('CRNN A', $rowA);
        $this->assertStringContainsString('100,0%', $rowA, 'presisi crnn_a');
        $this->assertStringContainsString('80,0%', $rowA, 'recall crnn_a');
        $this->assertStringContainsString('88,9%', $rowA, 'F1 crnn_a');
        $rowB = $this->part($html, 'data-row', 'crnn_b');
        $this->assertStringNotContainsString('88,9%', $rowB);
        $this->assertStringNotContainsString('–', $rowB, 'semua metrik crnn_b terisi');
        $this->assertStringContainsString('100,0%', $this->part($html, 'data-blind', 'crnn_a'));
        $definition = $this->part($html, 'data-part', 'definisi-metrik');
        $this->assertStringContainsString('recall = bagian karakter label yang terbaca benar', $definition);
        $this->assertStringNotContainsString('export_results.py', $definition, 'semua pipeline punya metrik: tanpa petunjuk ekspor ulang');

        // Kesalahan aksara: tabel per aksara pipeline resmi (crnn_a), terlemah dulu; MIN_REF dilonggarkan untuk data kecil.
        $html = $this->get('/kesalahan')->assertOk()->getContent();
        $summary = $this->part($html, 'data-part', 'ringkasan-aksara');
        // macro-F1 crnn_a = rata-rata F1 ka (1,0), na (0,667), ta (1,0) = 0,889; mikro dari tabel metrics.
        $this->assertStringContainsString('macro-F1 88,9% atas 3 aksara di label', $summary);
        $this->assertStringContainsString('mikro: presisi 100,0% · recall 80,0% · F1 88,9%', $summary);
        $this->assertSame(0, $this->xpath($html)->query("//*[@data-part='tanpa-per-aksara']")->length);
        // Dengan MIN_REF 10 tidak ada aksara yang cukup sering di data uji sekecil ini: daftar kosong, tanpa galat.
        $this->assertSame(0, $this->xpath($html)->query('//tr[@data-char]')->length);
    }

    public function test_weakest_table_lists_lowest_f1_first_for_the_official_pipeline(): void
    {
        $this->artisan('aksara:import', ['--path' => $this->contract()])->assertSuccessful();
        // Perbanyak kemunculan supaya melewati MIN_REF; crnn_b (bukan resmi) diberi angka berbeda dan tidak boleh tampil.
        ClassMetric::where('pipeline', 'crnn_a')->update(['ref' => 50, 'hyp' => 50]);
        ClassMetric::where('pipeline', 'crnn_b')->update(['ref' => 50, 'hyp' => 50, 'f1' => 0.01]);
        $this->actingAs(User::factory()->create());

        $html = $this->get('/kesalahan')->assertOk()->getContent();
        $rows = $this->xpath($html)->query('//tr[@data-char]');
        $this->assertSame(3, $rows->length);
        $this->assertSame('U+A9A4', $rows->item(0)->getAttribute('data-char'), 'na (F1 66,7%) paling lemah, di atas');
        $this->assertStringContainsString('66,7%', $this->squash($rows->item(0)));
        $this->assertStringContainsString('50', $this->squash($rows->item(0)));
        foreach ($rows as $row) {
            $this->assertStringNotContainsString('1,0%', $this->squash($row), 'angka crnn_b (F1 1,0%) tidak tampil');
        }
    }

    public function test_old_exports_without_character_metrics_still_import_and_render(): void
    {
        $this->artisan('aksara:import', ['--path' => $this->contract(withChars: false)])->assertSuccessful();

        $a = Metric::where('scope', 'nusaaksara_745')->where('pipeline', 'crnn_a')->first();
        $this->assertNull($a->precision);
        $this->assertNull($a->f1);
        $this->assertSame(0, ClassMetric::count());

        $this->actingAs(User::factory()->create());
        $html = $this->get('/perbandingan')->assertOk()->getContent();
        $this->assertStringContainsString('–', $this->part($html, 'data-row', 'crnn_a'));
        $this->assertStringContainsString('export_results.py', $this->part($html, 'data-part', 'definisi-metrik'));
        $html = $this->get('/kesalahan')->assertOk()->getContent();
        $this->assertStringContainsString('ekspor sebelum metrik per aksara ada', $this->part($html, 'data-part', 'tanpa-per-aksara'));
        $this->assertSame(0, $this->xpath($html)->query("//*[@data-part='ringkasan-aksara']")->length);
    }

    public function test_reimport_replaces_character_metrics(): void
    {
        $this->artisan('aksara:import', ['--path' => $this->contract()])->assertSuccessful();
        $this->artisan('aksara:import', ['--path' => $this->contract()])->assertSuccessful();
        $this->assertSame(6, ClassMetric::count(), 'tabel per aksara diganti utuh, bukan ditumpuk');
        $this->assertSame(3, Metric::where('scope', '!=', 'translit_draft')->count());
    }
}
