<?php

namespace Tests\Unit;

use App\Support\TextMetrics;
use App\Support\Transliterator;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

class TransliteratorTest extends TestCase
{
    public static function samples(): array
    {
        return [
            'ngoko & pepet' => ['ꦲꦏꦸ ꦧꦺꦴꦠꦼꦤ꧀', 'aku botên'],
            'cakra & pangkon' => ['ꦥꦿꦺꦴꦒ꧀ꦫꦩ꧀', 'program'],
            'pasangan' => ['ꦏꦤ꧀ꦠꦺꦴꦤ꧀', 'kanton'],
            'geminasi majalah' => ['ꦩꦿꦶꦥꦠ꧀ꦠꦺ', 'mripatté'],
            'pada lungsi' => ['ꦕꦼꦫꦶꦠ꧉', 'cêrita.'],
            'layar, wignyan, cecak' => ['ꦧꦂꦲꦸꦮꦃꦲꦸꦮꦁ', 'barhuwahhuwang'],
            'pa cerek & keret' => ['ꦉꦏꦽꦠ', 'rêkrêta'],
            'angka' => ['꧇꧑꧙꧔꧕꧇', '1945'],
            'cecak telu' => ['ꦥ꦳ꦏꦸꦭ꧀ꦠꦱ꧀', 'fakultas'],
        ];
    }

    #[DataProvider('samples')]
    public function test_transliterates_to_latin(string $aksara, string $latin): void
    {
        $this->assertSame($latin, Transliterator::toLatin($aksara));
    }

    public function test_keeps_non_javanese_text(): void
    {
        $this->assertSame('ok 2026', Transliterator::toLatin('ok 2026'));
    }

    public function test_cer_counts_code_points_not_bytes(): void
    {
        $this->assertSame(1, TextMetrics::distance('ꦲꦏꦸ', 'ꦲꦏꦶ'));
        $this->assertSame(1, TextMetrics::distance('botên', 'boten'));
        $this->assertEqualsWithDelta(1 / 3, TextMetrics::cer(['ꦲꦏꦸ'], ['ꦲꦏ']), 1e-9);
        $this->assertSame(0.0, TextMetrics::cer([], []));
    }

    public function test_letters_only_normalises_for_transliteration_comparison(): void
    {
        $this->assertSame('andhêlikaképangrasané', TextMetrics::lettersOnly('Andhêlikaké pangrasané.'));
        $this->assertSame('yanupaya', TextMetrics::lettersOnly('-yan upaya,'));
    }
}
