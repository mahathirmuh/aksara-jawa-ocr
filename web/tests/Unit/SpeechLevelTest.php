<?php

namespace Tests\Unit;

use App\Support\SpeechLevel;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\TestCase;

class SpeechLevelTest extends TestCase
{
    public static function sentences(): array
    {
        return [
            'krama & krama inggil' => ['Kula badhe tindak dhateng griya panjenengan.', 'krama'],
            'ngoko' => ['Aku arep lunga menyang omahmu, ora suwe.', 'ngoko'],
            'madya' => ['Sampeyan ajeng teng pundi niki?', 'madya'],
            'campur' => ['Aku ora ngerti, nanging panjenengan sampun rawuh.', 'campur'],
            'afiks krama' => ['Griyanipun dipunresiki saben enjing.', 'krama'],
            'NusaAksara (krama)' => ['-yan upaya ngantos dumugi ngriki punika,', 'krama'],
        ];
    }

    #[DataProvider('sentences')]
    public function test_classifies_speech_level(string $latin, string $level): void
    {
        $this->assertSame($level, SpeechLevel::classify($latin)['level']);
    }

    public function test_evidence_explains_the_decision(): void
    {
        $result = SpeechLevel::classify('Kula mboten saged.');
        $this->assertSame([['kula', 'krama'], ['mboten', 'krama'], ['saged', 'krama']], $result['evidence']);
        $this->assertSame('cukup', $result['confidence']);
    }

    public function test_no_markers_means_undetermined(): void
    {
        $result = SpeechLevel::classify('pagi-pagi Jakarta');
        $this->assertNull($result['level']);
        $this->assertSame('rendah', $result['confidence']);
    }

    public function test_unspaced_text_uses_only_long_markers_and_is_low_confidence(): void
    {
        $result = SpeechLevel::classify('panjenenganbadhesampunmboten');
        $this->assertSame('krama', $result['level']);
        $this->assertSame('rendah', $result['confidence']);
        $this->assertFalse($result['spaced']);
        // "aku" terlalu pendek untuk dicari di teks tanpa spasi: bisa muncul di dalam kata lain.
        $this->assertNull(SpeechLevel::classify('akuoraarep')['level']);
    }
}
