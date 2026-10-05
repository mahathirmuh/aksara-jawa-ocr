<?php

namespace App\Support;

use Normalizer;

/**
 * Aksara Jawa -> Latin, deterministik, DRAF (tahap 2 alur).
 *
 * Hanya bantu baca: tidak dipakai untuk metrik OCR. Ejaan mengikuti transliterasi NusaAksara:
 * taling = é, pepet = ê, taling-tarung = o, konsonan ganda ditulis apa adanya.
 * Batasan yang diketahui: ꦲ di awal kata dibaca vokal, tetapi teks tanpa spasi tidak punya batas kata
 * selain awal baris, sehingga "ꦲ" di tengah baris tetap menjadi "h".
 */
final class Transliterator
{
    private const CONS = [
        'ꦏ' => 'k', 'ꦐ' => 'q', 'ꦑ' => 'k', 'ꦒ' => 'g', 'ꦓ' => 'g', 'ꦔ' => 'ng', 'ꦕ' => 'c', 'ꦖ' => 'c',
        'ꦗ' => 'j', 'ꦘ' => 'ny', 'ꦙ' => 'jh', 'ꦚ' => 'ny', 'ꦛ' => 'th', 'ꦜ' => 'th', 'ꦝ' => 'dh', 'ꦞ' => 'dh',
        'ꦟ' => 'n', 'ꦠ' => 't', 'ꦡ' => 't', 'ꦢ' => 'd', 'ꦣ' => 'dh', 'ꦤ' => 'n', 'ꦥ' => 'p', 'ꦦ' => 'p',
        'ꦧ' => 'b', 'ꦨ' => 'b', 'ꦩ' => 'm', 'ꦪ' => 'y', 'ꦫ' => 'r', 'ꦬ' => 'r', 'ꦭ' => 'l', 'ꦮ' => 'w',
        'ꦯ' => 's', 'ꦰ' => 's', 'ꦱ' => 's', 'ꦲ' => 'h',
    ];

    private const NUKTA = ['p' => 'f', 'w' => 'v', 'j' => 'z', 'k' => 'kh', 'g' => 'gh'];

    private const SWARA = [
        'ꦄ' => 'a', 'ꦅ' => 'i', 'ꦆ' => 'i', 'ꦇ' => 'i', 'ꦈ' => 'u', 'ꦉ' => 'rê', 'ꦊ' => 'lê', 'ꦋ' => 'lêu',
        'ꦌ' => 'é', 'ꦍ' => 'ai', 'ꦎ' => 'o',
    ];

    private const VOWEL = ['ꦶ' => 'i', 'ꦷ' => 'i', 'ꦸ' => 'u', 'ꦹ' => 'u', 'ꦼ' => 'ê', 'ꦻ' => 'ai'];

    private const FINAL = ['ꦁ' => 'ng', 'ꦂ' => 'r', 'ꦃ' => 'h', 'ꦀ' => 'ng'];

    private const PUNCT = [
        '꧈' => ',', '꧉' => '.', '꧌' => '(', '꧍' => ')', 'ꧏ' => '²', '꧇' => '', '꧊' => '', '꧋' => '',
        '꧃' => '', '꧄' => '', '꧅' => '', '꧆' => '', '꧁' => '', '꧂' => '',
    ];

    private const TALING = 'ꦺ';
    private const TARUNG = 'ꦴ';
    private const PANGKON = '꧀';
    private const NUKTA_SIGN = '꦳';
    private const PENGKAL = 'ꦾ';
    private const CAKRA = 'ꦿ';
    private const KERET = 'ꦽ';

    public static function toLatin(string $text): string
    {
        $text = Normalizer::normalize($text, Normalizer::FORM_C) ?: $text;
        $chars = mb_str_split($text);
        $n = count($chars);
        $out = [];
        $i = 0;
        while ($i < $n) {
            $ch = $chars[$i];
            if (isset(self::CONS[$ch]) || isset(self::SWARA[$ch])) {
                $prev = $i > 0 ? $chars[$i - 1] : null;
                $wordStart = $prev === null || ! self::isJavanese($prev) || isset(self::PUNCT[$prev]);
                if (isset(self::SWARA[$ch])) {
                    [$onset, $vowel] = ['', self::SWARA[$ch]];
                } else {
                    [$onset, $vowel] = [self::CONS[$ch], 'a'];
                    if ($ch === 'ꦲ' && $wordStart) {
                        $onset = '';
                    }
                }
                [$medial, $final, $taling] = ['', '', false];
                $j = $i + 1;
                while ($j < $n && self::isMark($chars[$j])) {
                    $m = $chars[$j];
                    match (true) {
                        $m === self::NUKTA_SIGN => $onset = self::NUKTA[$onset] ?? $onset,
                        $m === self::PENGKAL => $medial .= 'y',
                        $m === self::CAKRA => $medial .= 'r',
                        $m === self::KERET => [$medial, $vowel] = [$medial.'r', 'ê'],
                        $m === self::PANGKON => $vowel = '',
                        $m === self::TALING => [$taling, $vowel] = [true, 'é'],
                        $m === self::TARUNG => $vowel = $taling ? 'o' : 'a',
                        isset(self::VOWEL[$m]) => $vowel = self::VOWEL[$m],
                        isset(self::FINAL[$m]) => $final .= self::FINAL[$m],
                        default => null,
                    };
                    $j++;
                }
                // Sengau di depan ca/ja ditulis nya + pangkon (ꦥꦚ꧀ꦕꦶ), tetapi dibaca dan dilatinkan "n": panci, panjenengan.
                if ($ch === 'ꦚ' && $vowel === '' && $medial === '' && $final === '' && in_array($chars[$j] ?? '', ['ꦕ', 'ꦗ'], true)) {
                    $onset = 'n';
                }
                $out[] = $onset.$medial.$vowel.$final;
                $i = $j;
            } elseif (($cp = mb_ord($ch)) >= 0xA9D0 && $cp <= 0xA9D9) {
                $out[] = (string) ($cp - 0xA9D0);
                $i++;
            } elseif (isset(self::PUNCT[$ch])) {
                $out[] = self::PUNCT[$ch];
                $i++;
            } elseif (self::isMark($ch)) {
                $i++; // tanda tanpa aksara dasar
            } else {
                $out[] = $ch;
                $i++;
            }
        }

        return implode('', $out);
    }

    private static function isMark(string $ch): bool
    {
        return isset(self::VOWEL[$ch]) || isset(self::FINAL[$ch])
            || in_array($ch, [self::TALING, self::TARUNG, self::PANGKON, self::NUKTA_SIGN, self::PENGKAL, self::CAKRA, self::KERET], true);
    }

    private static function isJavanese(string $ch): bool
    {
        $cp = mb_ord($ch);

        return $cp >= 0xA980 && $cp <= 0xA9DF;
    }
}
