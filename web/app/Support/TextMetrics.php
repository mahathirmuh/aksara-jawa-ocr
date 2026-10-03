<?php

namespace App\Support;

/**
 * CER per titik kode Unicode (bukan per byte; levenshtein() bawaan PHP bekerja per byte).
 *
 * Hanya untuk tahap alur yang hidup di web (transliterasi draf vs manusia). Angka OCR tetap dari Python.
 */
final class TextMetrics
{
    public static function distance(string $a, string $b): int
    {
        $x = mb_str_split($a);
        $y = mb_str_split($b);
        if ($x === []) {
            return count($y);
        }
        $prev = range(0, count($y));
        foreach ($x as $i => $cx) {
            $cur = [$i + 1];
            foreach ($y as $j => $cy) {
                $cur[] = min($prev[$j + 1] + 1, $cur[$j] + 1, $prev[$j] + ($cx === $cy ? 0 : 1));
            }
            $prev = $cur;
        }

        return $prev[count($y)];
    }

    /** CER korpus: total jarak dibagi total panjang referensi. */
    public static function cer(array $references, array $hypotheses): float
    {
        $dist = $len = 0;
        foreach ($references as $k => $ref) {
            $dist += self::distance($ref, $hypotheses[$k] ?? '');
            $len += mb_strlen($ref);
        }

        return $len > 0 ? $dist / $len : 0.0;
    }

    /** Untuk membandingkan transliterasi: huruf kecil, hanya huruf (spasi, tanda hubung, tanda baca dibuang). */
    public static function lettersOnly(string $text): string
    {
        return preg_replace('/[^\p{L}]/u', '', mb_strtolower($text));
    }
}
