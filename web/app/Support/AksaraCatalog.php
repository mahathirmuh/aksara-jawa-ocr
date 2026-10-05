<?php

namespace App\Support;

use IntlChar;

/**
 * Kamus aksara: codepoint yang dibaca model OCR beserta nama, jenis, dan bacaan Latin draf.
 *
 * Sumber utama: "charset" di data/tokenizer.json repo OCR (urutan kelas keluaran model). Bila berkas itu tidak ada
 * (mis. saat test, yang tidak membawa repo OCR), dipakai semua codepoint terisi blok Javanese U+A980–U+A9DF ditambah
 * spasi; saat ini isinya sama persis dengan charset tokenizer.
 *
 * Bacaan Latin berasal dari Transliterator (draf bantu baca), bukan dari kamus baku dan bukan hasil OCR.
 */
final class AksaraCatalog
{
    /** Jenis menurut kategori umum Unicode, urut seperti ditampilkan. */
    public const KINDS = [
        'aksara' => 'Aksara',
        'sandhangan' => 'Sandhangan dan tanda',
        'angka' => 'Angka',
        'pada' => 'Pada (tanda baca)',
    ];

    /** Nama Unicode yang masih berbahasa Inggris, diindonesiakan untuk kamus. */
    private const NAMES = [
        'left rerenggan' => 'rerenggan kiri',
        'right rerenggan' => 'rerenggan kanan',
        'turned pada piseleh' => 'pada piseleh terbalik',
    ];

    /** Aksara ka: alas untuk memperlihatkan sandhangan dan supaya aksara yang diuji tidak dianggap awal kata. */
    private const BASE = 0xA98F;

    /**
     * @return array{source: string, classes: int, entries: list<array{char: string, code: string, name: string, kind: string, latin: string}>}
     */
    public static function load(): array
    {
        [$codes, $source] = self::codepoints();
        $entries = [];
        foreach ($codes as $cp) {
            if ($cp === 0x20) {
                continue;   // spasi ikut dihitung sebagai kelas, tetapi bukan aksara
            }
            $ch = mb_chr($cp);
            $kind = self::kind($cp);
            $entries[] = [
                'char' => $ch,
                'code' => sprintf('U+%04X', $cp),
                'name' => strtr(aksara_name($ch), self::NAMES),
                'kind' => $kind,
                'latin' => self::reading($ch, $cp, $kind),
            ];
        }

        // +1: kelas kosong (blank) CTC di indeks 0.
        return ['source' => $source, 'classes' => count($codes) + 1, 'entries' => $entries];
    }

    /**
     * Berapa kali tiap karakter muncul di kumpulan teks (mis. label uji).
     *
     * @param  iterable<string|null>  $texts
     * @return array<string, int>
     */
    public static function countIn(iterable $texts): array
    {
        $counts = [];
        foreach ($texts as $text) {
            foreach (mb_str_split((string) $text) as $ch) {
                $counts[$ch] = ($counts[$ch] ?? 0) + 1;
            }
        }

        return $counts;
    }

    public static function hasJavanese(string $text): bool
    {
        return preg_match('/[\x{A980}-\x{A9DF}]/u', $text) === 1;
    }

    /** @return array{0: list<int>, 1: string} */
    private static function codepoints(): array
    {
        $path = rtrim((string) config('aksara.ocr_repo'), '/\\').DIRECTORY_SEPARATOR.'data'.DIRECTORY_SEPARATOR.'tokenizer.json';
        if (is_file($path)) {
            $charset = json_decode((string) file_get_contents($path), true)['charset'] ?? null;
            $codes = [];
            foreach (is_array($charset) ? $charset : [] as $code) {
                if (is_string($code) && preg_match('/^U\+([0-9A-F]{4,6})$/i', $code, $m)) {
                    $codes[] = hexdec($m[1]);
                }
            }
            if ($codes) {
                return [$codes, 'tokenizer'];
            }
        }

        $codes = [0x20];
        for ($cp = 0xA980; $cp <= 0xA9DF; $cp++) {
            if (IntlChar::isdefined($cp)) {
                $codes[] = $cp;
            }
        }

        return [$codes, 'unicode'];
    }

    private static function kind(int $cp): string
    {
        return match (IntlChar::charType($cp)) {
            IntlChar::CHAR_CATEGORY_NON_SPACING_MARK, IntlChar::CHAR_CATEGORY_COMBINING_SPACING_MARK => 'sandhangan',
            IntlChar::CHAR_CATEGORY_DECIMAL_DIGIT_NUMBER => 'angka',
            IntlChar::CHAR_CATEGORY_OTHER_LETTER => 'aksara',
            default => 'pada',   // tanda baca (Po) dan pangrangkep (Lm)
        };
    }

    /**
     * Bacaan Latin draf; string kosong bila Transliterator tidak punya aturannya.
     * Sandhangan dibaca di atas aksara ka (wulu -> "ki"), karena tanda tidak punya bunyi tanpa aksara dasar.
     */
    private static function reading(string $ch, int $cp, string $kind): string
    {
        $base = mb_chr(self::BASE);

        if ($kind === 'angka') {
            return (string) ($cp - 0xA9D0);
        }
        if ($kind === 'sandhangan') {
            $latin = Transliterator::toLatin($base.$ch);

            return self::hasJavanese($latin) ? '' : $latin;
        }
        if ($kind === 'aksara') {
            // Didahului ka supaya ꦲ dibaca "ha", bukan vokal awal kata; "ka" di depan lalu dibuang.
            $latin = Transliterator::toLatin($base.$ch);

            return self::hasJavanese($latin) || ! str_starts_with($latin, 'ka') ? '' : substr($latin, 2);
        }
        $latin = Transliterator::toLatin($ch);

        return self::hasJavanese($latin) ? '' : $latin;
    }
}
