<?php

namespace App\Support;

use Normalizer;

/**
 * Latin (bahasa Jawa) -> aksara Jawa menurut aturan, pasangan Transliterator (arah sebaliknya).
 *
 * Hanya mengganti SISTEM TULISAN, tidak menerjemahkan. Ejaan Latin yang dibaca: é/è = taling, e tanpa tanda =
 * pepet (ê juga), dh/th/ng/ny = satu aksara. Hasilnya draf: teks yang tidak membedakan é dari e (kebanyakan teks
 * Latin sehari-hari, juga keluaran mesin penerjemah) akan mendapat pepet di tempat yang seharusnya taling.
 *
 * Aturan per kata:
 *  - suku kata = konsonan (+ cakra r / pengkal y) + vokal; vokal tanpa konsonan pembuka ditulis dengan ha (ꦲ);
 *  - ng, r, h penutup suku kata menjadi cecak, layar, wignyan; konsonan lain yang tidak diikuti vokal diberi
 *    pangkon, sehingga konsonan berikutnya menjadi pasangan dan konsonan penutup kata terbaca mati;
 *  - ejaan baku ($standard): rê dan lê ditulis pa cerek (ꦉ) dan nga lelet (ꦊ), konsonan + rê memakai keret (ꦽ),
 *    n di depan c/j ditulis nya (ꦥꦚ꧀ꦕꦶ "panci"), dan angka diapit pada pangkat. Keempatnya dicocokkan dengan ejaan
 *    aksara 2.134 lema kamus (Wiktionary): tanpa pengecualian untuk keret, cerek, dan nc/nj. Tanpa $standard hasilnya
 *    mengikuti konvensi korpus OCR (scripts/translit/translit.js), dipakai untuk mencocokkan aturan dengan alat lama.
 *
 * Yang sengaja tidak dilakukan: aksara murda dan swara untuk nama, pada lingsa yang dilebur dengan pangkon, dan
 * tanda tanya/seru (aksara Jawa tidak punya; dibiarkan apa adanya).
 */
final class AksaraWriter
{
    /**
     * Seberapa sering aturan ini (ejaan baku) menghasilkan ejaan aksara yang sama dengan lema kamus bahasa Jawa:
     * [sama, dari]. Dijaga tests/Unit/AksaraWriterTest terhadap database/dictionaries/jv.jsonl.gz; sisanya nama
     * beraksara murda/swara dan lema yang é-nya tidak ditandai.
     */
    public const DICTIONARY_AGREEMENT = [2154, 2212];

    private const CONSONANTS = [
        'ng' => 'ꦔ', 'ny' => 'ꦚ', 'dh' => 'ꦝ', 'th' => 'ꦛ',
        'h' => 'ꦲ', 'n' => 'ꦤ', 'c' => 'ꦕ', 'r' => 'ꦫ', 'k' => 'ꦏ', 'd' => 'ꦢ', 't' => 'ꦠ', 's' => 'ꦱ', 'w' => 'ꦮ',
        'l' => 'ꦭ', 'p' => 'ꦥ', 'j' => 'ꦗ', 'y' => 'ꦪ', 'm' => 'ꦩ', 'g' => 'ꦒ', 'b' => 'ꦧ',
        // Bunyi serapan: aksara rekan (cecak telu) untuk f, v, z; q dibaca k; x = ks.
        'f' => 'ꦥ꦳', 'v' => 'ꦮ꦳', 'z' => 'ꦗ꦳', 'q' => 'ꦏ', 'x' => 'ꦏ꧀ꦱ',
    ];

    private const VOWELS = ['a' => '', 'i' => 'ꦶ', 'u' => 'ꦸ', 'e' => 'ꦼ', 'é' => 'ꦺ', 'è' => 'ꦺ', 'o' => 'ꦺꦴ'];

    /** Penutup suku kata yang punya sandhangan sendiri (hanya sesudah vokal). */
    private const FINALS = ['ng' => 'ꦁ', 'r' => 'ꦂ', 'h' => 'ꦃ'];

    private const PUNCTUATION = [',' => '꧈', '.' => '꧉', ':' => '꧇', '(' => '꧌', ')' => '꧍', ';' => '꧈'];

    /** Ejaan Latin lain yang maknanya sama: pepet bertanda, transliterasi ilmiah (ḍ, ṭ, ṅ), vokal beraksen. */
    private const LATIN_VARIANTS = [
        'ê' => 'e', 'ě' => 'e', 'ĕ' => 'e', 'ə' => 'e', 'ë' => 'e', 'ē' => 'e',
        'å' => 'a', 'â' => 'a', 'ā' => 'a', 'á' => 'a', 'à' => 'a',
        'ô' => 'o', 'ō' => 'o', 'ó' => 'o', 'ò' => 'o',
        'î' => 'i', 'ī' => 'i', 'í' => 'i', 'ì' => 'i', 'û' => 'u', 'ū' => 'u', 'ú' => 'u', 'ù' => 'u',
        'ḍ' => 'dh', 'ṭ' => 'th', 'ṅ' => 'ng', 'ñ' => 'ny', 'ś' => 's', 'ṣ' => 's',
        "'" => '', '’' => '', '‘' => '', '`' => '', 'ʼ' => '',
    ];

    private const PANGKON = '꧀';
    private const CAKRA = 'ꦿ';
    private const PENGKAL = 'ꦾ';
    private const KERET = 'ꦽ';
    private const PA_CEREK = 'ꦉ';
    private const NGA_LELET = 'ꦊ';
    private const PADA_PANGKAT = '꧇';

    public static function fromLatin(string $text, bool $standard = true): string
    {
        $text = mb_strtolower(Normalizer::normalize($text, Normalizer::FORM_C) ?: $text);
        $text = strtr($text, self::LATIN_VARIANTS);

        // Potongan: kata, bilangan (boleh berpemisah titik/koma/titik dua: 2,5 atau 10:30), atau satu karakter lain.
        $aksara = preg_replace_callback('/[a-zéè]+|[0-9]+(?:[.,:][0-9]+)*|(?P<join>(?<=[a-zéè])[-‐‑–—](?=[a-zéè]))|./us', function (array $m) use ($standard) {
            $piece = $m[0];
            // Tanda hubung di antara dua kata (kata ulang, gabungan): aksara Jawa tidak memakainya; kedua kata ditulis berurutan.
            if (($m['join'] ?? '') !== '') {
                return '';
            }
            if (preg_match('/^[a-zéè]/u', $piece)) {
                return self::word($piece, $standard);
            }
            if (ctype_digit($piece[0])) {
                $digits = preg_replace_callback('/[0-9]/', fn (array $d) => mb_chr(0xA9D0 + (int) $d[0]), $piece);

                return $standard ? self::PADA_PANGKAT.$digits.self::PADA_PANGKAT : $digits;
            }

            return self::PUNCTUATION[$piece] ?? $piece;
        }, $text) ?? $text;

        // Titik dua sesudah bilangan ("1945: ...") jatuh tepat di pada pangkat penutup bilangan itu: satu saja cukup.
        return preg_replace('/'.self::PADA_PANGKAT.'{2,}/u', self::PADA_PANGKAT, $aksara) ?? $aksara;
    }

    private static function word(string $word, bool $standard): string
    {
        preg_match_all('/ng|ny|dh|th|[bcdfghjklmnpqrstvwxyz]|[aiueoéè]/u', $word, $found);
        $tokens = $found[0];
        $count = count($tokens);
        $out = '';
        $afterVowel = false;      // token sebelumnya menutup dengan vokal: ng/r/h berikutnya bisa jadi penutup suku kata
        $afterPangkon = false;    // aksara berikutnya menjadi pasangan
        for ($i = 0; $i < $count;) {
            $token = $tokens[$i];
            if (isset(self::VOWELS[$token])) {
                $out .= 'ꦲ'.self::VOWELS[$token];
                [$afterVowel, $afterPangkon] = [true, false];
                $i++;

                continue;
            }
            $next = $tokens[$i + 1] ?? null;
            $afterNext = $tokens[$i + 2] ?? null;
            $closes = $afterVowel && isset(self::FINALS[$token]);
            if ($next !== null && isset(self::VOWELS[$next])) {
                $out .= self::syllable($token, '', $next, $standard, $afterPangkon);
                [$afterVowel, $afterPangkon] = [true, false];
                $i += 2;
            } elseif (! $closes && ($next === 'r' || $next === 'y') && $afterNext !== null && isset(self::VOWELS[$afterNext])) {
                $out .= self::syllable($token, $next, $afterNext, $standard, $afterPangkon);
                [$afterVowel, $afterPangkon] = [true, false];
                $i += 3;
            } elseif ($closes) {
                $out .= self::FINALS[$token];
                [$afterVowel, $afterPangkon] = [false, false];
                $i++;
            } else {
                // Sengau di depan c/j ditulis nya: panci = pa-nya-pangkon-ci.
                $nasal = $standard && $token === 'n' && ($next === 'c' || $next === 'j') ? 'ny' : $token;
                $out .= self::CONSONANTS[$nasal].self::PANGKON;
                [$afterVowel, $afterPangkon] = [false, true];
                $i++;
            }
        }

        return $out;
    }

    private static function syllable(string $consonant, string $medial, string $vowel, bool $standard, bool $afterPangkon): string
    {
        $base = self::CONSONANTS[$consonant];
        $pepet = $vowel === 'e';
        if ($medial === 'r') {
            return $standard && $pepet ? $base.self::KERET : $base.self::CAKRA.self::VOWELS[$vowel];
        }
        if ($medial === 'y') {
            return $base.self::PENGKAL.self::VOWELS[$vowel];
        }
        // rê dan lê punya aksara sendiri; sebagai pasangan tetap ditulis ra/la berpepet.
        if ($standard && $pepet && ! $afterPangkon && ($consonant === 'r' || $consonant === 'l')) {
            return $consonant === 'r' ? self::PA_CEREK : self::NGA_LELET;
        }

        return $base.self::VOWELS[$vowel];
    }
}
