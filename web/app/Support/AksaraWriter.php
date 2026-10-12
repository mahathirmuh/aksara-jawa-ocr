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
 * Nama (pilihan, $names; bawaannya mati supaya hasil lama dan angka kecocokan kamus tidak berubah). Kata berhuruf
 * besar ditulis berlapis:
 *  1. nama yang ada di kamus dengan ejaan aksara bermurda/berswara memakai ejaan kamus itu apa adanya (Bima,
 *     Suwarnabhumi, Arjuna), juga di awal kalimat, kecuali di awal kalimat bila kamus juga memuatnya sebagai kata
 *     biasa berhuruf kecil (agung, ukir, arga): di situ huruf besarnya tidak membedakan nama dari kata biasa;
 *  2. kata berhuruf besar lain yang bukan kata pertama kalimat dianggap nama: berawalan vokal -> aksara swara (A, I,
 *     U, É, O; pepet tidak punya), selain itu SATU aksara murda pada konsonan pertama yang punya bentuk murda (na,
 *     ka, ta, sa, pa, nya, ga, ba) dan tidak menutup suku kata (Thomas, Marwan: tanpa murda). Lapis ini kaidah umum, BELUM diperiksa pembaca aksara: kamus hanya memuat belasan
 *     nama bermurda/berswara, terlalu sedikit untuk menurunkan aturannya, dan di situ nama berswara tidak diberi murda.
 * Huruf besar di awal kalimat bukan tanda nama, jadi kata pertama kalimat hanya kena lapis 1.
 *
 * Yang sengaja tidak dilakukan: pada lingsa yang dilebur dengan pangkon, dan tanda tanya/seru (aksara Jawa tidak
 * punya; dibiarkan apa adanya).
 */
final class AksaraWriter
{
    /**
     * Seberapa sering aturan ini (ejaan baku) menghasilkan ejaan aksara yang sama dengan lema kamus bahasa Jawa:
     * [sama, dari]. Dihitung pada lema berejaan baku; entri yang hanya varian ejaan ("dated spelling of ...",
     * "nonstandard spelling of ...", "romanization of ...") tidak ikut, karena ejaan Latinnya memang bukan ejaan baku.
     * Dijaga tests/Unit/AksaraWriterTest terhadap database/dictionaries/jv.jsonl.gz; sisanya nama beraksara
     * murda/swara dan lema yang é-nya tidak ditandai.
     */
    public const DICTIONARY_AGREEMENT = [2173, 2231];

    /** Arti pertama entri kamus yang hanya menunjuk ejaan lain (lihat DICTIONARY_AGREEMENT). */
    public const SPELLING_VARIANT = '/^(?:\([^()]*\) )?(?:(?:\w+ )?(?:spelling|form) of |romanization of )/u';

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

    /** Konsonan yang punya bentuk murda. */
    private const MURDA = ['n' => 'ꦟ', 'k' => 'ꦑ', 't' => 'ꦡ', 's' => 'ꦯ', 'p' => 'ꦦ', 'ny' => 'ꦘ', 'g' => 'ꦓ', 'b' => 'ꦨ'];

    /** Vokal pembuka nama: aksara swara. Pepet tidak punya aksara swara. */
    private const SWARA = ['a' => 'ꦄ', 'i' => 'ꦆ', 'u' => 'ꦈ', 'é' => 'ꦌ', 'è' => 'ꦌ', 'o' => 'ꦎ'];

    /** Penanda "kata ini berhuruf besar", disisipkan sebelum teks dijadikan huruf kecil. */
    private const CAPITAL = "\x01";

    private const DICTIONARY = __DIR__.'/../../database/dictionaries/jv.jsonl.gz';

    /** @var array<string, string>|null nama di kamus (huruf kecil) => ejaan aksaranya */
    private static ?array $names = null;

    /** @var array<string, true>|null nama di kamus yang juga lema biasa berhuruf kecil di kamus yang sama */
    private static ?array $ordinary = null;

    public static function fromLatin(string $text, bool $standard = true, bool $names = false): string
    {
        return self::write($text, $standard, $names)['aksara'];
    }

    /**
     * Seperti fromLatin() dengan nama, ditambah kata mana yang ditulis sebagai nama dan atas dasar apa:
     * 'dictionary' = ejaan kamus, 'rule' = kaidah murda/swara (belum diperiksa pembaca aksara).
     *
     * @return array{aksara: string, dictionary: list<string>, rule: list<string>}
     */
    public static function write(string $text, bool $standard = true, bool $names = true): array
    {
        $text = Normalizer::normalize($text, Normalizer::FORM_C) ?: $text;
        $text = str_replace(self::CAPITAL, '', $text);
        if ($names) {
            $text = preg_replace('/(?<![\p{L}\p{M}])(?=\p{Lu})/u', self::CAPITAL, $text) ?? $text;
        }
        $text = strtr(mb_strtolower($text), self::LATIN_VARIANTS);
        $known = $names ? self::dictionaryNames() : [];
        $ordinary = $names ? self::ordinaryNames() : [];
        $found = ['dictionary' => [], 'rule' => []];
        $first = true;   // kata berikutnya adalah kata pertama kalimat

        // Potongan: kata, bilangan (boleh berpemisah titik/koma/titik dua: 2,5 atau 10:30), atau satu karakter lain.
        $pattern = '/\x01?[a-zéè]+|[0-9]+(?:[.,:][0-9]+)*|(?P<join>(?<=[a-zéè])[-‐‑–—](?=\x01?[a-zéè]))|./us';
        $aksara = preg_replace_callback($pattern, function (array $m) use ($standard, $known, $ordinary, &$found, &$first) {
            $piece = $m[0];
            // Tanda hubung di antara dua kata (kata ulang, gabungan): aksara Jawa tidak memakainya; kedua kata ditulis berurutan.
            if (($m['join'] ?? '') !== '') {
                return '';
            }
            $capital = str_starts_with($piece, self::CAPITAL);
            $piece = $capital ? substr($piece, 1) : $piece;
            if ($piece === '') {
                return '';
            }
            if (preg_match('/^[a-zéè]/u', $piece)) {
                [$opens, $first] = [$first, false];
                // Di awal kalimat huruf besar bukan tanda nama: kata yang juga kata biasa di kamus ditulis biasa.
                if ($capital && isset($known[$piece]) && ! ($opens && isset($ordinary[$piece]))) {
                    $found['dictionary'][] = $piece;

                    return $known[$piece];
                }
                $plain = self::word($piece, $standard);
                if ($capital && ! $opens) {
                    $named = self::word($piece, $standard, true);
                    // Nama tanpa vokal pembuka berswara dan tanpa konsonan bermurda (Rara, Laila) tertulis sama.
                    if ($named !== $plain) {
                        $found['rule'][] = $piece;
                    }

                    return $named;
                }

                return $plain;
            }
            if (ctype_digit($piece[0])) {
                $first = false;
                $digits = preg_replace_callback('/[0-9]/', fn (array $d) => mb_chr(0xA9D0 + (int) $d[0]), $piece);

                return $standard ? self::PADA_PANGKAT.$digits.self::PADA_PANGKAT : $digits;
            }
            if (in_array($piece, ['.', '!', '?', "\n"], true)) {
                $first = true;
            }

            return self::PUNCTUATION[$piece] ?? $piece;
        }, $text) ?? $text;

        // Titik dua sesudah bilangan ("1945: ...") jatuh tepat di pada pangkat penutup bilangan itu: satu saja cukup.
        $aksara = preg_replace('/'.self::PADA_PANGKAT.'{2,}/u', self::PADA_PANGKAT, $aksara) ?? $aksara;

        return ['aksara' => $aksara, 'dictionary' => array_values(array_unique($found['dictionary'])),
            'rule' => array_values(array_unique($found['rule']))];
    }

    /**
     * Nama satu kata di kamus bahasa Jawa (lema berhuruf besar) yang ejaan aksaranya memakai aksara murda atau swara.
     *
     * @return array<string, string>
     */
    public static function dictionaryNames(): array
    {
        if (self::$names !== null) {
            return self::$names;
        }
        self::$names = [];
        $plain = [];
        $special = '/['.implode('', self::MURDA).implode('', array_unique(self::SWARA)).']/u';
        foreach (is_file(self::DICTIONARY) ? gzfile(self::DICTIONARY) : [] as $line) {
            $entry = json_decode($line, true);
            $word = Normalizer::normalize((string) ($entry['word'] ?? ''), Normalizer::FORM_C) ?: '';
            $aksara = (string) ($entry['aksara'] ?? '');
            if (preg_match('/^\p{Ll}[\p{L}\p{M}]*$/u', $word)) {
                $plain[strtr($word, self::LATIN_VARIANTS)] = true;
            }
            if ($aksara === '' || ! preg_match('/^\p{Lu}[\p{L}\p{M}]*$/u', $word) || ! preg_match($special, $aksara)) {
                continue;
            }
            self::$names[strtr(mb_strtolower($word), self::LATIN_VARIANTS)] ??= $aksara;
        }
        self::$ordinary = array_intersect_key($plain, self::$names);

        return self::$names;
    }

    /**
     * Nama di dictionaryNames() yang di kamus juga lema biasa berhuruf kecil (agung, ukir, arga): di awal kalimat
     * kata-kata itu tidak ditulis sebagai nama.
     *
     * @return array<string, true>
     */
    public static function ordinaryNames(): array
    {
        self::dictionaryNames();

        return self::$ordinary ?? [];
    }

    private static function word(string $word, bool $standard, bool $name = false): string
    {
        preg_match_all('/ng|ny|dh|th|[bcdfghjklmnpqrstvwxyz]|[aiueoéè]/u', $word, $found);
        $tokens = $found[0];
        $count = count($tokens);
        $out = '';
        $afterVowel = false;      // token sebelumnya menutup dengan vokal: ng/r/h berikutnya bisa jadi penutup suku kata
        $afterPangkon = false;    // aksara berikutnya menjadi pasangan
        // Nama: vokal pembuka -> aksara swara (lalu tanpa murda); selain itu satu murda pada konsonan pertama yang punya.
        $murda = $name;
        for ($i = 0; $i < $count;) {
            $token = $tokens[$i];
            if (isset(self::VOWELS[$token])) {
                if ($name && $i === 0 && isset(self::SWARA[$token])) {
                    $out .= self::SWARA[$token];
                    [$afterVowel, $afterPangkon, $murda] = [true, false, false];
                    $i++;

                    continue;
                }
                $out .= 'ꦲ'.self::VOWELS[$token];
                [$afterVowel, $afterPangkon] = [true, false];
                $i++;

                continue;
            }
            $next = $tokens[$i + 1] ?? null;
            $afterNext = $tokens[$i + 2] ?? null;
            $closes = $afterVowel && isset(self::FINALS[$token]);
            // Murda hanya pada konsonan yang membuka suku kata (diikuti vokal, atau gugus di awal kata: Njoto), bukan
            // pada konsonan penutup suku kata (s di Thomas, n di Marwan).
            $opensSyllable = ($next !== null && isset(self::VOWELS[$next]))
                || (($next === 'r' || $next === 'y') && $afterNext !== null && isset(self::VOWELS[$afterNext]));
            $big = $murda && ! $closes && isset(self::MURDA[$token]) && ($opensSyllable || ! $afterVowel);
            $murda = $murda && ! $big;
            if ($next !== null && isset(self::VOWELS[$next])) {
                $out .= self::syllable($token, '', $next, $standard, $afterPangkon, $big);
                [$afterVowel, $afterPangkon] = [true, false];
                $i += 2;
            } elseif (! $closes && ($next === 'r' || $next === 'y') && $afterNext !== null && isset(self::VOWELS[$afterNext])) {
                $out .= self::syllable($token, $next, $afterNext, $standard, $afterPangkon, $big);
                [$afterVowel, $afterPangkon] = [true, false];
                $i += 3;
            } elseif ($closes) {
                $out .= self::FINALS[$token];
                [$afterVowel, $afterPangkon] = [false, false];
                $i++;
            } else {
                // Sengau di depan c/j ditulis nya: panci = pa-nya-pangkon-ci.
                $nasal = $standard && $token === 'n' && ($next === 'c' || $next === 'j') ? 'ny' : $token;
                $out .= ($big ? self::MURDA[$nasal] : self::CONSONANTS[$nasal]).self::PANGKON;
                [$afterVowel, $afterPangkon] = [false, true];
                $i++;
            }
        }

        return $out;
    }

    private static function syllable(string $consonant, string $medial, string $vowel, bool $standard, bool $afterPangkon, bool $murda = false): string
    {
        $base = $murda ? self::MURDA[$consonant] : self::CONSONANTS[$consonant];
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
