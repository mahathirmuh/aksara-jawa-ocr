<?php

namespace App\Support;

/**
 * Tahap 4 alur: tingkat tutur bahasa Jawa (ngoko / madya / krama / campur) dari teks Latin.
 *
 * Baseline leksikon yang transparan (PLAN.md §11: "leksikon + fitur afiks"): setiap keputusan membawa
 * daftar kata penanda yang cocok, jadi bisa diperiksa pembaca. Krama inggil dihitung sebagai krama.
 * Belum dievaluasi terhadap label manusia sampai ada data berlabel (lihat halaman Penjelajah).
 *
 * Teks tanpa spasi (transliterasi keluaran OCR; 87% label NusaAksara juga tanpa spasi di aksaranya)
 * tidak punya batas kata: hanya penanda panjang (>= 5 huruf) dan afiks krama yang dicari, dan
 * keyakinannya selalu "rendah".
 */
final class SpeechLevel
{
    public const LEVELS = ['ngoko', 'madya', 'krama', 'campur'];

    /** Kata penanda per tingkat. Kata netral (ing, lan, bapak, ...) sengaja tidak dimasukkan. */
    private const MARKERS = [
        'ngoko' => [
            'aku', 'kowe', 'iki', 'iku', 'kuwi', 'kae', 'ora', 'wis', 'uwis', 'arep', 'ana', 'sing', 'saka', 'marang',
            'karo', 'banjur', 'ya', 'iya', 'apa', 'piye', 'kepriye', 'endi', 'kok', 'dhewe', 'dheweke', 'mangan', 'turu',
            'lunga', 'weruh', 'ngerti', 'bisa', 'isa', 'durung', 'dudu', 'uga', 'kabeh', 'akeh', 'cilik', 'gedhe',
            'omah', 'mung', 'wae', 'bae', 'maneh', 'menyang', 'mengko', 'saiki', 'kene', 'kana', 'kono', 'ngene',
            'ngono', 'lagi', 'bakal', 'amarga', 'merga', 'yen', 'nek', 'dadi', 'weneh', 'menehi', 'takon', 'omong',
            'ngomong', 'kandha', 'mulih', 'teka', 'gawe', 'duwe', 'ndeleng', 'krungu', 'sapa', 'pira', 'kapan',
        ],
        'madya' => [
            'niki', 'niku', 'nika', 'napa', 'ajeng', 'empun', 'mpun', 'mawon', 'sampeyan', 'onten', 'teng', 'mriki',
            'mrika', 'ngoten', 'ngaten', 'pripun', 'riyin', 'nggih', 'kalih', 'kajenge',
        ],
        'krama' => [
            'kula', 'kawula', 'panjenengan', 'menika', 'punika', 'mboten', 'boten', 'sampun', 'badhe', 'wonten',
            'ingkang', 'saking', 'dhateng', 'kaliyan', 'lajeng', 'inggih', 'menapa', 'punapa', 'pundi', 'piyambak',
            'piyambakipun', 'nedha', 'tilem', 'kesah', 'sumerep', 'mangertos', 'saged', 'dereng', 'sanes', 'ugi',
            'sedaya', 'kathah', 'alit', 'ageng', 'griya', 'namung', 'kemawon', 'malih', 'sami', 'ananging',
            'amargi', 'awit', 'bilih', 'menawi', 'manawi', 'sakedhik', 'samenika', 'sakmenika', 'sinten', 'rumiyin',
            'mekaten', 'makaten', 'kados', 'mangkeh', 'wau', 'tuwin', 'saha', 'dados', 'nyuwun', 'maringi', 'sae',
            'ngendikan', 'matur', 'wangsul', 'dugi', 'ngantos', 'dumugi', 'ngriki', 'ngrika', 'nalika', 'kalawau',
            // krama inggil
            'dhahar', 'sare', 'tindak', 'priksa', 'pirsa', 'ngendika', 'dhawuh', 'rawuh', 'kondur', 'asma', 'yuswa',
            'garwa', 'putra', 'ngasta', 'paring', 'kersa', 'dalem', 'panjenenganipun', 'kagem', 'ngunjuk', 'sumanggem',
        ],
    ];

    /** Urutan pencocokan bila satu kata kelak masuk lebih dari satu daftar: tingkat pertama menang. */
    private const ORDER = ['krama', 'madya', 'ngoko'];

    /**
     * Akurasi terhadap label manusia. $pairs: [[label manusia, prediksi], ...].
     * Prediksi null (tanpa penanda) dihitung salah, bukan dibuang, supaya angka tidak menipu.
     */
    public static function evaluate(array $pairs): array
    {
        $n = count($pairs);
        $confusion = [];
        $correct = 0;
        foreach ($pairs as [$human, $predicted]) {
            $predicted ??= 'tak tentu';
            $confusion[$human][$predicted] = ($confusion[$human][$predicted] ?? 0) + 1;
            $correct += $human === $predicted ? 1 : 0;
        }

        return ['lines' => $n, 'accuracy' => $n ? $correct / $n : null, 'confusion' => $confusion];
    }

    public static function classify(?string $latin): array
    {
        $text = self::normalise((string) $latin);
        $words = preg_split('/[^a-z]+/', $text, -1, PREG_SPLIT_NO_EMPTY);
        $spaced = count($words) > 1 && str_contains(trim((string) $latin), ' ');
        $evidence = [];

        if ($spaced) {
            foreach ($words as $word) {
                if ($level = self::levelOf($word)) {
                    $evidence[] = [$word, $level];
                } elseif (strlen($word) > 5 && (str_ends_with($word, 'ipun') || str_starts_with($word, 'dipun'))) {
                    $evidence[] = [$word, 'krama'];   // afiks krama: -ipun, dipun-
                }
            }
        } else {
            $flat = implode('', $words);
            foreach (self::ORDER as $level) {
                foreach (self::MARKERS[$level] as $marker) {
                    if (strlen($marker) >= 5 && ($n = substr_count($flat, $marker)) > 0) {
                        array_push($evidence, ...array_fill(0, $n, [$marker, $level]));
                        $flat = str_replace($marker, '|', $flat);   // jangan hitung dua kali di tingkat lain
                    }
                }
            }
            if (($n = substr_count($flat, 'ipun')) > 0) {
                array_push($evidence, ...array_fill(0, $n, ['-ipun', 'krama']));
            }
        }

        $counts = array_fill_keys(['ngoko', 'madya', 'krama'], 0);
        foreach ($evidence as [, $level]) {
            $counts[$level]++;
        }

        return [
            'level' => self::decide($counts),
            'counts' => $counts,
            'evidence' => $evidence,
            'confidence' => $spaced && array_sum($counts) >= 3 ? 'cukup' : 'rendah',
            'spaced' => $spaced,
        ];
    }

    private static function levelOf(string $word): ?string
    {
        foreach (self::ORDER as $level) {
            if (in_array($word, self::MARKERS[$level], true)) {
                return $level;
            }
        }

        return null;
    }

    /** null bila tidak ada penanda sama sekali. */
    private static function decide(array $counts): ?string
    {
        $total = array_sum($counts);
        if ($total === 0) {
            return null;
        }
        $polite = $counts['krama'] + $counts['madya'];
        // Campur: ngoko dan (madya/krama) sama-sama bermakna, masing-masing >= 25% penanda dan >= 2 kata.
        if ($counts['ngoko'] >= 2 && $polite >= 2 && min($counts['ngoko'], $polite) / $total >= 0.25) {
            return 'campur';
        }
        arsort($counts);

        return array_key_first($counts);
    }

    private static function normalise(string $text): string
    {
        $text = mb_strtolower($text);

        return strtr($text, ['é' => 'e', 'è' => 'e', 'ê' => 'e', 'ë' => 'e', 'ô' => 'o', 'â' => 'a', 'ā' => 'a', 'ī' => 'i', 'ū' => 'u']);
    }
}
