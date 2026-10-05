<?php

namespace App\Support;

/**
 * Memulihkan tanda é/è (taling) pada bahasa Jawa beraksara Latin yang menulis semua bunyi itu "e".
 *
 * Aksara Jawa membedakan taling dari pepet, tetapi teks Latin sehari-hari dan keluaran mesin penerjemah jarang
 * menandainya. Leksikon database/dictionaries/jv-taling.json (kata tanpa tanda -> ejaan bertandanya, dibangun
 * tools/taling_lexicon.py dari Wikipedia bahasa Jawa) dipakai per kata. Kata yang sudah bertanda dan kata di luar
 * leksikon tidak diubah, jadi sisanya tetap dibaca pepet. Terukur pada artikel yang ditahan: kata ber-e yang
 * ejaannya benar naik dari 53% menjadi 90% (angka pastinya di kunci "meta" berkas leksikon).
 */
final class TalingRestorer
{
    /** @var array<string, array<string, string>> leksikon per jalur berkas */
    private static array $loaded = [];

    /**
     * @return array{text: string, changed: array<string, string>} teks yang dipulihkan + kata yang berubah (asal => hasil)
     */
    public static function restore(string $text): array
    {
        $words = self::words();
        $changed = [];
        if ($words === []) {
            return ['text' => $text, 'changed' => $changed];
        }
        $restored = preg_replace_callback('/\p{L}+/u', function (array $m) use ($words, &$changed) {
            $word = $m[0];
            $marked = $words[mb_strtolower($word)] ?? null;
            // Hanya kata yang seluruhnya tanpa tanda: kata bertanda adalah pilihan penulisnya.
            if ($marked === null || mb_strlen($marked) !== mb_strlen($word) || preg_match('/[éèÉÈ]/u', $word)) {
                return $word;
            }
            // Tanda dipindahkan huruf demi huruf supaya huruf besar di teks asal tetap.
            $out = '';
            foreach (mb_str_split($word) as $i => $char) {
                $target = mb_substr($marked, $i, 1);
                $out .= $target === 'é' || $target === 'è' ? ($char === 'E' ? mb_strtoupper($target) : $target) : $char;
            }
            if ($out !== $word) {
                $changed[$word] = $out;
            }

            return $out;
        }, $text);

        return ['text' => $restored ?? $text, 'changed' => $changed];
    }

    /** Ukuran dan asal leksikon (kunci "meta" berkasnya), untuk keterangan di halaman; null bila berkas tidak ada. */
    public static function meta(): ?array
    {
        $path = config('aksara.taling_lexicon');
        if (! is_file($path)) {
            return null;
        }
        $data = json_decode(file_get_contents($path), true) ?: [];

        return ($data['meta'] ?? []) + ['words' => count($data['words'] ?? [])];
    }

    private static function words(): array
    {
        $path = config('aksara.taling_lexicon');

        return self::$loaded[$path] ??= is_file($path) ? (json_decode(file_get_contents($path), true)['words'] ?? []) : [];
    }
}
