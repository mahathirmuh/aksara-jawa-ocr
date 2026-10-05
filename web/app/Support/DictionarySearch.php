<?php

namespace App\Support;

use App\Models\DictionaryEntry;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Support\Collection;

/**
 * Pencarian di kamus kata. Semua pencocokan memakai kolom turunan (`lookup`, `gloss_text`) dan LIKE biasa, supaya
 * sama di PostgreSQL dan SQLite. Urutan hasil: kata yang persis sama, lalu yang berawalan sama, lalu yang memuatnya;
 * "balik" = kata lain yang ARTINYA memuat kata yang dicari sebagai kata utuh (mis. "rumah" -> omah, griya).
 */
final class DictionarySearch
{
    /** Di bawah panjang ini pencarian "memuat" dan pencarian balik terlalu berisik (satu-dua huruf ada di mana-mana). */
    private const MIN_CONTAINS = 3;

    private const REVERSE_LIMIT = 12;

    /** Kata per kueri yang diartikan satu per satu bila kuerinya berupa beberapa kata. */
    public const MAX_WORDS = 8;

    /**
     * @return array{matches: Collection<int, DictionaryEntry>, total: int, reverse: Collection<int, DictionaryEntry>, reverse_total: int}
     */
    public static function search(string $dictionary, string $term, int $limit, ?string $aksara = null): array
    {
        $needle = DictionaryEntry::normalize($term);
        $empty = ['matches' => collect(), 'total' => 0, 'reverse' => collect(), 'reverse_total' => 0];
        if ($needle === '' && $aksara === null) {
            return $empty;
        }
        $base = fn (): Builder => DictionaryEntry::where('dictionary', $dictionary);
        $like = self::escape($needle);
        $contains = mb_strlen($needle) >= self::MIN_CONTAINS;

        $matches = collect();
        if ($aksara !== null) {
            $matches = $base()->where('aksara', $aksara)->orderBy('id')->limit($limit)->get();
        }
        if ($needle !== '') {
            $matches = $matches->concat(self::ranked($base()->where('lookup', $needle), $limit));
            $matches = $matches->concat(self::ranked($base()->where('lookup', '<>', $needle)
                ->whereRaw("lookup LIKE ? ESCAPE '\\'", [$like.'%']), $limit));
            if ($contains) {
                $matches = $matches->concat(self::ranked($base()->whereRaw("lookup LIKE ? ESCAPE '\\'", ['%'.$like.'%'])
                    ->whereRaw("lookup NOT LIKE ? ESCAPE '\\'", [$like.'%']), $limit));
            }
        }
        $matches = $matches->unique('id')->take($limit)->values();
        if ($needle === '') {
            return ['matches' => $matches, 'total' => $matches->count()] + $empty;
        }

        $total = $base()->whereRaw("lookup LIKE ? ESCAPE '\\'", [($contains ? '%' : '').$like.'%'])->count();
        $reverse = collect();
        $reverseTotal = 0;
        if ($contains) {
            $inGloss = fn (): Builder => $base()->whereRaw("gloss_text LIKE ? ESCAPE '\\'", ['% '.$like.' %'])
                ->whereRaw("lookup NOT LIKE ? ESCAPE '\\'", ['%'.$like.'%']);
            $reverseTotal = $inGloss()->count();
            // Arti yang pendek biasanya padanan langsung ("rumah"), arti yang panjang hanya menyebut katanya.
            $reverse = $inGloss()->orderByRaw('length(gloss_text), lookup, id')->limit(self::REVERSE_LIMIT)->get();
        }

        return ['matches' => $matches, 'total' => max($total, $matches->count()), 'reverse' => $reverse, 'reverse_total' => $reverseTotal];
    }

    /**
     * Arti per kata untuk kueri berupa beberapa kata: hanya entri yang katanya persis sama.
     *
     * @param  list<string>  $words
     * @return list<array{word: string, entries: Collection<int, DictionaryEntry>}>
     */
    public static function gloss(string $dictionary, array $words, int $perWord = 3): array
    {
        $rows = [];
        foreach ($words as $word) {
            $needle = DictionaryEntry::normalize($word);
            $rows[] = ['word' => $word, 'entries' => $needle === '' ? collect()
                : self::ranked(DictionaryEntry::where('dictionary', $dictionary)->where('lookup', $needle), $perWord)];
        }

        return $rows;
    }

    /** Kata-kata sebuah kueri (huruf, angka, tanda hubung, apostrof), tanpa pengulangan, paling banyak MAX_WORDS. */
    public static function words(string $text): array
    {
        preg_match_all("/[\\p{L}\\p{M}\\p{N}][\\p{L}\\p{M}\\p{N}'’-]*/u", $text, $found);
        $seen = [];
        foreach ($found[0] as $word) {
            $seen[DictionaryEntry::normalize($word)] ??= $word;
        }
        unset($seen['']);

        return array_slice(array_values($seen), 0, self::MAX_WORDS);
    }

    /** Arti berbahasa Indonesia didahulukan, lalu kata yang lebih pendek. */
    private static function ranked(Builder $query, int $limit): Collection
    {
        return $query->orderByRaw("case gloss_lang when 'id' then 0 when 'jv' then 1 else 2 end, length(lookup), lookup, id")
            ->limit($limit)->get();
    }

    /** Karakter pola LIKE di kata yang diketik diperlakukan sebagai huruf biasa. */
    private static function escape(string $needle): string
    {
        return addcslashes($needle, '\\%_');
    }
}
