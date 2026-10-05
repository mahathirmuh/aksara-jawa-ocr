<?php

namespace App\Support;

use App\Models\DictionaryEntry;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\DB;

/**
 * Pencarian di kamus kata. Semua pencocokan memakai kolom turunan (`lookup`, `gloss_text`) dan LIKE biasa, supaya
 * sama di PostgreSQL dan SQLite. Urutan hasil: kata yang persis sama, lalu yang berawalan sama, lalu yang memuatnya;
 * "balik" = kata lain yang ARTINYA memuat kata yang dicari sebagai kata utuh (mis. "house" -> omah, griya).
 */
final class DictionarySearch
{
    /** Di bawah panjang ini pencarian "memuat" dan pencarian balik terlalu berisik (satu-dua huruf ada di mana-mana). */
    private const MIN_CONTAINS = 3;

    private const REVERSE_LIMIT = 12;

    /** Kata per kueri yang diartikan satu per satu bila kuerinya berupa beberapa kata. */
    public const MAX_WORDS = 8;

    /**
     * @param  string  $term  kata atau frasa seperti diketik (boleh berawalan tanda hubung: "-an", "ke- -an")
     * @param  string|null  $aksara  ejaan aksara yang juga dicocokkan langsung (kueri beraksara, kamus Jawa)
     * @return array{matches: Collection<int, DictionaryEntry>, total: int, reverse: Collection<int, DictionaryEntry>, reverse_total: int}
     */
    public static function search(string $dictionary, string $term, int $limit, ?string $aksara = null): array
    {
        $needle = DictionaryEntry::normalize(self::unquote($term));
        $empty = ['matches' => collect(), 'total' => 0, 'reverse' => collect(), 'reverse_total' => 0];
        if ($needle === '' && $aksara === null) {
            return $empty;
        }
        $base = fn (): Builder => DictionaryEntry::where('dictionary', $dictionary);
        $like = self::escape($needle);
        $contains = mb_strlen($needle) >= self::MIN_CONTAINS;
        // Entri yang katanya cocok: memuat kata itu (>= 3 huruf) atau berawalan kata itu.
        $byWord = fn (Builder $query): Builder => $query->whereRaw("lookup LIKE ? ESCAPE '\\'", [($contains ? '%' : '').$like.'%']);
        // Entri yang ejaan aksaranya persis kueri; null aman karena perbandingan dengan NULL tidak pernah benar.
        $bySpelling = fn (Builder $query): Builder => $query->where('aksara', $aksara);

        $matches = $aksara !== null ? self::ranked($bySpelling($base()), $limit) : collect();
        if ($needle !== '') {
            $matches = $matches->concat(self::ranked($base()->where('lookup', $needle), $limit));
            $matches = $matches->concat(self::ranked($base()->where('lookup', '<>', $needle)
                ->whereRaw("lookup LIKE ? ESCAPE '\\'", [$like.'%']), $limit));
            if ($contains) {
                $matches = $matches->concat(self::ranked($byWord($base())->whereRaw("lookup NOT LIKE ? ESCAPE '\\'", [$like.'%']), $limit));
            }
        }
        $matches = $matches->unique('id')->take($limit)->values();

        // Jumlah = gabungan cocok-kata dan cocok-ejaan, tanpa menghitung dua kali entri yang kena keduanya.
        $total = $needle !== '' ? $byWord($base())->count() : 0;
        if ($aksara !== null) {
            $spelled = $bySpelling($base());
            $total += $needle !== '' ? $spelled->whereRaw("lookup NOT LIKE ? ESCAPE '\\'", [($contains ? '%' : '').$like.'%'])->count() : $spelled->count();
        }

        $reverse = collect();
        $reverseTotal = 0;
        // Arti disimpan tanpa tanda baca ("son in law"), jadi kata yang dicari diperlakukan sama sebelum dicocokkan.
        // Syarat panjangnya berlaku untuk bentuk itu: "-an" menjadi "an", yang ada di hampir semua arti berbahasa Inggris.
        $inGloss = trim(DictionaryEntry::glossText([$needle]));
        if (mb_strlen($inGloss) >= self::MIN_CONTAINS) {
            $found = fn (): Builder => $base()->whereRaw("gloss_text LIKE ? ESCAPE '\\'", ['% '.self::escape($inGloss).' %'])
                ->whereRaw("lookup NOT LIKE ? ESCAPE '\\'", ['%'.$like.'%'])
                ->when($aksara !== null, fn (Builder $query) => $query->where(fn (Builder $q) => $q->whereNull('aksara')->orWhere('aksara', '<>', $aksara)));
            $reverseTotal = $found()->count();
            // Arti yang pendek biasanya padanan langsung ("house"), arti yang panjang hanya menyebut katanya.
            $reverse = $found()->orderByRaw('length(gloss_text), '.self::lookupColumn().', id')->limit(self::REVERSE_LIMIT)->get();
        }

        return ['matches' => $matches, 'total' => $total, 'reverse' => $reverse, 'reverse_total' => $reverseTotal];
    }

    /**
     * Arti per kata untuk kueri berupa beberapa kata: hanya entri yang katanya persis sama. Dipakai juga untuk
     * mencari frasa utuh (satu "kata" berisi seluruh kueri, kata berulang dan panjangnya tidak diubah).
     *
     * @param  list<string>  $words
     * @return list<array{word: string, entries: Collection<int, DictionaryEntry>}>
     */
    public static function gloss(string $dictionary, array $words, int $perWord = 3): array
    {
        $exact = fn (string $needle): Collection => $needle === '' ? collect()
            : self::ranked(DictionaryEntry::where('dictionary', $dictionary)->where('lookup', $needle), $perWord);
        $rows = [];
        foreach ($words as $word) {
            $needle = DictionaryEntry::normalize(self::unquote($word));
            $entries = $exact($needle);
            // "-an" dicari sebagai imbuhan; kata biasa yang kebetulan didahului tanda hubung dicari tanpa tanda itu.
            if ($entries->isEmpty() && str_starts_with($needle, '-')) {
                $entries = $exact(ltrim($needle, '- '));
            }
            $rows[] = ['word' => $word, 'entries' => $entries];
        }

        return $rows;
    }

    /**
     * Kata-kata sebuah kueri (huruf, angka, tanda hubung, apostrof), tanpa pengulangan, paling banyak $max.
     * Tanda hubung di depan kata ikut ("-an", "ke- -an"): begitulah imbuhan ditulis di kamus.
     */
    public static function words(string $text, int $max = self::MAX_WORDS): array
    {
        preg_match_all("/-?[\\p{L}\\p{M}\\p{N}][\\p{L}\\p{M}\\p{N}'’-]*/u", self::unquote($text), $found);
        $seen = [];
        foreach ($found[0] as $word) {
            $seen[DictionaryEntry::normalize($word)] ??= $word;
        }
        unset($seen['']);

        return array_slice(array_values($seen), 0, $max);
    }

    /**
     * Buang tanda kutip yang MEMBUNGKUS kata atau frasa ('rumah', ‘omah’ lan ‘griya’); apostrof yang bagian dari kata
     * (Al-Qur'an, furu') tidak disentuh karena tidak punya pasangan pembuka.
     */
    public static function unquote(string $text): string
    {
        $text = preg_replace('/(?<![\p{L}\p{N}])[\'‘’`]+([^\'‘’`]*[\p{L}\p{N}])[\'’`]+(?![\p{L}\p{N}])/u', '$1', $text) ?? $text;

        return trim($text, " \t\n\r\0\x0B\"“”«»");
    }

    /** Arti berbahasa Indonesia didahulukan, lalu kata yang lebih pendek. */
    private static function ranked(Builder $query, int $limit): Collection
    {
        return $query->orderByRaw("case gloss_lang when 'id' then 0 when 'jv' then 1 else 2 end, length(lookup), ".self::lookupColumn().', id')
            ->limit($limit)->get();
    }

    /** Urutan kata yang sama di PostgreSQL (kolasi sistem menaruh "-" sebelum spasi) dan SQLite: urutan byte. */
    private static function lookupColumn(): string
    {
        return DB::connection()->getDriverName() === 'pgsql' ? 'lookup COLLATE "C"' : 'lookup';
    }

    /** Karakter pola LIKE di kata yang diketik diperlakukan sebagai huruf biasa. */
    private static function escape(string $needle): string
    {
        return addcslashes($needle, '\\%_');
    }
}
