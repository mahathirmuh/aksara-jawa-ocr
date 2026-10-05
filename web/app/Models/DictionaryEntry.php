<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Normalizer;

/**
 * Satu entri kamus kata (bahasa Jawa atau bahasa Indonesia) dari sumber pihak ketiga.
 * `lookup` dan `gloss_text` diturunkan saat impor lewat normalize() / glossText(): pencarian memakai fungsi yang
 * sama pada kata yang diketik, jadi "gêdhé", "Gedhe", dan "gedhe" bertemu di entri yang sama.
 */
class DictionaryEntry extends Model
{
    public const DICTIONARIES = ['jv' => 'Kamus bahasa Jawa', 'id' => 'Kamus bahasa Indonesia'];

    public const GLOSS_LANGUAGES = ['id' => 'Indonesia', 'en' => 'Inggris', 'jv' => 'Jawa'];

    public $timestamps = false;

    protected $guarded = [];

    protected $casts = ['glosses' => 'array', 'examples' => 'array'];

    /** Bentuk yang dicari: huruf kecil, tanpa diakritik (é, è, ê, å -> e, e, e, a), tanda hubung dan apostrof dipertahankan. */
    public static function normalize(string $text): string
    {
        $text = Normalizer::normalize($text, Normalizer::FORM_D) ?: $text;
        $text = preg_replace('/\p{Mn}+/u', '', $text) ?? $text;
        $text = mb_strtolower(strtr($text, ['’' => "'", '‘' => "'", 'ʼ' => "'", '`' => "'"]));
        $text = preg_replace("/[^\\p{L}\\p{N}' -]+/u", ' ', $text) ?? $text;

        return trim(preg_replace('/\s+/u', ' ', $text) ?? $text);
    }

    /** Semua arti sebagai deret kata yang diapit spasi, supaya `LIKE '% kata %'` hanya mengenai kata utuh. */
    public static function glossText(array $glosses): string
    {
        $words = preg_replace('/[^\p{L}\p{N}]+/u', ' ', self::normalize(implode(' ', $glosses))) ?? '';
        $words = trim(preg_replace('/\s+/u', ' ', $words) ?? $words);

        return $words === '' ? ' ' : ' '.$words.' ';
    }
}
