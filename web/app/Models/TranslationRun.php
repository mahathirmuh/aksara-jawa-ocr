<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Collection;

/** Kualitas korpus satu run terjemahan (chrF & BLEU sacrebleu terhadap terjemahan manusia). */
class TranslationRun extends Model
{
    protected $guarded = [];

    protected $casts = ['chrf' => 'float', 'bleu' => 'float'];

    /** Run terbaru tiap sumber, berkunci nama sumbernya ("label" atau kunci pipeline OCR). */
    public static function latestBySource(): Collection
    {
        return static::latest('id')->get()->unique('source')->keyBy('source');
    }

    /**
     * Terjemahan "dari keluaran OCR" yang ditampilkan: atas keluaran pipeline resmi bila sudah dijalankan, kalau belum
     * atas keluaran pipeline resmi yang dibaca beam + LM, kalau belum terjemahan terbaru atas keluaran pipeline lain.
     * Satu aturan untuk Ringkasan dan Metode, supaya dua halaman tidak mengutip run yang berbeda.
     */
    public static function forOcr(?string $officialKey, ?Collection $runs = null): ?self
    {
        $runs ??= static::latestBySource();

        return ($officialKey !== null ? $runs->get($officialKey) ?? $runs->get($officialKey.'_beam') : null)
            ?? $runs->first(fn (self $run) => $run->source !== 'label');
    }
}
