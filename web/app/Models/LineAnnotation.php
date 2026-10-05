<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Collection;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * Transliterasi & arti buatan manusia (NusaAksara), plus label tingkat tutur dari manusia
 * (speech_level, diisi lewat Penjelajah; menjadi data uji tahap 4).
 */
class LineAnnotation extends Model
{
    protected $guarded = [];

    protected $casts = ['labeled_at' => 'datetime'];

    /**
     * Label tingkat tutur manusia yang bisa dinilai: labelnya ada, alih aksara manusianya ada (itulah masukan
     * leksikon), dan barisnya termasuk baris uji yang diimpor. Satu kueri untuk Ringkasan dan Metode, supaya kedua
     * halaman menghitung akurasi pada baris yang sama.
     */
    public static function scorableSpeechLabels(): Collection
    {
        return static::whereNotNull('speech_level')->whereNotNull('transliteration')
            ->whereIn('external_id', Line::select('external_id'))
            ->orderBy('id')->get(['external_id', 'transliteration', 'speech_level']);
    }

    public function labeler(): BelongsTo
    {
        return $this->belongsTo(User::class, 'labeled_by');
    }
}
