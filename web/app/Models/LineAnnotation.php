<?php

namespace App\Models;

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

    public function labeler(): BelongsTo
    {
        return $this->belongsTo(User::class, 'labeled_by');
    }
}
