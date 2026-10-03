<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/** Keluaran satu pipeline untuk satu baris. CER & segmen beda dihitung di Python. */
class Prediction extends Model
{
    public $timestamps = false;

    protected $guarded = [];

    protected $casts = ['segments' => 'array', 'spans' => 'array', 'cer' => 'float'];

    public function line(): BelongsTo
    {
        return $this->belongsTo(Line::class);
    }
}
