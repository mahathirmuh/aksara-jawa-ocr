<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\HasOne;

/** Satu potongan baris nyata beserta label aksaranya. Sumber: lines.jsonl. */
class Line extends Model
{
    protected $guarded = [];

    protected $casts = ['tags' => 'array'];

    public function predictions(): HasMany
    {
        return $this->hasMany(Prediction::class);
    }

    public function annotation(): HasOne
    {
        // external_id unik per berkas citra (Jawa_62_25.png); saat ini hanya satu dataset.
        return $this->hasOne(LineAnnotation::class, 'external_id', 'external_id');
    }

    public function label(): string
    {
        return pathinfo($this->external_id, PATHINFO_FILENAME);
    }
}
