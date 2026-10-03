<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/** Terjemahan mesin satu baris (tahap 3). source "label" = dari transliterasi manusia. */
class MachineTranslation extends Model
{
    protected $guarded = [];

    protected $casts = ['chrf' => 'float'];
}
