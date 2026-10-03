<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/** Satu run ablasi augmentasi Fase 5 (out/ablation.json). */
class AblationRun extends Model
{
    public $timestamps = false;

    protected $guarded = [];

    protected $casts = ['clean' => 'float', 'heavy' => 'float', 'g3' => 'float', 'g3_no_space' => 'float'];
}
