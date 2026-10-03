<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/** Target proyek G1–G4 (PLAN.md §1.2). */
class Gate extends Model
{
    public $timestamps = false;

    protected $guarded = [];

    protected $casts = ['passed' => 'boolean', 'value' => 'float', 'target' => 'float'];
}
