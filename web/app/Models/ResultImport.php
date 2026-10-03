<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/** Jejak setiap impor out/results/. */
class ResultImport extends Model
{
    protected $guarded = [];

    protected $casts = ['counts' => 'array', 'limited' => 'boolean'];
}
