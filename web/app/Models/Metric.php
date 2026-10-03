<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/** CER agregat per (cakupan, pipeline). */
class Metric extends Model
{
    public $timestamps = false;

    protected $guarded = [];

    protected $casts = ['cer' => 'float', 'cer_no_space' => 'float', 'exact' => 'float'];
}
