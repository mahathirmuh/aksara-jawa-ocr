<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/** CER agregat per (cakupan, pipeline), dan NED (0-1) bila ekspornya menghitungnya. */
class Metric extends Model
{
    public $timestamps = false;

    protected $guarded = [];

    protected $casts = ['cer' => 'float', 'ned' => 'float', 'cer_no_space' => 'float', 'exact' => 'float'];
}
