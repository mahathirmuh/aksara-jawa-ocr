<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/**
 * CER agregat per (cakupan, pipeline), plus recall, presisi, F1, dan akurasi karakter (null pada ekspor lama).
 * Semua dari Python (scripts/export_results.py); F1 dan kawan-kawannya dari penjajaran yang sama dengan CER.
 */
class Metric extends Model
{
    public $timestamps = false;

    protected $guarded = [];

    protected $casts = ['cer' => 'float', 'cer_no_space' => 'float', 'exact' => 'float',
        'precision' => 'float', 'recall' => 'float', 'f1' => 'float', 'char_accuracy' => 'float'];
}
