<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/**
 * Recall, presisi, F1 satu karakter untuk satu pipeline pada satu cakupan (manifest `class_metrics`, dihitung
 * scripts/export_results.py dari penjajaran yang sama dengan CER). tp = terbaca benar; fn = ada di label tetapi
 * tertukar/hilang; fp = dikeluarkan model padahal label lain/tidak ada.
 */
class ClassMetric extends Model
{
    public $timestamps = false;

    protected $guarded = [];

    protected $casts = ['precision' => 'float', 'recall' => 'float', 'f1' => 'float'];
}
