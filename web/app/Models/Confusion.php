<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/** Aksara tertukar (sub), hilang (del), atau tambahan (ins) pada keluaran OCR. */
class Confusion extends Model
{
    public $timestamps = false;

    protected $guarded = [];
}
