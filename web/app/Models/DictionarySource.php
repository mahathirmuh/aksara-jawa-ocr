<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/** Sumber data sebuah kamus kata (nama, lisensi, tautan, tanggal diambil): dasar atribusi di halaman Kamus. */
class DictionarySource extends Model
{
    protected $guarded = [];

    protected $casts = ['entries' => 'integer'];
}
