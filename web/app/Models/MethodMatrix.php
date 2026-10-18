<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/** Matriks metode halaman Matriks (out/results/matrix.json dari scripts/method_matrix.py export), utuh di `payload`. */
class MethodMatrix extends Model
{
    protected $guarded = [];

    protected $casts = ['payload' => 'array'];

    /** Matriks yang berlaku; null bila belum pernah diimpor. */
    public static function current(): ?self
    {
        return static::latest('id')->first();
    }
}
