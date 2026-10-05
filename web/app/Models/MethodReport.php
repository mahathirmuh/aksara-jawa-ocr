<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/** Kartu metode halaman Metode (out/results/methods.json dari scripts/export_methods.py), disimpan utuh di `payload`. */
class MethodReport extends Model
{
    protected $guarded = [];

    protected $casts = ['payload' => 'array'];

    /** Kartu metode yang berlaku; null bila belum pernah diimpor. */
    public static function current(): ?self
    {
        return static::latest('id')->first();
    }
}
