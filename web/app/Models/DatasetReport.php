<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/** Kartu data halaman Dataset (out/results/datasets.json dari scripts/export_datasets.py), disimpan utuh di `payload`. */
class DatasetReport extends Model
{
    protected $guarded = [];

    protected $casts = ['payload' => 'array'];

    /** Kartu data yang berlaku; null bila belum pernah diimpor. */
    public static function current(): ?self
    {
        return static::latest('id')->first();
    }
}
