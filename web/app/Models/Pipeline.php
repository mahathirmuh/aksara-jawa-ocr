<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/** Cara membaca citra baris (CRNN greedy, beam + LM, VLM, ...). Sumber: manifest.json. */
class Pipeline extends Model
{
    protected $guarded = [];

    public function isDone(): bool
    {
        return $this->status === 'done';
    }
}
