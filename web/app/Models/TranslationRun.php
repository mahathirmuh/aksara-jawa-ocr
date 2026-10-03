<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/** Kualitas korpus satu run terjemahan (chrF & BLEU sacrebleu terhadap terjemahan manusia). */
class TranslationRun extends Model
{
    protected $guarded = [];

    protected $casts = ['chrf' => 'float', 'bleu' => 'float'];
}
