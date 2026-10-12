<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

/** Cara membaca citra baris (CRNN greedy, beam + LM, VLM, ...). Sumber: manifest.json. */
class Pipeline extends Model
{
    /** Pipeline resmi bila manifest tidak menyebutnya (ekspor sebelum kunci "official" ada). */
    public const DEFAULT_OFFICIAL = 'crnn_fonts';

    protected $guarded = [];

    protected $casts = ['official' => 'boolean'];

    public function isDone(): bool
    {
        return $this->status === 'done';
    }

    /**
     * Pipeline yang angkanya dipakai sebagai angka resmi: gerbang G1–G3 dan kesalahan aksara dihitung darinya
     * (scripts/export_results.py, OFFICIAL_RUN). null bila belum ada hasil yang diimpor.
     */
    public static function official(): ?self
    {
        return static::where('official', true)->first() ?? static::where('key', self::DEFAULT_OFFICIAL)->first();
    }

    /**
     * Pasangan [kunci greedy, kunci beam + LM] yang dibandingkan halaman: milik pipeline resmi bila beam + LM-nya
     * (kunci "<resmi>_beam") sudah diekspor, kalau tidak pasangan lama fase5_fonts.
     *
     * @return array{string, string}
     */
    public static function beamPair(): array
    {
        $official = static::official()?->key;
        $done = $official !== null && static::where('key', $official.'_beam')->where('status', 'done')->exists();

        return $done ? [$official, $official.'_beam'] : [self::DEFAULT_OFFICIAL, self::DEFAULT_OFFICIAL.'_beam'];
    }
}
