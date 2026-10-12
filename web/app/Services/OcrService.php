<?php

namespace App\Services;

use Illuminate\Http\Client\ConnectionException;
use Illuminate\Http\Client\RequestException;
use Illuminate\Support\Facades\Http;
use RuntimeException;

/** Klien layanan model FastAPI (src/serve.py di repo OCR). */
class OcrService
{
    /**
     * Cara membaca yang disediakan layanan (kunci = PIPELINES di src/serve.py). Greedy = jalur resmi. Bobot LM untuk
     * beam disetel per checkpoint di src/serve.py; halaman Demo membacanya dari layanan (beamWeights()), tidak
     * menuliskannya sendiri. Nama checkpoint juga ditambahkan halaman Demo dari /health layanan.
     */
    public const PIPELINES = [
        'crnn_greedy' => 'greedy (jalur resmi)',
        'crnn_beam_lm' => 'beam + LM',
    ];

    /**
     * Bobot beam + LM yang dipakai layanan, dari jawaban /predict atau /health (kunci "beam"); sumber pertama yang
     * memuatnya menang. Null bila layanan tidak berjalan atau versi lama yang belum melaporkannya.
     *
     * @return array{alpha: float, beta: float}|null
     */
    public static function beamWeights(?array ...$answers): ?array
    {
        foreach ($answers as $answer) {
            $beam = $answer['beam'] ?? null;
            if (is_array($beam) && is_numeric($beam['alpha'] ?? null) && is_numeric($beam['beta'] ?? null)) {
                return ['alpha' => (float) $beam['alpha'], 'beta' => (float) $beam['beta']];
            }
        }

        return null;
    }

    /** 0.5 -> "0,5"; 1.0 -> "1". */
    public static function weight(float $value): string
    {
        return str_replace('.', ',', rtrim(rtrim(number_format($value, 4, '.', ''), '0'), '.'));
    }

    /** "out/checkpoints/fase7_track/last_snapshot.pt" -> "fase7_track"; null bila layanan tidak berjalan. */
    public static function checkpointName(?array $health): ?string
    {
        $path = $health['checkpoint'] ?? null;

        return $path ? basename(dirname($path)) : null;
    }

    public function url(): string
    {
        return rtrim(config('aksara.service_url'), '/');
    }

    /** null bila layanan tidak berjalan. */
    public function health(): ?array
    {
        try {
            return Http::timeout(3)->get($this->url().'/health')->throw()->json();
        } catch (ConnectionException|RequestException) {
            return null;
        }
    }

    public function predict(string $imagePath, string $pipeline): array
    {
        if (! array_key_exists($pipeline, self::PIPELINES)) {
            throw new RuntimeException("Pipeline tidak dikenal: {$pipeline}");
        }
        try {
            $response = Http::timeout(config('aksara.service_timeout'))
                ->attach('file', file_get_contents($imagePath), basename($imagePath))
                ->post($this->url().'/predict', ['pipeline' => $pipeline]);
        } catch (ConnectionException) {
            throw new RuntimeException('Layanan model tidak berjalan di '.$this->url().'. Jalankan di folder repo OCR: '
                .'.venv/Scripts/python -m uvicorn src.serve:app --host 127.0.0.1 --port 8011');
        }
        if ($response->failed()) {
            throw new RuntimeException('Layanan model menolak citra: '.($response->json('detail') ?? $response->status()));
        }

        return $response->json();
    }
}
