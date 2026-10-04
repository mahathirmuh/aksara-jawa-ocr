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
     * Cara membaca yang disediakan layanan (kunci = PIPELINES di src/serve.py). Greedy = jalur resmi; bobot LM untuk
     * beam (alpha 0,25, beta 1,0) disetel di dev sintetis dengan checkpoint fase5_fonts dan belum diuji ulang untuk
     * checkpoint lain. Nama checkpoint ditambahkan halaman Demo dari /health layanan.
     */
    public const PIPELINES = [
        'crnn_greedy' => 'greedy (jalur resmi)',
        'crnn_beam_lm' => 'beam + LM (bobot LM disetel pada fase5_fonts)',
    ];

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
