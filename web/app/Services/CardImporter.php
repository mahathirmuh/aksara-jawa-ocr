<?php

namespace App\Services;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\QueryException;
use Illuminate\Support\Facades\DB;
use JsonException;
use RuntimeException;
use Throwable;

/**
 * Dasar impor "kartu" dari repo OCR: satu berkas JSON buatan skrip Python, disimpan utuh sebagai satu baris.
 *
 * Web tidak menghitung apa pun dari isinya. Dua pengaman supaya berkas yang salah ditolak saat impor dan bukan menjadi
 * galat saat halaman dibuka:
 *  1. bentuk: kunci di `shape()` wajib ada dan bertipe benar, dengan pesan yang menyebut kuncinya;
 *  2. uji tampil: sebelum disimpan, halaman dirender sekali dengan kartu calon (`preview()`); bila gagal karena apa
 *     pun (kunci yang tidak terdaftar di `shape()`, tipe yang tidak terduga), impor dibatalkan.
 * Kartu yang sudah tersimpan tidak disentuh bila kartu baru ditolak.
 */
abstract class CardImporter
{
    /** Nama berkas kartu di folder hasil; diisi turunan. */
    public const FILE = '';

    private const TYPES = [
        'number' => 'angka', 'string' => 'teks', 'bool' => 'benar/salah', 'list' => 'daftar', 'map' => 'objek',
    ];

    /** Kolom teks tabel kartu (varchar 255): nilai lebih panjang ditolak PostgreSQL saat disimpan. */
    private const COLUMN_LIMIT = 255;

    /** Kelas model Eloquent tempat kartu disimpan. */
    abstract protected function model(): string;

    /** Nama kartu untuk pesan, huruf kecil: "kartu data". */
    abstract protected function label(): string;

    /** Kunci konfigurasi nomor skema yang didukung. */
    abstract protected function schemaConfig(): string;

    /** Skrip Python yang membuat berkasnya, untuk petunjuk di pesan. */
    abstract protected function exporter(): string;

    /**
     * Bentuk kartu: kunci (notasi titik, "*" = setiap butir daftar atau objek) => tipe (number, string, bool, list,
     * map; akhiran "?" = boleh null, untuk nilai yang ekspornya menulis null bila tidak dihitung).
     *
     * @return array<string, string>
     */
    abstract protected function shape(): array;

    /** Ringkasan untuk pesan perintah artisan. */
    abstract protected function summary(array $card): array;

    /** Render halaman sekali dengan kartu yang baru disimpan (masih di dalam transaksi); lempar apa pun yang gagal. */
    abstract protected function preview(Model $report): void;

    /** Pemeriksaan tambahan yang tidak bisa dinyatakan `shape()`; lempar RuntimeException bila gagal. */
    protected function check(array $card): void {}

    public static function path(string $dir): string
    {
        return rtrim($dir, '/\\').DIRECTORY_SEPARATOR.static::FILE;
    }

    public function import(string $dir): array
    {
        $path = static::path($dir);
        $label = $this->label();
        if (! is_file($path)) {
            throw new RuntimeException("Berkas tidak ditemukan: {$path}. Jalankan dulu: python {$this->exporter()}");
        }
        try {
            $card = json_decode(file_get_contents($path), true, flags: JSON_THROW_ON_ERROR);
        } catch (JsonException $e) {
            throw new RuntimeException("{$path} bukan JSON yang sah: {$e->getMessage()}");
        }
        if (! is_array($card) || array_is_list($card)) {
            throw new RuntimeException("{$path} bukan {$label}: isinya harus objek JSON.");
        }
        $expected = config($this->schemaConfig());
        if (($card['schema'] ?? null) !== $expected) {
            throw new RuntimeException('Skema '.$label.' '.json_encode($card['schema'] ?? null)." tidak didukung (butuh {$expected}).");
        }
        $this->validate($card, $path);
        $this->check($card);

        $model = $this->model();
        try {
            DB::transaction(function () use ($card, $dir, $model, $label) {
                $model::query()->delete();
                $report = $model::create([
                    'schema' => $card['schema'], 'generated_at' => $card['generated'],
                    'official_run' => $card['official']['run'], 'source_path' => $dir, 'payload' => $card,
                ]);
                try {
                    $this->preview($report);
                } catch (Throwable $e) {
                    throw new RuntimeException(ucfirst($label).' tidak bisa ditampilkan halamannya ('.$e->getMessage()
                        .'); kartu sebelumnya dipertahankan.', previous: $e);
                }
            });
        } catch (QueryException $e) {
            throw new RuntimeException(ucfirst($label).' tidak bisa disimpan: '.$e->getMessage());
        }

        return $this->summary($card);
    }

    private function validate(array $card, string $path): void
    {
        $label = ucfirst($this->label());
        // Kunci pipeline boleh kosong: kartu untuk run yang tidak dikenal ekspor hasil, yang ditandai halamannya.
        $shape = ['generated' => 'string', 'official.run' => 'string', 'official.pipeline' => 'string?'] + $this->shape();
        foreach ($shape as $key => $type) {
            $nullable = str_ends_with($type, '?');
            $type = rtrim($type, '?');
            $values = str_contains($key, '*') ? data_get($card, $key) : [data_get($card, $key)];
            if (! is_array($values)) {
                throw new RuntimeException("{$label} tidak lengkap: kunci {$key} tidak ada di {$path}.");
            }
            foreach ($values as $value) {
                if ($value === null) {
                    if ($nullable) {
                        continue;
                    }
                    throw new RuntimeException("{$label} tidak lengkap: kunci {$key} tidak ada di {$path}.");
                }
                if (! self::matches($type, $value)) {
                    throw new RuntimeException("{$label} tidak sah: kunci {$key} harus ".self::TYPES[$type]." di {$path}.");
                }
            }
        }
        foreach (['generated', 'official.run'] as $key) {
            if (mb_strlen(data_get($card, $key)) > self::COLUMN_LIMIT) {
                throw new RuntimeException("{$label} tidak sah: kunci {$key} lebih panjang dari ".self::COLUMN_LIMIT.' karakter.');
            }
        }
    }

    private static function matches(string $type, mixed $value): bool
    {
        return match ($type) {
            'number' => is_int($value) || is_float($value),
            'string' => is_string($value),
            'bool' => is_bool($value),
            'list' => is_array($value) && array_is_list($value),
            'map' => is_array($value),
        };
    }
}
