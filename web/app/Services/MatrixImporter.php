<?php

namespace App\Services;

use App\Models\MethodMatrix;
use Illuminate\Support\Facades\DB;
use RuntimeException;

/**
 * Impor matriks metode (out/results/matrix.json, skema 1) ke tabel method_matrices. Angkanya dihitung Python
 * (scripts/method_matrix.py); di sini hanya diperiksa bentuknya, supaya berkas rusak ditolak saat impor dan matriks
 * lama tetap dipakai, bukan menjadi galat saat halaman dibuka.
 */
class MatrixImporter
{
    public const SCHEMA = 1;

    public const GROUPS = ['kita', 'vlm', 'umum'];

    /** @return array{methods: int, architecture: int, ablation: int, augment: int} */
    public function import(string $dir): array
    {
        $path = rtrim($dir, '/\\').DIRECTORY_SEPARATOR.'matrix.json';
        if (! is_file($path)) {
            throw new RuntimeException("Berkas matriks tidak ada: {$path}. Jalankan dulu: python scripts/method_matrix.py export");
        }
        $data = json_decode((string) file_get_contents($path), true);
        if (! is_array($data)) {
            throw new RuntimeException("Matriks bukan JSON yang sah: {$path}");
        }
        self::check($data);

        DB::transaction(function () use ($data, $path) {
            MethodMatrix::query()->delete();
            MethodMatrix::create([
                'schema' => $data['schema'], 'generated_at' => $data['generated'], 'source_path' => $path, 'payload' => $data,
            ]);
        });

        return ['methods' => count($data['methods']), 'architecture' => count($data['architecture']),
            'ablation' => count($data['ablation']), 'augment' => count($data['augment'])];
    }

    /** Bentuk minimum yang dibaca halaman; pesan menyebut kunci yang salah. */
    public static function check(array $data): void
    {
        $fail = fn (string $what) => throw new RuntimeException("Matriks tidak sah: {$what}");
        if (($data['schema'] ?? null) !== self::SCHEMA) {
            $fail('schema harus '.self::SCHEMA);
        }
        foreach (['generated', 'reference', 'prompt', 'rules'] as $key) {
            if (! is_string($data[$key] ?? null) || $data[$key] === '') {
                $fail("{$key} harus teks");
            }
        }
        foreach (['methods', 'architecture', 'ablation', 'augment'] as $key) {
            if (! is_array($data[$key] ?? null) || ! array_is_list($data[$key])) {
                $fail("{$key} harus daftar");
            }
        }
        if ($data['methods'] === []) {
            $fail('methods kosong');
        }
        foreach (['cetak', 'tulisan_tangan', 'semua'] as $kind) {
            $s = $data['subsets'][$kind] ?? null;
            if (! self::count($s['lines'] ?? null) || ! self::count($s['pages'] ?? null)) {
                $fail("subsets.{$kind} harus punya lines dan pages");
            }
        }
        if (! in_array($data['reference'], array_column($data['methods'], 'key'), true)) {
            $fail("pembanding {$data['reference']} tidak ada di methods");
        }
        $keys = [];
        foreach ($data['methods'] as $i => $m) {
            if (! is_string($m['key'] ?? null) || ! is_string($m['label'] ?? null) || $m['label'] === '') {
                $fail("methods.{$i} harus punya key dan label");
            }
            if (! in_array($m['group'] ?? null, self::GROUPS, true)) {
                $fail("methods.{$i}.group harus salah satu dari ".implode(', ', self::GROUPS));
            }
            if (isset($keys[$m['key']])) {
                $fail("methods.{$i}.key berulang: {$m['key']}");
            }
            $keys[$m['key']] = true;
            foreach (['cetak', 'tulisan_tangan', 'semua'] as $kind) {
                if (($m[$kind] ?? null) !== null && ! self::cell($m[$kind], $m['key'] !== $data['reference'])) {
                    $fail("methods.{$i}.{$kind} tidak sah");
                }
            }
            if (($m['cetak'] ?? null) === null && ! is_array($m['partial'] ?? null)) {
                $fail("methods.{$i} tidak punya angka cetak maupun bagian yang dibaca");
            }
        }
        foreach ($data['ablation'] as $i => $a) {
            if (! is_string($a['label'] ?? null) || ! is_string($a['status'] ?? null)) {
                $fail("ablation.{$i} harus punya label dan status");
            }
            foreach (['cetak', 'tulisan_tangan', 'semua'] as $kind) {
                $c = $a[$kind] ?? null;
                if (! self::number($c['delta'] ?? null) || ! self::interval($c['ci95_pages'] ?? null)) {
                    $fail("ablation.{$i}.{$kind} harus punya delta dan ci95_pages");
                }
            }
        }
        foreach ($data['augment'] as $i => $r) {
            if (! is_string($r['op'] ?? null) || ! self::number($r['g3'] ?? null) || ! self::number($r['heavy'] ?? null)) {
                $fail("augment.{$i} harus punya op, g3, dan heavy");
            }
        }
        $stages = $data['architecture_stages'] ?? [];
        if (! is_array($stages) || ! array_is_list($stages)) {
            $fail('architecture_stages harus daftar');
        }
        foreach ($stages as $i => $s) {
            $c = $s['cetak'] ?? null;
            $ok = is_string($s['stage'] ?? null) && $s['stage'] !== '' && is_array($c)
                && self::number($c['cer_t'] ?? null) && self::number($c['cer_c'] ?? null)
                && self::number($c['delta'] ?? null) && self::interval($c['ci95'] ?? null)
                && self::number($s['semua']['cer_t'] ?? null) && self::number($s['semua']['cer_c'] ?? null);
            if (! $ok) {
                $fail("architecture_stages.{$i} harus punya stage, cetak (cer_t, cer_c, delta, ci95), dan semua");
            }
        }
        foreach ($data['architecture'] as $i => $r) {
            if (! is_string($r['label'] ?? null) || ! self::count($r['params'] ?? null)) {
                $fail("architecture.{$i} harus punya label dan params");
            }
        }
    }

    private static function cell(mixed $c, bool $withDelta): bool
    {
        return is_array($c) && self::number($c['cer'] ?? null) && $c['cer'] >= 0 && self::count($c['lines'] ?? null)
            && self::number($c['ned'] ?? null) && $c['ned'] >= 0 && $c['ned'] <= 1
            && (! $withDelta || (self::number($c['delta'] ?? null) && self::interval($c['ci95'] ?? null)
                && self::number($c['ned_delta'] ?? null) && self::interval($c['ned_ci95'] ?? null)));
    }

    private static function number(mixed $x): bool
    {
        return is_int($x) || is_float($x);
    }

    private static function count(mixed $x): bool
    {
        return is_int($x) && $x > 0;
    }

    private static function interval(mixed $x): bool
    {
        return is_array($x) && array_is_list($x) && count($x) === 2 && self::number($x[0]) && self::number($x[1]) && $x[0] <= $x[1];
    }
}
