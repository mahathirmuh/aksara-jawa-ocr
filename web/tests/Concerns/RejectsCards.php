<?php

namespace Tests\Concerns;

use App\Services\CardImporter;
use RuntimeException;

/**
 * Pernyataan "kartu ini harus ditolak importir" yang benar-benar bisa gagal.
 *
 * Bentuk yang lazim, `try { import(); $this->fail(...); } catch (RuntimeException $e) { assert pesan }`, TIDAK bisa
 * gagal: galat `fail()` PHPUnit adalah turunan RuntimeException, jadi ikut tertangkap dan pesannya memuat teks yang
 * diharapkan. Kartu yang DITERIMA pun lolos (ditemukan tinjauan 2026-10-06: lima mutan importir lolos karenanya).
 * Di sini pengecualian ditangkap dulu, baru dinyatakan di luar blok try, dan kartu yang tersimpan diperiksa tidak
 * berubah sesudah TIAP penolakan.
 */
trait RejectsCards
{
    /**
     * @param  class-string  $model  model tempat kartu disimpan
     * @param  class-string  $exception  kelas pengecualian yang diharapkan, persis (CardStorageException untuk galat
     *                                   database; bawaannya RuntimeException = kartu yang ditolak)
     */
    protected function assertCardRejected(CardImporter $importer, string $model, string $dir, string $expected,
        ?array $card = null, ?string $raw = null, string $exception = RuntimeException::class): void
    {
        file_put_contents($dir.'/'.$importer::FILE, $raw ?? json_encode($card, JSON_UNESCAPED_UNICODE));
        $before = $model::query()->orderBy('id')->get()->toArray();
        $caught = null;
        try {
            $importer->import($dir);
        } catch (RuntimeException $e) {
            $caught = $e;
        }
        $this->assertNotNull($caught, "Kartu seharusnya ditolak ({$expected}), tetapi diterima.");
        // Kelas PERSIS, bukan turunannya: CardStorageException turunan RuntimeException, dan kartu yang ditolak harus
        // terbedakan dari galat penyimpanan (perintah impor memberi kode keluar yang berbeda untuk keduanya).
        $this->assertSame($exception, $caught::class, $caught->getMessage());
        $this->assertStringContainsString($expected, $caught->getMessage());
        $this->assertSame($before, $model::query()->orderBy('id')->get()->toArray(), "Kartu yang tersimpan berubah sesudah penolakan: {$expected}");
    }
}
