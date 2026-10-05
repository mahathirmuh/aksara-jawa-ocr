<?php

namespace App\Services;

use RuntimeException;

/**
 * Kartu tidak bisa disimpan atau diperiksa karena keadaan database web (tabel belum dimigrasi, database tidak
 * terjangkau), bukan karena isi kartunya. Dibedakan dari kartu yang ditolak supaya perintah impor keluar dengan kode
 * gagal dan pesannya menunjuk ke `php artisan migrate`, bukan ke skrip ekspor.
 */
class CardStorageException extends RuntimeException {}
