<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/*
 * Kartu data halaman Dataset: isi out/results/datasets.json buatan scripts/export_datasets.py (repo OCR), disimpan
 * utuh. Satu baris saja, diganti tiap `php artisan aksara:datasets`; bisa dibangun ulang kapan saja.
 *
 * Kolom payload bertipe json, bukan jsonb: jsonb di PostgreSQL mengurutkan ulang kunci objek (train/val/test menjadi
 * val/test/train), json menyimpan dokumennya apa adanya. Halaman tetap tidak bergantung pada urutan kunci.
 */
return new class extends Migration
{
    public function up(): void
    {
        Schema::create('dataset_reports', function (Blueprint $table) {
            $table->id();
            $table->unsignedSmallInteger('schema');
            $table->string('generated_at');
            $table->string('official_run');             // run yang pemakaiannya dihitung (OFFICIAL_RUN saat ekspor)
            $table->string('source_path');
            $table->json('payload');
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('dataset_reports');
    }
};
