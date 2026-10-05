<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/*
 * Kartu metode halaman Metode: isi out/results/methods.json buatan scripts/export_methods.py (repo OCR), disimpan
 * utuh. Satu baris saja, diganti tiap `php artisan aksara:methods`; bisa dibangun ulang kapan saja.
 *
 * Bentuknya sama dengan dataset_reports, termasuk kolom payload bertipe json (bukan jsonb, yang mengurutkan ulang
 * kunci objek di PostgreSQL).
 */
return new class extends Migration
{
    public function up(): void
    {
        Schema::create('method_reports', function (Blueprint $table) {
            $table->id();
            $table->unsignedSmallInteger('schema');
            $table->string('generated_at');
            $table->string('official_run');             // run yang modelnya diuraikan (OFFICIAL_RUN saat ekspor)
            $table->string('source_path');
            $table->json('payload');
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('method_reports');
    }
};
