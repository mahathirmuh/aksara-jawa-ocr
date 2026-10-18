<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/*
 * Matriks metode halaman Matriks: isi out/results/matrix.json buatan `scripts/method_matrix.py export` (repo OCR),
 * disimpan utuh. Satu baris saja, diganti tiap `php artisan aksara:matrix`; bisa dibangun ulang kapan saja.
 * payload bertipe json (bukan jsonb) supaya urutan kunci tetap seperti berkasnya, sama dengan kartu data dan metode.
 */
return new class extends Migration
{
    public function up(): void
    {
        Schema::create('method_matrices', function (Blueprint $table) {
            $table->id();
            $table->unsignedSmallInteger('schema');
            $table->string('generated_at');
            $table->string('source_path');
            $table->json('payload');
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('method_matrices');
    }
};
