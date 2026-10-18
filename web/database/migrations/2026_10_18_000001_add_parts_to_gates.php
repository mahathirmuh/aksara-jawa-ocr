<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        // Rincian sebuah gerbang per kelompok baris (G3: baris cetak dan tulisan tangan; keputusan user 2026-10-18),
        // dilaporkan di samping angka gerbang, bukan menggantikannya. json, bukan jsonb: urutan butir dipertahankan.
        Schema::table('gates', function (Blueprint $table) {
            $table->json('parts')->nullable();
        });
    }

    public function down(): void
    {
        Schema::table('gates', function (Blueprint $table) {
            $table->dropColumn('parts');
        });
    }
};
