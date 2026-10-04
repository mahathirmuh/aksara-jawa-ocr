<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        // Pipeline yang angkanya menjadi angka resmi (gerbang G1–G3, kesalahan aksara): kunci "official" di
        // manifest.json. Sebelum kolom ini ada, angka resmi tertanam di kode sebagai crnn_fonts.
        Schema::table('pipelines', function (Blueprint $table) {
            $table->boolean('official')->default(false);
        });
    }

    public function down(): void
    {
        Schema::table('pipelines', function (Blueprint $table) {
            $table->dropColumn('official');
        });
    }
};
