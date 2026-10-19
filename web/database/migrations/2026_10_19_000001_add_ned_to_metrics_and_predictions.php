<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        // NED (suntingan / panjang terpanjang label-keluaran, selalu 0-1) di samping CER yang tidak berbatas atas
        // (permintaan user 2026-10-19: skala 0-100%). Dihitung scripts/export_results.py; ekspor lama tanpa kunci = null.
        Schema::table('metrics', function (Blueprint $table) {
            $table->double('ned')->nullable();
        });
        Schema::table('predictions', function (Blueprint $table) {
            $table->double('ned')->nullable();
            $table->index(['pipeline', 'ned']);
        });
    }

    public function down(): void
    {
        Schema::table('predictions', function (Blueprint $table) {
            $table->dropIndex(['pipeline', 'ned']);
            $table->dropColumn('ned');
        });
        Schema::table('metrics', function (Blueprint $table) {
            $table->dropColumn('ned');
        });
    }
};
