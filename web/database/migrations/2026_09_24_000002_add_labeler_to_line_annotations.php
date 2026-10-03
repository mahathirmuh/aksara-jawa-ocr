<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/* Label tingkat tutur dari manusia (data uji tahap 4): siapa dan kapan. */
return new class extends Migration
{
    public function up(): void
    {
        Schema::table('line_annotations', function (Blueprint $table) {
            $table->foreignId('labeled_by')->nullable()->constrained('users')->nullOnDelete();
            $table->timestamp('labeled_at')->nullable();
        });
    }

    public function down(): void
    {
        Schema::table('line_annotations', function (Blueprint $table) {
            $table->dropConstrainedForeignId('labeled_by');
            $table->dropColumn('labeled_at');
        });
    }
};
