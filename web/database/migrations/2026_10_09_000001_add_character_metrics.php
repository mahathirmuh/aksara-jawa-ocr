<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/**
 * Recall, presisi, F1, dan akurasi karakter di samping CER (manifest skema 1, kunci opsional di tiap metrik), plus
 * tabel per karakter (manifest kunci `class_metrics`). Ekspor lama tanpa kunci itu tetap bisa diimpor: kolomnya null
 * dan halaman menampilkan "–".
 */
return new class extends Migration
{
    public function up(): void
    {
        Schema::table('metrics', function (Blueprint $table) {
            $table->double('precision')->nullable()->after('exact');
            $table->double('recall')->nullable()->after('precision');
            $table->double('f1')->nullable()->after('recall');
            $table->double('char_accuracy')->nullable()->after('f1');
        });

        Schema::create('class_metrics', function (Blueprint $table) {
            $table->id();
            $table->string('scope');             // nusaaksara_745
            $table->string('pipeline');
            $table->string('char', 8);
            $table->string('code', 12);          // U+XXXX
            $table->unsignedInteger('ref');      // kemunculan di referensi (tp + fn)
            $table->unsignedInteger('hyp');      // kemunculan di keluaran (tp + fp)
            $table->unsignedInteger('tp');
            $table->unsignedInteger('fn');
            $table->unsignedInteger('fp');
            $table->double('precision')->nullable();
            $table->double('recall')->nullable();
            $table->double('f1')->nullable();
            $table->unique(['scope', 'pipeline', 'char']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('class_metrics');
        Schema::table('metrics', function (Blueprint $table) {
            $table->dropColumn(['precision', 'recall', 'f1', 'char_accuracy']);
        });
    }
};
