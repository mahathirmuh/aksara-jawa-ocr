<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/* Tahap 3 (arti): terjemahan mesin per baris dan ringkasan kualitasnya terhadap terjemahan manusia. */
return new class extends Migration
{
    public function up(): void
    {
        Schema::create('machine_translations', function (Blueprint $table) {
            $table->id();
            $table->string('dataset');
            $table->string('external_id');
            $table->string('source');            // label = transliterasi manusia; lain = kunci pipeline OCR
            $table->text('input');               // teks Latin yang diterjemahkan
            $table->text('output');
            $table->double('chrf')->nullable();  // sentence chrF terhadap terjemahan manusia
            $table->string('model');
            $table->timestamps();
            $table->unique(['dataset', 'external_id', 'source']);
        });

        Schema::create('translation_runs', function (Blueprint $table) {
            $table->id();
            $table->string('source');
            $table->string('model');
            $table->unsignedInteger('lines');
            $table->double('chrf');
            $table->double('bleu');
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('translation_runs');
        Schema::dropIfExists('machine_translations');
    }
};
