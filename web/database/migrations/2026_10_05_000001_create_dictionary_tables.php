<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/*
 * Kamus kata di halaman Kamus: bahasa Jawa (jv) dan bahasa Indonesia (id). Isinya bukan hasil hitung proyek ini,
 * melainkan data pihak ketiga yang dibangun tools/dictionaries.py dan diimpor `php artisan aksara:dictionary`;
 * bisa dibangun ulang kapan saja.
 */
return new class extends Migration
{
    public function up(): void
    {
        Schema::create('dictionary_sources', function (Blueprint $table) {
            $table->id();
            $table->string('dictionary');              // jv | id
            $table->string('key');                     // kunci sumber yang dipakai entri (mis. wikidata)
            $table->string('name');
            $table->string('license');
            $table->string('url');
            $table->string('retrieved')->nullable();   // tanggal data diambil, YYYY-MM-DD
            $table->unsignedInteger('entries')->default(0);
            $table->timestamps();
            $table->unique(['dictionary', 'key']);
        });

        Schema::create('dictionary_entries', function (Blueprint $table) {
            $table->id();
            $table->string('dictionary');
            $table->string('source');
            $table->string('word');
            $table->string('lookup');                  // kata huruf kecil tanpa diakritik: kolom yang dicari
            $table->string('pos')->nullable();         // kelas kata
            $table->string('aksara')->nullable();      // ejaan aksara Jawa, bila sumbernya punya
            $table->string('register')->nullable();    // ragam: ngoko, krama, krama inggil, ...
            $table->string('gloss_lang', 8);           // bahasa arti: id | en | jv
            $table->json('glosses');
            $table->text('gloss_text');                // arti tanpa tanda baca, diapit spasi: pencarian balik per kata utuh
            $table->json('examples')->nullable();     // contoh pemakaian dari sumbernya (paling banyak tiga)
            $table->text('note')->nullable();          // keterangan bebas dari sumbernya, mis. padanan antar-ragam (ngoko omah · krama griya)
            $table->string('url', 500)->nullable();
            $table->index(['dictionary', 'lookup']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('dictionary_entries');
        Schema::dropIfExists('dictionary_sources');
    }
};
