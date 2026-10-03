<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('pipelines', function (Blueprint $table) {
            $table->id();
            $table->string('key')->unique();
            $table->string('label');
            $table->string('config');
            $table->string('kind');
            $table->string('status');            // done | planned
            $table->unsignedSmallInteger('sort')->default(0);
            $table->timestamps();
        });

        Schema::create('lines', function (Blueprint $table) {
            $table->id();
            $table->string('dataset');
            $table->string('external_id');
            $table->string('image_path');        // relatif terhadap repo OCR
            $table->unsignedInteger('width');
            $table->unsignedInteger('height');
            $table->text('reference');
            $table->string('condition')->nullable();
            $table->string('source_id')->nullable();
            $table->jsonb('tags');
            $table->timestamps();
            $table->unique(['dataset', 'external_id']);
        });

        // Dikunci dengan (dataset, external_id), bukan line_id, supaya bertahan saat hasil diimpor ulang.
        Schema::create('line_annotations', function (Blueprint $table) {
            $table->id();
            $table->string('dataset');
            $table->string('external_id');
            $table->text('transliteration')->nullable();
            $table->text('translation')->nullable();
            $table->string('speech_level')->nullable();   // ngoko | madya | krama | campur (tahap 4, belum ada)
            $table->string('source');
            $table->timestamps();
            $table->unique(['dataset', 'external_id']);
        });

        Schema::create('predictions', function (Blueprint $table) {
            $table->id();
            $table->foreignId('line_id')->constrained()->cascadeOnDelete();
            $table->string('pipeline');
            $table->text('text');
            $table->double('cer');
            $table->jsonb('segments');
            $table->jsonb('spans')->nullable();
            $table->string('confidence')->nullable();
            $table->unique(['line_id', 'pipeline']);
            $table->index(['pipeline', 'cer']);
        });

        Schema::create('metrics', function (Blueprint $table) {
            $table->id();
            $table->string('scope');             // nusaaksara_745 | blind_50 | synth_dev | synth_heldout | translit_draft
            $table->string('pipeline')->nullable();
            $table->unsignedInteger('lines');
            $table->double('cer');
            $table->double('cer_no_space')->nullable();
            $table->double('exact')->nullable();
            $table->unsignedInteger('better')->nullable();
            $table->unsignedInteger('worse')->nullable();
            $table->unique(['scope', 'pipeline']);
        });

        Schema::create('gates', function (Blueprint $table) {
            $table->id();
            $table->string('code')->unique();
            $table->string('name');
            $table->double('value');
            $table->double('target');
            $table->boolean('passed');
            $table->string('basis');
            $table->string('source');
        });

        Schema::create('ablation_runs', function (Blueprint $table) {
            $table->id();
            $table->unsignedSmallInteger('k')->unique();
            $table->string('added');
            $table->string('augment');
            $table->double('clean');
            $table->double('heavy');
            $table->double('g3')->nullable();
            $table->double('g3_no_space')->nullable();
        });

        Schema::create('confusions', function (Blueprint $table) {
            $table->id();
            $table->string('kind');              // sub | del | ins
            $table->string('ref', 8);
            $table->string('hyp', 8);
            $table->unsignedInteger('count');
            $table->index(['kind', 'count']);
        });

        Schema::create('result_imports', function (Blueprint $table) {
            $table->id();
            $table->unsignedSmallInteger('schema');
            $table->string('generated_at');
            $table->string('source_path');
            $table->boolean('limited')->default(false);
            $table->jsonb('counts');
            $table->timestamps();
        });
    }

    public function down(): void
    {
        foreach (['result_imports', 'confusions', 'ablation_runs', 'gates', 'metrics', 'predictions',
            'line_annotations', 'lines', 'pipelines'] as $table) {
            Schema::dropIfExists($table);
        }
    }
};
