<?php

namespace App\Services;

use App\Models\Line;
use App\Models\LineAnnotation;
use App\Models\MachineTranslation;
use App\Models\Prediction;
use App\Models\TranslationRun;
use App\Support\Transliterator;
use Illuminate\Http\Client\ConnectionException;
use Illuminate\Http\Client\RequestException;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Process;
use RuntimeException;

/**
 * Tahap 3 (arti). Batch: web menyiapkan teks Latin, tools/translate_batch.py (NLLB lokal) menerjemahkan dan
 * menghitung chrF/BLEU, web mengimpor hasilnya. Demo: layanan tools/translate_service.py.
 */
class TranslationService
{
    /**
     * Batch `aksara:translate` masih mengirim pepet bertanda "ê" ke model (lihat forModel()), jadi chrF dan BLEU yang
     * tersimpan kemungkinan terlalu rendah. Halaman Metode menyebutnya selama nilai ini true. Ubah menjadi false HANYA
     * bersamaan dengan menjalankan ulang batchnya: inputs() mengikuti nilai ini, dan angka lama tidak lagi sebanding.
     */
    public const BATCH_KEEPS_PEPET_MARK = true;

    /**
     * Teks Latin yang diterjemahkan per sumber:
     *   label       -> transliterasi manusia NusaAksara (kualitas model terjemahan saja)
     *   <pipeline>  -> transliterasi draf dari keluaran OCR pipeline itu (alur ujung ke ujung)
     */
    public function inputs(array $sources, int $limit = 0): array
    {
        $lines = Line::with('annotation')->orderBy('external_id')->when($limit, fn ($q) => $q->limit($limit))->get();
        $rows = [];
        foreach ($sources as $source) {
            $predictions = $source === 'label' ? collect()
                : Prediction::where('pipeline', $source)->whereIn('line_id', $lines->pluck('id'))->pluck('text', 'line_id');
            foreach ($lines as $line) {
                $input = $source === 'label'
                    ? $line->annotation?->transliteration
                    : (isset($predictions[$line->id]) ? Transliterator::toLatin($predictions[$line->id]) : null);
                if ($input) {
                    $rows[] = ['dataset' => $line->dataset, 'external_id' => $line->external_id, 'source' => $source,
                        'input' => self::BATCH_KEEPS_PEPET_MARK ? $input : self::forModel($input),
                        'reference' => $line->annotation?->translation];
                }
            }
        }

        return $rows;
    }

    /** Folder berkas batch (config aksara.mt_dir). Semua akses berkas batch lewat sini supaya test bisa mengalihkannya. */
    public function dir(): string
    {
        return config('aksara.mt_dir');
    }

    /** @return array{string, string, string} input.jsonl, output.jsonl, summary.json */
    public function files(): array
    {
        $dir = $this->dir();

        return ["{$dir}/input.jsonl", "{$dir}/output.jsonl", "{$dir}/summary.json"];
    }

    public function runBatch(array $rows, ?callable $onOutput = null): array
    {
        is_dir($this->dir()) || mkdir($this->dir(), 0775, true);
        [$in, $out, $summary] = $this->files();
        file_put_contents($in, implode('', array_map(fn ($r) => json_encode($r, JSON_UNESCAPED_UNICODE)."\n", $rows)));
        @unlink($out);

        $result = Process::path(base_path())->forever()
            ->env(['PYTHONIOENCODING' => 'utf-8'])
            ->run([config('aksara.python'), base_path('tools/translate_batch.py'), $in, $out, $summary],
                fn ($type, $buffer) => $onOutput && $onOutput($buffer));
        if ($result->failed()) {
            throw new RuntimeException('Terjemahan batch gagal: '.trim($result->errorOutput() ?: $result->output()));
        }

        return $this->import($out, $summary);
    }

    public function import(string $outputPath, string $summaryPath): array
    {
        $summary = json_decode(file_get_contents($summaryPath), true, flags: JSON_THROW_ON_ERROR);
        $rows = array_map(fn ($l) => json_decode($l, true, flags: JSON_THROW_ON_ERROR),
            array_filter(file($outputPath, FILE_IGNORE_NEW_LINES), 'strlen'));

        DB::transaction(function () use ($rows, $summary) {
            foreach ($rows as $r) {
                MachineTranslation::updateOrCreate(
                    ['dataset' => $r['dataset'], 'external_id' => $r['external_id'], 'source' => $r['source']],
                    ['input' => $r['input'], 'output' => $r['output'], 'chrf' => $r['chrf'] ?? null, 'model' => $summary['model']],
                );
            }
            foreach ($summary['runs'] as $run) {
                TranslationRun::create(['model' => $summary['model']] + $run);
            }
        });

        return $summary;
    }

    /**
     * Kebalikan import(): tulis output.jsonl & summary.json dari database (hanya SELECT) dalam format
     * tools/translate_batch.py, supaya --import-only bisa membangun ulang kedua tabel tanpa menjalankan NLLB.
     * Ringkasan = run terakhir per sumber, sama dengan yang tampil di halaman. Lama run tidak tersimpan di
     * database, jadi `seconds` null; `dumped_at` menandai berkas hasil pemulihan (import() tidak membacanya).
     */
    public function dump(): array
    {
        $rows = MachineTranslation::orderBy('id')->get();   // urutan id = urutan baris berkas asal
        $runs = TranslationRun::latest('id')->get()->unique('source')->sortBy('id')->values();
        if ($rows->isEmpty() || $runs->isEmpty()) {
            throw new RuntimeException('Database tidak memuat terjemahan mesin beserta ringkasan run-nya; tidak ada berkas yang ditulis.');
        }
        $models = $rows->pluck('model')->merge($runs->pluck('model'))->unique()->values();
        if ($models->count() > 1) {
            throw new RuntimeException('Terjemahan berasal dari lebih dari satu model ('.$models->join(', ')
                .'), sedangkan summary.json hanya memuat satu; tidak ada berkas yang ditulis.');
        }
        [, $out, $summaryPath] = $this->files();
        // Berkas yang lebih lengkap dari database (mis. database baru diisi sebagian) adalah salinan yang lebih baik.
        $existing = is_file($out) ? count(array_filter(file($out, FILE_IGNORE_NEW_LINES), 'strlen')) : 0;
        if ($existing > $rows->count()) {
            throw new RuntimeException("{$out} memuat {$existing} baris, lebih banyak dari database ({$rows->count()}); tidak ditimpa.");
        }

        // `reference` tidak dibaca import(), tapi ada di berkas asal: terjemahan manusia, seperti di inputs().
        $references = LineAnnotation::get(['dataset', 'external_id', 'translation'])
            ->mapWithKeys(fn ($a) => ["{$a->dataset}|{$a->external_id}" => $a->translation]);
        $jsonl = $rows->map(fn (MachineTranslation $r) => json_encode([
            'dataset' => $r->dataset, 'external_id' => $r->external_id, 'source' => $r->source, 'input' => $r->input,
            'reference' => $references["{$r->dataset}|{$r->external_id}"] ?? null, 'output' => $r->output, 'chrf' => $r->chrf,
        ], JSON_UNESCAPED_UNICODE | JSON_THROW_ON_ERROR)."\n")->join('');
        $summary = ['model' => $models->first(), 'seconds' => null, 'dumped_at' => now()->toIso8601String(),
            'runs' => $runs->map(fn (TranslationRun $r) => ['source' => $r->source, 'lines' => (int) $r->lines,
                'chrf' => $r->chrf, 'bleu' => $r->bleu])->all()];

        is_dir($this->dir()) || mkdir($this->dir(), 0775, true);
        file_put_contents($out, $jsonl);
        file_put_contents($summaryPath, json_encode($summary, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT | JSON_THROW_ON_ERROR)."\n");

        return ['rows' => $rows->count()] + $summary;
    }

    public function url(): string
    {
        return rtrim(config('aksara.translate_service_url'), '/');
    }

    /**
     * Bahasa Jawa beraksara Latin seperti yang dikenal model: pepet tanpa tanda. Draf transliterasi menulis pepet
     * "ê", ejaan yang nyaris tidak ada di data latih NLLB: terukur 2026-10-05, "pêkên" diterjemahkan "ke sana" dan
     * "sêga" menjadi "tiga buah", sedangkan "peken" -> "pasar" dan "sega" -> "nasi". Tanda é/è tidak mengganggu.
     * Dipakai semua panggilan langsung ke layanan; batch (inputs()) belum, supaya angka chrF yang sudah dilaporkan
     * tetap bisa diulang sampai dijalankan ulang.
     */
    public static function forModel(string $latin): string
    {
        return strtr($latin, ['ê' => 'e', 'Ê' => 'E']);
    }

    /** Keadaan layanan terjemahan: {status, model, loaded, directions}; null bila tidak berjalan. */
    public function health(): ?array
    {
        try {
            return Http::timeout(3)->get($this->url().'/health')->throw()->json();
        } catch (ConnectionException|RequestException) {
            return null;
        }
    }

    /**
     * Terjemahan beberapa kalimat sekaligus, arah mana pun antara 'jv' (Jawa beraksara Latin) dan 'id'.
     * Null bila layanan tidak bisa dihubungi; galat dari layanan (mis. versi lama yang hanya satu arah) dilempar.
     *
     * @param  list<string>  $texts
     * @return array{translations: list<string>, model: string, elapsed_ms: float}|null
     */
    public function translateMany(array $texts, string $source, string $target): ?array
    {
        try {
            $body = Http::timeout(180)->post($this->url().'/translate', ['texts' => array_map(self::forModel(...), array_values($texts)), 'source' => $source, 'target' => $target])
                ->throw()->json();
        } catch (ConnectionException) {
            return null;
        } catch (RequestException $e) {
            throw new RuntimeException('Layanan terjemahan menolak permintaan: '.($e->response->json('detail') ?? 'HTTP '.$e->response->status()));
        }
        if (! is_array($body['translations'] ?? null) || count($body['translations']) !== count($texts)) {
            // Layanan versi lama mengabaikan "texts" dan hanya mengenal satu arah.
            throw new RuntimeException('Layanan terjemahan yang berjalan versi lama (hanya Jawa → Indonesia); nyalakan ulang layanannya.');
        }

        return $body;
    }

    /** Terjemahan satu kalimat lewat layanan Demo; null bila layanan tidak berjalan. */
    public function translateOne(string $latin): ?array
    {
        try {
            return Http::timeout(120)->post($this->url().'/translate', ['text' => self::forModel($latin)])->throw()->json();
        } catch (ConnectionException|RequestException) {
            return null;
        }
    }
}
