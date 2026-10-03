<?php

namespace App\Console\Commands;

use App\Services\TranslationService;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\DB;
use RuntimeException;

class TranslateLines extends Command
{
    protected $signature = 'aksara:translate
        {--source=label,crnn_fonts_beam : sumber teks: "label" (transliterasi manusia) dan/atau kunci pipeline OCR}
        {--limit=0 : hanya N baris pertama (uji cepat)}
        {--import-only : impor ulang hasil terakhir di storage/app/mt tanpa menjalankan NLLB (mis. setelah ganti database)}
        {--dump : kebalikan --import-only: tulis ulang output.jsonl & summary.json di storage/app/mt dari database (hanya membaca database)}';

    protected $description = 'Tahap 3: terjemahkan baris ke bahasa Indonesia (NLLB lokal) dan nilai terhadap terjemahan manusia';

    public function handle(TranslationService $mt): int
    {
        if ($this->option('dump') && $this->option('import-only')) {
            $this->error('Pilih salah satu: --dump (database -> berkas) atau --import-only (berkas -> database).');

            return self::FAILURE;
        }
        if ($this->option('dump')) {
            try {
                $summary = $mt->dump();
            } catch (RuntimeException $e) {
                $this->error($e->getMessage());

                return self::FAILURE;
            }
            $db = DB::connection();
            $this->info("{$summary['rows']} terjemahan dari database {$db->getDriverName()}:{$db->getDatabaseName()} ditulis ke {$mt->dir()}");
            $this->printRuns($summary['runs']);

            return self::SUCCESS;
        }
        if ($this->option('import-only')) {
            [, $out, $summary] = $mt->files();
            if (! is_file($out) || ! is_file($summary)) {
                $this->error("Belum ada hasil terjemahan di {$mt->dir()}. Pulihkan dengan --dump dari database yang masih "
                    .'memuatnya, atau jalankan aksara:translate tanpa --import-only (NLLB, ~1 jam CPU).');

                return self::FAILURE;
            }
            $this->printRuns($mt->import($out, $summary)['runs']);

            return self::SUCCESS;
        }
        $sources = array_filter(array_map('trim', explode(',', $this->option('source'))));
        $rows = $mt->inputs($sources, (int) $this->option('limit'));
        if (! $rows) {
            $this->error('Tidak ada teks untuk diterjemahkan. Jalankan dulu aksara:import dan aksara:annotations.');

            return self::FAILURE;
        }
        $this->info(count($rows).' teks dikirim ke NLLB lokal (model dimuat dulu, lalu ~1 detik per kalimat di CPU)...');
        try {
            $summary = $mt->runBatch($rows, fn ($buffer) => $this->output->write($buffer));
        } catch (RuntimeException $e) {
            $this->error($e->getMessage());

            return self::FAILURE;
        }
        $this->printRuns($summary['runs']);

        return self::SUCCESS;
    }

    private function printRuns(array $runs): void
    {
        foreach ($runs as $run) {
            $this->line("{$run['source']}: chrF {$run['chrf']} · BLEU {$run['bleu']} ({$run['lines']} baris)");
        }
    }
}
