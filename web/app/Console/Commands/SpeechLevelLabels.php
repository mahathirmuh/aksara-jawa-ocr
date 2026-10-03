<?php

namespace App\Console\Commands;

use App\Models\LineAnnotation;
use App\Models\User;
use App\Support\SpeechLevel;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\DB;

/**
 * Cadangan label tingkat tutur dari manusia. Hanya data ini (dan akun) yang tidak bisa dibangun ulang
 * dari file hasil, jadi disalin ke JSON yang ikut repo.
 */
class SpeechLevelLabels extends Command
{
    protected $signature = 'aksara:labels {action : export atau import} {--path= : default database/labels/speech_levels.json}';

    protected $description = 'Ekspor/impor label tingkat tutur buatan manusia (data uji tahap 4)';

    public function handle(): int
    {
        $path = $this->option('path') ?: database_path('labels/speech_levels.json');

        return match ($this->argument('action')) {
            'export' => $this->export($path),
            'import' => $this->import($path),
            default => $this->fail('Aksi harus "export" atau "import".'),
        };
    }

    private function export(string $path): int
    {
        $rows = LineAnnotation::with('labeler')->whereNotNull('speech_level')->orderBy('external_id')->get()
            ->map(fn ($a) => ['dataset' => $a->dataset, 'external_id' => $a->external_id, 'speech_level' => $a->speech_level,
                'labeled_by' => $a->labeler?->email, 'labeled_at' => $a->labeled_at?->toIso8601String()])
            ->values();
        is_dir(dirname($path)) || mkdir(dirname($path), 0775, true);
        file_put_contents($path, json_encode($rows, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT)."\n");
        $this->info("{$rows->count()} label diekspor ke {$path}");

        return self::SUCCESS;
    }

    private function import(string $path): int
    {
        if (! is_file($path)) {
            $this->error("Berkas tidak ada: {$path}");

            return self::FAILURE;
        }
        $rows = json_decode(file_get_contents($path), true, flags: JSON_THROW_ON_ERROR);
        $users = User::pluck('id', 'email');
        DB::transaction(function () use ($rows, $users) {
            foreach ($rows as $r) {
                abort_unless(in_array($r['speech_level'], SpeechLevel::LEVELS, true), 422, "Label tidak sah: {$r['speech_level']}");
                $existing = LineAnnotation::where('dataset', $r['dataset'])->where('external_id', $r['external_id'])->first();
                LineAnnotation::updateOrCreate(
                    ['dataset' => $r['dataset'], 'external_id' => $r['external_id']],
                    ['speech_level' => $r['speech_level'], 'labeled_by' => $users[$r['labeled_by'] ?? ''] ?? null,
                        'labeled_at' => $r['labeled_at'], 'source' => $existing?->source ?? 'label manusia (web)'],
                );
            }
        });
        $this->info(count($rows)." label diimpor dari {$path}");

        return self::SUCCESS;
    }
}
