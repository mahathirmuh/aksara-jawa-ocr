<div class="mx-auto flex max-w-6xl flex-col gap-6">
    @include('partials.page-head', ['eyebrow' => 'Pipeline', 'heading' => 'Perbandingan pembaca',
        'aside' => 'Setiap hasil berlabel konfigurasi pipeline lengkap'])

    @if (! $hasData)
        @include('partials.no-results')
    @else
        <section class="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900">
            <flux:heading size="lg">745 baris nyata (G3 penuh)</flux:heading>
            <p class="mt-1 max-w-prose text-sm text-zinc-500">Tiap baris = satu checkpoint dengan satu cara membaca (tertulis di bawah namanya), diukur pada baris yang sama. Beam + LM membaca checkpoint fase5_fonts yang sama dengan greedy; bobot LM-nya disetel di dev sintetis, bukan di NusaAksara.</p>
            <div class="mt-3 overflow-x-auto">
                <table class="data-table w-full text-sm">
                    <thead><tr><th>Pipeline</th><th class="r">CER</th><th class="r">CER tanpa spasi</th><th class="r">Baris persis</th><th>Catatan</th></tr></thead>
                    <tbody>
                        @foreach ($full as $m)
                            @php($p = $pipelines[$m->pipeline] ?? null)
                            <tr>
                                <td><div class="font-semibold">{{ $p?->label ?? $m->pipeline }}</div><div class="font-mono text-[11px] text-zinc-500">{{ $p?->config }}</div></td>
                                <td class="r num font-medium">{{ pct($m->cer, 2) }}</td>
                                <td class="r num">{{ pct($m->cer_no_space, 2) }}</td>
                                <td class="r num">{{ pct($m->exact, 1) }}</td>
                                <td class="text-zinc-500">
                                    @if ($m->pipeline === $official?->key) angka G3 resmi{{ isset($runNotes[$m->pipeline]) ? ' · '.$runNotes[$m->pipeline] : '' }}
                                    @elseif (isset($runNotes[$m->pipeline])) {{ $runNotes[$m->pipeline] }}
                                    @elseif ($m->better !== null) {{ $m->better }} baris membaik, {{ $m->worse }} memburuk dibanding greedy checkpoint yang sama
                                    @elseif ($m->pipeline === 'crnn_fonts') angka G3 resmi sebelumnya
                                    @elseif ($m->pipeline === 'crnn_core') tanpa 8 font tambahan
                                    @elseif ($m->pipeline === 'crnn_4b') tanpa augmentasi, 2 font
                                    @endif
                                </td>
                            </tr>
                        @endforeach
                    </tbody>
                </table>
            </div>
            {{-- Run lanjutan (fase6, fase7; daftar dan catatannya di Perbandingan::FOLLOWUP_RUNS) yang ada di tabel.
                 Bila pipeline resmi sendiri salah satu run itu, ia juga satu run yang diukur pada test set ini. --}}
            @if ($followups->isNotEmpty())
                <p class="mt-3 max-w-prose rounded-lg bg-[var(--warn-wash)] px-3 py-2 text-xs text-[var(--warn)]">
                    @if ($officialIsFollowup)
                        {{ $followups->join(', ', ' dan ') }}: satu run per kondisi, diukur pada test set yang sama dengan angka resmi. Baca
                        selisihnya sebagai arah. Angka G3 resmi = {{ $official->label }} (greedy), juga satu run; NusaAksara dipakai
                        berulang untuk membandingkan run, jadi angka terbaiknya sedikit optimistis.
                    @else
                        {{ $followups->join(', ', ' dan ') }}: satu run per kondisi, diukur lagi pada test set yang sama. Baca selisihnya
                        sebagai arah, bukan angka resmi; angka G3 resmi tetap {{ $official?->label ?? 'CRNN fase5_fonts' }} (greedy).
                    @endif
                </p>
            @endif
        </section>

        <section class="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900">
            <flux:heading size="lg">Tahap 2–3: transliterasi & arti</flux:heading>
            <p class="mt-1 max-w-prose text-sm text-zinc-500">Dinilai terhadap transliterasi dan terjemahan manusia NusaAksara. Baris "dari transliterasi manusia" mengukur model tahap itu saja; baris "dari keluaran OCR" mengukur alur ujung ke ujung.</p>
            <div class="mt-3 overflow-x-auto">
                <table class="data-table w-full text-sm">
                    <thead><tr><th>Tahap</th><th>Masukan</th><th class="r">Metrik</th><th class="r">Baris</th></tr></thead>
                    <tbody>
                        @if ($translit)
                            <tr><td>2 · transliterasi draf (aturan)</td><td>label aksara</td><td class="r num">CER {{ pct($translit->cer) }}</td><td class="r num">{{ $translit->lines }}</td></tr>
                        @endif
                        @forelse ($mtRuns as $run)
                            <tr>
                                <td>3 · terjemahan <span class="font-mono text-[11px] text-zinc-500">{{ $run->model }}</span></td>
                                <td>{{ $run->source === 'label' ? 'transliterasi manusia' : 'keluaran OCR ('.($pipelines[$run->source]->label ?? $run->source).') → Latin draf' }}</td>
                                <td class="r num">chrF {{ number_format($run->chrf, 1, ',', '.') }} · BLEU {{ number_format($run->bleu, 1, ',', '.') }}</td>
                                <td class="r num">{{ $run->lines }}</td>
                            </tr>
                        @empty
                            <tr><td colspan="4" class="text-zinc-500">Terjemahan mesin belum dijalankan: <code class="font-mono text-xs">php artisan aksara:translate</code></td></tr>
                        @endforelse
                    </tbody>
                </table>
            </div>
        </section>

        <div class="grid gap-6 lg:grid-cols-2">
            <section class="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900">
                <flux:heading size="lg">50 baris uji buta</flux:heading>
                <p class="mt-1 text-sm text-zinc-500">Satu-satunya subset yang juga dibaca VLM. CER, makin pendek makin baik.</p>
                {{-- Tinggi ikut jumlah batang supaya kerapatannya tetap: tiap run lanjutan yang diekspor menambah satu
                     batang. Paling kecil 16rem (h-64 semula, cukup untuk 7 batang). --}}
                <div class="relative mt-3" style="height: {{ max(16, 3 + 1.75 * $blind->count()) }}rem" x-data="chart('bars', @js($blindChart))" wire:ignore>
                    <canvas x-ref="canvas" role="img" aria-label="CER per pipeline pada 50 baris uji buta"></canvas>
                </div>
                <table class="data-table mt-2 w-full text-sm">
                    <thead><tr><th>Pipeline</th><th class="r">CER</th><th class="r">Tanpa spasi</th></tr></thead>
                    <tbody>
                        @foreach ($blind as $m)
                            <tr><td>{{ $pipelines[$m->pipeline]->label ?? $m->pipeline }}</td><td class="r num">{{ pct($m->cer) }}</td><td class="r num">{{ pct($m->cer_no_space) }}</td></tr>
                        @endforeach
                    </tbody>
                </table>
            </section>

            <div class="flex flex-col gap-6">
                <section class="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900">
                    <flux:heading size="lg">Pipeline yang akan masuk</flux:heading>
                    <p class="mt-1 text-sm text-zinc-500">Muncul otomatis di halaman ini setelah hasilnya diekspor dan diimpor.</p>
                    <table class="data-table mt-3 w-full text-sm">
                        <tbody>
                            @foreach ($pipelines->where('status', 'planned') as $p)
                                <tr>
                                    <td><div class="font-semibold text-zinc-500">{{ $p->label }}</div><div class="font-mono text-[11px] text-zinc-500">{{ $p->config }}</div></td>
                                    <td class="r"><span class="pill pill-wait">… belum</span></td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </section>

                <section class="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900">
                    <flux:heading size="lg">Data sintetis</flux:heading>
                    <p class="mt-1 text-sm text-zinc-500">Beam + LM juga membantu di luar data nyata (checkpoint fase5_fonts).</p>
                    <table class="data-table mt-3 w-full text-sm">
                        <thead><tr><th>Set</th><th class="r">Greedy</th><th class="r">Beam + LM</th></tr></thead>
                        <tbody>
                            @foreach (['synth_dev' => 'Dev sintetis (aug fase5)', 'synth_heldout' => 'Font held-out bersih'] as $scope => $name)
                                @if ($synth->has($scope))
                                    <tr>
                                        <td>{{ $name }} <span class="text-zinc-500">({{ $synth[$scope]->first()->lines }} baris)</span></td>
                                        <td class="r num">{{ pct($synth[$scope]['crnn_fonts']->cer ?? null, 2) }}</td>
                                        <td class="r num">{{ pct($synth[$scope]['crnn_fonts_beam']->cer ?? null, 2) }}</td>
                                    </tr>
                                @endif
                            @endforeach
                        </tbody>
                    </table>
                </section>
            </div>
        </div>
    @endif
</div>
