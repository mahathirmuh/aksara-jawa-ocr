<div class="mx-auto flex max-w-6xl flex-col gap-6">
    @include('partials.page-head', ['eyebrow' => 'Gerbang proyek', 'heading' => 'Ringkasan',
        'aside' => $import ? 'Hasil diekspor '.$import->generated_at.' · diimpor '.$import->created_at->diffForHumans() : null])

    @if (! $import)
        @include('partials.no-results')
    @else
        @if ($import->limited)
            <div class="rounded-lg bg-[var(--warn-wash)] px-4 py-2 text-sm text-[var(--warn)]">
                Ekspor ini dibatasi (<span class="font-mono">--limit</span>): angka bukan hasil 745 baris penuh.
            </div>
        @endif

        {{-- G1–G4 --}}
        <div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            @foreach ($gates as $gate)
                <div class="flex flex-col gap-2 rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-900">
                    <div class="flex items-center justify-between">
                        <span class="font-mono text-sm text-zinc-500">{{ $gate->code }}</span>
                        <span class="pill {{ $gate->passed ? 'pill-ok' : 'pill-no' }}">{{ $gate->passed ? '✓ Lulus' : '✕ Belum' }}</span>
                    </div>
                    <div class="font-medium">{{ $gate->name }}</div>
                    <div class="num text-3xl font-semibold tracking-tight">
                        {{ $gate->code === 'G4' ? pct($gate->value, 0) : pct($gate->value, $gate->code === 'G3' ? 1 : 2) }}
                        <span class="text-sm font-medium tracking-normal text-zinc-500">target {{ $gate->code === 'G4' ? '= 100%' : '< '.pct($gate->target, 0) }}</span>
                    </div>
                    <div class="mt-auto text-xs text-zinc-600 dark:text-zinc-300">{{ $gate->basis }}</div>
                    <div class="break-all font-mono text-[11px] text-zinc-500">{{ $gate->source }}</div>
                </div>
            @endforeach
        </div>

        {{-- Jarak G3 ke target: titik pada satu sumbu CER. Titik berdekatan (fonts, beam, run lanjutan fase6/fase7)
             membuat label di samping titik bertumpuk, apalagi di layar sempit: hanya titik resmi yang berlabel (di
             bawah sumbu, karena label target di atas), semua nilai ada di daftar di bawah sumbu. --}}
        @php
            $marker = fn ($p) => $p['official'] ? 'border-white bg-[var(--series-1)] dark:border-zinc-900' : 'border-[var(--series-1)] bg-white dark:bg-zinc-900';
        @endphp
        <section class="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900">
            <flux:heading size="lg">Jarak G3 ke target</flux:heading>
            <p class="mt-1 text-sm text-zinc-500">CER pada 745 baris NusaAksara. Makin ke kiri makin baik; titik penuh = angka G3 resmi (fase5_fonts, greedy), titik berongga = pembanding, nilainya di daftar bawah.</p>
            <div class="relative mx-2 mt-10 mb-12 h-2 rounded-full bg-zinc-100 dark:bg-zinc-800" role="img"
                 aria-label="CER G3: {{ $points->map(fn ($p) => $p['label'].' '.pct($p['value']))->join(', ') }}; target 8%">
                <div class="absolute inset-y-0 left-0 rounded-l-full bg-[var(--good-wash)]" style="width: 8%"></div>
                <div class="absolute -top-7 bottom-0 w-0.5 bg-[var(--good)]" style="left: 8%">
                    <span class="absolute -top-1 left-1.5 whitespace-nowrap text-xs text-[var(--good)]">target 8%</span>
                </div>
                @foreach ($points as $p)
                    <div class="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 {{ $p['official'] ? 'z-10' : '' }}" style="left: {{ $p['x'] }}%"
                         title="{{ $p['label'] }}: CER {{ pct($p['value'], 2) }}">
                        <span class="block size-3.5 rounded-full border-2 {{ $marker($p) }}"></span>
                        @if ($p['official'])
                            {{-- Dekat ujung sumbu, label dirapatkan ke dalam supaya tidak keluar kartu. --}}
                            <span class="absolute top-5 whitespace-nowrap text-xs text-zinc-600 dark:text-zinc-300 {{ $p['x'] < 15 ? 'left-0' : ($p['x'] > 85 ? 'right-0' : 'left-1/2 -translate-x-1/2') }}">
                                {{ $p['label'] }} {{ pct($p['value']) }}
                            </span>
                        @endif
                    </div>
                @endforeach
            </div>
            <div class="flex justify-between font-mono text-[11px] text-zinc-500"><span>0%</span><span>50%</span><span>100%</span></div>
            <ul class="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-zinc-600 dark:text-zinc-300" aria-label="CER G3 per pipeline, terbaik dulu">
                @foreach ($points as $p)
                    <li class="inline-flex items-center gap-1.5">
                        <span class="inline-block size-3 shrink-0 rounded-full border-2 {{ $marker($p) }}"></span>
                        <span>{{ $p['label'] }}</span><span class="num font-medium">{{ pct($p['value']) }}</span>
                    </li>
                @endforeach
            </ul>
        </section>

        {{-- Seluruh alur --}}
        <section class="flex flex-col gap-3">
            <flux:heading size="lg">Alur: citra → aksara → Latin → arti → tingkat tutur</flux:heading>
            <div class="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                @php
                    $stages = [
                        ['1', 'OCR aksara', 'Citra baris → Unicode aksara Jawa', $best->has('crnn_fonts') ? 'CER '.pct($best['crnn_fonts']->cer).' (greedy)' : '–',
                            $best->has('crnn_fonts_beam') ? 'beam + LM '.pct($best['crnn_fonts_beam']->cer).' · target < 8%' : 'target < 8%', 'wait', 'Berjalan, belum capai target'],
                        ['2', 'Transliterasi', 'Aksara → Latin', $translit ? 'CER draf '.pct($translit->cer).' vs manusia' : 'belum dinilai',
                            "transliterasi manusia: {$humanTranslit}/{$lines} baris · draf aturan di web", 'wait', 'Draf aturan'],
                        ['3', 'Arti', 'Jawa → bahasa Indonesia',
                            $mtRuns->has('label') ? 'chrF '.number_format($mtRuns['label']->chrf, 1, ',', '.').' (NLLB, dari transliterasi manusia)' : "{$humanTranslation}/{$lines} baris punya arti manusia",
                            $mtRuns->has('label')
                                ? "{$humanTranslation}/{$lines} baris punya arti manusia"
                                    .($mtRuns->has('crnn_fonts_beam') ? ' · dari keluaran OCR chrF '.number_format($mtRuns['crnn_fonts_beam']->chrf, 1, ',', '.') : '')
                                : 'model terjemahan belum dijalankan (php artisan aksara:translate)',
                            $mtRuns->has('label') ? 'wait' : 'no', $mtRuns->has('label') ? 'Model dasar' : 'Belum dijalankan'],
                        ['4', 'Tingkat tutur', 'ngoko · madya · krama · campur',
                            $speechEval['lines'] ? 'akurasi leksikon '.pct($speechEval['accuracy']).' pada '.$speechEval['lines'].' baris' : 'leksikon belum dievaluasi',
                            "{$speechEval['lines']}/{$lines} baris berlabel manusia · prediksi per baris: "
                                .$speechAuto->map(fn ($n, $level) => "{$level} {$n}")->join(', ')
                                .' · per halaman ('.$speechPages->sum().'): '.$speechPages->map(fn ($n, $level) => "{$level} {$n}")->join(', '),
                            'wait', 'Baseline leksikon'],
                    ];
                @endphp
                @foreach ($stages as [$n, $name, $io, $main, $sub, $state, $stateLabel])
                    <div class="flex flex-col gap-1.5 rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-700 dark:bg-zinc-900">
                        <div class="flex items-center justify-between gap-2">
                            <span class="font-mono text-xs text-zinc-500">tahap {{ $n }}</span>
                            <span class="pill pill-{{ $state }}">{{ $stateLabel }}</span>
                        </div>
                        <div class="font-semibold">{{ $name }}</div>
                        <div class="text-xs text-zinc-500">{{ $io }}</div>
                        <div class="num mt-1 font-medium">{{ $main }}</div>
                        <div class="text-xs text-zinc-600 dark:text-zinc-300">{{ $sub }}</div>
                    </div>
                @endforeach
            </div>
            <p class="max-w-prose text-xs text-zinc-500">Tahap 2–4 dinilai dari label (teks benar) terpisah dari keluaran OCR, supaya kesalahan OCR tidak terbaca sebagai kesalahan tahap berikutnya.</p>
        </section>

        {{-- Fase --}}
        <section class="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900">
            <flux:heading size="lg">Fase pengerjaan OCR</flux:heading>
            <div class="mt-3 overflow-x-auto">
                <table class="data-table w-full text-sm">
                    <thead><tr><th>Fase</th><th>Isi</th><th>Status</th><th>Bukti</th></tr></thead>
                    <tbody>
                        @foreach ($phases as [$f, $isi, $state, $label, $bukti])
                            <tr>
                                <td class="font-mono">{{ $f }}</td>
                                <td>{{ $isi }}</td>
                                <td><span class="pill pill-{{ $state }}">{{ ['ok' => '✓', 'no' => '✕', 'wait' => '…'][$state] }} {{ $label }}</span></td>
                                <td class="text-zinc-500">{{ $bukti }}</td>
                            </tr>
                        @endforeach
                    </tbody>
                </table>
            </div>
        </section>
    @endif
</div>
