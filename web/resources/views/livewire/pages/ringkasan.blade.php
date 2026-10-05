<div class="flex flex-col gap-4">
    @include('partials.page-head', ['eyebrow' => 'Gerbang proyek', 'heading' => 'Ringkasan',
        'aside' => $import ? 'Hasil diekspor '.$import->generated_at.' · diimpor '.$import->created_at->diffForHumans() : null])

    @if (! $import)
        @include('partials.no-results')
    @else
        @if ($import->limited)
            <div class="alert alert-warning">
                <span>Ekspor ini dibatasi (<span class="font-mono">--limit</span>): angka bukan hasil 745 baris penuh.</span>
            </div>
        @endif

        {{-- G1–G4: ubin ringkas. Kotak kode berwarna hijau bila gerbang lulus, merah bila belum. --}}
        <div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            @foreach ($gates as $gate)
                <div class="stat-card shadow-[var(--shadow-card)]">
                    <div class="stat-head">
                        <span class="stat-icon {{ $gate->passed ? 'stat-icon-green' : 'stat-icon-red' }}">{{ $gate->code }}</span>
                        <div class="min-w-0 flex-1">
                            <div class="stat-title">{{ $gate->name }}</div>
                            <div class="flex flex-wrap items-baseline gap-x-2">
                                <span class="stat-value">{{ $gate->code === 'G4' ? pct($gate->value, 0) : pct($gate->value, $gate->code === 'G3' ? 1 : 2) }}</span>
                                <span class="stat-note">target {{ $gate->code === 'G4' ? '= 100%' : '< '.pct($gate->target, 0) }}</span>
                            </div>
                        </div>
                    </div>
                    {{-- Garis pemisah sejajar di keempat ubin; jalur berkas menempel di dasar walau keterangannya beda panjang. --}}
                    <div class="flex flex-1 flex-col gap-1.5 border-t border-[color:var(--row-line)] pt-3">
                        <div class="flex items-start gap-2">
                            <span class="status {{ $gate->passed ? 'status-green' : 'status-red' }} shrink-0">{{ $gate->passed ? '✓ Lulus' : '✕ Belum' }}</span>
                            <span class="stat-hint min-w-0 pt-0.5">{{ $gate->basis }}</span>
                        </div>
                        <div class="code mt-auto break-all">{{ $gate->source }}</div>
                    </div>
                </div>
            @endforeach
        </div>

        {{-- Jarak G3 ke target: titik pada satu sumbu CER. Titik berdekatan (fonts, beam, run lanjutan fase6/fase7)
             membuat label di samping titik bertumpuk, apalagi di layar sempit: hanya titik resmi yang berlabel (di
             bawah sumbu, karena label target di atas), semua nilai ada di daftar di bawah sumbu. --}}
        @php
            $marker = fn ($p) => $p['official'] ? 'border-[var(--surface)] bg-[var(--series-1)]' : 'border-[var(--series-1)] bg-[var(--surface)]';
        @endphp
        <section class="card">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">Jarak G3 ke target</h2>
                    <p class="card-subtitle">CER pada 745 baris NusaAksara. Makin ke kiri makin baik; titik penuh = angka G3 resmi ({{ $official?->label ?? 'belum ditetapkan' }}, greedy), titik berongga = pembanding, nilainya di daftar bawah.</p>
                </div>
                <div class="relative mx-2 mt-10 mb-12 h-2 rounded-full bg-zinc-100 dark:bg-zinc-700" role="img"
                     aria-label="CER G3: {{ $points->map(fn ($p) => $p['label'].' '.pct($p['value']))->join(', ') }}; target 8%">
                    <div class="absolute inset-y-0 left-0 rounded-l-full bg-[var(--good-wash)]" style="width: 8%"></div>
                    <div class="absolute -top-7 bottom-0 w-0.5 bg-[var(--good-solid)]" style="left: 8%">
                        <span class="absolute -top-1 left-1.5 whitespace-nowrap text-xs font-medium text-[var(--good)]">target 8%</span>
                    </div>
                    @foreach ($points as $p)
                        <div class="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 {{ $p['official'] ? 'z-10' : '' }}" style="left: {{ $p['x'] }}%"
                             title="{{ $p['label'] }}: CER {{ pct($p['value'], 2) }}">
                            <span class="block size-3.5 rounded-full border-2 {{ $marker($p) }}"></span>
                            @if ($p['official'])
                                {{-- Dekat ujung sumbu, label dirapatkan ke dalam supaya tidak keluar kartu. --}}
                                <span class="absolute top-5 whitespace-nowrap text-xs font-medium text-zinc-800 dark:text-zinc-100 {{ $p['x'] < 15 ? 'left-0' : ($p['x'] > 85 ? 'right-0' : 'left-1/2 -translate-x-1/2') }}">
                                    {{ $p['label'] }} {{ pct($p['value']) }}
                                </span>
                            @endif
                        </div>
                    @endforeach
                </div>
                <div class="code flex justify-between"><span>0%</span><span>50%</span><span>100%</span></div>
                <ul class="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-xs text-zinc-600 dark:text-zinc-300" aria-label="CER G3 per pipeline, terbaik dulu">
                    @foreach ($points as $p)
                        <li class="inline-flex items-center gap-1.5">
                            <span class="inline-block size-3 shrink-0 rounded-full border-2 {{ $marker($p) }}"></span>
                            <span>{{ $p['label'] }}</span><span class="num font-semibold text-zinc-800 dark:text-zinc-100">{{ pct($p['value']) }}</span>
                        </li>
                    @endforeach
                </ul>
            </div>
        </section>

        {{-- Seluruh alur --}}
        <section class="card">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">Alur: citra → aksara → Latin → arti → tingkat tutur</h2>
                </div>
                <div class="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                    @php
                        $stages = [
                            ['1', 'OCR aksara', 'Citra baris → Unicode aksara Jawa', $officialMetric ? 'CER '.pct($officialMetric->cer).' (greedy)' : '–',
                                ($officialMetric ? $official->label.' · ' : '').($officialBeam ? 'beam + LM '.pct($officialBeam->cer).' · ' : '').'target < 8%',
                                'wait', 'Berjalan, belum capai target'],
                            ['2', 'Transliterasi', 'Aksara → Latin', $translit ? 'CER draf '.pct($translit->cer).' vs manusia' : 'belum dinilai',
                                "transliterasi manusia: {$humanTranslit}/{$lines} baris · draf aturan di web", 'wait', 'Draf aturan'],
                            ['3', 'Arti', 'Jawa → bahasa Indonesia',
                                $mtRuns->has('label') ? 'chrF '.number_format($mtRuns['label']->chrf, 1, ',', '.').' (NLLB, dari transliterasi manusia)' : "{$humanTranslation}/{$lines} baris punya arti manusia",
                                $mtRuns->has('label')
                                    ? "{$humanTranslation}/{$lines} baris punya arti manusia"
                                        .($ocrRun ? ' · dari keluaran OCR ('.$ocrRunLabel.') chrF '.number_format($ocrRun->chrf, 1, ',', '.') : '')
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
                        {{-- Pil status menempel di dasar ubin supaya sejajar antar-tahap walau isinya beda panjang. --}}
                        <div class="stat-card">
                            <div>
                                <div class="subheader">tahap {{ $n }}</div>
                                <div class="text-sm font-semibold">{{ $name }}</div>
                                <div class="stat-hint">{{ $io }}</div>
                            </div>
                            <div class="num text-sm font-semibold">{{ $main }}</div>
                            <div class="stat-hint">{{ $sub }}</div>
                            <div class="mt-auto"><span class="status pill-{{ $state }}">{{ $stateLabel }}</span></div>
                        </div>
                    @endforeach
                </div>
                <p class="mt-3 max-w-prose text-xs text-zinc-500 dark:text-zinc-400">Tahap 2–4 dinilai dari label (teks benar) terpisah dari keluaran OCR, supaya kesalahan OCR tidak terbaca sebagai kesalahan tahap berikutnya.</p>
            </div>
        </section>

        {{-- Fase --}}
        <section class="card">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">Fase pengerjaan OCR</h2>
                </div>
                <div class="table-wrap">
                    <table class="data-table">
                        <thead><tr><th>Fase</th><th>Isi</th><th>Status</th><th>Bukti</th></tr></thead>
                        <tbody>
                            @foreach ($phases as [$f, $isi, $state, $label, $bukti])
                                <tr>
                                    <td class="num font-semibold">{{ $f }}</td>
                                    <td>{{ $isi }}</td>
                                    <td><span class="status pill-{{ $state }}">{{ ['ok' => '✓', 'no' => '✕', 'wait' => '…'][$state] }} {{ $label }}</span></td>
                                    <td class="text-zinc-500 dark:text-zinc-400">{{ $bukti }}</td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
    @endif
</div>
