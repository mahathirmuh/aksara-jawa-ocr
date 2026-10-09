<div class="flex flex-col gap-4">
    @include('partials.page-head', ['eyebrow' => 'Pipeline', 'heading' => 'Perbandingan pembaca',
        'aside' => 'Setiap hasil berlabel konfigurasi pipeline lengkap'])

    @if (! $hasData)
        @include('partials.no-results')
    @else
        <section class="card">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">745 baris nyata (G3 penuh)</h2>
                    <p class="card-subtitle max-w-4xl">Tiap baris = satu checkpoint dengan satu cara membaca (tertulis di bawah namanya), diukur pada baris yang sama. Beam + LM membaca checkpoint fase5_fonts yang sama dengan greedy; bobot LM-nya disetel di dev sintetis, bukan di NusaAksara.</p>
                </div>
                {{-- Kolom pipeline dan catatan diberi lebar tetap; di ponsel kolom pipeline dipersempit supaya CER
                     ikut terlihat tanpa menggeser tabel. Baris pipeline resmi disorot (.is-highlight). Presisi, recall,
                     F1 = metrik karakter dari penjajaran yang sama dengan CER (src/metrics.py); null pada ekspor lama. --}}
                <div class="table-wrap">
                    <table class="data-table min-w-[68rem]">
                        <thead><tr><th class="w-[14.5rem] sm:w-[32%]">Pipeline</th><th class="r">CER</th><th class="r">CER tanpa spasi</th><th class="r">Baris persis</th><th class="r">Akurasi karakter</th><th class="r">Presisi</th><th class="r">Recall</th><th class="r">F1</th><th class="sm:w-[22%]">Catatan</th></tr></thead>
                        <tbody>
                            @foreach ($full as $m)
                                @php($p = $pipelines[$m->pipeline] ?? null)
                                <tr @class(['is-highlight' => $m->pipeline === $official?->key]) data-row="{{ $m->pipeline }}">
                                    <td><div class="text-sm font-semibold">{{ $p?->label ?? $m->pipeline }}</div><div class="code mt-0.5">{{ $p?->config }}</div></td>
                                    <td class="r num text-sm font-semibold" data-cell="cer">{{ pct($m->cer, 2) }}</td>
                                    <td class="r num" data-cell="cer_no_space">{{ pct($m->cer_no_space, 2) }}</td>
                                    <td class="r num" data-cell="exact">{{ pct($m->exact, 1) }}</td>
                                    <td class="r num" data-cell="char_accuracy">{{ pct($m->char_accuracy, 1) }}</td>
                                    <td class="r num" data-cell="precision">{{ pct($m->precision, 1) }}</td>
                                    <td class="r num" data-cell="recall">{{ pct($m->recall, 1) }}</td>
                                    <td class="r num font-semibold" data-cell="f1">{{ pct($m->f1, 1) }}</td>
                                    <td class="text-muted">
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
                {{-- Definisi metrik karakter, sekali di bawah tabel; "–" bila hasil diimpor dari ekspor sebelum metrik ini ada. --}}
                <p class="text-muted mt-3 max-w-4xl text-xs" data-part="definisi-metrik">
                    CER = (tertukar + hilang + tambahan) / panjang label. Dari penjajaran karakter yang sama:
                    recall = bagian karakter label yang terbaca benar (1 − recall = tertukar + hilang per panjang label),
                    presisi = bagian keluaran model yang benar (turun bila model menambah karakter), F1 = rata-rata harmonik
                    keduanya, akurasi karakter = cocok / (cocok + tertukar + hilang + tambahan). Baris persis = akurasi
                    tingkat baris (seluruh baris sama dengan label). Semua dihitung per karakter atas seluruh baris, termasuk spasi.
                    @if ($full->contains(fn ($m) => $m->f1 === null))
                        "–" = hasil diimpor dari ekspor sebelum metrik ini ada; jalankan ulang <code class="code">export_results.py</code> lalu <code class="code">aksara:import</code>.
                    @endif
                </p>
                {{-- Run lanjutan (fase6, fase7; daftar dan catatannya di Perbandingan::FOLLOWUP_RUNS) yang ada di tabel.
                     Bila pipeline resmi sendiri salah satu run itu, ia juga satu run yang diukur pada test set ini. --}}
                @if ($followups->isNotEmpty())
                    <div class="alert alert-warning mt-4" role="note">
                        <p class="min-w-0">
                            @if ($officialIsFollowup)
                                {{ $followups->join(', ', ' dan ') }}: satu run per kondisi, diukur pada test set yang sama dengan angka resmi. Baca
                                selisihnya sebagai arah. Angka G3 resmi = {{ $official->label }} (greedy), juga satu run; NusaAksara dipakai
                                berulang untuk membandingkan run, jadi angka terbaiknya sedikit optimistis.
                            @else
                                {{ $followups->join(', ', ' dan ') }}: satu run per kondisi, diukur lagi pada test set yang sama. Baca selisihnya
                                sebagai arah, bukan angka resmi; angka G3 resmi tetap {{ $official?->label ?? 'CRNN fase5_fonts' }} (greedy).
                            @endif
                        </p>
                    </div>
                @endif
            </div>
        </section>

        <section class="card">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">Tahap 2–3: transliterasi & arti</h2>
                    <p class="card-subtitle max-w-4xl">Dinilai terhadap transliterasi dan terjemahan manusia NusaAksara. Baris "dari transliterasi manusia" mengukur model tahap itu saja; baris "dari keluaran OCR" mengukur alur ujung ke ujung.</p>
                </div>
                <div class="table-wrap">
                    <table class="data-table min-w-[34rem]">
                        <thead><tr><th>Tahap</th><th>Masukan</th><th class="r">Metrik</th><th class="r">Baris</th></tr></thead>
                        <tbody>
                            @if ($translit)
                                <tr><td>2 · transliterasi draf (aturan)</td><td>label aksara</td><td class="r num whitespace-nowrap">CER {{ pct($translit->cer) }}</td><td class="r num">{{ $translit->lines }}</td></tr>
                            @endif
                            @forelse ($mtRuns as $run)
                                <tr>
                                    <td>3 · terjemahan <span class="code">{{ $run->model }}</span></td>
                                    <td>{{ $run->source === 'label' ? 'transliterasi manusia' : 'keluaran OCR ('.($pipelines[$run->source]->label ?? $run->source).') → Latin draf' }}</td>
                                    <td class="r num whitespace-nowrap">chrF {{ number_format($run->chrf, 1, ',', '.') }} · BLEU {{ number_format($run->bleu, 1, ',', '.') }}</td>
                                    <td class="r num">{{ $run->lines }}</td>
                                </tr>
                            @empty
                                <tr><td colspan="4" class="text-muted">Terjemahan mesin belum dijalankan: <code class="code">php artisan aksara:translate</code></td></tr>
                            @endforelse
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <div class="grid gap-4 lg:grid-cols-2">
            <section class="card">
                <div class="card-body">
                    <div class="mb-3">
                        <h2 class="section-title">50 baris uji buta</h2>
                        <p class="card-subtitle">Satu-satunya subset yang juga dibaca VLM. CER, makin pendek makin baik.</p>
                    </div>
                    {{-- Tinggi ikut jumlah batang supaya kerapatannya tetap: tiap run lanjutan yang diekspor menambah satu
                         batang. Paling kecil 16rem (h-64 semula, cukup untuk 7 batang). --}}
                    <div class="relative" style="height: {{ max(16, 3 + 1.75 * $blind->count()) }}rem" x-data="chart('bars', @js($blindChart))" wire:ignore>
                        <canvas x-ref="canvas" role="img" aria-label="CER per pipeline pada 50 baris uji buta"></canvas>
                    </div>
                    <div class="table-wrap mt-4">
                        <table class="data-table">
                            <thead><tr><th>Pipeline</th><th class="r">CER</th><th class="r">Tanpa spasi</th><th class="r">Akurasi</th><th class="r">Presisi</th><th class="r">Recall</th><th class="r">F1</th></tr></thead>
                            <tbody>
                                @foreach ($blind as $m)
                                    <tr @class(['is-highlight' => $m->pipeline === $official?->key]) data-blind="{{ $m->pipeline }}"><td>{{ $pipelines[$m->pipeline]->label ?? $m->pipeline }}</td><td class="r num" data-cell="cer">{{ pct($m->cer) }}</td><td class="r num" data-cell="cer_no_space">{{ pct($m->cer_no_space) }}</td><td class="r num" data-cell="char_accuracy">{{ pct($m->char_accuracy) }}</td><td class="r num" data-cell="precision">{{ pct($m->precision) }}</td><td class="r num" data-cell="recall">{{ pct($m->recall) }}</td><td class="r num font-semibold" data-cell="f1">{{ pct($m->f1) }}</td></tr>
                                @endforeach
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>

            <div class="flex flex-col gap-4">
                <section class="card">
                    <div class="card-body">
                        <div class="mb-3">
                            <h2 class="section-title">Pipeline yang akan masuk</h2>
                            <p class="card-subtitle">Muncul otomatis di halaman ini setelah hasilnya diekspor dan diimpor.</p>
                        </div>
                        <ul class="table-wrap text-xs">
                            @foreach ($pipelines->where('status', 'planned') as $p)
                                <li class="flex items-center justify-between gap-3 border-t border-[var(--row-line)] p-3 first:border-t-0">
                                    <div class="min-w-0"><div class="text-sm font-semibold">{{ $p->label }}</div><div class="code mt-0.5">{{ $p->config }}</div></div>
                                    <span class="status status-yellow">… belum</span>
                                </li>
                            @endforeach
                        </ul>
                    </div>
                </section>

                <section class="card">
                    <div class="card-body">
                        <div class="mb-3">
                            <h2 class="section-title">Data sintetis</h2>
                            <p class="card-subtitle">Beam + LM juga membantu di luar data nyata (checkpoint fase5_fonts).</p>
                        </div>
                        <div class="table-wrap">
                            <table class="data-table">
                                <thead><tr><th>Set</th><th class="r">Greedy</th><th class="r">Beam + LM</th></tr></thead>
                                <tbody>
                                    @foreach (['synth_dev' => 'Dev sintetis (aug fase5)', 'synth_heldout' => 'Font held-out bersih'] as $scope => $name)
                                        @if ($synth->has($scope))
                                            <tr>
                                                <td>{{ $name }} <span class="text-muted">({{ $synth[$scope]->first()->lines }} baris)</span></td>
                                                <td class="r num">{{ pct($synth[$scope]['crnn_fonts']->cer ?? null, 2) }}</td>
                                                <td class="r num">{{ pct($synth[$scope]['crnn_fonts_beam']->cer ?? null, 2) }}</td>
                                            </tr>
                                        @endif
                                    @endforeach
                                </tbody>
                            </table>
                        </div>
                    </div>
                </section>
            </div>
        </div>
    @endif
</div>
