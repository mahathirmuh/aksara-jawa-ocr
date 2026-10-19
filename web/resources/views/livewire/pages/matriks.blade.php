@php
    // Selisih dalam poin persentase, dua desimal, bertanda (tanda minus tipografis).
    $pp = fn (?float $x) => $x === null ? '–' : ($x > 0 ? '+' : ($x < 0 ? '−' : '')).number_format(abs($x) * 100, 2, ',', '.');
    $ci = fn (?array $c) => $c ? '['.$pp($c[0]).'; '.$pp($c[1]).']' : '';
    $yes = fn ($v) => $v === null ? '–' : ($v ? 'ya' : 'tidak');
@endphp
<div class="flex flex-col gap-4">
    @include('partials.page-head', ['eyebrow' => 'Pipeline', 'heading' => 'Matriks metode',
        'aside' => 'OCR aksara Jawa: metode kita lawan metode lain pada baris nyata yang sama'])

    @if (! $matrix)
        <section class="card"><div class="card-body text-muted">
            Matriks belum diimpor. Di repo OCR: <code class="code">python scripts/method_matrix.py export</code>, lalu di folder web:
            <code class="code">php artisan aksara:matrix</code>.
        </div></section>
    @else
        <section class="card">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">Perbandingan metode OCR</h2>
                    <p class="card-subtitle max-w-4xl">
                        Potongan baris NusaAksara yang sama untuk semua metode ({{ $matrix['subsets']['cetak']['lines'] }} baris cetak dari
                        {{ $matrix['subsets']['cetak']['pages'] }} halaman sebagai ukuran utama; {{ $matrix['subsets']['semua']['lines'] }}
                        baris = G3). Semua keluaran dinilai dengan aturan yang sama: {{ $matrix['rules'] }}.
                        <b>NED</b> (normalized edit distance) = jumlah suntingan dibagi jumlah panjang terpanjang label/keluaran tiap baris,
                        selalu 0–100% (kompetisi ICDAR 2019 merata-ratakan rasio yang sama per baris);
                        <b>CER</b> = suntingan dibagi panjang label, angka gerbang resmi, bisa lewat 100% (100% = keluaran kosong). Selisih
                        terhadap model resmi dengan selang kepercayaan 95% bootstrap per halaman; negatif = lebih baik.
                    </p>
                </div>
                <div class="table-wrap">
                    <table class="data-table min-w-[72rem]">
                        <thead><tr>
                            <th class="w-[24%]">Metode</th><th>Kelompok</th><th class="r">NED cetak</th><th class="r">Δ NED cetak, SK 95%</th>
                            <th class="r">CER cetak</th><th class="r">Δ CER cetak, SK 95%</th><th class="r">CER tulisan tangan</th>
                            <th class="r">CER semua (G3)</th><th class="r">Baris persis</th>
                        </tr></thead>
                        <tbody>
                            @foreach ($methods as $r)
                                <tr data-method="{{ $r['key'] }}" @class(['is-highlight' => $r['key'] === $matrix['reference']])>
                                    <td>
                                        <div class="text-sm font-semibold">{{ $r['label'] }}</div>
                                        <div class="text-muted mt-0.5 text-xs">{{ $r['note'] }}</div>
                                    </td>
                                    <td><span class="status {{ \App\Livewire\Pages\Matriks::GROUP_STATUS[$r['group']] }}">{{ \App\Livewire\Pages\Matriks::GROUP_LABELS[$r['group']] }}</span></td>
                                    @if ($r['cetak'])
                                        <td class="r num text-sm font-semibold" data-cell="ned">{{ pct($r['cetak']['ned'], 2) }}</td>
                                        <td class="r num whitespace-nowrap" data-cell="ned_delta">
                                            @if (isset($r['cetak']['ned_delta'])) {{ $pp($r['cetak']['ned_delta']) }} <span class="text-muted">{{ $ci($r['cetak']['ned_ci95']) }}</span>
                                            @else <span class="text-muted">pembanding</span> @endif
                                        </td>
                                        <td class="r num" data-cell="cetak">{{ pct($r['cetak']['cer'], 2) }}</td>
                                        <td class="r num whitespace-nowrap" data-cell="delta">
                                            @if (isset($r['cetak']['delta'])) {{ $pp($r['cetak']['delta']) }} <span class="text-muted">{{ $ci($r['cetak']['ci95']) }}</span>
                                            @else <span class="text-muted">pembanding</span> @endif
                                        </td>
                                        <td class="r num" data-cell="tulisan_tangan">{{ $r['tulisan_tangan'] ? pct($r['tulisan_tangan']['cer'], 2) : '–' }}</td>
                                        <td class="r num" data-cell="semua">{{ $r['semua'] ? pct($r['semua']['cer'], 2) : '–' }}</td>
                                        <td class="r num">{{ pct($r['cetak']['exact'], 1) }}</td>
                                    @else
                                        <td colspan="7" class="text-muted" data-cell="partial">
                                            hanya {{ $r['partial']['lines'] }} baris dibaca: NED {{ pct($r['partial']['ned'] ?? null, 1) }}, CER {{ pct($r['partial']['cer'], 1) }}
                                            (model resmi pada baris yang sama: NED {{ pct($r['partial']['reference_ned'] ?? null, 1) }}, CER {{ pct($r['partial']['reference_cer'], 1) }})
                                        </td>
                                    @endif
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
                <div class="relative mt-4" style="height: {{ max(12, 3 + 2 * count($chart['labels'])) }}rem" x-data="chart('bars', @js($chart))" wire:ignore>
                    <canvas x-ref="canvas" role="img" aria-label="NED baris cetak per metode"></canvas>
                </div>
            </div>
        </section>

        <section class="card">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">Ciri praktis</h2>
                    <p class="card-subtitle max-w-4xl">Waktu baca diukur per baris di laptop ini (CPU untuk pembaca lokal, termasuk antrean API untuk VLM). Biaya hanya bila harga publiknya diketahui.</p>
                </div>
                <div class="table-wrap">
                    <table class="data-table min-w-[44rem]">
                        <thead><tr><th>Metode</th><th>Berjalan offline</th><th>Dilatih untuk aksara Jawa</th><th class="r">Parameter</th><th class="r">Detik per baris</th><th class="r">Token keluaran per baris</th><th class="r">USD per 1.000 baris</th></tr></thead>
                        <tbody>
                            @foreach ($methods as $r)
                                <tr data-practical="{{ $r['key'] }}">
                                    <td class="text-sm font-semibold">{{ $r['label'] }}</td>
                                    <td>{{ $yes($r['offline']) }}</td>
                                    <td>{{ $yes($r['trained']) }}</td>
                                    <td class="r num">{{ $r['params'] ? nfmt($r['params']) : '–' }}</td>
                                    <td class="r num">{{ isset($r['seconds_per_line']) ? number_format($r['seconds_per_line'], 2, ',', '.') : '–' }}</td>
                                    <td class="r num">{{ isset($r['output_tokens_per_line']) ? nfmt($r['output_tokens_per_line']) : '–' }}</td>
                                    <td class="r num">{{ isset($r['usd_per_1000_lines']) ? number_format($r['usd_per_1000_lines'], 2, ',', '.') : '–' }}</td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        @if ($matrix['architecture'])
            <section class="card">
                <div class="card-body">
                    <div class="mb-3">
                        <h2 class="section-title">Arsitektur: BiLSTM lawan Transformer</h2>
                        <p class="card-subtitle max-w-4xl">
                            CNN, data, langkah, dan jadwal belajar sama; yang beda hanya pemodel urutan sesudah CNN. Pasangan pertama
                            memakai resep run <span class="code">base</span> (1.500 langkah dari nol, 2 font, tanpa augmentasi: belum mengenal
                            cetakan nyata); pasangan resep penuh melatih Transformer ulang lewat rantai yang sama dengan model resmi.
                        </p>
                    </div>
                    <div class="table-wrap">
                        <table class="data-table min-w-[44rem]">
                            <thead><tr><th>Model</th><th class="r">Parameter</th><th class="r">Val bersih (font latih)</th><th class="r">G1 javatext bersih</th><th class="r">G2 javatext berat</th><th class="r">Cetak nyata, NED / CER</th></tr></thead>
                            <tbody>
                                @foreach ($matrix['architecture'] as $r)
                                    <tr data-arch="{{ $r['key'] }}">
                                        <td>
                                            <div class="text-sm font-semibold">{{ $r['label'] }}</div>
                                            <div class="text-muted mt-0.5 text-xs">{{ $r['note'] ?? '' }}</div>
                                        </td>
                                        <td class="r num">{{ nfmt($r['params']) }}</td>
                                        <td class="r num">{{ pct($r['val_clean'] ?? null, 2) }}</td>
                                        <td class="r num">{{ pct($r['g1'] ?? null, 2) }}</td>
                                        <td class="r num">{{ pct($r['g2'] ?? null, 2) }}</td>
                                        <td class="r num">{{ $r['cetak'] ? pct($r['cetak']['ned'] ?? null, 1).' / '.pct($r['cetak']['cer'], 1) : '–' }}</td>
                                    </tr>
                                @endforeach
                            </tbody>
                        </table>
                    </div>
                    @if ($stages)
                        <h3 class="section-title mt-5 mb-2">Per tahap rantai model resmi</h3>
                        <p class="card-subtitle mb-3 max-w-4xl">Transformer lawan CRNN sesudah tiap tahap yang sama. Selisih CER baris cetak (Transformer − CRNN) dengan SK 95% per halaman; positif = CRNN lebih baik.</p>
                        <div class="table-wrap">
                            <table class="data-table min-w-[48rem]">
                                <thead><tr><th>Tahap</th><th class="r">Val bersih, Transformer / CRNN</th><th class="r">CER cetak, Transformer / CRNN</th><th class="r">Selisih CER cetak, SK 95%</th><th class="r">CER semua, Transformer / CRNN</th></tr></thead>
                                <tbody>
                                    @foreach ($stages as $s)
                                        <tr data-stage="{{ $s['stage'] }}">
                                            <td class="code">{{ $s['stage'] }}</td>
                                            <td class="r num">{{ pct($s['val_transformer'] ?? null, 2) }} / {{ pct($s['val_crnn'] ?? null, 2) }}</td>
                                            <td class="r num">{{ pct($s['cetak']['cer_t'], 1) }} / {{ pct($s['cetak']['cer_c'], 1) }}</td>
                                            <td class="r num whitespace-nowrap">{{ $pp($s['cetak']['delta']) }} <span class="text-muted">{{ $ci($s['cetak']['ci95']) }}</span></td>
                                            <td class="r num">{{ pct($s['semua']['cer_t'], 1) }} / {{ pct($s['semua']['cer_c'], 1) }}</td>
                                        </tr>
                                    @endforeach
                                </tbody>
                            </table>
                        </div>
                    @endif
                </div>
            </section>
        @endif

        <section class="card">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">Ablasi komponen metode kita</h2>
                    <p class="card-subtitle max-w-4xl">Tiap baris = satu perlakuan lawan kontrolnya (beda satu hal, langkah dan titik lanjut sama), pada baris nyata yang sama. Selisih CER dalam poin persen, SK 95% per halaman; negatif = komponen itu membantu. Dua baris rujukan menunjukkan besarnya derau.</p>
                </div>
                <div class="table-wrap">
                    <table class="data-table min-w-[52rem]">
                        <thead><tr><th>Komponen</th><th>Perlakuan − kontrol</th><th class="r">Cetak</th><th class="r">Tulisan tangan</th><th class="r">Semua</th><th>Status</th></tr></thead>
                        @foreach (\App\Livewire\Pages\Matriks::ABLATION_GROUPS as $group => $name)
                            @if (! empty($ablation[$group]))
                                <tbody>
                                    <tr><th colspan="6" class="text-muted text-xs uppercase tracking-wide">{{ $name }}</th></tr>
                                    @foreach ($ablation[$group] as $a)
                                        <tr data-pair="{{ $a['a'] }}">
                                            <td class="text-sm font-semibold">{{ $a['label'] }}</td>
                                            <td class="code">{{ str_replace('crnn_', '', $a['a']) }} − {{ str_replace('crnn_', '', $a['b']) }}</td>
                                            @foreach (['cetak', 'tulisan_tangan', 'semua'] as $kind)
                                                <td class="r num whitespace-nowrap" data-cell="{{ $kind }}">{{ $pp($a[$kind]['delta']) }} <span class="text-muted">{{ $ci($a[$kind]['ci95_pages']) }}</span></td>
                                            @endforeach
                                            <td>{{ $a['status'] }}</td>
                                        </tr>
                                    @endforeach
                                </tbody>
                            @endif
                        @endforeach
                    </table>
                </div>
            </div>
        </section>

        @if ($matrix['augment'])
            <section class="card">
                <div class="card-body">
                    <div class="mb-3">
                        <h2 class="section-title">Ablasi augmentasi</h2>
                        <p class="card-subtitle max-w-4xl">12 run 600 langkah dari run <span class="code">base</span>; tiap baris MENAMBAH satu operasi ke baris sebelumnya (kumulatif, satu seed), jadi selisihnya memuat interaksi dan derau antar-run. G3 di sini seluruh 745 baris.</p>
                    </div>
                    <div class="table-wrap">
                        <table class="data-table min-w-[36rem]">
                            <thead><tr><th class="r">k</th><th>Operasi ditambahkan</th><th class="r">Val berat</th><th class="r">G3</th><th class="r">Δ G3</th></tr></thead>
                            <tbody>
                                @foreach ($matrix['augment'] as $i => $r)
                                    <tr data-augment="{{ $r['k'] }}">
                                        <td class="r num">{{ $r['k'] }}</td>
                                        <td>{{ $r['op'] }}</td>
                                        <td class="r num">{{ pct($r['heavy'], 1) }}</td>
                                        <td class="r num">{{ pct($r['g3'], 1) }}</td>
                                        <td class="r num">{{ $i ? $pp($r['g3'] - $matrix['augment'][$i - 1]['g3']) : '' }}</td>
                                    </tr>
                                @endforeach
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>
        @endif

        <div class="alert alert-warning" role="note">
            <div class="min-w-0 text-sm">
                <p><strong>Batasan.</strong> NusaAksara adalah data uji yang sudah dibaca berulang untuk memilih run, jadi angka model kita sedikit optimistis; tiap metode pembanding membacanya sekali. VLM mungkin pernah melihat NusaAksara saat pretraining. Pembagian cetak / tulisan tangan per halaman adalah klasifikasi visual yang belum diperiksa manusia.</p>
                <details class="mt-2"><summary class="cursor-pointer">Prompt VLM (sama untuk semua VLM, sidik {{ $matrix['prompt_sha'] }})</summary><p class="code mt-1 whitespace-pre-wrap">{{ $matrix['prompt'] }}</p></details>
                <p class="text-muted mt-2 text-xs">Diekspor {{ $matrix['generated'] }} · pembanding selisih: {{ $matrix['reference'] }}</p>
            </div>
        </div>
    @endif
</div>
