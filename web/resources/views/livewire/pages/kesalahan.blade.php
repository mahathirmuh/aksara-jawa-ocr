<div class="flex flex-col gap-4">
    @include('partials.page-head', ['eyebrow' => ($pipeline?->label ?? 'CRNN').' · greedy · 745 baris', 'heading' => 'Kesalahan aksara',
        'aside' => $totals->isEmpty() ? null : nfmt($totals['sub'] ?? 0).' tertukar · '.nfmt($totals['del'] ?? 0).' hilang · '.nfmt($totals['ins'] ?? 0).' tambahan'])

    @if ($totals->isEmpty())
        @include('partials.no-results')
    @else
        {{-- Tiga kartu sejajar bila wadahnya cukup lebar (ikut lebar isi, jadi benar juga saat sidebar diciutkan).
             Daftar Tertukar lebih panjang (14 vs 10 baris): kartunya membentang dua baris grid dan catatan mengisi
             ruang di bawah dua kartu lainnya. Warna ikon = warna tanda beda di Penjelajah (beda, hilang, tambahan). --}}
        <div class="@container">
            <div class="grid items-start gap-4 @[60rem]:grid-cols-3 @[60rem]:grid-rows-[auto_1fr]">
                @foreach ([
                    'sub' => ['Tertukar', 'Aksara label → yang dibaca model', 'stat-icon-yellow', 'M7 10h14l-4 -4M17 14h-14l4 4'],
                    'del' => ['Hilang', 'Ada di label, tidak dibaca', '', 'M5 12l14 0'],
                    'ins' => ['Tambahan', 'Dibaca model, tidak ada di label', 'stat-icon-red', 'M12 5l0 14M5 12l14 0'],
                ] as $kind => [$title, $desc, $tone, $icon])
                    @php($items = $lists[$kind])
                    @php($max = max(1, $items->max('count')))
                    <section @class(['card', '@[60rem]:row-span-2' => $kind === 'sub'])>
                        <div class="card-body">
                            {{-- Kepala kartu = angka ringkas: ikon, judul, jumlah seluruh kesalahan jenis ini. --}}
                            <div class="stat-head">
                                <span class="stat-icon {{ $tone }}" aria-hidden="true">
                                    <svg class="size-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="{{ $icon }}" /></svg>
                                </span>
                                <div class="min-w-0">
                                    <h2 class="stat-title">{{ $title }}</h2>
                                    <div class="stat-value">{{ nfmt($totals[$kind] ?? 0) }}</div>
                                </div>
                            </div>
                            <p class="stat-hint mt-3">{{ $desc }}</p>

                            <ul class="mt-3">
                                @foreach ($items as $c)
                                    <li class="grid min-h-[3.25rem] grid-cols-[auto_minmax(0,1fr)] items-center gap-x-4 border-t border-[var(--row-line)] py-2">
                                        @if ($kind === 'sub')
                                            <span class="jv grid w-[5.5rem] grid-cols-[1fr_auto_1fr] items-baseline justify-items-center gap-x-1 text-2xl leading-snug">
                                                <span>{{ $c->ref }}</span><span class="font-sans text-sm text-muted">→</span><span>{{ $c->hyp }}</span>
                                            </span>
                                        @else
                                            <span class="jv w-[3.125rem] text-center text-2xl leading-snug">{{ $kind === 'del' ? $c->ref : $c->hyp }}</span>
                                        @endif
                                        <span class="min-w-0">
                                            <span class="flex items-baseline justify-between gap-3">
                                                <span class="truncate text-xs">
                                                    {{ $kind === 'sub' ? aksara_name($c->ref).' → '.aksara_name($c->hyp) : aksara_name($kind === 'del' ? $c->ref : $c->hyp) }}
                                                </span>
                                                <span class="num text-sm font-semibold">{{ nfmt($c->count) }}</span>
                                            </span>
                                            <span class="mt-1.5 block h-1.5 rounded-full bg-[var(--secondary-wash)]">
                                                <span class="block h-full rounded-full bg-[var(--series-1)]" style="width: {{ max(4, $c->count / $max * 100) }}%"></span>
                                            </span>
                                        </span>
                                    </li>
                                @endforeach
                            </ul>
                        </div>
                    </section>
                @endforeach

                <div class="alert alert-info @[60rem]:col-span-2">
                    <p class="max-w-[85ch]">Spasi diabaikan di tiga daftar di atas. Penghapusan mendominasi: model paling sering melewatkan tanda kecil seperti wulu dan cecak, terutama pada potongan bertekstur raster.</p>
                </div>
            </div>
        </div>

        {{-- Per aksara: recall, presisi, F1 tiap karakter dari penjajaran yang sama dengan CER (manifest class_metrics,
             pipeline resmi). Daftar = aksara yang cukup sering di label (>= MIN_REF) dengan F1 terendah. Ekspor lama tidak
             punya tabel ini: diberi petunjuk ekspor ulang, bukan tabel kosong. --}}
        <section class="card" data-part="per-aksara">
            <div class="card-body">
                <div class="mb-3 flex flex-wrap items-end justify-between gap-x-4 gap-y-1">
                    <div>
                        <h2 class="section-title">Per aksara: recall, presisi, F1</h2>
                        <p class="card-subtitle max-w-4xl">Recall = bagian kemunculan aksara itu di label yang terbaca benar; presisi = bagian keluaran model untuk aksara itu yang benar; F1 = rata-rata harmonik keduanya. Dihitung dari penjajaran karakter yang sama dengan CER, termasuk spasi.</p>
                    </div>
                    @if ($hasClasses)
                        <p class="page-aside" data-part="ringkasan-aksara">
                            macro-F1 {{ pct($macroF1, 1) }} atas {{ $classCount }} aksara di label
                            @if ($micro) · mikro: presisi {{ pct($micro->precision, 1) }} · recall {{ pct($micro->recall, 1) }} · F1 {{ pct($micro->f1, 1) }} · akurasi karakter {{ pct($micro->char_accuracy, 1) }} @endif
                        </p>
                    @endif
                </div>
                @if (! $hasClasses)
                    <div class="alert alert-warning" role="note" data-part="tanpa-per-aksara">
                        <p class="min-w-0">Hasil yang diimpor berasal dari ekspor sebelum metrik per aksara ada. Jalankan ulang <code class="code">scripts/export_results.py</code> lalu <code class="code">php artisan aksara:import</code>.</p>
                    </div>
                @else
                    <p class="text-muted mb-2 text-xs">{{ $weakest->count() }} aksara dengan F1 terendah di antara yang muncul paling sedikit {{ App\Livewire\Pages\Kesalahan::MIN_REF }} kali di label. Aksara yang tidak pernah dikeluarkan model: recall 0, presisi tidak terdefinisi ("–"), F1 0. Daftar kebingungan di atas dan tabel ini dihitung dari penjajaran karakter yang sama.</p>
                    <div class="table-wrap">
                        <table class="data-table min-w-[40rem]">
                            <thead><tr><th>Aksara</th><th>Nama</th><th class="r">Di label</th><th class="r">Dikeluarkan</th><th class="r">Recall</th><th class="r">Presisi</th><th class="r">F1</th></tr></thead>
                            <tbody>
                                @foreach ($weakest as $c)
                                    <tr data-char="{{ $c->code }}">
                                        <td class="jv text-2xl leading-snug">{{ $c->char === ' ' ? '␣' : $c->char }}</td>
                                        <td class="text-xs">{{ aksara_name($c->char) }} <span class="code">{{ $c->code }}</span></td>
                                        <td class="r num" data-cell="ref">{{ nfmt($c->ref) }}</td>
                                        <td class="r num" data-cell="hyp">{{ nfmt($c->hyp) }}</td>
                                        <td class="r num" data-cell="recall">{{ pct($c->recall, 1) }}</td>
                                        <td class="r num" data-cell="precision">{{ pct($c->precision, 1) }}</td>
                                        <td class="r num font-semibold" data-cell="f1">{{ pct($c->f1, 1) }}</td>
                                    </tr>
                                @endforeach
                            </tbody>
                        </table>
                    </div>
                @endif
            </div>
        </section>
    @endif
</div>
