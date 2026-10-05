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
                    <p class="max-w-[85ch]">Spasi diabaikan di halaman ini. Penghapusan mendominasi: model paling sering melewatkan tanda kecil seperti wulu dan cecak, terutama pada potongan bertekstur raster.</p>
                </div>
            </div>
        </div>
    @endif
</div>
