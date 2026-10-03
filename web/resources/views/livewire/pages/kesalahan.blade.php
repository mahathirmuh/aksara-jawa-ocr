<div class="mx-auto flex max-w-6xl flex-col gap-6">
    @include('partials.page-head', ['eyebrow' => ($pipeline?->label ?? 'CRNN').' · greedy · 745 baris', 'heading' => 'Kesalahan aksara',
        'aside' => $totals->isEmpty() ? null : nfmt($totals['sub'] ?? 0).' tertukar · '.nfmt($totals['del'] ?? 0).' hilang · '.nfmt($totals['ins'] ?? 0).' tambahan'])

    @if ($totals->isEmpty())
        @include('partials.no-results')
    @else
        <div class="grid gap-6 lg:grid-cols-3">
            @foreach (['sub' => ['Tertukar', 'Aksara label → yang dibaca model'], 'del' => ['Hilang', 'Ada di label, tidak dibaca'], 'ins' => ['Tambahan', 'Dibaca model, tidak ada di label']] as $kind => [$title, $desc])
                @php($items = $lists[$kind])
                @php($max = max(1, $items->max('count')))
                <section class="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900">
                    <flux:heading size="lg">{{ $title }}</flux:heading>
                    <p class="mt-1 text-sm text-zinc-500">{{ $desc }}</p>
                    <ul class="mt-3 flex flex-col">
                        @foreach ($items as $c)
                            <li class="grid grid-cols-[auto_1fr_auto] items-center gap-3 border-t border-zinc-100 py-2 dark:border-zinc-800">
                                <span class="jv min-w-[2.4em] text-center text-2xl leading-snug">
                                    @if ($kind === 'sub'){{ $c->ref }}<span class="px-1 font-sans text-sm text-zinc-500">→</span>{{ $c->hyp }}@else{{ $kind === 'del' ? $c->ref : $c->hyp }}@endif
                                </span>
                                <span class="min-w-0">
                                    <span class="block truncate text-xs text-zinc-500">
                                        {{ $kind === 'sub' ? aksara_name($c->ref).' → '.aksara_name($c->hyp) : aksara_name($kind === 'del' ? $c->ref : $c->hyp) }}
                                    </span>
                                    <span class="mt-1 block h-1.5 rounded-r bg-[var(--series-1)]" style="width: {{ max(4, $c->count / $max * 100) }}%"></span>
                                </span>
                                <span class="num text-right text-sm font-semibold">{{ nfmt($c->count) }}</span>
                            </li>
                        @endforeach
                    </ul>
                </section>
            @endforeach
        </div>
        <p class="max-w-prose text-sm text-zinc-500">Spasi diabaikan di halaman ini. Penghapusan mendominasi: model paling sering melewatkan tanda kecil seperti wulu dan cecak, terutama pada potongan bertekstur raster.</p>
    @endif
</div>
