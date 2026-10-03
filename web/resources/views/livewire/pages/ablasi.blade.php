<div class="mx-auto flex max-w-6xl flex-col gap-6">
    @include('partials.page-head', ['eyebrow' => 'Fase 5', 'heading' => 'Ablasi augmentasi',
        'aside' => '12 run kumulatif dari checkpoint 4b · 600 langkah · 10 font · satu seed'])

    @if ($runs->isEmpty())
        @include('partials.no-results')
    @else
        <section class="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900">
            <div class="relative h-80" x-data="chart('ablation', @js($chartData))" wire:ignore>
                <canvas x-ref="canvas" role="img" aria-label="CER G3 dan val berat untuk setiap run ablasi"></canvas>
            </div>
            <p class="mt-3 max-w-prose text-sm text-zinc-500">
                Augmentasi preset menurunkan val berat tetapi hampir tidak menggerakkan G3. Augmentasi peniru pindaian
                (tight, stroke) yang menurunkan G3. Kolom G3 memakai test set, jadi dibaca sebagai arah, bukan dasar memilih konfigurasi.
            </p>
        </section>

        <section class="overflow-x-auto rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900">
            <table class="data-table w-full text-sm">
                <thead><tr><th>k</th><th>Ditambahkan</th><th class="r">Val bersih</th><th class="r">Val berat</th><th class="r">Δ berat</th><th class="r">G3</th><th class="r">Δ G3</th><th class="r">G3 tanpa spasi</th></tr></thead>
                <tbody>
                    @foreach ($runs as $i => $run)
                        @php($prev = $i ? $runs[$i - 1] : null)
                        @php($dg = $prev ? $run->g3 - $prev->g3 : null)
                        <tr @class(['bg-[var(--band)]' => $run->k >= 8])>
                            <td class="font-mono">{{ $run->k }}</td>
                            <td>{{ $run->added }}</td>
                            <td class="r num">{{ pct($run->clean, 2) }}</td>
                            <td class="r num">{{ pct($run->heavy) }}</td>
                            <td class="r num text-zinc-500">{{ $prev ? pt($run->heavy - $prev->heavy) : '' }}</td>
                            <td class="r num">{{ pct($run->g3) }}</td>
                            <td @class(['r num', 'font-semibold text-[var(--good)]' => $dg !== null && $dg <= -0.1])>{{ pt($dg) }}</td>
                            <td class="r num">{{ pct($run->g3_no_space) }}</td>
                        </tr>
                    @endforeach
                </tbody>
            </table>
        </section>
    @endif
</div>
