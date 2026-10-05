<div class="flex flex-col gap-4">
    @include('partials.page-head', ['eyebrow' => 'Fase 5', 'heading' => 'Ablasi augmentasi',
        'aside' => '12 run kumulatif dari checkpoint 4b · 600 langkah · 10 font · satu seed'])

    @if ($runs->isEmpty())
        @include('partials.no-results')
    @else
        {{-- Grafik: G3 vs val berat per run; pita "peniru pindaian" digambar app.js mulai k = 8. --}}
        <section class="card">
            <div class="card-body">
                <div class="relative h-80" x-data="chart('ablation', @js($chartData))" wire:ignore>
                    <canvas x-ref="canvas" role="img" aria-label="CER G3 dan val berat untuk setiap run ablasi"></canvas>
                </div>
            </div>
            <div class="card-footer">
                <p class="max-w-[85ch]">
                    Augmentasi preset menurunkan val berat tetapi hampir tidak menggerakkan G3. Augmentasi peniru pindaian
                    (tight, stroke) yang menurunkan G3. Kolom G3 memakai test set, jadi dibaca sebagai arah, bukan dasar memilih konfigurasi.
                </p>
            </div>
        </section>

        {{-- Tabel: baris bertanda = augmentasi peniru pindaian (pita yang sama dengan grafik). --}}
        <section class="card">
            <div class="card-body">
                <div class="table-wrap">
                    <table class="data-table">
                        <thead><tr><th class="normal-case">k</th><th>Ditambahkan</th><th class="r">Val bersih</th><th class="r">Val berat</th><th class="r">Δ berat</th><th class="r">G3</th><th class="r">Δ G3</th><th class="r">G3 tanpa spasi</th></tr></thead>
                        <tbody>
                            @foreach ($runs as $i => $run)
                                @php($prev = $i ? $runs[$i - 1] : null)
                                @php($dg = $prev ? $run->g3 - $prev->g3 : null)
                                <tr @class(['is-highlight' => $run->k >= 8])>
                                    <td class="num text-muted">{{ $run->k }}</td>
                                    <td class="whitespace-nowrap font-medium">{{ $run->added }}</td>
                                    <td class="r num">{{ pct($run->clean, 2) }}</td>
                                    <td class="r num">{{ pct($run->heavy) }}</td>
                                    <td class="r num whitespace-nowrap text-muted">{{ $prev ? pt($run->heavy - $prev->heavy) : '' }}</td>
                                    <td class="r num">{{ pct($run->g3) }}</td>
                                    <td @class(['r num whitespace-nowrap', 'font-semibold text-[var(--good)]' => $dg !== null && $dg <= -0.1])>{{ pt($dg) }}</td>
                                    <td class="r num">{{ pct($run->g3_no_space) }}</td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
    @endif
</div>
