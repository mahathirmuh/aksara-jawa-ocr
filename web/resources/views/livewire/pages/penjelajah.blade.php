<div class="flex flex-col gap-4">
    <div class="flex flex-wrap items-end justify-between gap-x-4 gap-y-1">
        <div class="min-w-0">
            <div class="page-pretitle">NusaAksara · {{ nfmt($total) }} baris nyata</div>
            <h1 class="page-title">Penjelajah baris</h1>
        </div>
        <label class="inline-flex cursor-pointer items-center gap-2 text-sm text-zinc-600 dark:text-zinc-300" for="latin-toggle">
            <input id="latin-toggle" type="checkbox" wire:model.live="latin" class="size-4 accent-[var(--color-accent)]">
            Bantu baca Latin (draf)
        </label>
    </div>

    @if (! $total)
        @include('partials.no-results')
    @else
        {{-- Saringan & pipeline. Chip aktif ditandai aria-pressed (gayanya di app.css: .chip[aria-pressed='true']). --}}
        <section class="card">
            <div class="card-body flex flex-col gap-3">
                <div class="flex flex-wrap items-center gap-2">
                    <button type="button" wire:click="$set('tag', 'semua')" aria-pressed="{{ $tag === 'semua' ? 'true' : 'false' }}" class="chip">
                        semua <span class="chip-count num">{{ $total }}</span>
                    </button>
                    @foreach ($tagCounts as $t => $n)
                        <button type="button" wire:click="$set('tag', @js($t))" aria-pressed="{{ $tag === $t ? 'true' : 'false' }}" class="chip">
                            {{ $t }} <span class="chip-count num">{{ $n }}</span>
                        </button>
                    @endforeach
                    <label class="ml-auto inline-flex items-center gap-2 text-xs font-medium text-zinc-500 dark:text-zinc-400" for="sort">Urutkan
                        <select id="sort" wire:model.live="sort" class="explorer-select">
                            @foreach ($sorts as $key => $name)
                                <option value="{{ $key }}">{{ $name }}</option>
                            @endforeach
                        </select>
                    </label>
                </div>
                <div class="flex flex-wrap items-center gap-2 border-t border-[color:var(--row-line)] pt-3">
                    <span class="text-xs font-medium text-zinc-500 dark:text-zinc-400">Tampilkan</span>
                    @foreach ($pipelines as $p)
                        @if ($p->isDone())
                            <button type="button" wire:click="togglePipe(@js($p->key))" title="{{ $p->config }}" aria-pressed="{{ in_array($p->key, $pipes, true) ? 'true' : 'false' }}"
                                    class="chip">{{ $p->label }}</button>
                        @else
                            <span title="{{ $p->config }}" class="chip is-disabled">{{ $p->label }} <span class="opacity-70">belum</span></span>
                        @endif
                    @endforeach
                </div>
            </div>
        </section>

        <div class="grid items-start gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
            {{-- Daftar --}}
            <div class="flex flex-col gap-2">
                <div class="card explorer-list" role="listbox" aria-label="Daftar baris">
                    @forelse ($page as $l)
                        <button type="button" wire:key="li-{{ $l->id }}" wire:click="select(@js($l->external_id))" role="option" aria-selected="{{ $current?->id === $l->id ? 'true' : 'false' }}"
                                class="explorer-row">
                            <span class="font-mono text-xs">{{ $l->label() }}</span>
                            <span class="num text-right text-xs text-zinc-500 dark:text-zinc-400">CRNN {{ pct($l->official_cer, 0) }}@if ($l->vlm_cer !== null) · VLM {{ pct($l->vlm_cer, 0) }}@endif</span>
                            <span class="col-span-2 truncate text-[11px] text-zinc-500 dark:text-zinc-400">{{ collect($l->tags)->reject(fn ($t) => $t === 'label tanpa spasi')->join(' · ') ?: ' ' }}</span>
                        </button>
                    @empty
                        <p class="p-4 text-sm text-zinc-500 dark:text-zinc-400">Tidak ada baris dengan saringan ini.</p>
                    @endforelse
                </div>
                {{-- Paginasi ringkas: komponen bawaan terlalu lebar untuk kolom daftar. --}}
                <div class="flex items-center justify-between gap-2 text-xs text-zinc-500 dark:text-zinc-400">
                    <button type="button" wire:click="previousPage" @disabled($page->onFirstPage()) class="btn-previous">‹ Sebelumnya</button>
                    <span class="num">{{ $page->total() ? $page->firstItem().'–'.$page->lastItem() : 0 }} dari {{ nfmt($page->total()) }}</span>
                    <button type="button" wire:click="nextPage" @disabled(! $page->hasMorePages()) class="btn-next">Berikutnya ›</button>
                </div>
            </div>

            {{-- Detail --}}
            @if ($current)
                <article class="card min-w-0" wire:key="detail-{{ $current->id }}" x-data="{ band: null, pin: null }">
                    <div class="card-header">
                        <div class="flex min-w-0 flex-wrap items-center gap-2">
                            <h2 class="font-mono text-sm font-semibold">{{ $current->external_id }}</h2>
                            @foreach ($current->tags as $t)
                                <span class="status status-gray">{{ $t }}</span>
                            @endforeach
                        </div>
                        <span class="text-xs text-zinc-500 dark:text-zinc-400">{{ $current->source_id }} · {{ $current->condition }}</span>
                    </div>

                    <div class="card-body flex flex-col gap-3">
                        <figure class="card-inset m-0 overflow-x-auto p-3">
                            <div class="crop">
                                <img src="{{ route('line.image', $current) }}" alt="Potongan baris {{ $current->external_id }}" width="{{ $current->width }}" height="{{ $current->height }}">
                                <span class="band" x-show="band" x-cloak :style="band ? `left:${band[0] * 100}%;width:${Math.max(0.4, (band[1] - band[0]) * 100)}%` : ''"></span>
                            </div>
                        </figure>
                        <p class="text-xs text-zinc-500 dark:text-zinc-400">Arahkan kursor ke suku kata pada baris CRNN greedy untuk melihat kolom citra yang dibacanya (alignment CTC). Klik untuk menyematkan.</p>

                        <div class="flex flex-col">
                            {{-- Label dan tahap 2–4 dari label --}}
                            @php($ann = $current->annotation)
                            @php($row = 'detail-row')
                            <div class="{{ $row }}">
                                <div><div class="detail-name">Label</div><div class="code">anotasi NusaAksara · tahap 1</div></div>
                                <div class="min-w-0">
                                    <div class="jv break-all text-2xl leading-[1.9]">{{ $current->reference }}</div>
                                    @if ($latin)<div class="text-sm italic text-zinc-600 dark:text-zinc-300"><span class="not-italic text-[11px] text-zinc-500 dark:text-zinc-400">draf · </span>{{ \App\Support\Transliterator::toLatin($current->reference) }}</div>@endif
                                </div>
                            </div>
                            <div class="{{ $row }}">
                                <div><div class="detail-name">Transliterasi</div><div class="code">manusia · tahap 2</div></div>
                                <div class="text-sm">{{ $ann?->transliteration ?? '–' }}</div>
                            </div>
                            <div class="{{ $row }}">
                                <div><div class="detail-name">Arti</div><div class="code">manusia & mesin · tahap 3</div></div>
                                <div class="flex flex-col gap-1 text-sm">
                                    <div>{{ $ann?->translation ?? '–' }}</div>
                                    @if ($machine->has('label'))
                                        <div class="text-zinc-600 dark:text-zinc-300">
                                            <span class="text-[11px] text-zinc-500 dark:text-zinc-400">mesin (NLLB, dari transliterasi manusia) · </span>{{ $machine['label']->output }}
                                            @if ($machine['label']->chrf !== null)<span class="num text-[11px] text-zinc-500 dark:text-zinc-400"> · chrF {{ number_format($machine['label']->chrf, 1, ',', '.') }}</span>@endif
                                        </div>
                                    @endif
                                </div>
                            </div>
                            <div class="{{ $row }}">
                                <div><div class="detail-name">Tingkat tutur</div><div class="code">tahap 4</div></div>
                                @php($auto = \App\Support\SpeechLevel::classify($ann?->transliteration))
                                <div class="flex flex-col gap-2 text-sm">
                                    <div class="flex flex-wrap items-center gap-2">
                                        <span class="text-xs text-zinc-500 dark:text-zinc-400">leksikon (dari transliterasi manusia):</span>
                                        <span class="status {{ $auto['level'] ? 'status-blue' : 'status-gray' }}">{{ $auto['level'] ?? 'tak tentu' }}</span>
                                        <span class="text-xs text-zinc-500 dark:text-zinc-400">keyakinan {{ $auto['confidence'] }}</span>
                                    </div>
                                    @if ($auto['evidence'])
                                        <div class="flex flex-wrap gap-1">
                                            @foreach ($auto['evidence'] as [$word, $level])
                                                <span class="rounded-sm bg-[var(--secondary-wash)] px-1.5 py-0.5 text-[11px]">{{ $word }} · {{ $level }}</span>
                                            @endforeach
                                        </div>
                                    @endif
                                    <div class="flex flex-wrap items-center gap-1.5">
                                        <span class="text-xs text-zinc-500 dark:text-zinc-400">label manusia:</span>
                                        @foreach (\App\Support\SpeechLevel::LEVELS as $lv)
                                            <button type="button" wire:click="setSpeechLevel(@js($current->external_id), @js($lv))"
                                                    aria-pressed="{{ $ann?->speech_level === $lv ? 'true' : 'false' }}"
                                                    class="chip">{{ $lv }}</button>
                                        @endforeach
                                        @if ($ann?->speech_level)
                                            <button type="button" wire:click="setSpeechLevel(@js($current->external_id), null)" class="text-xs text-[var(--primary)] hover:underline">hapus</button>
                                            <span class="text-[11px] text-zinc-500 dark:text-zinc-400">oleh {{ $ann->labeler?->name ?? '–' }} · {{ $ann->labeled_at?->diffForHumans() }}</span>
                                        @else
                                            <span class="text-[11px] text-zinc-500 dark:text-zinc-400">belum dilabel (data uji akurasi leksikon)</span>
                                        @endif
                                    </div>
                                </div>
                            </div>

                            {{-- Keluaran pipeline --}}
                            @php($best = $predictions->min('cer'))
                            @foreach ($shown as $p)
                                @php($pred = $predictions[$p->key])
                                <div class="{{ $row }}" wire:key="row-{{ $current->id }}-{{ $p->key }}">
                                    <div class="flex flex-col gap-1">
                                        <div class="detail-name flex flex-wrap items-center gap-1.5">{{ $p->label }}
                                            <span class="status {{ $pred->cer == $best ? 'status-blue' : 'status-gray' }} num">CER {{ pct($pred->cer, 0) }}</span>
                                        </div>
                                        <div class="code">{{ $p->config }}@if ($pred->confidence) · keyakinan {{ $pred->confidence }}@endif</div>
                                    </div>
                                    <div class="min-w-0">
                                        <div class="jv break-all text-2xl leading-[1.9]">{!! segments_html($pred->segments, $pred->spans) !!}</div>
                                        @if ($latin)
                                            @php($ocrLatin = \App\Support\Transliterator::toLatin($pred->text))
                                            @php($ocrTutur = \App\Support\SpeechLevel::classify($ocrLatin))
                                            <div class="text-sm italic text-zinc-600 dark:text-zinc-300"><span class="not-italic text-[11px] text-zinc-500 dark:text-zinc-400">draf · </span>{{ $ocrLatin }}</div>
                                            <div class="text-[11px] text-zinc-500 dark:text-zinc-400">tutur dari keluaran ini: {{ $ocrTutur['level'] ?? 'tak tentu' }} (keyakinan {{ $ocrTutur['confidence'] }})</div>
                                            @if ($machine->has($p->key))
                                                <div class="text-[11px] text-zinc-500 dark:text-zinc-400">arti dari keluaran ini (mesin): <span class="text-zinc-700 dark:text-zinc-300">{{ $machine[$p->key]->output }}</span>@if ($machine[$p->key]->chrf !== null) · chrF {{ number_format($machine[$p->key]->chrf, 1, ',', '.') }}@endif</div>
                                            @endif
                                        @endif
                                    </div>
                                </div>
                            @endforeach
                            @foreach ($missing as $p)
                                <div class="{{ $row }} text-sm text-zinc-500 dark:text-zinc-400">
                                    <div class="font-semibold">{{ $p->label }}</div>
                                    <div>Tidak ada prediksi untuk baris ini{{ $p->key === 'vlm_zeroshot' ? ' (VLM hanya membaca 50 baris uji buta)' : '' }}.</div>
                                </div>
                            @endforeach
                        </div>
                    </div>

                    <div class="card-footer flex flex-wrap items-center gap-4">
                        <span><span class="sy jv text-base">ꦲ</span> sama</span>
                        <span><span class="sy sy-sub jv text-base">ꦲ</span> beda</span>
                        <span><span class="sy sy-ins jv text-base">ꦲ</span> tambahan</span>
                        <span><span class="sy sy-del jv text-base">ꦲ</span> hilang</span>
                        <span><span class="sy sy-sp">␣</span> spasi tidak ada di label</span>
                    </div>
                </article>
            @endif
        </div>
    @endif
</div>
