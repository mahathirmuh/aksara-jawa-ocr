<div class="mx-auto flex max-w-7xl flex-col gap-5">
    <div class="flex flex-wrap items-end justify-between gap-3">
        <div>
            <div class="eyebrow">NusaAksara · {{ nfmt($total) }} baris nyata</div>
            <flux:heading size="xl" level="1" class="!mt-1">Penjelajah baris</flux:heading>
        </div>
        <label class="inline-flex cursor-pointer items-center gap-2 text-sm text-zinc-600 dark:text-zinc-300" for="latin-toggle">
            <input id="latin-toggle" type="checkbox" wire:model.live="latin" class="size-4 accent-[var(--color-accent)]">
            Bantu baca Latin (draf)
        </label>
    </div>

    @if (! $total)
        @include('partials.no-results')
    @else
        {{-- Saringan & pipeline --}}
        <div class="flex flex-col gap-3">
            <div class="flex flex-wrap items-center gap-2">
                @php($chip = 'inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs transition')
                @php($on = 'border-accent bg-accent text-accent-foreground')
                @php($off = 'border-zinc-300 text-zinc-600 hover:border-accent dark:border-zinc-600 dark:text-zinc-300')
                <button type="button" wire:click="$set('tag', 'semua')" aria-pressed="{{ $tag === 'semua' ? 'true' : 'false' }}" class="{{ $chip }} {{ $tag === 'semua' ? $on : $off }}">
                    semua <span class="num opacity-70">{{ $total }}</span>
                </button>
                @foreach ($tagCounts as $t => $n)
                    <button type="button" wire:click="$set('tag', @js($t))" aria-pressed="{{ $tag === $t ? 'true' : 'false' }}" class="{{ $chip }} {{ $tag === $t ? $on : $off }}">
                        {{ $t }} <span class="num opacity-70">{{ $n }}</span>
                    </button>
                @endforeach
                <label class="ml-auto inline-flex items-center gap-2 text-xs text-zinc-500" for="sort">Urutkan
                    <select id="sort" wire:model.live="sort" class="rounded-md border border-zinc-300 bg-transparent px-2 py-1 text-sm dark:border-zinc-600">
                        @foreach ($sorts as $key => $name)
                            <option value="{{ $key }}">{{ $name }}</option>
                        @endforeach
                    </select>
                </label>
            </div>
            <div class="flex flex-wrap items-center gap-2">
                <span class="text-xs text-zinc-500">Tampilkan</span>
                @foreach ($pipelines as $p)
                    @if ($p->isDone())
                        <button type="button" wire:click="togglePipe(@js($p->key))" title="{{ $p->config }}" aria-pressed="{{ in_array($p->key, $pipes, true) ? 'true' : 'false' }}"
                                class="{{ $chip }} {{ in_array($p->key, $pipes, true) ? $on : $off }}">{{ $p->label }}</button>
                    @else
                        <span title="{{ $p->config }}" class="{{ $chip }} cursor-not-allowed border-dashed border-zinc-300 text-zinc-400 dark:border-zinc-600">{{ $p->label }} <span class="opacity-70">belum</span></span>
                    @endif
                @endforeach
            </div>
        </div>

        <div class="grid items-start gap-5 lg:grid-cols-[300px_minmax(0,1fr)]">
            {{-- Daftar --}}
            <div class="flex flex-col gap-2">
                <div class="max-h-[42rem] overflow-y-auto rounded-xl border border-zinc-200 bg-white dark:border-zinc-700 dark:bg-zinc-900" role="listbox" aria-label="Daftar baris">
                    @forelse ($page as $l)
                        <button type="button" wire:key="li-{{ $l->id }}" wire:click="select(@js($l->external_id))" role="option" aria-selected="{{ $current?->id === $l->id ? 'true' : 'false' }}"
                                @class(['grid w-full grid-cols-[1fr_auto] gap-x-2 border-b border-l-4 border-zinc-100 px-3 py-2 text-left dark:border-zinc-800',
                                    'border-l-accent bg-[color-mix(in_srgb,var(--color-accent)_10%,transparent)]' => $current?->id === $l->id,
                                    'border-l-transparent hover:bg-zinc-50 dark:hover:bg-zinc-800' => $current?->id !== $l->id])>
                            <span class="font-mono text-xs">{{ $l->label() }}</span>
                            <span class="num text-right text-xs text-zinc-500">CRNN {{ pct($l->official_cer, 0) }}@if ($l->vlm_cer !== null) · VLM {{ pct($l->vlm_cer, 0) }}@endif</span>
                            <span class="col-span-2 truncate text-[11px] text-zinc-500">{{ collect($l->tags)->reject(fn ($t) => $t === 'label tanpa spasi')->join(' · ') ?: ' ' }}</span>
                        </button>
                    @empty
                        <p class="p-4 text-sm text-zinc-500">Tidak ada baris dengan saringan ini.</p>
                    @endforelse
                </div>
                {{-- Paginasi ringkas: komponen bawaan terlalu lebar untuk kolom 300 px. --}}
                <div class="flex items-center justify-between gap-2 text-xs text-zinc-500">
                    <button type="button" wire:click="previousPage" @disabled($page->onFirstPage())
                            class="rounded-md border border-zinc-300 px-2 py-1 hover:border-accent disabled:opacity-40 disabled:hover:border-zinc-300 dark:border-zinc-600">‹ Sebelumnya</button>
                    <span class="num">{{ $page->total() ? $page->firstItem().'–'.$page->lastItem() : 0 }} dari {{ nfmt($page->total()) }}</span>
                    <button type="button" wire:click="nextPage" @disabled(! $page->hasMorePages())
                            class="rounded-md border border-zinc-300 px-2 py-1 hover:border-accent disabled:opacity-40 disabled:hover:border-zinc-300 dark:border-zinc-600">Berikutnya ›</button>
                </div>
            </div>

            {{-- Detail --}}
            @if ($current)
                <article class="flex min-w-0 flex-col gap-4 rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900"
                         wire:key="detail-{{ $current->id }}" x-data="{ band: null, pin: null }">
                    <div class="flex flex-wrap items-center gap-2">
                        <h2 class="font-mono text-base font-medium">{{ $current->external_id }}</h2>
                        @foreach ($current->tags as $t)
                            <span class="pill pill-plain">{{ $t }}</span>
                        @endforeach
                        <span class="ml-auto text-xs text-zinc-500">{{ $current->source_id }} · {{ $current->condition }}</span>
                    </div>

                    <figure class="m-0 overflow-x-auto rounded-lg border border-zinc-200 bg-zinc-50 p-3 dark:border-zinc-700 dark:bg-zinc-800">
                        <div class="crop">
                            <img src="{{ route('line.image', $current) }}" alt="Potongan baris {{ $current->external_id }}" width="{{ $current->width }}" height="{{ $current->height }}">
                            <span class="band" x-show="band" x-cloak :style="band ? `left:${band[0] * 100}%;width:${Math.max(0.4, (band[1] - band[0]) * 100)}%` : ''"></span>
                        </div>
                    </figure>
                    <p class="-mt-2 text-xs text-zinc-500">Arahkan kursor ke suku kata pada baris CRNN greedy untuk melihat kolom citra yang dibacanya (alignment CTC). Klik untuk menyematkan.</p>

                    <div class="flex flex-col">
                        {{-- Label dan tahap 2–4 dari label --}}
                        @php($ann = $current->annotation)
                        @php($row = 'grid gap-x-5 gap-y-1 border-t border-zinc-200 py-3 sm:grid-cols-[200px_minmax(0,1fr)] dark:border-zinc-700')
                        <div class="{{ $row }}">
                            <div><div class="font-semibold">Label</div><div class="font-mono text-[11px] text-zinc-500">anotasi NusaAksara · tahap 1</div></div>
                            <div class="min-w-0">
                                <div class="jv break-all text-2xl leading-[1.9]">{{ $current->reference }}</div>
                                @if ($latin)<div class="text-sm italic text-zinc-600 dark:text-zinc-300"><span class="not-italic text-[11px] text-zinc-500">draf · </span>{{ \App\Support\Transliterator::toLatin($current->reference) }}</div>@endif
                            </div>
                        </div>
                        <div class="{{ $row }}">
                            <div><div class="font-semibold">Transliterasi</div><div class="font-mono text-[11px] text-zinc-500">manusia · tahap 2</div></div>
                            <div class="text-sm">{{ $ann?->transliteration ?? '–' }}</div>
                        </div>
                        <div class="{{ $row }}">
                            <div><div class="font-semibold">Arti</div><div class="font-mono text-[11px] text-zinc-500">manusia & mesin · tahap 3</div></div>
                            <div class="flex flex-col gap-1 text-sm">
                                <div>{{ $ann?->translation ?? '–' }}</div>
                                @if ($machine->has('label'))
                                    <div class="text-zinc-600 dark:text-zinc-300">
                                        <span class="text-[11px] text-zinc-500">mesin (NLLB, dari transliterasi manusia) · </span>{{ $machine['label']->output }}
                                        @if ($machine['label']->chrf !== null)<span class="num text-[11px] text-zinc-500"> · chrF {{ number_format($machine['label']->chrf, 1, ',', '.') }}</span>@endif
                                    </div>
                                @endif
                            </div>
                        </div>
                        <div class="{{ $row }}">
                            <div><div class="font-semibold">Tingkat tutur</div><div class="font-mono text-[11px] text-zinc-500">tahap 4</div></div>
                            @php($auto = \App\Support\SpeechLevel::classify($ann?->transliteration))
                            <div class="flex flex-col gap-2 text-sm">
                                <div class="flex flex-wrap items-center gap-2">
                                    <span class="text-xs text-zinc-500">leksikon (dari transliterasi manusia):</span>
                                    <span class="pill {{ $auto['level'] ? 'pill-acc' : 'pill-plain' }}">{{ $auto['level'] ?? 'tak tentu' }}</span>
                                    <span class="text-xs text-zinc-500">keyakinan {{ $auto['confidence'] }}</span>
                                </div>
                                @if ($auto['evidence'])
                                    <div class="flex flex-wrap gap-1">
                                        @foreach ($auto['evidence'] as [$word, $level])
                                            <span class="rounded bg-zinc-100 px-1.5 py-0.5 text-[11px] dark:bg-zinc-800">{{ $word }} · {{ $level }}</span>
                                        @endforeach
                                    </div>
                                @endif
                                <div class="flex flex-wrap items-center gap-1.5">
                                    <span class="text-xs text-zinc-500">label manusia:</span>
                                    @foreach (\App\Support\SpeechLevel::LEVELS as $lv)
                                        <button type="button" wire:click="setSpeechLevel(@js($current->external_id), @js($lv))"
                                                aria-pressed="{{ $ann?->speech_level === $lv ? 'true' : 'false' }}"
                                                class="{{ $chip }} {{ $ann?->speech_level === $lv ? $on : $off }}">{{ $lv }}</button>
                                    @endforeach
                                    @if ($ann?->speech_level)
                                        <button type="button" wire:click="setSpeechLevel(@js($current->external_id), null)" class="text-xs text-zinc-500 underline">hapus</button>
                                        <span class="text-[11px] text-zinc-500">oleh {{ $ann->labeler?->name ?? '–' }} · {{ $ann->labeled_at?->diffForHumans() }}</span>
                                    @else
                                        <span class="text-[11px] text-zinc-500">belum dilabel (data uji akurasi leksikon)</span>
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
                                    <div class="flex flex-wrap items-center gap-1.5 font-semibold">{{ $p->label }}
                                        <span class="pill {{ $pred->cer == $best ? 'pill-acc' : 'pill-plain' }} num">CER {{ pct($pred->cer, 0) }}</span>
                                    </div>
                                    <div class="font-mono text-[11px] text-zinc-500">{{ $p->config }}@if ($pred->confidence) · keyakinan {{ $pred->confidence }}@endif</div>
                                </div>
                                <div class="min-w-0">
                                    <div class="jv break-all text-2xl leading-[1.9]">{!! segments_html($pred->segments, $pred->spans) !!}</div>
                                    @if ($latin)
                                        @php($ocrLatin = \App\Support\Transliterator::toLatin($pred->text))
                                        @php($ocrTutur = \App\Support\SpeechLevel::classify($ocrLatin))
                                        <div class="text-sm italic text-zinc-600 dark:text-zinc-300"><span class="not-italic text-[11px] text-zinc-500">draf · </span>{{ $ocrLatin }}</div>
                                        <div class="text-[11px] text-zinc-500">tutur dari keluaran ini: {{ $ocrTutur['level'] ?? 'tak tentu' }} (keyakinan {{ $ocrTutur['confidence'] }})</div>
                                        @if ($machine->has($p->key))
                                            <div class="text-[11px] text-zinc-500">arti dari keluaran ini (mesin): <span class="text-zinc-700 dark:text-zinc-300">{{ $machine[$p->key]->output }}</span>@if ($machine[$p->key]->chrf !== null) · chrF {{ number_format($machine[$p->key]->chrf, 1, ',', '.') }}@endif</div>
                                        @endif
                                    @endif
                                </div>
                            </div>
                        @endforeach
                        @foreach ($missing as $p)
                            <div class="{{ $row }} text-sm text-zinc-500">
                                <div class="font-semibold">{{ $p->label }}</div>
                                <div>Tidak ada prediksi untuk baris ini{{ $p->key === 'vlm_zeroshot' ? ' (VLM hanya membaca 50 baris uji buta)' : '' }}.</div>
                            </div>
                        @endforeach
                    </div>

                    <div class="flex flex-wrap items-center gap-4 border-t border-zinc-200 pt-3 text-xs text-zinc-600 dark:border-zinc-700 dark:text-zinc-300">
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
