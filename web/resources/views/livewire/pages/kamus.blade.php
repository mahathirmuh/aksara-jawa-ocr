<div class="flex flex-col gap-4">
    @include('partials.page-head', ['eyebrow' => 'Rujukan alur', 'heading' => 'Kamus',
        'aside' => nfmt($total).' aksara dan tanda · '.nfmt($lexiconTotal->sum()).' kata penanda tingkat tutur'])

    {{-- Pencarian: menyaring kamus aksara dan leksikon; aksara yang ditempel diurai per codepoint. --}}
    <section class="card">
        <div class="card-body flex flex-col gap-3">
            <flux:input wire:model.live.debounce.300ms="q" id="kamus-q" type="search" icon="ti-search" autocomplete="off"
                        aria-label="Cari di kamus"
                        placeholder="Cari nama, kode, bacaan Latin, atau kata; bisa juga menempel aksara (mis. wulu, U+A9B6, kula, {{ mb_chr(0xA98F).mb_chr(0xA9B6) }})" />

            @if ($probe)
                <div class="card-inset flex flex-col gap-2 p-3" wire:key="uraian">
                    @if ($probe['aksara'])
                        <div class="flex flex-wrap items-center gap-2">
                            <span class="subheader">Uraian</span>
                            @foreach ($probe['chars'] as $c)
                                <span class="inline-flex items-center gap-1.5 rounded-sm border border-[color:var(--border-soft)] bg-[var(--surface)] px-2">
                                    <span class="jv text-xl leading-[1.9]">{{ $c['char'] }}</span>
                                    <span class="text-xs">{{ $c['name'] }}</span>
                                </span>
                            @endforeach
                        </div>
                        <div class="text-sm">
                            <span class="text-xs text-zinc-500 dark:text-zinc-400">Latin (draf aturan):</span>
                            <span class="font-semibold">{{ $probe['latin'] !== '' ? $probe['latin'] : '–' }}</span>
                        </div>
                    @endif
                    @if ($probe['speech']['spaced'] || $probe['speech']['evidence'])
                        <div class="flex flex-wrap items-center gap-2 text-sm">
                            <span class="text-xs text-zinc-500 dark:text-zinc-400">Tingkat tutur menurut leksikon:</span>
                            <span class="status {{ $probe['speech']['level'] ? 'status-blue' : 'status-gray' }}">{{ $probe['speech']['level'] ?? 'tak tentu' }}</span>
                            <span class="text-xs text-zinc-500 dark:text-zinc-400">keyakinan {{ $probe['speech']['confidence'] }}</span>
                            @foreach ($probe['speech']['evidence'] as [$word, $level])
                                <span class="kamus-word">{{ $word }} · {{ $level }}</span>
                            @endforeach
                        </div>
                    @elseif (! $probe['aksara'])
                        <p class="text-xs text-zinc-500 dark:text-zinc-400">Hasil pencarian "{{ $q }}" ada di kamus aksara dan leksikon di bawah.</p>
                    @endif
                </div>
            @else
                <p class="text-xs text-zinc-500 dark:text-zinc-400">Pencarian menyaring kamus aksara dan leksikon di bawah. Aksara Jawa yang ditempel diurai per aksara dan diberi bacaan Latin draf.</p>
            @endif

            {{-- Halaman ini panjang: lompat ke bagiannya. --}}
            <nav class="flex flex-wrap gap-2" aria-label="Bagian kamus">
                <a href="#kamus-aksara" class="chip">Kamus aksara <span class="chip-count num">{{ nfmt($shown) }}</span></a>
                <a href="#kamus-koreksi" class="chip">Kamus koreksi OCR</a>
                <a href="#leksikon-tutur" class="chip">Leksikon tingkat tutur <span class="chip-count num">{{ nfmt($lexicon->sum(fn ($words) => $words->count())) }}</span></a>
            </nav>
        </div>
    </section>

    {{-- 1. Aksara yang dibaca model --}}
    <section class="card scroll-mt-20" id="kamus-aksara">
        <div class="card-body">
            <div class="mb-3">
                <h2 class="section-title">Kamus aksara</h2>
                <p class="card-subtitle">Semua yang bisa dikeluarkan model OCR: {{ nfmt($total) }} aksara dan tanda blok Javanese, ditambah spasi dan satu kelas kosong CTC ({{ $catalog['classes'] }} kelas). Bacaan Latin adalah draf aturan transliterasi, bukan hasil OCR.@if ($lines) Angka di kanan tiap ubin: berapa kali aksara itu muncul di {{ nfmt($lines) }} label uji.@endif</p>
            </div>

            @if ($catalog['source'] !== 'tokenizer')
                <div class="alert alert-info mb-3">
                    <span><span class="font-mono">data/tokenizer.json</span> tidak ditemukan di repo OCR; daftar ini diambil dari blok Unicode Javanese.</span>
                </div>
            @endif

            <div class="flex flex-col gap-4">
                @foreach ($kinds as $kind => $label)
                    @if ($groups->has($kind))
                        <div wire:key="jenis-{{ $kind }}">
                            <div class="subheader mb-2">{{ $label }} · {{ $groups[$kind]->count() }}@if ($kind === 'sandhangan') · bacaan dicontohkan pada aksara ka @endif</div>
                            <div class="kamus-grid">
                                @foreach ($groups[$kind] as $e)
                                    <div class="kamus-tile" wire:key="ak-{{ $e['code'] }}">
                                        <span class="jv kamus-glyph">{{ $e['char'] }}</span>
                                        <span class="min-w-0 flex-1">
                                            <span class="block truncate text-xs font-semibold">{{ $e['name'] }}</span>
                                            <span class="code block">{{ $e['code'] }}</span>
                                        </span>
                                        <span class="shrink-0 text-right">
                                            <span class="block text-xs font-medium text-[var(--primary)]">{{ $e['latin'] !== '' ? $e['latin'] : '–' }}</span>
                                            @if ($lines)<span class="num block text-[11px] text-zinc-500 dark:text-zinc-400">{{ nfmt($e['count']) }}×</span>@endif
                                        </span>
                                    </div>
                                @endforeach
                            </div>
                        </div>
                    @endif
                @endforeach
                @if (! $shown)
                    <p class="text-sm text-zinc-500 dark:text-zinc-400">Tidak ada aksara yang cocok dengan pencarian ini.</p>
                @endif
            </div>
        </div>
    </section>

    {{-- 2. Kamus koreksi OCR: model bahasa tingkat karakter untuk beam search --}}
    <section class="card scroll-mt-20" id="kamus-koreksi">
        <div class="card-body">
            <div class="mb-3">
                <h2 class="section-title">Kamus koreksi OCR</h2>
                <p class="card-subtitle">Bukan kamus kata.@if ($lines) {{ nfmt($unspaced) }} dari {{ nfmt($lines) }} label uji ditulis tanpa spasi, jadi tidak ada batas kata untuk dicocokkan.@endif Kamus di alur ini adalah model bahasa tingkat karakter yang dibangun dari baris teks split train (<span class="font-mono">python -m src.charlm</span>). Saat membaca, beam search menimbang skor CTC dari citra dengan skor model bahasa itu, sehingga urutan aksara yang lazim lebih dipilih. Bobotnya disetel di dev sintetis, bukan di NusaAksara.</p>
            </div>

            @forelse ($lm as $m)
                <div class="flex flex-col gap-3" wire:key="lm-{{ $m['beam']->key }}">
                    <div class="code">{{ $m['beam']->config }}</div>
                    <div class="table-wrap">
                        <table class="data-table">
                            <thead>
                                <tr><th>Data</th><th class="r">Baris</th><th class="r">Greedy</th><th class="r">{{ $m['beam']->label }}</th><th class="r">Selisih</th><th>Catatan</th></tr>
                            </thead>
                            <tbody>
                                @foreach ($m['rows'] as $r)
                                    <tr>
                                        <td class="font-semibold">{{ $r['label'] }}</td>
                                        <td class="r num">{{ nfmt($r['lines']) }}</td>
                                        <td class="r num">{{ pct($r['greedy'], 2) }}</td>
                                        <td class="r num font-semibold">{{ pct($r['beam'], 2) }}</td>
                                        <td class="r num font-semibold {{ $r['beam'] < $r['greedy'] ? 'text-[var(--good)]' : 'text-[var(--bad)]' }}">{{ pt($r['beam'] - $r['greedy']) }}</td>
                                        {{-- @endif tidak boleh menempel pada huruf: Blade hanya mengenali direktif yang tidak didahului karakter kata. --}}
                                        <td class="text-zinc-500 dark:text-zinc-400">
                                            @if ($r['better'] !== null)
                                                {{ nfmt($r['better']) }} baris membaik, {{ nfmt($r['worse']) }} memburuk
                                            @endif
                                        </td>
                                    </tr>
                                @endforeach
                            </tbody>
                        </table>
                    </div>
                    <div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
                        <a href="{{ route('penjelajah', ['tag' => 'beam membaik']) }}" wire:navigate class="text-[var(--primary)] hover:underline">Lihat baris yang membaik</a>
                        <a href="{{ route('penjelajah', ['tag' => 'beam memburuk']) }}" wire:navigate class="text-[var(--primary)] hover:underline">Lihat baris yang memburuk</a>
                    </div>
                    @if ($official && $m['base'] && $official->key !== $m['base']->key)
                        <div class="alert alert-warning">
                            <span>Angka di atas membandingkan greedy dan beam pada checkpoint yang sama, {{ $m['base']->label }}. Kamus belum diukur di atas pipeline resmi {{ $official->label }}, dan bobotnya disetel pada {{ $m['base']->label }}.</span>
                        </div>
                    @endif
                </div>
            @empty
                <div class="card card-dashed">
                    <div class="card-body text-sm text-zinc-600 dark:text-zinc-300">
                        Belum ada hasil beam + model bahasa yang diimpor. Jalankan <span class="font-mono">php artisan aksara:import</span> setelah ekspor.
                    </div>
                </div>
            @endforelse
        </div>
    </section>

    {{-- 3. Leksikon tingkat tutur (tahap 4) --}}
    <section class="card scroll-mt-20" id="leksikon-tutur">
        <div class="card-body">
            <div class="mb-3">
                <h2 class="section-title">Leksikon tingkat tutur</h2>
                <p class="card-subtitle">Kata penanda yang dipakai tahap 4 untuk menebak ngoko, madya, atau krama dari teks Latin. Kata netral sengaja tidak dimasukkan; krama inggil dihitung krama; kata berakhiran <span class="whitespace-nowrap">-ipun</span> atau berawalan <span class="whitespace-nowrap">dipun-</span> juga dihitung krama. Pada teks tanpa spasi hanya penanda sepanjang lima huruf atau lebih yang dicari, dan keyakinannya selalu rendah.</p>
            </div>
            <div class="grid gap-4 lg:grid-cols-3">
                @foreach ($lexicon as $level => $words)
                    <div class="card-inset p-3" wire:key="tutur-{{ $level }}">
                        <div class="mb-2 flex items-center justify-between gap-2">
                            <span class="status status-blue">{{ $level }}</span>
                            <span class="num text-xs text-zinc-500 dark:text-zinc-400">{{ nfmt($words->count()) }}@if ($words->count() !== $lexiconTotal[$level]) dari {{ nfmt($lexiconTotal[$level]) }}@endif kata</span>
                        </div>
                        <div class="flex flex-wrap gap-1.5">
                            @forelse ($words as $w)
                                <span class="kamus-word">{{ $w }}</span>
                            @empty
                                <span class="text-xs text-zinc-500 dark:text-zinc-400">Tidak ada yang cocok.</span>
                            @endforelse
                        </div>
                    </div>
                @endforeach
            </div>
        </div>
    </section>
</div>
