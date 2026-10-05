@php
    // Anak judul: isi tiap kamus, kamus kata hanya disebut bila sudah diimpor.
    $aside = collect($dictionaries)->filter(fn ($d) => $d['entries'] > 0)
        ->map(fn ($d, $key) => nfmt($d['entries']).' entri '.($key === 'jv' ? 'bahasa Jawa' : 'bahasa Indonesia'))
        ->push(nfmt($total).' aksara dan tanda', nfmt($lexiconTotal->sum()).' kata penanda tingkat tutur')->implode(' · ');
@endphp
<div class="flex flex-col gap-4">
    @include('partials.page-head', ['eyebrow' => 'Rujukan alur', 'heading' => 'Kamus', 'aside' => $aside])

    {{-- Pencarian: satu kotak untuk semua kamus. Kata dicari di kamus kata, nama/kode di kamus aksara dan leksikon;
         aksara yang ditempel diurai per codepoint lalu dicari lewat bacaan Latin drafnya. --}}
    <section class="card">
        <div class="card-body flex flex-col gap-3">
            <flux:input wire:model.live.debounce.300ms="q" id="kamus-q" type="search" icon="ti-search" autocomplete="off"
                        aria-label="Cari di kamus"
                        placeholder="Cari kata Jawa atau Indonesia, nama atau kode aksara; bisa juga menempel aksara (mis. omah, rumah, wulu, U+A9B6, {{ mb_chr(0xA98F).mb_chr(0xA9B6) }})" />

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
                        <p class="text-xs text-zinc-500 dark:text-zinc-400">Hasil pencarian "{{ $q }}" ada di tiap kamus di bawah.</p>
                    @endif
                </div>
            @else
                <p class="text-xs text-zinc-500 dark:text-zinc-400">Satu kotak untuk semua kamus di bawah. Beberapa kata sekaligus (mis. satu baris transliterasi) diartikan per kata; aksara Jawa yang ditempel diurai per aksara dan dicari lewat bacaan Latin drafnya.</p>
            @endif

            {{-- Halaman ini panjang: lompat ke bagiannya. Angka = jumlah yang cocok dengan pencarian. --}}
            <nav class="flex flex-wrap gap-2" aria-label="Bagian kamus">
                @foreach ($dictionaries as $key => $d)
                    <a href="#kamus-{{ $key }}" class="chip" wire:key="lompat-{{ $key }}">{{ $d['label'] }}
                        @if ($d['mode'] === 'search' || $d['mode'] === 'words')
                            <span class="chip-count num">{{ nfmt($d['found']) }}</span>
                        @endif
                    </a>
                @endforeach
                <a href="#kamus-aksara" class="chip">Kamus aksara <span class="chip-count num">{{ nfmt($shown) }}</span></a>
                <a href="#kamus-koreksi" class="chip">Kamus koreksi OCR</a>
                <a href="#leksikon-tutur" class="chip">Leksikon tingkat tutur <span class="chip-count num">{{ nfmt($lexicon->sum(fn ($words) => $words->count())) }}</span></a>
            </nav>
        </div>
    </section>

    {{-- Kamus kata: bahasa Jawa dan bahasa Indonesia. Data pihak ketiga (lihat sumber di kaki tiap bagian). --}}
    @foreach ($dictionaries as $key => $d)
        @php
            // Di tiap entri cukup nama pendek sumbernya (tanpa keterangan dalam kurung); nama lengkap dan lisensinya di kaki bagian.
            $sourceNames = $d['sources']->mapWithKeys(fn ($source) => [$source->key => trim(explode('(', $source->name)[0])])->all();
            // Bahasa arti: bila kamus ini hanya punya satu bahasa arti, cukup disebut sekali di anak judul.
            $languages = collect($d['languages'])->map(fn ($code) => \App\Models\DictionaryEntry::GLOSS_LANGUAGES[$code] ?? $code);
            $mixed = $languages->count() > 1;
        @endphp
        <section class="card scroll-mt-20" id="kamus-{{ $key }}" wire:key="kamus-kata-{{ $key }}">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">{{ $d['label'] }}</h2>
                    <p class="card-subtitle">
                        @if ($key === 'jv')
                            Kata bahasa Jawa dengan ejaan aksara (bila sumbernya punya), kelas kata, ragam ngoko atau krama, dan artinya.
                        @else
                            Kata bahasa Indonesia dengan kelas kata dan artinya, untuk membaca hasil tahap arti.
                        @endif
                        @if ($languages->isNotEmpty())
                            Arti berbahasa {{ $languages->join(', ', ' dan ') }}{{ $mixed && $languages->contains('Indonesia') ? '; yang berbahasa Indonesia didahulukan' : '' }}. Bisa juga dicari dari artinya: ketik kata dalam bahasa itu untuk menemukan {{ $key === 'jv' ? 'padanan Jawanya' : 'kata Indonesianya' }}.
                        @endif
                        Ini data pihak ketiga, bukan hasil OCR atau hitungan proyek ini.
                    </p>
                </div>

                @if ($d['mode'] === 'missing')
                    <div class="card card-dashed">
                        <div class="card-body text-sm text-zinc-600 dark:text-zinc-300">
                            {{ $d['label'] }} belum diimpor. Bangun datanya dengan <span class="font-mono">python web/tools/dictionaries.py</span>, lalu jalankan <span class="font-mono">php artisan aksara:dictionary</span>.
                        </div>
                    </div>
                @elseif ($d['mode'] === 'idle')
                    <div class="flex flex-wrap items-center gap-2 text-sm">
                        <span class="text-xs text-zinc-500 dark:text-zinc-400">{{ nfmt($d['entries']) }} entri. Ketik kata di kotak pencarian, atau coba:</span>
                        @foreach ($d['examples'] as $example)
                            {{-- Tombol ini hilang begitu pencarian jalan, jadi fokus dipindah ke kotak cari (kalau tidak, jatuh ke badan halaman). --}}
                            <button type="button" class="chip" wire:click="$set('q', '{{ $example }}')" wire:key="contoh-{{ $key }}-{{ $example }}"
                                    x-on:click="document.getElementById('kamus-q')?.focus({ preventScroll: true })">{{ $example }}</button>
                        @endforeach
                    </div>
                @elseif ($d['mode'] === 'words')
                    {{-- Beberapa kata: arti per kata, hanya entri yang katanya persis sama. --}}
                    @if ($d['phrase']->isNotEmpty())
                        <div class="dict-list mb-3">
                            @foreach ($d['phrase'] as $entry)
                                @include('partials.dictionary-entry', ['entry' => $entry, 'sourceNames' => $sourceNames, 'showLanguage' => $mixed])
                            @endforeach
                        </div>
                    @endif
                    <div class="table-wrap">
                        <table class="data-table">
                            <thead>
                                <tr><th>Kata</th><th>Arti per kata</th></tr>
                            </thead>
                            <tbody>
                                @foreach ($d['rows'] as $i => $row)
                                    <tr wire:key="per-kata-{{ $key }}-{{ $i }}">
                                        <td class="whitespace-nowrap align-top font-semibold">{{ $row['word'] }}</td>
                                        <td>
                                            @forelse ($row['entries'] as $entry)
                                                <div class="dict-inline">
                                                    @if ($entry->aksara)
                                                        <span class="jv dict-aksara" lang="jv-Java">{{ $entry->aksara }}</span>
                                                    @endif
                                                    @if ($entry->pos)
                                                        <span class="text-xs text-zinc-500 dark:text-zinc-400">{{ $entry->pos }}</span>
                                                    @endif
                                                    @if ($entry->register)
                                                        <span class="status status-blue">{{ $entry->register }}</span>
                                                    @endif
                                                    <span lang="{{ $entry->gloss_lang }}">{{ implode('; ', array_slice($entry->glosses, 0, 2)) }}</span>
                                                    @if ($mixed && $entry->gloss_lang !== 'id')
                                                        <span class="text-[11px] text-zinc-500 dark:text-zinc-400">({{ \App\Models\DictionaryEntry::GLOSS_LANGUAGES[$entry->gloss_lang] ?? $entry->gloss_lang }})</span>
                                                    @endif
                                                    @if ($entry->url)
                                                        <a href="{{ $entry->url }}" target="_blank" rel="noopener noreferrer" class="text-[11px] text-[var(--primary)] hover:underline"
                                                           aria-label="Sumber entri {{ $entry->word }}: {{ $sourceNames[$entry->source] ?? $entry->source }}">sumber</a>
                                                    @endif
                                                </div>
                                            @empty
                                                <span class="text-zinc-500 dark:text-zinc-400">tidak ada di kamus ini</span>
                                            @endforelse
                                        </td>
                                    </tr>
                                @endforeach
                            </tbody>
                        </table>
                    </div>
                    <p class="mt-2 text-xs text-zinc-500 dark:text-zinc-400">Paling banyak {{ \App\Support\DictionarySearch::MAX_WORDS }} kata pertama yang diartikan, dan hanya entri yang katanya persis sama. Cari satu kata untuk melihat entri lengkapnya.</p>
                @else
                    @if ($d['matches']->isEmpty() && $d['reverse']->isEmpty())
                        <p class="text-sm text-zinc-500 dark:text-zinc-400">Tidak ada entri yang cocok dengan pencarian ini.</p>
                    @endif
                    {{-- Direktif Blade tidak dikenali bila menempel pada huruf, jadi @if selalu di baris sendiri. --}}
                    @if ($d['matches']->isNotEmpty())
                        <div class="subheader mb-1">
                            {{ nfmt($d['total']) }} entri cocok dengan katanya
                            @if ($d['total'] > $d['matches']->count())
                                · {{ nfmt($d['matches']->count()) }} ditampilkan
                            @endif
                        </div>
                        <div class="dict-list">
                            @foreach ($d['matches'] as $entry)
                                @include('partials.dictionary-entry', ['entry' => $entry, 'sourceNames' => $sourceNames, 'showLanguage' => $mixed])
                            @endforeach
                        </div>
                        @if ($d['total'] > $d['matches']->count() && $d['limit'] < $d['max'])
                            <div class="mt-3">
                                <button type="button" class="chip" wire:click="more('{{ $key }}')">Tampilkan lebih banyak</button>
                            </div>
                        @endif
                    @endif
                    @if ($d['reverse']->isNotEmpty())
                        <div class="subheader mb-1 mt-4">
                            Kata lain yang artinya memuat kata ini · {{ nfmt($d['reverse_total']) }}
                            @if ($d['reverse_total'] > $d['reverse']->count())
                                · {{ nfmt($d['reverse']->count()) }} ditampilkan
                            @endif
                        </div>
                        <div class="dict-list">
                            @foreach ($d['reverse'] as $entry)
                                @include('partials.dictionary-entry', ['entry' => $entry, 'sourceNames' => $sourceNames, 'showLanguage' => $mixed, 'limit' => 2])
                            @endforeach
                        </div>
                    @endif
                @endif

                {{-- KBBI tidak boleh disalin (hak cipta Badan Bahasa), jadi definisi resminya hanya ditautkan. --}}
                @if ($key === 'id' && $d['mode'] === 'search' && $d['term'] !== '' && ! ($probe['aksara'] ?? false))
                    <p class="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
                        Definisi resmi:
                        <a href="https://kbbi.kemendikdasmen.go.id/entri/{{ rawurlencode($d['term']) }}" target="_blank" rel="noopener noreferrer" class="text-[var(--primary)] hover:underline">"{{ $d['term'] }}" di KBBI Daring</a>
                        (situs Badan Bahasa, dibuka di tab baru; isinya tidak disalin ke sini).
                    </p>
                @endif

                @if ($d['sources']->isNotEmpty())
                    <p class="dict-sources">
                        Sumber:
                        @foreach ($d['sources'] as $source)
                            <a href="{{ $source->url }}" target="_blank" rel="noopener noreferrer" class="text-[var(--primary)] hover:underline">{{ $source->name }}</a>
                            ({{ $source->license }}; {{ nfmt($source->entries) }} entri{{ $source->retrieved ? '; diambil '.$source->retrieved : '' }}){{ $loop->last ? '.' : ',' }}
                        @endforeach
                    </p>
                @endif
            </div>
        </section>
    @endforeach

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
