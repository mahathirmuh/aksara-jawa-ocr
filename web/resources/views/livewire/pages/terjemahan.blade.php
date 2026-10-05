@php
    $loaded = (bool) ($service['loaded'] ?? false);
    // Layanan versi lama hanya mengenal Jawa -> Indonesia dan tidak melaporkan "directions".
    $twoWay = in_array('id-jv', $service['directions'] ?? [], true);
    [$fromLabel, $toLabel] = $directions[$direction];
@endphp
<div class="flex flex-col gap-4">
    @include('partials.page-head', ['eyebrow' => 'Alat · terjemahan mesin dan alih aksara', 'heading' => 'Terjemahan',
        'aside' => 'Hasil di halaman ini draf mesin, bukan bagian dari angka OCR.'])

    {{-- Status layanan model terjemahan. Alih aksara (Latin ↔ aksara) dikerjakan web dan tetap jalan tanpa layanan ini. --}}
    <div @class(['alert', 'alert-success' => $service && $loaded && $twoWay, 'alert-warning' => ! $service || ! $twoWay, 'alert-info' => $service && $twoWay && ! $loaded])>
        @if (! $service)
            <span class="alert-title">… Layanan terjemahan belum berjalan</span>
            <span>Alih aksara tetap bisa dipakai. Untuk menerjemahkan, jalankan di folder <span class="font-mono">web</span>:</span>
            <code class="code min-w-0 break-words">../.venv/Scripts/python -m uvicorn translate_service:app --app-dir tools --host 127.0.0.1 --port 8012</code>
        @elseif (! $twoWay)
            <span class="alert-title">Layanan terjemahan versi lama</span>
            <span>Yang berjalan hanya mengenal Jawa → Indonesia. Nyalakan ulang layanannya supaya dua arah.</span>
        @elseif (! $loaded)
            <span class="alert-title">Model sedang dimuat</span>
            <span>Butuh sekitar satu menit; terjemahan pertama menunggu sampai selesai.</span>
        @else
            <span class="alert-title">✓ Layanan terjemahan berjalan</span>
            <span class="code min-w-0 break-all">{{ $service['model'] ?? '' }} · lokal, tidak ada teks yang dikirim ke luar</span>
        @endif
        <flux:button size="sm" wire:click="checkService" class="ml-auto">Periksa lagi</flux:button>
    </div>

    <section class="card">
        <div class="card-body flex flex-col gap-4">
            {{-- Arah terjemahan --}}
            <div class="flex flex-wrap items-center gap-2" role="group" aria-label="Arah terjemahan">
                @foreach ($directions as $key => [$from, $to])
                    <button type="button" class="chip" wire:click="$set('direction', '{{ $key }}')" aria-pressed="{{ $direction === $key ? 'true' : 'false' }}" wire:key="arah-{{ $key }}">
                        {{ $from }} → {{ $to }}
                    </button>
                @endforeach
                <button type="button" class="chip" wire:click="swap" title="Tukar arah; hasil yang ada menjadi masukan">⇄ Tukar</button>
            </div>

            <div class="translate-grid">
                {{-- Masukan --}}
                <form wire:submit="translate" class="flex min-w-0 flex-col gap-2">
                    <label class="subheader" for="terjemahan-teks">{{ $fromLabel }}</label>
                    <flux:textarea id="terjemahan-teks" wire:model.live.debounce.400ms="text" rows="7" maxlength="{{ \App\Livewire\Pages\Terjemahan::MAX_CHARS }}"
                                   class="{{ $inputIsAksara ? 'jv translate-aksara-input' : '' }}"
                                   placeholder="{{ $toJavanese ? 'Tulis kalimat bahasa Indonesia, mis. Saya akan pergi ke pasar besok pagi.' : 'Tempel aksara Jawa atau tulis bahasa Jawa beraksara Latin, mis. Aku arep lunga menyang pasar.' }}" />
                    @error('text')<p class="text-xs text-[var(--bad)]">{{ $message }}</p>@enderror
                    <div class="flex flex-wrap items-center gap-3">
                        <flux:button type="submit" variant="primary">Terjemahkan</flux:button>
                        <flux:button type="button" wire:click="clear">Kosongkan</flux:button>
                        <span wire:loading wire:target="translate" class="text-xs text-muted">Menerjemahkan… model lokal butuh 1–3 detik per kalimat.</span>
                        <span class="num ml-auto text-xs text-muted">{{ nfmt(mb_strlen($text)) }} / {{ nfmt(\App\Livewire\Pages\Terjemahan::MAX_CHARS) }}</span>
                    </div>
                    <p class="text-xs text-muted">Paling banyak {{ \App\Livewire\Pages\Terjemahan::MAX_SENTENCES }} kalimat sekali jalan; tiap kalimat diterjemahkan sendiri.</p>
                    @if ($latinDraft !== null)
                        {{-- Jawa beraksara: dilatinkan dulu oleh aturan, baru diterjemahkan. --}}
                        <div class="card-inset p-3" wire:key="latin-draf">
                            <div class="subheader">Latin (draf aturan)</div>
                            <div class="mt-1 break-words text-sm italic">{{ $latinDraft !== '' ? $latinDraft : '–' }}</div>
                            <div class="mt-1 text-[11px] text-muted">Teks inilah yang diterjemahkan. Aksara tanpa spasi tidak punya batas kata, jadi terjemahannya lebih lemah.</div>
                        </div>
                    @endif
                </form>

                {{-- Hasil --}}
                <div class="flex min-w-0 flex-col gap-3">
                    @if ($error)
                        <div class="alert alert-danger" wire:key="galat">
                            <span>{{ $error }}</span>
                        </div>
                    @endif

                    @if ($toJavanese)
                        <div class="flex flex-col gap-2">
                            <label class="subheader" for="terjemahan-jawa">Bahasa Jawa (Latin)</label>
                            <flux:textarea id="terjemahan-jawa" wire:model.live.debounce.400ms="javanese" rows="4" maxlength="2000"
                                           placeholder="Hasil terjemahan muncul di sini dan boleh disunting. Bisa juga langsung menulis bahasa Jawa: aksaranya mengikuti." />
                            <div class="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
                                <span>é/è = taling, e = pepet.</span>
                                <button type="button" class="text-[var(--primary)] hover:underline" wire:click="restoreTaling">Pulihkan tanda é</button>
                                @if ($restored)
                                    <span>{{ nfmt(count($restored)) }} kata dipulihkan: {{ collect($restored)->take(8)->map(fn ($to, $from) => "{$from} → {$to}")->implode(', ') }}{{ count($restored) > 8 ? ', …' : '' }}</span>
                                @endif
                            </div>
                        </div>
                        <div class="flex flex-col gap-2" x-data="{ copied: false }">
                            <div class="flex items-center justify-between gap-2">
                                <span class="subheader">Aksara Jawa</span>
                                @if ($aksara !== '')
                                    <button type="button" class="text-xs text-[var(--primary)] hover:underline"
                                            x-on:click="navigator.clipboard.writeText($refs.aksara.innerText).then(() => { copied = true; setTimeout(() => copied = false, 1500) })">
                                        <span x-show="! copied">Salin</span><span x-show="copied" x-cloak>Tersalin</span>
                                    </button>
                                @endif
                            </div>
                            <div class="translate-output jv" x-ref="aksara" lang="jv-Java" wire:key="hasil-aksara">{{ $aksara !== '' ? $aksara : '–' }}</div>
                        </div>
                    @else
                        <div class="flex flex-col gap-2" x-data="{ copied: false }">
                            <div class="flex items-center justify-between gap-2">
                                <span class="subheader">Bahasa Indonesia</span>
                                @if ($indonesian !== '')
                                    <button type="button" class="text-xs text-[var(--primary)] hover:underline"
                                            x-on:click="navigator.clipboard.writeText($refs.indonesia.innerText).then(() => { copied = true; setTimeout(() => copied = false, 1500) })">
                                        <span x-show="! copied">Salin</span><span x-show="copied" x-cloak>Tersalin</span>
                                    </button>
                                @endif
                            </div>
                            <div class="translate-output" x-ref="indonesia" wire:key="hasil-indonesia">{{ $indonesian !== '' ? $indonesian : '–' }}</div>
                        </div>
                    @endif

                    <div class="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
                        @if ($meta)
                            <span>NLLB lokal · {{ nfmt($meta['sentences']) }} kalimat · {{ number_format($meta['elapsed_ms'] / 1000, 1, ',', '.') }} detik</span>
                            @if ($meta['long'])
                                <span class="text-[var(--warn)]">{{ nfmt($meta['long']) }} kalimat lebih dari {{ \App\Livewire\Pages\Terjemahan::LONG_SENTENCE }} karakter: terjemahannya bisa terpotong.</span>
                            @endif
                        @endif
                        @if ($speech && ($speech['spaced'] || $speech['evidence']))
                            <span>Tingkat tutur menurut leksikon: <span class="status {{ $speech['level'] ? 'status-blue' : 'status-gray' }}">{{ $speech['level'] ?? 'tak tentu' }}</span> keyakinan {{ $speech['confidence'] }}</span>
                        @endif
                        @if ($kamusQuery !== '')
                            <a href="{{ route('kamus', ['q' => $kamusQuery]) }}" wire:navigate class="text-[var(--primary)] hover:underline">Arti per kata di Kamus</a>
                        @endif
                    </div>
                </div>
            </div>
        </div>
    </section>

    {{-- Apa yang dikerjakan siapa, dan seberapa bisa dipercaya --}}
    <section class="card">
        <div class="card-body">
            <h2 class="section-title">Cara kerja dan batasnya</h2>
            <div class="mt-3 grid gap-4 lg:grid-cols-3">
                <div class="card-inset p-3">
                    <div class="subheader">1 · Menerjemahkan</div>
                    <p class="mt-1 text-sm">Indonesia ↔ Jawa beraksara Latin oleh model NLLB-200 (600M) yang berjalan di laptop ini. Model kecil: kalimat sederhana biasanya benar, tetapi makna bisa meleset atau ada kata yang hilang. Perlakukan sebagai draf dan baca ulang.</p>
                </div>
                <div class="card-inset p-3">
                    <div class="subheader">2 · Memulihkan tanda é</div>
                    @if ($lexicon)
                        <p class="mt-1 text-sm">Aksara Jawa membedakan taling (é) dari pepet (e); teks Latin jarang. Leksikon {{ nfmt($lexicon['words']) }} kata dari Wikipedia bahasa Jawa memulihkannya: pada artikel yang ditahan, kata ber-e yang ejaannya benar naik dari {{ pct($lexicon['held_out']['accuracy_without'] ?? null, 0) }} menjadi {{ pct($lexicon['held_out']['accuracy_with'] ?? null, 0) }}. Kata di luar leksikon tetap dibaca pepet.</p>
                    @else
                        <p class="mt-1 text-sm">Leksikon pemulih tanda é belum dibangun (<span class="font-mono">tools/taling_lexicon.py</span>), jadi setiap e tanpa tanda dibaca pepet.</p>
                    @endif
                </div>
                <div class="card-inset p-3">
                    <div class="subheader">3 · Alih aksara</div>
                    <p class="mt-1 text-sm">Latin → aksara dan sebaliknya memakai aturan pasti di web, tanpa model. Aturan Latin → aksara menghasilkan ejaan yang sama dengan {{ nfmt($agreement[0]) }} dari {{ nfmt($agreement[1]) }} lema kamus bahasa Jawa ({{ pct($agreement[0] / $agreement[1]) }}). Tidak memakai aksara murda dan swara untuk nama.</p>
                </div>
            </div>
        </div>
    </section>
</div>
