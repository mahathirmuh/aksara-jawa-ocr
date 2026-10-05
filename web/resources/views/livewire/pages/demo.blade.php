<div class="flex flex-col gap-4">
    @include('partials.page-head', ['eyebrow' => 'Layanan model · FastAPI /predict', 'heading' => 'Baca potongan baris',
        'aside' => 'Hanya potongan satu baris. Deteksi baris dari foto halaman di luar scope.'])

    {{-- Status layanan --}}
    <div @class(['alert', 'alert-success' => $service, 'alert-warning' => ! $service])>
        @if ($service)
            <span class="alert-title">✓ Layanan model berjalan</span>
            <span class="code min-w-0 break-all">{{ $service['checkpoint'] ?? '' }} · {{ $service['lm'] ?? '' }}</span>
        @else
            <span class="alert-title">… Layanan model belum berjalan</span>
            <span>Jalankan di folder repo OCR:</span>
            <code class="code min-w-0 break-words">.venv/Scripts/python -m uvicorn src.serve:app --host 127.0.0.1 --port 8011</code>
        @endif
        <flux:button size="sm" wire:click="checkService" class="ml-auto">Periksa lagi</flux:button>
    </div>

    {{-- Tahap pipeline (pola ala Docling: setiap tahap bisa diganti). Panah di antara ubin digambar CSS (.demo-steps). --}}
    <section class="card">
        <div class="card-body">
            <div class="mb-3"><h2 class="section-title">Pipeline</h2></div>
            @php($beam = $pipeline === 'crnn_beam_lm')
            <ol class="demo-steps">
                @foreach ([['Praproses', 'polaritas · margin 0', false], ['Pembaca', $reader.($beam ? ' · beam 16' : ' · greedy'), true],
                    ['Penggabung', $beam ? 'LM karakter o5 · α 0,25 · β 1,0' : 'tidak ada', $beam], ['Pascaproses', 'NFC · urutan logis', false]] as [$name, $opt, $active])
                    <li @class(['demo-step card-inset', 'card-active' => $active])>
                        {{-- Ikon Tabler (MIT) per tahap: hiasan saja, urutannya mengikuti daftar di atas. --}}
                        <span class="demo-step-icon" aria-hidden="true">
                            @switch($loop->index)
                                @case(0)
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 6m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" /><path d="M4 6l8 0" /><path d="M16 6l4 0" /><path d="M8 12m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" /><path d="M4 12l2 0" /><path d="M10 12l10 0" /><path d="M17 18m-2 0a2 2 0 1 0 4 0a2 2 0 1 0 -4 0" /><path d="M4 18l11 0" /><path d="M19 18l1 0" /></svg>
                                    @break
                                @case(1)
                                    <flux:icon.ti-scan />
                                    @break
                                @case(2)
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7h5l3.5 5h9.5" /><path d="M3 17h5l3.495 -5" /><path d="M18 15l3 -3l-3 -3" /></svg>
                                    @break
                                @default
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l4 -4l4 4m-4 -4v14" /><path d="M21 15l-4 4l-4 -4m4 4v-14" /></svg>
                            @endswitch
                        </span>
                        <span class="min-w-0">
                            <span class="block text-sm font-semibold">{{ $name }}</span>
                            <span class="code block break-words">{{ $opt }}</span>
                        </span>
                    </li>
                @endforeach
            </ol>
            <p class="mt-3 text-xs text-muted">VLM fine-tune, gabungan, dan korektor LLM akan menjadi pilihan di sini setelah dibangun.</p>
        </div>
    </section>

    {{-- Sebelum ada hasil kedua kartu sama tinggi; sesudahnya kartu Unggah tidak ikut memanjang. --}}
    <div @class(['grid gap-4 lg:grid-cols-2 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]', 'items-start' => $result])>
        <section class="card">
            <div class="card-body flex flex-col gap-4">
                <h2 class="section-title">Unggah</h2>
                <form wire:submit="read" class="flex flex-col gap-4">
                    <div class="grid gap-2">
                        <label class="text-xs font-medium" for="photo">Potongan satu baris (PNG/JPG, maks. 8 MB)</label>
                        {{-- Input berkas bawaan bergaya area lepas-berkas: berkas juga bisa diseret ke kotak ini. --}}
                        <input id="photo" type="file" wire:model="photo" accept="image/*" class="demo-drop">
                        @error('photo')<p class="text-xs text-[var(--bad)]">{{ $message }}</p>@enderror
                    </div>
                    <div class="grid gap-2">
                        <label class="text-xs font-medium" for="pipeline">Pembaca</label>
                        <flux:select id="pipeline" wire:model.live="pipeline">
                            @foreach ($pipelines as $key => $name)
                                <option value="{{ $key }}">{{ $name }}</option>
                            @endforeach
                        </flux:select>
                    </div>
                    <div class="flex flex-wrap items-center gap-3">
                        <flux:button type="submit" variant="primary" :disabled="! $photo">Baca</flux:button>
                        <span wire:loading wire:target="read" class="text-xs text-muted">Membaca…</span>
                        <span wire:loading wire:target="photo" class="text-xs text-muted">Mengunggah…</span>
                    </div>
                </form>
                @if ($error)
                    <div class="alert alert-danger" role="alert"><span class="min-w-0 break-words">{{ $error }}</span></div>
                @endif
            </div>
        </section>

        <section class="card" x-data="{ band: null, pin: null }">
            <div class="card-body @container flex flex-col gap-4">
                <h2 class="section-title">Hasil</h2>
                @if ($photo && ! $errors->has('photo'))
                    <div class="card-inset overflow-x-auto p-3">
                        <div class="crop">
                            <img src="{{ $photo->temporaryUrl() }}" alt="Potongan baris yang diunggah">
                            <span class="band" x-show="band" x-cloak :style="band ? `left:${band[0] * 100}%;width:${Math.max(0.4, (band[1] - band[0]) * 100)}%` : ''"></span>
                        </div>
                    </div>
                @endif
                @if ($result)
                    @php($segments = collect($result['syllables'] ?? [])->map(fn ($s) => ['t' => $s['text'], 's' => 'ok'])->all())
                    @php($spans = collect($result['syllables'] ?? [])->map(fn ($s) => [$s['x0'], $s['x1']])->all())
                    <div>
                        <div class="subheader">Tahap 1 · aksara <span class="code font-normal normal-case tracking-normal">({{ $result['config'] }})</span></div>
                        <div class="jv mt-1 break-all text-3xl leading-[1.9]">{{ $result['text'] }}</div>
                        @if (($result['candidates'][0]['text'] ?? null) !== null)
                            <div class="mt-1 text-xs text-muted">Suku kata greedy (arahkan kursor untuk melihat kolom citranya):</div>
                            <div class="jv break-all text-xl leading-[1.9]">{!! segments_html($segments, $spans) !!}</div>
                        @endif
                    </div>
                    @php($latinText = \App\Support\Transliterator::toLatin($result['text']))
                    @php($tutur = \App\Support\SpeechLevel::classify($latinText))
                    <div class="grid gap-3 @xl:grid-cols-3">
                        <div class="card-inset min-w-0 p-3">
                            <div class="subheader">Tahap 2 · Latin</div>
                            <div class="mt-1 break-words text-sm italic">{{ $latinText }}</div>
                            <div class="mt-1 text-[11px] text-muted">draf aturan</div>
                        </div>
                        <div class="card-inset min-w-0 p-3">
                            <div class="subheader">Tahap 3 · arti</div>
                            @if ($translation)
                                <div class="mt-1 break-words text-sm">{{ $translation['translation'] }}</div>
                                <div class="mt-1 text-[11px] text-muted">NLLB lokal · {{ number_format($translation['elapsed_ms'], 0, ',', '.') }} ms</div>
                            @else
                                <div class="mt-1 text-sm text-muted">layanan terjemahan belum berjalan</div>
                                <code class="code mt-1 block break-words">../.venv/Scripts/python -m uvicorn translate_service:app --app-dir tools --port 8012</code>
                            @endif
                        </div>
                        <div class="card-inset min-w-0 p-3">
                            <div class="subheader">Tahap 4 · tingkat tutur</div>
                            <div class="mt-1 text-sm">{{ $tutur['level'] ?? 'tak tentu' }}</div>
                            <div class="mt-1 text-[11px] text-muted">leksikon · keyakinan {{ $tutur['confidence'] }}{{ $tutur['spaced'] ? '' : ' (teks tanpa spasi)' }}</div>
                        </div>
                    </div>
                    <p class="text-xs text-muted">Tahap 2–4 di sini berjalan dari keluaran OCR, jadi kesalahan OCR ikut terbawa. Perbandingan dengan teks benar ada di Penjelajah baris.</p>
                    <div class="table-wrap">
                        <table class="data-table">
                            <thead><tr><th>Kandidat</th><th>Teks</th><th class="r">log P CTC</th><th class="r">log P LM</th><th class="r">Skor</th></tr></thead>
                            <tbody>
                                @foreach ($result['candidates'] as $c)
                                    <tr @class(['is-highlight font-semibold' => $c['chosen'] ?? false])>
                                        <td class="whitespace-nowrap"><span class="code">{{ $c['source'] }}</span>@if ($c['chosen'] ?? false) <span class="status status-blue">dipilih</span>@endif</td>
                                        {{-- Lebar minimum: tanpa ini, di layar sempit kolom teks menyusut sampai satu suku kata per baris;
                                             tabelnya menggulung mendatar di dalam .table-wrap. --}}
                                        <td class="jv min-w-[16rem] text-lg font-normal">{{ $c['text'] }}</td>
                                        <td class="r num">{{ number_format($c['ctc_logp'], 2, ',', '.') }}</td>
                                        <td class="r num">{{ number_format($c['lm_logp'], 2, ',', '.') }}</td>
                                        <td class="r num">{{ number_format($c['score'], 2, ',', '.') }}</td>
                                    </tr>
                                @endforeach
                            </tbody>
                        </table>
                    </div>
                    <p class="text-xs text-muted">Skor = log P CTC + 0,25 × log P LM + 1,0 × panjang. CTC mengukur seberapa didukung piksel; kandidat dari sumber mana pun (VLM, LLM) kelak dinilai dengan skor yang sama. {{ number_format($result['elapsed_ms'], 0, ',', '.') }} ms.</p>
                @elseif (! $photo)
                    <div class="flex flex-1 flex-col items-center justify-center gap-3 py-6 text-center">
                        <span class="stat-icon" aria-hidden="true"><flux:icon.ti-scan /></span>
                        <p class="text-sm text-muted">Unggah potongan baris, lalu tekan Baca.</p>
                    </div>
                @endif
            </div>
        </section>
    </div>
</div>
