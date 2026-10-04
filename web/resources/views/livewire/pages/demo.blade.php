<div class="mx-auto flex max-w-6xl flex-col gap-6">
    @include('partials.page-head', ['eyebrow' => 'Layanan model · FastAPI /predict', 'heading' => 'Baca potongan baris',
        'aside' => 'Hanya potongan satu baris. Deteksi baris dari foto halaman di luar scope.'])

    {{-- Status layanan --}}
    <div @class(['flex flex-wrap items-center gap-3 rounded-lg px-4 py-3 text-sm',
            'bg-[var(--good-wash)] text-[var(--good)]' => $service, 'bg-[var(--warn-wash)] text-[var(--warn)]' => ! $service])>
        @if ($service)
            <span class="font-semibold">✓ Layanan model berjalan</span>
            <span class="font-mono text-xs">{{ $service['checkpoint'] ?? '' }} · {{ $service['lm'] ?? '' }}</span>
        @else
            <span class="font-semibold">… Layanan model belum berjalan</span>
            <span>Jalankan di folder repo OCR:</span>
            <code class="font-mono text-xs">.venv/Scripts/python -m uvicorn src.serve:app --host 127.0.0.1 --port 8011</code>
        @endif
        <button type="button" wire:click="checkService" class="ml-auto rounded-md border border-current px-2 py-0.5 text-xs">Periksa lagi</button>
    </div>

    {{-- Tahap pipeline (pola ala Docling: setiap tahap bisa diganti) --}}
    <section class="rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900">
        <flux:heading size="lg">Pipeline</flux:heading>
        @php($beam = $pipeline === 'crnn_beam_lm')
        <div class="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            @foreach ([['Praproses', 'polaritas · margin 0', false], ['Pembaca', $reader.($beam ? ' · beam 16' : ' · greedy'), true],
                ['Penggabung', $beam ? 'LM karakter o5 · α 0,25 · β 1,0' : 'tidak ada', $beam], ['Pascaproses', 'NFC · urutan logis', false]] as [$name, $opt, $active])
                <div @class(['flex flex-col gap-1 rounded-lg border p-3', 'border-accent bg-[color-mix(in_srgb,var(--color-accent)_10%,transparent)]' => $active,
                        'border-zinc-200 bg-zinc-50 dark:border-zinc-700 dark:bg-zinc-800' => ! $active])>
                    <span class="text-sm font-semibold">{{ $name }}</span>
                    <span class="font-mono text-[11px] text-zinc-600 dark:text-zinc-300">{{ $opt }}</span>
                </div>
            @endforeach
        </div>
        <p class="mt-2 text-xs text-zinc-500">VLM fine-tune, gabungan, dan korektor LLM akan menjadi pilihan di sini setelah dibangun.</p>
    </section>

    <div class="grid gap-6 lg:grid-cols-2">
        <section class="flex flex-col gap-4 rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900">
            <flux:heading size="lg">Unggah</flux:heading>
            <form wire:submit="read" class="flex flex-col gap-4">
                <label class="flex flex-col gap-1 text-sm" for="photo">Potongan satu baris (PNG/JPG, maks. 8 MB)
                    <input id="photo" type="file" wire:model="photo" accept="image/*" class="text-sm file:mr-3 file:rounded-md file:border-0 file:bg-zinc-100 file:px-3 file:py-1.5 dark:file:bg-zinc-800">
                </label>
                @error('photo')<p class="text-sm text-[var(--bad)]">{{ $message }}</p>@enderror
                <label class="flex flex-col gap-1 text-sm" for="pipeline">Pembaca
                    <select id="pipeline" wire:model.live="pipeline" class="rounded-md border border-zinc-300 bg-transparent px-2 py-1.5 dark:border-zinc-600">
                        @foreach ($pipelines as $key => $name)
                            <option value="{{ $key }}">{{ $name }}</option>
                        @endforeach
                    </select>
                </label>
                <div class="flex items-center gap-3">
                    <flux:button type="submit" variant="primary" :disabled="! $photo">Baca</flux:button>
                    <span wire:loading wire:target="read" class="text-sm text-zinc-500">Membaca…</span>
                    <span wire:loading wire:target="photo" class="text-sm text-zinc-500">Mengunggah…</span>
                </div>
            </form>
            @if ($error)
                <p class="rounded-lg bg-[var(--bad-wash)] px-3 py-2 text-sm text-[var(--bad)]">{{ $error }}</p>
            @endif
        </section>

        <section class="flex min-w-0 flex-col gap-4 rounded-xl border border-zinc-200 bg-white p-5 dark:border-zinc-700 dark:bg-zinc-900"
                 x-data="{ band: null, pin: null }">
            <flux:heading size="lg">Hasil</flux:heading>
            @if ($photo && ! $errors->has('photo'))
                <div class="overflow-x-auto rounded-lg border border-zinc-200 bg-zinc-50 p-3 dark:border-zinc-700 dark:bg-zinc-800">
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
                    <div class="eyebrow">Tahap 1 · aksara ({{ $result['config'] }})</div>
                    <div class="jv mt-1 break-all text-3xl leading-[1.9]">{{ $result['text'] }}</div>
                    @if (($result['candidates'][0]['text'] ?? null) !== null)
                        <div class="mt-1 text-xs text-zinc-500">Suku kata greedy (arahkan kursor untuk melihat kolom citranya):</div>
                        <div class="jv break-all text-xl leading-[1.9]">{!! segments_html($segments, $spans) !!}</div>
                    @endif
                </div>
                @php($latinText = \App\Support\Transliterator::toLatin($result['text']))
                @php($tutur = \App\Support\SpeechLevel::classify($latinText))
                <div class="grid gap-3 sm:grid-cols-3">
                    <div><div class="eyebrow">Tahap 2 · Latin</div><div class="mt-1 text-sm italic">{{ $latinText }}</div><div class="text-[11px] text-zinc-500">draf aturan</div></div>
                    <div><div class="eyebrow">Tahap 3 · arti</div>
                        @if ($translation)
                            <div class="mt-1 text-sm">{{ $translation['translation'] }}</div>
                            <div class="text-[11px] text-zinc-500">NLLB lokal · {{ number_format($translation['elapsed_ms'], 0, ',', '.') }} ms</div>
                        @else
                            <div class="mt-1 text-sm text-zinc-500">layanan terjemahan belum berjalan</div>
                            <code class="text-[11px] text-zinc-500">../.venv/Scripts/python -m uvicorn translate_service:app --app-dir tools --port 8012</code>
                        @endif
                    </div>
                    <div><div class="eyebrow">Tahap 4 · tingkat tutur</div>
                        <div class="mt-1 text-sm">{{ $tutur['level'] ?? 'tak tentu' }}</div>
                        <div class="text-[11px] text-zinc-500">leksikon · keyakinan {{ $tutur['confidence'] }}{{ $tutur['spaced'] ? '' : ' (teks tanpa spasi)' }}</div>
                    </div>
                </div>
                <p class="text-xs text-zinc-500">Tahap 2–4 di sini berjalan dari keluaran OCR, jadi kesalahan OCR ikut terbawa. Perbandingan dengan teks benar ada di Penjelajah baris.</p>
                <div class="overflow-x-auto">
                    <table class="data-table w-full text-sm">
                        <thead><tr><th>Kandidat</th><th>Teks</th><th class="r">log P CTC</th><th class="r">log P LM</th><th class="r">Skor</th></tr></thead>
                        <tbody>
                            @foreach ($result['candidates'] as $c)
                                <tr @class(['font-semibold' => $c['chosen'] ?? false])>
                                    <td class="font-mono text-xs">{{ $c['source'] }}@if ($c['chosen'] ?? false) <span class="pill pill-acc">dipilih</span>@endif</td>
                                    <td class="jv text-lg">{{ $c['text'] }}</td>
                                    <td class="r num">{{ number_format($c['ctc_logp'], 2, ',', '.') }}</td>
                                    <td class="r num">{{ number_format($c['lm_logp'], 2, ',', '.') }}</td>
                                    <td class="r num">{{ number_format($c['score'], 2, ',', '.') }}</td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
                <p class="text-xs text-zinc-500">Skor = log P CTC + 0,25 × log P LM + 1,0 × panjang. CTC mengukur seberapa didukung piksel; kandidat dari sumber mana pun (VLM, LLM) kelak dinilai dengan skor yang sama. {{ number_format($result['elapsed_ms'], 0, ',', '.') }} ms.</p>
            @elseif (! $photo)
                <p class="text-sm text-zinc-500">Unggah potongan baris, lalu tekan Baca.</p>
            @endif
        </section>
    </div>
</div>
