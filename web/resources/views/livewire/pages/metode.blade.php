<div class="flex flex-col gap-4">
    @include('partials.page-head', ['eyebrow' => 'Metode', 'heading' => 'Metode dan machine learning',
        'aside' => $report ? 'Kartu metode diekspor '.$report->generated_at.' · run resmi '.$report->official_run : null])

    @if (! $report)
        <div class="card card-dashed">
            <div class="card-body">
                <h2 class="section-title">Kartu metode belum diimpor</h2>
                <p class="card-subtitle max-w-prose">
                    Daftar metode, pengaturannya, dan bukti terukurnya disusun di repo Python dari checkpoint, kode, dan hasil evaluasi,
                    lalu diimpor ke web. Jalankan dari folder repo OCR:
                </p>
                <pre class="code-block mt-3">.venv/Scripts/python scripts/export_methods.py
cd web
php artisan aksara:methods</pre>
            </div>
        </div>
    @else
        @foreach ($stale as $note)
            <div class="alert alert-warning" data-stale>
                <span class="alert-title">Kartu metode dan hasil tidak sejalan</span>
                <span>{{ $note }} Samakan keduanya dengan mengulang ekspor dan impor yang tertinggal:
                    <span class="font-mono">scripts/export_results.py</span> dengan <span class="font-mono">php artisan aksara:import</span>, atau
                    <span class="font-mono">scripts/export_methods.py</span> dengan <span class="font-mono">php artisan aksara:methods</span>.</span>
            </div>
        @endforeach

        {{-- Empat hal pokok: arsitektur, ukuran model, banyaknya butir, dan model apa saja yang dipakai. --}}
        <div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            @foreach ($tiles as $tile)
                <div class="stat-card shadow-[var(--shadow-card)]">
                    <div class="stat-head">
                        <span class="stat-icon stat-icon-blue" aria-hidden="true"><flux:icon :name="$tile['icon']" /></span>
                        <div class="min-w-0 flex-1">
                            <div class="stat-title">{{ $tile['title'] }}</div>
                            <div class="flex flex-wrap items-baseline gap-x-2">
                                <span class="stat-value">{{ $tile['value'] }}</span>
                                @if ($tile['note'] !== '')
                                    <span class="stat-note">{{ $tile['note'] }}</span>
                                @endif
                            </div>
                        </div>
                    </div>
                    <p class="stat-hint flex-1 border-t border-[color:var(--row-line)] pt-3">{{ $tile['hint'] }}</p>
                </div>
            @endforeach
        </div>

        {{-- Alur: urutan langkah dari citra sampai tingkat tutur; kotak bertanda = memakai model hasil belajar. --}}
        <section class="card">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">Alur dari citra ke arti</h2>
                    <p class="card-subtitle max-w-[85ch]">
                        Kotak biru memakai model hasil belajar (machine learning). Kotak lain memakai aturan tetap atau leksikon.
                    </p>
                </div>
                <ol class="flow" aria-label="Urutan langkah: {{ collect($flow)->pluck(0)->join(', ') }}">
                    @foreach ($flow as [$name, $detail, $learned])
                        <li @class(['flow-step', 'is-learned' => $learned]) data-step="{{ $name }}">
                            <span class="flow-name">{{ $name }}</span>
                            <span class="flow-detail">{{ $detail }}</span>
                            @if ($learned)
                                <span class="sr-only">(machine learning)</span>
                            @endif
                        </li>
                    @endforeach
                </ol>
            </div>
        </section>

        {{-- Satu kartu per tahap. Tiap butir: nama, jenis, status; penjelasan dan bukti terukur; pengaturan dan berkas kodenya. --}}
        @foreach ($groups as $group)
            <section class="card" id="metode-{{ $group['key'] }}" data-group="{{ $group['key'] }}">
                <div class="card-body">
                    <div class="mb-1">
                        <h2 class="section-title">{{ $group['title'] }}</h2>
                        <p class="card-subtitle max-w-[85ch]">{{ $group['intro'] }}</p>
                    </div>
                    <div class="method-list">
                        @foreach ($group['methods'] as $method)
                            @php
                                [$statusLabel, $statusClass] = \App\Livewire\Pages\Metode::STATUSES[$method['status']] ?? [$method['status'], 'status-gray'];
                            @endphp
                            <article class="method-row" data-method="{{ $method['key'] }}">
                                <div class="min-w-0">
                                    <h3 class="method-name">{{ $method['name'] }}</h3>
                                    <div class="mt-1.5 flex flex-wrap items-center gap-1.5">
                                        <span class="badge badge-outline">{{ \App\Livewire\Pages\Metode::KINDS[$method['kind']] ?? $method['kind'] }}</span>
                                        <span class="status {{ $statusClass }}">{{ $statusLabel }}</span>
                                    </div>
                                </div>
                                <div class="min-w-0">
                                    <p class="method-summary">{{ $method['summary'] }}</p>
                                    @if ($method['evidence'] ?? null)
                                        <p class="method-evidence">
                                            <span class="font-semibold">Terukur.</span> {{ $method['evidence'] }}
                                            @if ($method['evidence_source'] ?? null)
                                                <span class="code block">Sumber: {{ \App\Livewire\Pages\Metode::breakable($method['evidence_source']) }}</span>
                                            @endif
                                        </p>
                                    @endif
                                </div>
                                <dl class="method-settings">
                                    @foreach ($method['settings'] as [$label, $value])
                                        <div>
                                            <dt>{{ $label }}</dt>
                                            <dd class="num">{{ $value }}</dd>
                                        </div>
                                    @endforeach
                                    @if ($method['files'] ?? [])
                                        <div>
                                            <dt>Kode</dt>
                                            <dd class="code">
                                                @foreach ($method['files'] as $file)
                                                    <span class="block">{{ \App\Livewire\Pages\Metode::breakable($file) }}</span>
                                                @endforeach
                                            </dd>
                                        </div>
                                    @endif
                                </dl>
                            </article>
                        @endforeach
                    </div>
                </div>
            </section>
        @endforeach

        {{-- Rencana: pipeline berstatus "planned" di ekspor hasil. Belum ada angkanya. --}}
        @if ($planned)
            <section class="card" id="metode-rencana">
                <div class="card-body">
                    <div class="mb-3">
                        <h2 class="section-title">Direncanakan, belum dikerjakan</h2>
                        <p class="card-subtitle max-w-[85ch]">Metode yang sudah dirancang sebagai pembaca kedua atau pengoreksi, tetapi belum dijalankan dan belum punya angka.</p>
                    </div>
                    <div class="table-wrap">
                        <table class="data-table">
                            <thead><tr><th>Metode</th><th>Rancangan</th><th>Status</th></tr></thead>
                            <tbody>
                                @foreach ($planned as $plan)
                                    <tr data-plan="{{ $plan['key'] ?? $plan['label'] }}">
                                        <td class="whitespace-nowrap font-medium">{{ $plan['label'] }}</td>
                                        <td class="code min-w-64">{{ $plan['config'] }}</td>
                                        <td><span class="status status-gray">direncanakan</span></td>
                                    </tr>
                                @endforeach
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>
        @endif
    @endif
</div>
