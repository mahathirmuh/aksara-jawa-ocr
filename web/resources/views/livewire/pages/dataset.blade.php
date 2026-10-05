<div class="flex flex-col gap-4">
    @include('partials.page-head', ['eyebrow' => 'Data', 'heading' => 'Dataset',
        'aside' => $report ? 'Kartu data diekspor '.$report->generated_at.' · pemakaian dihitung untuk run '.$report->official_run : null])

    @if (! $report)
        <div class="card card-dashed">
            <div class="card-body">
                <h2 class="section-title">Kartu data belum diimpor</h2>
                <p class="card-subtitle max-w-prose">
                    Jumlah baris, pembagian, dan pemakaian data dihitung di repo Python, lalu diimpor ke web. Jalankan dari folder repo OCR:
                </p>
                <pre class="code-block mt-3">.venv/Scripts/python scripts/export_datasets.py
cd web
php artisan aksara:datasets</pre>
            </div>
        </div>
    @else
        @if ($stale)
            <div class="alert alert-warning">
                <span class="alert-title">Kartu data tertinggal</span>
                <span>Pemakaian di halaman ini dihitung untuk run {{ $stale['run'] }}, sedangkan angka resmi web sekarang milik {{ $stale['official'] }}.
                    Jalankan ulang <span class="font-mono">python scripts/export_datasets.py</span> lalu <span class="font-mono">php artisan aksara:datasets</span>.</span>
            </div>
        @endif
        @foreach ($card['warnings'] as $warning)
            <div class="alert alert-warning"><span>{{ $warning }}</span></div>
        @endforeach

        {{-- Empat angka pokok: besar korpus, pembagian, data nyata, font. --}}
        <div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            @foreach ($tiles as $tile)
                <div class="stat-card shadow-[var(--shadow-card)]">
                    <div class="stat-head">
                        <span class="stat-icon stat-icon-blue" aria-hidden="true"><flux:icon :name="$tile['icon']" /></span>
                        <div class="min-w-0 flex-1">
                            <div class="stat-title">{{ $tile['title'] }}</div>
                            <div class="flex flex-wrap items-baseline gap-x-2">
                                <span class="stat-value">{{ $tile['value'] }}</span>
                                <span class="stat-note">{{ $tile['note'] }}</span>
                            </div>
                        </div>
                    </div>
                    <p class="stat-hint flex-1 border-t border-[color:var(--row-line)] pt-3">{{ $tile['hint'] }}</p>
                </div>
            @endforeach
        </div>

        {{-- Pembagian: batang berskala jumlah baris, lalu arti tiap bagian dan berapa yang dipakai run resmi. --}}
        <section class="card">
            <div class="card-body">
                <div class="mb-4">
                    <h2 class="section-title">Pembagian latih, validasi, uji</h2>
                    <p class="card-subtitle max-w-[85ch]">
                        {{ nfmt($card['corpus']['lines']) }} baris teks dibagi menurut artikel asalnya, jadi satu artikel tidak pernah muncul di dua bagian
                        dan teks uji tidak pernah dilihat saat latih.
                    </p>
                </div>
                <div class="split-bar" role="img"
                     aria-label="{{ collect($parts)->map(fn ($p) => $p['label'].' '.pct($p['share']))->join(', ') }}">
                    @foreach ($parts as $part)
                        <span class="split-{{ $part['key'] }}" style="width: {{ round($part['share'] * 100, 3) }}%"></span>
                    @endforeach
                </div>
                {{-- Di layar lebar tiap bagian memakai dua baris kisi induk (subgrid), supaya ketiga kotak pemakaian sejajar
                     walau panjang penjelasannya berbeda; di ponsel tiap bagian menumpuk sendiri. --}}
                <div class="mt-4 grid gap-x-8 gap-y-6 md:grid-cols-3 md:gap-y-3">
                    @foreach ($parts as $part)
                        <div class="flex min-w-0 flex-col gap-3 md:row-span-2 md:grid md:grid-rows-subgrid">
                            <div>
                                <div class="flex flex-wrap items-baseline gap-x-2">
                                    <span class="split-dot split-{{ $part['key'] }} self-center" aria-hidden="true"></span>
                                    <h3 class="text-sm font-semibold">{{ $part['label'] }}</h3>
                                    <span class="num text-base font-bold">{{ pct($part['share']) }}</span>
                                    <span class="num text-muted text-xs">{{ nfmt($part['lines']) }} baris</span>
                                </div>
                                <p class="text-muted mt-1 text-sm">{{ $part['meaning'] }}</p>
                            </div>
                            <div class="card-inset px-3 py-2.5">
                                <div class="subheader">Dipakai run resmi</div>
                                <div class="mt-1 text-sm">
                                    <span class="num font-semibold">{{ nfmt($part['used']) }}</span>
                                    <span class="text-muted">{{ $part['caption'] }}</span>
                                </div>
                                <div class="meter mt-2" role="img" aria-label="{{ pct($part['used_share']) }} bagian {{ mb_strtolower($part['label']) }}">
                                    <span class="split-{{ $part['key'] }}" style="width: {{ round($part['used_share'] * 100, 3) }}%"></span>
                                </div>
                                <div class="text-muted mt-1.5 text-xs">
                                    <span class="num">{{ pct($part['used_share']) }}</span> bagian {{ mb_strtolower($part['label']) }}. {{ $part['detail'] }}
                                </div>
                            </div>
                        </div>
                    @endforeach
                </div>
            </div>
            <div class="card-footer">
                <p class="max-w-[95ch]">
                    Kenapa bukan 80/20 atau 70/15/15? Rasio itu untuk data berjumlah ribuan. Di sini bagian uji saja sudah
                    {{ nfmt($parts[2]['lines']) }} baris, padahal gerbang hanya memakai {{ nfmt($parts[2]['used']) }}, dan bagian latih tidak pernah habis terpakai.
                    Membagi ulang juga memindahkan teks yang sudah dilatih ke bagian uji, sehingga angka lama tidak bisa lagi dibandingkan dengan yang baru.
                </p>
            </div>
        </section>

        {{-- Asal korpus: langkah pembangunan dan sebaran panjang baris. --}}
        <section class="card">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">Dari artikel ke baris korpus</h2>
                    <p class="card-subtitle max-w-[85ch]">
                        Teks Latin Wikipedia dialihaksarakan ke aksara Jawa, disaring, lalu dibagi. Citra baris tidak disimpan: dirender dari teks ini setiap kali dibutuhkan.
                    </p>
                </div>
                <div class="grid gap-x-10 gap-y-6 lg:grid-cols-2">
                    <dl>
                        @foreach ($funnel as [$label, $value, $hint])
                            <div class="flex items-baseline justify-between gap-4 border-t border-[color:var(--row-line)] py-2 first:border-t-0 first:pt-0">
                                <dt class="min-w-0">
                                    <span class="font-medium">{{ $label }}</span>
                                    @if ($hint)
                                        <span class="text-muted block text-xs">{{ $hint }}</span>
                                    @endif
                                </dt>
                                <dd class="num shrink-0 font-semibold">{{ $value }}</dd>
                            </div>
                        @endforeach
                        <div class="flex items-baseline justify-between gap-4 border-t-2 border-[color:var(--border)] pt-2">
                            <dt class="font-semibold">Baris korpus</dt>
                            <dd class="num shrink-0 text-base font-bold">{{ nfmt($card['corpus']['lines']) }}</dd>
                        </div>
                    </dl>
                    <div>
                        <div class="subheader mb-2">Panjang baris (jumlah codepoint)</div>
                        <div class="flex flex-col gap-2" role="img"
                             aria-label="Sebaran panjang baris: {{ collect($histogram)->map(fn ($b) => $b['range'].' codepoint '.pct($b['share'], 0))->join(', ') }}">
                            @foreach ($histogram as $bin)
                                <div class="grid grid-cols-[3rem_minmax(0,1fr)_7.5rem] items-center gap-3 text-xs">
                                    <span class="num text-muted">{{ $bin['range'] }}</span>
                                    <span class="meter meter-lg"><span class="split-train" style="width: {{ $bin['width'] }}%"></span></span>
                                    <span class="num text-right">{{ nfmt($bin['lines']) }} <span class="text-muted">· {{ pct($bin['share'], 0) }}</span></span>
                                </div>
                            @endforeach
                        </div>
                    </div>
                </div>
            </div>
        </section>

        {{-- Rantai checkpoint: tiap run melanjutkan bobot run sebelumnya; baris bertanda = run resmi. --}}
        <section class="card">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">Data latih di rantai checkpoint run resmi</h2>
                    <p class="card-subtitle max-w-[85ch]">
                        Checkpoint {{ $card['official']['run'] }} melanjutkan bobot {{ count($runs) - 1 }} run sebelumnya. Semua run mengambil barisnya dari kumpulan acak yang sama,
                        jadi sepanjang rantai model hanya pernah melihat sebagian kecil bagian latih.
                    </p>
                </div>
                <div class="table-wrap">
                    <table class="data-table">
                        <thead><tr><th>Run</th><th class="r">Langkah</th><th class="r">Sampel</th><th class="r">Kumpulan baris</th><th class="r">Baris berbeda</th><th class="r">Font</th><th>Resep data</th></tr></thead>
                        <tbody>
                            @foreach ($runs as $run)
                                <tr @class(['is-highlight' => $run['official']])>
                                    <td class="whitespace-nowrap">
                                        <span class="font-medium">{{ $run['run'] }}</span>
                                        @if ($run['official'])
                                            <span class="status status-blue ml-1">resmi</span>
                                        @endif
                                    </td>
                                    <td class="r num">{{ nfmt($run['steps']) }}</td>
                                    <td class="r num">{{ nfmt($run['samples']) }}</td>
                                    <td class="r num">{{ nfmt($run['pool']) }}</td>
                                    <td class="r num">{{ nfmt($run['distinct_lines']) }}</td>
                                    <td class="r num">{{ $run['fonts'] }}</td>
                                    <td class="min-w-56">{{ $run['recipe'] }}</td>
                                </tr>
                            @endforeach
                            <tr class="font-semibold">
                                <td>Seluruh rantai</td>
                                <td class="r num">{{ nfmt($card['lineage']['steps']) }}</td>
                                <td class="r num">{{ nfmt($card['lineage']['samples']) }}</td>
                                <td class="r num">{{ nfmt($card['lineage']['pool']) }}</td>
                                <td class="r num">{{ nfmt($card['lineage']['distinct_lines']) }}</td>
                                <td></td>
                                <td class="text-muted font-normal">{{ pct($trainLines ? $card['lineage']['distinct_lines'] / $trainLines : 0) }} bagian latih pernah dilihat</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="card-footer">
                <p class="max-w-[95ch]">
                    Sampel = langkah × ukuran batch. Baris berbeda lebih sedikit daripada sampel bila sebuah run dihentikan lalu dilanjutkan:
                    tiap proses mengulang urutan batch dari awal, jadi baris awal dilihat lagi dengan render yang berbeda.
                    @unless ($card['lineage']['complete'])
                        Checkpoint leluhur sudah tidak ada, jadi rantai di atas tidak lengkap dan jumlahnya batas bawah.
                    @endunless
                </p>
            </div>
        </section>

        {{-- Daftar dataset: apa isinya, untuk apa dipakai, dari mana, boleh disebar atau tidak. --}}
        <section class="card">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">Daftar dataset</h2>
                    <p class="card-subtitle max-w-[85ch]">Data yang dipakai melatih dan menguji pembaca aksara. Tidak ada yang ikut repo publik selain font inti.</p>
                </div>
                <div class="table-wrap">
                    <table class="data-table">
                        <thead><tr><th>Dataset</th><th class="r">Jumlah</th><th>Peran</th><th>Sumber &amp; lisensi</th><th>Disebar</th></tr></thead>
                        <tbody>
                            @foreach ($datasets as $dataset)
                                <tr class="align-top">
                                    <td class="min-w-64 max-w-md">
                                        <div class="font-medium">{{ $dataset['name'] }}</div>
                                        <div class="text-muted mt-0.5">{{ $dataset['content'] }}</div>
                                    </td>
                                    <td class="r whitespace-nowrap">
                                        <div class="num font-semibold">{{ nfmt($dataset['count']) }}</div>
                                        <div class="text-muted">{{ $dataset['unit'] }}</div>
                                        @isset($dataset['articles'])
                                            <div class="num text-muted">{{ nfmt($dataset['articles']) }} artikel</div>
                                        @endisset
                                        @isset($dataset['pages'])
                                            <div class="num text-muted">{{ nfmt($dataset['pages']) }} halaman</div>
                                        @endisset
                                        @isset($dataset['verified'])
                                            <div class="num text-muted">{{ nfmt($dataset['verified']) }} terverifikasi</div>
                                        @endisset
                                    </td>
                                    <td class="min-w-40">
                                        <div class="flex flex-col items-start gap-1">
                                            @foreach ($dataset['roles'] as $role => $lines)
                                                <span class="role-chip"><span class="split-dot split-{{ $role }}" aria-hidden="true"></span>{{ \App\Livewire\Pages\Dataset::ROLES[$role] ?? $role }} <span class="num text-muted">{{ nfmt($lines) }}</span></span>
                                            @endforeach
                                            @if (($dataset['status'] ?? '') === 'pending')
                                                <span class="status status-yellow">menunggu verifikasi</span>
                                                <span class="text-muted">calon {{ collect($dataset['planned'] ?? [])->map(fn ($r) => \App\Livewire\Pages\Dataset::ROLES[$r] ?? $r)->join(' dan ') }}</span>
                                            @endif
                                            @if ($dataset['gates'])
                                                <span class="text-muted">gerbang {{ implode(', ', $dataset['gates']) }}</span>
                                            @endif
                                        </div>
                                    </td>
                                    <td class="min-w-56 max-w-sm">
                                        <div>
                                            @isset($dataset['source_url'])
                                                <a href="{{ $dataset['source_url'] }}" target="_blank" rel="noopener noreferrer" class="text-[color:var(--primary)] hover:underline">{{ $dataset['source'] }}</a>
                                            @else
                                                {{ $dataset['source'] }}
                                            @endisset
                                        </div>
                                        <div class="text-muted mt-0.5">Lisensi: {{ $dataset['license'] }}</div>
                                    </td>
                                    <td class="min-w-52 max-w-xs">
                                        <span class="status {{ $dataset['shareable'] ? 'status-green' : 'status-red' }}">{{ $dataset['shareable'] ? 'Boleh' : 'Tidak boleh' }}</span>
                                        <div class="text-muted mt-1">{{ $dataset['share_note'] }}</div>
                                    </td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        {{-- Font: bentuk huruf yang dilihat model saat latih, validasi, dan uji. --}}
        <section class="card">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">Font per peran</h2>
                    <p class="card-subtitle max-w-[85ch]">
                        Teks korpus baru menjadi data latih setelah dirender dengan font. Font latih menentukan bentuk huruf yang dikenal model;
                        validasi memakai font inti saja, dan gerbang sintetis memakai font uji yang tidak ikut merender data latih.
                    </p>
                </div>
                <div class="table-wrap">
                    <table class="data-table">
                        <thead><tr><th>Font</th><th>Peran</th><th>Lisensi</th><th>Catatan</th></tr></thead>
                        <tbody>
                            @foreach ($usedFonts as $font)
                                <tr class="align-top">
                                    <td class="whitespace-nowrap">
                                        <span class="font-medium">{{ pathinfo($font['file'], PATHINFO_FILENAME) }}</span>
                                        @if ($font['in_repo'])
                                            <span class="badge badge-outline ml-1">ikut repo</span>
                                        @endif
                                    </td>
                                    <td>
                                        <div class="flex flex-wrap gap-1">
                                            @foreach ($font['roles'] as $role)
                                                <span class="role-chip"><span class="split-dot split-{{ $role }}" aria-hidden="true"></span>{{ \App\Livewire\Pages\Dataset::ROLES[$role] ?? $role }}</span>
                                            @endforeach
                                        </div>
                                    </td>
                                    <td class="min-w-32">{{ $font['license'] }}</td>
                                    <td class="min-w-72 max-w-xl">
                                        @forelse ($font['remarks'] as $remark)
                                            <p @class(['mt-1' => ! $loop->first])>{{ $remark }}</p>
                                        @empty
                                            <span class="text-muted">–</span>
                                        @endforelse
                                    </td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="card-footer">
                <p class="max-w-[95ch]">
                    Font tambahan hanya dipakai merender data latih riset di mesin ini dan tidak disebarkan; pemeriksaan visualnya baru oleh Claude, belum oleh pembaca aksara.
                    @foreach ($unusedFonts as $group => $items)
                        Tidak dipakai karena {{ $group }}:
                        @foreach ($items as $font)
                            <span title="{{ $font['review'] ?? '' }}">{{ pathinfo($font['file'], PATHINFO_FILENAME) }}</span>{{ $loop->last ? '.' : ',' }}
                        @endforeach
                    @endforeach
                </p>
            </div>
        </section>

        {{-- Batasan: hanya yang syaratnya terpenuhi di kartu data. --}}
        @if ($limits)
            <section class="card">
                <div class="card-body">
                    <div class="mb-3">
                        <h2 class="section-title">Batasan data</h2>
                        <p class="card-subtitle max-w-[85ch]">Kelemahan pembagian dan isi data yang memengaruhi cara membaca angka gerbang.</p>
                    </div>
                    <ol class="flex flex-col gap-3">
                        @foreach ($limits as [$title, $body])
                            <li class="alert alert-warning block">
                                <h3 class="alert-title">{{ $title }}</h3>
                                <p class="mt-1 max-w-[95ch]">{{ $body }}</p>
                            </li>
                        @endforeach
                    </ol>
                </div>
            </section>
        @endif

        {{-- Data pendukung: bukan data latih pembaca aksara, tetapi dipakai tahap lain alur. --}}
        <section class="card">
            <div class="card-body">
                <div class="mb-3">
                    <h2 class="section-title">Data pendukung</h2>
                    <p class="card-subtitle max-w-[85ch]">Tidak dipakai melatih pembaca aksara; dipakai tahap lain alur dan alat di web.</p>
                </div>
                <div class="table-wrap">
                    <table class="data-table">
                        <thead><tr><th>Data</th><th>Dipakai untuk</th><th class="r">Jumlah</th><th>Asal</th></tr></thead>
                        <tbody>
                            @foreach ($support as [$name, $purpose, $amount, $origin])
                                <tr>
                                    <td class="whitespace-nowrap font-medium">{{ $name }}</td>
                                    <td class="min-w-44">{{ $purpose }}</td>
                                    <td class="r num whitespace-nowrap">{{ $amount }}</td>
                                    <td class="text-muted min-w-56">{{ $origin }}</td>
                                </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
    @endif
</div>
