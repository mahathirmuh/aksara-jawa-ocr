<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}" class="dark">
    <head>
        @include('partials.head')
    </head>
    @php
        // Foto panel kiri, dari Wikimedia Commons (sumber dan lisensi: public/img/login/SUMBER.md; dibuat
        // tools/login_images.py). pos = titik fokus saat foto dipotong mengikuti ukuran panel.
        $slides = [
            ['file' => 'naskah-sonobudoyo.jpg', 'pos' => '50% 60%',
                'alt' => 'Naskah beraksara Jawa yang terbuka di dalam etalase museum',
                'title' => 'Naskah beraksara Jawa, Museum Sonobudoyo', 'credit' => 'Foto: Candramawa99 · CC0'],
            ['file' => 'serat-damar-wulan.jpg', 'pos' => '50% 50%',
                'alt' => 'Halaman bergambar Serat Damar Wulan dengan dua baris aksara Jawa di bawahnya',
                'title' => 'Serat Damar Wulan, akhir abad ke-18', 'credit' => 'British Library, MSS Jav 89 · domain publik'],
            ['file' => 'kraton-yogyakarta.jpg', 'pos' => '50% 45%',
                'alt' => 'Gerbang Donopratono Kraton Yogyakarta di bawah langit biru',
                'title' => 'Gerbang Donopratono, Kraton Yogyakarta', 'credit' => 'Foto: Chainwit. · CC BY 4.0'],
        ];
    @endphp
    <body class="auth-page antialiased" style="--auth-backdrop: url('{{ asset('img/login/latar.jpg') }}')">
        {{-- Kartu lebar terbelah: korsel foto di kiri (di ponsel jadi pita di atas), formulir di kanan. --}}
        <main class="auth-card">
            {{-- Foto berganti tiap 6 detik; berhenti saat disorot atau difokus, dan tidak berputar bila pengguna
                 meminta gerak dikurangi. Tanpa JavaScript foto pertama tetap tampil. --}}
            <div class="auth-visual" role="group" aria-roledescription="carousel" aria-label="Foto aksara dan budaya Jawa"
                 x-data="{
                     i: 0, n: {{ count($slides) }}, timer: null,
                     play() {
                         if (this.timer || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
                         this.timer = setInterval(() => { this.i = (this.i + 1) % this.n }, 6000);
                     },
                     stop() { clearInterval(this.timer); this.timer = null },
                     destroy() { this.stop() },
                 }"
                 x-init="play()" x-on:mouseenter="stop()" x-on:mouseleave="play()" x-on:focusin="stop()" x-on:focusout="play()">
                @foreach ($slides as $k => $slide)
                    <figure @class(['auth-slide', 'is-active' => $k === 0]) x-bind:class="{ 'is-active': i === {{ $k }} }"
                            x-bind:aria-hidden="i === {{ $k }} ? 'false' : 'true'">
                        <img src="{{ asset('img/login/'.$slide['file']) }}" alt="{{ $slide['alt'] }}" width="1200" height="1320"
                             style="object-position: {{ $slide['pos'] }}">
                        <figcaption>
                            <span class="auth-slide-title">{{ $slide['title'] }}</span>
                            <span class="auth-slide-credit">{{ $slide['credit'] }}</span>
                        </figcaption>
                    </figure>
                @endforeach
                <div class="auth-thumbs">
                    @foreach ($slides as $k => $slide)
                        <button type="button" x-on:click="i = {{ $k }}" x-bind:aria-current="i === {{ $k }} ? 'true' : 'false'"
                                aria-current="{{ $k === 0 ? 'true' : 'false' }}" aria-label="Tampilkan foto {{ $k + 1 }}: {{ $slide['title'] }}"
                                style="background-image: url('{{ asset('img/login/'.$slide['file']) }}')"></button>
                    @endforeach
                </div>
            </div>

            <div class="auth-panel">
                <div class="auth-brand">
                    <a href="{{ route('home') }}" class="app-brand-link" wire:navigate>
                        <x-app-logo />
                    </a>
                    <span class="badge badge-outline">lokal</span>
                </div>

                <div class="auth-body">
                    <div class="auth-title">Aksara OCR Lab</div>
                    <hr class="auth-rule">
                    <div class="flex flex-col gap-6">
                        {{ $slot }}
                    </div>
                </div>

                <p class="auth-foot">&copy; {{ date('Y') }} Aksara OCR Lab</p>
            </div>
        </main>

        @fluxScripts
    </body>
</html>
