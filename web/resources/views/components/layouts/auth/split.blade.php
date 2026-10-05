<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}" class="dark">
    <head>
        @include('partials.head')
    </head>
    <body class="auth-page antialiased">
        @php
            // Deret hanacaraka (20 aksara dasar) sebagai tekstur latar; hiasan saja, tersembunyi dari pembaca layar.
            $carakan = implode(' ', array_map(
                fn (array $baris) => implode('', array_map('mb_chr', $baris)),
                [
                    [0xA9B2, 0xA9A4, 0xA995, 0xA9AB, 0xA98F],
                    [0xA9A2, 0xA9A0, 0xA9B1, 0xA9AE, 0xA9AD],
                    [0xA9A5, 0xA99D, 0xA997, 0xA9AA, 0xA99A],
                    [0xA9A9, 0xA992, 0xA9A7, 0xA99B, 0xA994],
                ],
            ));
        @endphp

        <div class="auth-backdrop" aria-hidden="true">{{ str_repeat($carakan.' ', 14) }}</div>

        {{-- Kartu lebar terbelah: panel gambar di kiri (layar lebar saja), formulir di kanan. --}}
        <main class="auth-card">
            <div class="auth-visual" aria-hidden="true">
                <div class="auth-visual-text">{{ str_repeat($carakan.' ', 3) }}</div>
                <div class="auth-visual-body">
                    <div class="auth-flow">
                        <span>citra baris</span><i>→</i><span>aksara Jawa</span><i>→</i><span>Latin</span><i>→</i><span>arti</span><i>→</i><span>tingkat tutur</span>
                    </div>
                    <p class="auth-visual-title">Membaca satu baris aksara Jawa cetak.</p>
                    <p class="auth-visual-sub">CRNN + CTC yang dilatih dari baris sintetis, lalu diuji pada baris cetak nyata.</p>
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
