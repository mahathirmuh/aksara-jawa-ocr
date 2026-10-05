<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}" class="dark">
    <head>
        @include('partials.head')
    </head>
    <body class="min-h-screen">
        @php
            // Menu sidebar per bagian: [nama rute, ikon Tabler, label].
            $menu = [
                'Hasil' => [
                    ['dashboard', 'ti-chart-pie', 'Ringkasan'],
                    ['perbandingan', 'ti-chart-bar', 'Perbandingan'],
                    ['ablasi', 'ti-flask', 'Ablasi'],
                ],
                'Analisis' => [
                    ['penjelajah', 'ti-list-search', 'Penjelajah baris'],
                    ['kesalahan', 'ti-arrows-exchange', 'Kesalahan aksara'],
                ],
                'Alat' => [
                    ['demo', 'ti-scan', 'Demo'],
                ],
            ];
            $pageTitle = $title ?? (request()->routeIs('settings.*') ? __('Settings') : 'Aksara OCR Lab');
            // Lima aksara pembuka deret hanacaraka (ha na ca ra ka), hanya hiasan dasar sidebar.
            $carakan = implode('', array_map('mb_chr', [0xA9B2, 0xA9A4, 0xA995, 0xA9AB, 0xA98F]));
        @endphp

        {{-- Sidebar: bisa diciutkan jadi rel ikon di desktop (tombol di bilah atas), laci geser di ponsel. --}}
        <flux:sidebar sticky collapsible class="app-sidebar w-60">
            <flux:sidebar.header class="app-brand">
                <a href="{{ route('dashboard') }}" class="app-brand-link" wire:navigate>
                    <x-app-logo />
                </a>
                <flux:sidebar.toggle class="lg:hidden" icon="ti-x" />
            </flux:sidebar.header>

            <flux:sidebar.nav>
                @foreach ($menu as $section => $items)
                    <div class="menu-section-header"><span class="subheader">{{ $section }}</span></div>
                    @foreach ($items as [$route, $icon, $label])
                        <flux:sidebar.item :icon="$icon" :href="route($route)" :current="request()->routeIs($route)" wire:navigate>{{ $label }}</flux:sidebar.item>
                    @endforeach
                @endforeach
            </flux:sidebar.nav>

            <flux:sidebar.spacer />

            <div class="sidebar-art" aria-hidden="true">{{ $carakan }}</div>
        </flux:sidebar>

        {{-- Bilah atas: tombol sidebar, judul halaman, tema, pengguna. --}}
        <flux:header sticky class="app-navbar">
            <button type="button" class="navbar-toggle" x-data x-on:click="$dispatch('flux-sidebar-toggle')"
                    aria-label="Buka atau tutup sidebar" data-flux-sidebar-toggle>
                <flux:icon.ti-layout-sidebar />
            </button>
            <div class="navbar-title">{{ $pageTitle }}</div>

            <flux:spacer />

            <span class="navbar-context max-md:hidden">citra → aksara → Latin → arti</span>

            <button type="button" class="navbar-icon" x-data x-on:click="$flux.dark = ! $flux.dark"
                    aria-label="Ganti tema terang atau gelap">
                <flux:icon.ti-moon class="dark:hidden" />
                <flux:icon.ti-sun class="hidden dark:block" />
            </button>

            <flux:dropdown position="bottom" align="end">
                <button type="button" class="navbar-user" aria-label="Menu pengguna">
                    <span class="navbar-avatar">{{ auth()->user()->initials() }}</span>
                    <span class="max-xl:hidden">
                        <span class="navbar-user-name block">{{ auth()->user()->name }}</span>
                        <span class="navbar-user-mail block">{{ auth()->user()->email }}</span>
                    </span>
                </button>

                <flux:menu class="w-[220px]">
                    <flux:menu.radio.group>
                        <div class="p-0 text-sm font-normal">
                            <div class="flex items-center gap-2 px-1 py-1.5 text-left text-sm">
                                <span class="navbar-avatar">{{ auth()->user()->initials() }}</span>

                                <div class="grid flex-1 text-left text-sm leading-tight">
                                    <span class="truncate font-semibold">{{ auth()->user()->name }}</span>
                                    <span class="truncate text-xs">{{ auth()->user()->email }}</span>
                                </div>
                            </div>
                        </div>
                    </flux:menu.radio.group>

                    <flux:menu.separator />

                    <flux:menu.radio.group>
                        <flux:menu.item href="/settings/profile" icon="ti-settings" wire:navigate>Settings</flux:menu.item>
                    </flux:menu.radio.group>

                    <flux:menu.separator />

                    <form method="POST" action="{{ route('logout') }}" class="w-full">
                        @csrf
                        <flux:menu.item as="button" type="submit" icon="ti-logout" class="w-full">
                            {{ __('Log Out') }}
                        </flux:menu.item>
                    </form>
                </flux:menu>
            </flux:dropdown>
        </flux:header>

        {{ $slot }}

        <footer class="app-footer">
            <div class="app-footer-inner">
                <span>Copyright &copy; {{ date('Y') }}, Aksara OCR Lab</span>
                <span aria-hidden="true">·</span>
                <span>riset non-komersial</span>
                <span class="badge badge-dark">lokal</span>
            </div>
        </footer>

        @fluxScripts
    </body>
</html>
