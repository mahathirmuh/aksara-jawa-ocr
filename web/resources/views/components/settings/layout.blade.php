{{-- Kartu pengaturan: tab bagian menempel di tepi atas kartu (tab garis bawah ala rujukan), lalu judul bagian
     dan isinya di badan kartu. --}}
@php
    $tabs = [
        'settings.profile' => 'Profile',
        'settings.password' => 'Password',
        'settings.appearance' => 'Appearance',
    ];
    // Saat Livewire memperbarui komponen (mis. sesudah "Save"), permintaannya menuju rute pembaruan Livewire,
    // jadi request()->routeIs() saja akan menghapus tanda tab aktif. Jalur halaman asal dibaca dari snapshot,
    // cara yang sama dipakai Flux untuk data-current.
    $path = trim(app('livewire')->originalPath(), '/');
    $isCurrent = fn (string $route) => request()->routeIs($route) || trim(route($route, absolute: false), '/') === $path;
@endphp

<div class="card">
    <nav class="nav-tabs px-6 pt-2 max-sm:px-4" aria-label="Settings">
        @foreach ($tabs as $route => $label)
            <a href="{{ route($route) }}" wire:navigate @if ($isCurrent($route)) aria-current="page" @endif
               class="dark:hover:text-[var(--brand-strong)] dark:aria-[current=page]:text-[var(--brand-strong)]">{{ $label }}</a>
        @endforeach
    </nav>

    <div class="card-body">
        <div class="mb-4">
            <h2 class="section-title">{{ $heading ?? '' }}</h2>
            <p class="card-subtitle">{{ $subheading ?? '' }}</p>
        </div>

        {{ $slot }}
    </div>
</div>
