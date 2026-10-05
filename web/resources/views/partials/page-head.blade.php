{{-- Kepala halaman: pratajuk, judul, keterangan kanan. --}}
<div class="flex flex-wrap items-end justify-between gap-x-4 gap-y-1">
    <div class="min-w-0">
        @isset($eyebrow)<div class="page-pretitle">{{ $eyebrow }}</div>@endisset
        <h1 class="page-title">{{ $heading }}</h1>
    </div>
    @isset($aside)<p class="page-aside">{{ $aside }}</p>@endisset
</div>
