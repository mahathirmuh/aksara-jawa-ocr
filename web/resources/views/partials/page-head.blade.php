{{-- Kepala halaman: eyebrow, judul, keterangan kanan. --}}
<div class="flex flex-wrap items-end justify-between gap-3">
    <div>
        @isset($eyebrow)<div class="eyebrow">{{ $eyebrow }}</div>@endisset
        <flux:heading size="xl" level="1" class="!mt-1">{{ $heading }}</flux:heading>
    </div>
    @isset($aside)<p class="text-sm text-zinc-500 dark:text-zinc-400">{{ $aside }}</p>@endisset
</div>
