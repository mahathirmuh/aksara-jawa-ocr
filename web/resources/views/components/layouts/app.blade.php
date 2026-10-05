<x-layouts.app.sidebar :title="$title ?? null">
    <flux:main class="app-main">
        <div class="container-xl">
            {{ $slot }}
        </div>
    </flux:main>
</x-layouts.app.sidebar>
