<a
    {{ $attributes->merge(['class' => 'text-sm text-[var(--primary)] underline-offset-2 hover:underline']) }}
    wire:navigate
>
    {{ $slot }}
</a>
