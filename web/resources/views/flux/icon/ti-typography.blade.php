{{-- Tabler Icons "typography" (MIT, https://tabler.io/icons). Dibuat dari SVG resminya; jangan disunting tangan. --}}

@props([
    'variant' => 'outline',
])

@php
    $classes = Flux::classes('shrink-0')->add(
        match ($variant) {
            'outline', 'solid' => '[:where(&)]:size-6',
            'mini' => '[:where(&)]:size-5',
            'micro' => '[:where(&)]:size-4',
        },
    );
@endphp

<svg
    {{ $attributes->class($classes) }}
    data-flux-icon
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
    data-slot="icon"
>
    <path d="M4 20l3 0" />
    <path d="M14 20l7 0" />
    <path d="M6.9 15l6.9 0" />
    <path d="M10.2 6.3l5.8 13.7" />
    <path d="M5 20l6 -16l2 0l7 16" />
</svg>
