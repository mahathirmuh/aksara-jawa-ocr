{{-- Satu entri kamus kata. $entry: App\Models\DictionaryEntry; $limit: berapa arti yang ditampilkan (bawaan 4);
     $showLanguage: tandai arti yang bukan bahasa Indonesia (dimatikan bila seluruh kamus satu bahasa). --}}
@php
    $glosses = $entry->glosses ?? [];
    $visible = array_slice($glosses, 0, $limit ?? 4);
@endphp
<div class="dict-entry" wire:key="kata-{{ $entry->id }}">
    <div class="dict-head">
        <span class="dict-word">{{ $entry->word }}</span>
        @if ($entry->aksara)
            <span class="jv dict-aksara" lang="jv-Java">{{ $entry->aksara }}</span>
        @endif
        @if ($entry->pos)
            <span class="badge badge-outline">{{ $entry->pos }}</span>
        @endif
        @if ($entry->register)
            <span class="status status-blue">{{ $entry->register }}</span>
        @endif
        @if (($showLanguage ?? true) && $entry->gloss_lang !== 'id')
            <span class="status status-gray">arti berbahasa {{ \App\Models\DictionaryEntry::GLOSS_LANGUAGES[$entry->gloss_lang] ?? $entry->gloss_lang }}</span>
        @endif
    </div>
    <ol class="dict-glosses {{ count($visible) === 1 ? 'is-single' : '' }}">
        @foreach ($visible as $gloss)
            <li>{{ $gloss }}</li>
        @endforeach
    </ol>
    @foreach (array_slice($entry->examples ?? [], 0, 2) as $example)
        <p class="dict-example">{{ $example }}</p>
    @endforeach
    @if ($entry->note)
        <p class="dict-note">{{ $entry->note }}</p>
    @endif
    <div class="dict-meta">
        @if (count($glosses) > count($visible))
            <span>{{ count($glosses) - count($visible) }} arti lain di sumbernya</span>
        @endif
        @if ($entry->url)
            <a href="{{ $entry->url }}" target="_blank" rel="noopener noreferrer" class="text-[var(--primary)] hover:underline">{{ $sourceNames[$entry->source] ?? $entry->source }}</a>
        @else
            <span>{{ $sourceNames[$entry->source] ?? $entry->source }}</span>
        @endif
    </div>
</div>
