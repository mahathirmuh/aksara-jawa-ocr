<?php

namespace App\Livewire\Pages;

use App\Models\Confusion;
use App\Models\Pipeline;
use Livewire\Attributes\Layout;
use Livewire\Attributes\Title;
use Livewire\Component;

#[Layout('components.layouts.app')]
#[Title('Kesalahan aksara')]
class Kesalahan extends Component
{
    public function render()
    {
        $limits = ['sub' => 14, 'del' => 10, 'ins' => 10];

        return view('livewire.pages.kesalahan', [
            'lists' => collect($limits)->map(fn ($n, $kind) => Confusion::where('kind', $kind)->orderByDesc('count')->limit($n)->get()),
            'totals' => Confusion::selectRaw('kind, sum(count) as total')->groupBy('kind')->pluck('total', 'kind'),
            // Ekspor menghitung kesalahan aksara dari pipeline resmi (manifest: confusion.pipeline = official).
            'pipeline' => Pipeline::official(),
        ]);
    }
}
