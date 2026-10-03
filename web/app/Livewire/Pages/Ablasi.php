<?php

namespace App\Livewire\Pages;

use App\Models\AblationRun;
use Livewire\Attributes\Layout;
use Livewire\Attributes\Title;
use Livewire\Component;

#[Layout('components.layouts.app')]
#[Title('Ablasi')]
class Ablasi extends Component
{
    public function render()
    {
        $runs = AblationRun::orderBy('k')->get();

        return view('livewire.pages.ablasi', [
            'runs' => $runs,
            'chartData' => [
                'labels' => $runs->pluck('k')->map(fn ($k) => (string) $k)->all(),
                'added' => $runs->pluck('added')->all(),
                'g3' => $runs->pluck('g3')->all(),
                'heavy' => $runs->pluck('heavy')->all(),
                'clean' => $runs->pluck('clean')->all(),
            ],
        ]);
    }
}
