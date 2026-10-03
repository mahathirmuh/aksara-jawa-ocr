<?php

namespace App\Livewire\Pages;

use App\Services\OcrService;
use App\Services\TranslationService;
use App\Support\Transliterator;
use Livewire\Attributes\Layout;
use Livewire\Attributes\Title;
use Livewire\Attributes\Validate;
use Livewire\Component;
use Livewire\WithFileUploads;
use RuntimeException;

#[Layout('components.layouts.app')]
#[Title('Demo')]
class Demo extends Component
{
    use WithFileUploads;

    #[Validate('required|image|max:8192', as: 'citra')]
    public $photo;

    public string $pipeline = 'crnn_beam_lm';

    public ?array $result = null;

    public ?string $error = null;

    /** Status layanan model; diperiksa saat halaman dibuka dan lewat tombol "Periksa lagi". */
    public ?array $service = null;

    public function mount(OcrService $ocr): void
    {
        $this->service = $ocr->health();
    }

    public function checkService(OcrService $ocr): void
    {
        $this->service = $ocr->health();
    }

    public function updatedPhoto(): void
    {
        $this->result = null;
        $this->error = null;
        $this->validateOnly('photo');
    }

    /** Tahap 3 dari keluaran OCR; null bila layanan terjemahan tidak berjalan. */
    public ?array $translation = null;

    public function read(OcrService $ocr, TranslationService $mt): void
    {
        $this->validate();
        $this->error = null;
        $this->translation = null;
        try {
            $this->result = $ocr->predict($this->photo->getRealPath(), $this->pipeline);
        } catch (RuntimeException $e) {
            $this->result = null;
            $this->error = $e->getMessage();

            return;
        }
        $this->translation = $mt->translateOne(Transliterator::toLatin($this->result['text']));
    }

    public function render()
    {
        return view('livewire.pages.demo', ['pipelines' => OcrService::PIPELINES]);
    }
}
