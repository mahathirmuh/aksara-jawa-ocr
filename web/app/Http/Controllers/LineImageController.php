<?php

namespace App\Http\Controllers;

use App\Models\Line;
use Symfony\Component\HttpFoundation\BinaryFileResponse;

/**
 * Citra potongan baris dibaca langsung dari repo OCR (tidak disalin ke web), hanya untuk pengguna login:
 * citra NusaAksara berlisensi non-komersial.
 */
class LineImageController extends Controller
{
    public function __invoke(Line $line): BinaryFileResponse
    {
        $root = realpath(config('aksara.ocr_repo'));
        $path = realpath($root.DIRECTORY_SEPARATOR.$line->image_path);
        abort_unless($root && $path && str_starts_with($path, $root.DIRECTORY_SEPARATOR) && is_file($path), 404);

        return response()->file($path, ['Cache-Control' => 'private, max-age=86400']);
    }
}
