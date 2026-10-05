<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />

<title>{{ isset($title) ? $title.' · Aksara OCR Lab' : 'Aksara OCR Lab' }}</title>

{{-- Ikon dibuat web/tools/favicon.py dari aksara ha Noto Sans Javanese. --}}
<link rel="icon" href="/favicon.ico" sizes="any" />
<link rel="icon" href="/favicon.svg" type="image/svg+xml" />
<link rel="apple-touch-icon" href="/apple-touch-icon.png" />
<meta name="theme-color" content="#3b82f6" />

{{-- Font antarmuka (Inter) dan font aksara dibundel/disajikan sendiri lewat app.css: tidak ada CDN font. --}}
@vite(['resources/css/app.css', 'resources/js/app.js'])
@fluxAppearance
