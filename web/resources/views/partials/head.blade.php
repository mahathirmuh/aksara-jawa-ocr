<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />

<title>{{ isset($title) ? $title.' · Aksara OCR Lab' : 'Aksara OCR Lab' }}</title>

{{-- Ikon dibuat web/tools/logo.py dari gambar lambang (resources/brand/logo.png); ?v= supaya browser memuat ulang bila berganti. --}}
<link rel="icon" href="/favicon.ico?v={{ filemtime(public_path('favicon.ico')) }}" sizes="16x16 32x32 48x48" />
<link rel="icon" href="/favicon-192.png?v={{ filemtime(public_path('favicon-192.png')) }}" type="image/png" sizes="192x192" />
<link rel="apple-touch-icon" href="/apple-touch-icon.png?v={{ filemtime(public_path('apple-touch-icon.png')) }}" />
<meta name="theme-color" content="#3b82f6" />

{{-- Font antarmuka (Inter) dan font aksara dibundel/disajikan sendiri lewat app.css: tidak ada CDN font. --}}
@vite(['resources/css/app.css', 'resources/js/app.js'])
@fluxAppearance
