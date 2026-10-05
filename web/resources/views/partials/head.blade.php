<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />

<title>{{ isset($title) ? $title.' · Aksara OCR Lab' : 'Aksara OCR Lab' }}</title>

{{-- Font antarmuka (Inter) dan font aksara dibundel/disajikan sendiri lewat app.css: tidak ada CDN font. --}}
@vite(['resources/css/app.css', 'resources/js/app.js'])
@fluxAppearance
