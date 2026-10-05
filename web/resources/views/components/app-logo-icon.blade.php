{{-- Lambang kecil: lambangnya saja di ubin putih (tulisan di gambar sumber tidak terbaca pada ukuran ini).
     Berkasnya dibuat tools/logo.py dari resources/brand/logo.png; ?v= supaya browser memuat ulang bila berganti. --}}
<img src="{{ asset('img/logo/mark.png') }}?v={{ filemtime(public_path('img/logo/mark.png')) }}" alt="" width="144" height="144" {{ $attributes }}>
