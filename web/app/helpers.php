<?php

/* Format angka gaya Indonesia untuk tampilan. */

if (! function_exists('pct')) {
    function pct(?float $x, int $decimals = 1): string
    {
        return $x === null ? '–' : number_format($x * 100, $decimals, ',', '.').'%';
    }
}

if (! function_exists('pt')) {
    /** Selisih dua proporsi dalam poin persentase, bertanda. */
    function pt(?float $x): string
    {
        if ($x === null) {
            return '';
        }

        return ($x > 0 ? '+' : ($x < 0 ? '−' : '')).number_format(abs($x) * 100, 1, ',', '.').' pt';
    }
}

if (! function_exists('nfmt')) {
    function nfmt(int|float $n): string
    {
        return number_format($n, 0, ',', '.');
    }
}

if (! function_exists('segments_html')) {
    /**
     * Segmen beda per suku kata (dari Python) -> HTML tanpa spasi di antara <span>: spasi di sana akan
     * tampil sebagai celah di tengah teks aksara. Suku kata yang punya kolom citra diberi handler Alpine
     * (butuh x-data="{ band: null, pin: null }" di induknya). Teks di-escape.
     */
    function segments_html(array $segments, ?array $spans = null): string
    {
        $out = '';
        $i = 0;
        foreach ($segments as $seg) {
            $text = e($seg['t']);
            if ($seg['s'] === 'del') {
                $out .= '<span class="sy sy-del" title="hilang">'.$text.'</span>';

                continue;
            }
            $span = $spans[$i++] ?? null;
            $attrs = '';
            if ($span) {
                [$a, $b] = [(float) $span[0], (float) $span[1]];
                $attrs = sprintf(' data-x0="%1$s" data-x1="%2$s" x-on:mouseenter="band=[%1$s,%2$s]" x-on:mouseleave="band=pin"'
                    .' x-on:click="pin=(pin&&pin[0]===%1$s)?null:[%1$s,%2$s];band=pin" x-bind:class="{\'is-pinned\':pin&&pin[0]===%1$s}"', $a, $b);
            }
            $isSpace = $seg['s'] === 'sp';
            $out .= '<span class="sy sy-'.e($seg['s']).'"'.$attrs.($isSpace ? ' title="spasi tidak ada di label"' : '').'>'
                .($isSpace ? '␣' : $text).'</span>';
        }

        return $out;
    }
}

if (! function_exists('aksara_name')) {
    /** Nama Unicode ringkas: "JAVANESE VOWEL SIGN WULU" -> "wulu". */
    function aksara_name(string $ch): string
    {
        if ($ch === ' ') {
            return 'spasi';
        }
        $name = IntlChar::charName($ch) ?: sprintf('U+%04X', mb_ord($ch));
        $name = str_replace(['JAVANESE ', 'VOWEL SIGN ', 'CONSONANT SIGN ', 'SIGN ', 'LETTER '], '', $name);
        $name = preg_replace_callback('/^DIGIT (\w+)$/', fn ($m) => 'angka '.array_search($m[1],
            ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE'], true), $name);

        return mb_strtolower($name);
    }
}
