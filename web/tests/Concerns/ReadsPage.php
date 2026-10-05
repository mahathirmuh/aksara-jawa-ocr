<?php

namespace Tests\Concerns;

use DOMDocument;
use DOMNode;
use DOMXPath;

/**
 * Membaca bagian halaman dari HTML-nya, supaya pernyataan test mengenai baris yang dimaksud saja.
 *
 * `assertSee` pada teks yang muncul di banyak tempat tetap lolos walau barisnya salah (pil terbalik, angka pindah ke
 * baris lain): tinjauan 2026-10-05 menemukan 10 kerusakan semacam itu yang tidak tertangkap test halaman Dataset.
 * Halaman menandai barisnya dengan atribut data-* dan test membaca teks baris itu saja.
 */
trait ReadsPage
{
    protected function xpath(string $html): DOMXPath
    {
        $dom = new DOMDocument;
        @$dom->loadHTML('<?xml encoding="utf-8"?>'.$html);

        return new DOMXPath($dom);
    }

    /** Teks sebuah simpul dengan spasi dan baris baru dirapatkan; null bila simpulnya tidak ada. */
    protected function squash(DOMNode|string|null $node): ?string
    {
        if ($node === null) {
            return null;
        }

        return trim(preg_replace('/\s+/u', ' ', is_string($node) ? $node : $node->textContent));
    }

    /** Teks satu elemen halaman yang ditandai atribut data-*; elemennya harus tepat satu. */
    protected function part(string $html, string $attribute, string $value): string
    {
        $nodes = $this->xpath($html)->query("//*[@{$attribute}='{$value}']");
        $this->assertSame(1, $nodes->length, "elemen {$attribute}={$value}");

        return $this->squash($nodes->item(0));
    }

    /** Teks ubin ringkas menurut judulnya. */
    protected function tile(string $html, string $title): string
    {
        $nodes = $this->xpath($html)->query("//div[contains(concat(' ', normalize-space(@class), ' '), ' stat-card ')][.//div[@class='stat-title' and normalize-space()='{$title}']]");
        $this->assertSame(1, $nodes->length, "ubin {$title}");

        return $this->squash($nodes->item(0));
    }
}
