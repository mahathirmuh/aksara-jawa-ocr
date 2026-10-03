<?php

/*
 * Hanya kunci yang berbeda dari bawaan Livewire; sisanya digabung dari konfigurasi paket.
 *
 * Starter kit ini ditulis untuk Livewire 3 (layout default components.layouts.app), sedangkan Composer
 * memasang Livewire 4 yang memakai layouts::app. Tanpa ini halaman Settings (Volt) error 500.
 */
return [
    'component_layout' => 'components.layouts.app',
];
