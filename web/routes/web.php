<?php

use App\Http\Controllers\LineImageController;
use App\Livewire\Pages;
use Illuminate\Support\Facades\Route;
use Livewire\Volt\Volt;

Route::redirect('/', '/ringkasan')->name('home');

Route::middleware(['auth'])->group(function () {
    // Nama "dashboard" dipakai alur login bawaan starter kit sebagai tujuan setelah masuk.
    Route::get('ringkasan', Pages\Ringkasan::class)->name('dashboard');
    Route::redirect('dashboard', '/ringkasan');
    Route::get('perbandingan', Pages\Perbandingan::class)->name('perbandingan');
    Route::get('ablasi', Pages\Ablasi::class)->name('ablasi');
    Route::get('dataset', Pages\Dataset::class)->name('dataset');
    Route::get('penjelajah', Pages\Penjelajah::class)->name('penjelajah');
    Route::get('kesalahan', Pages\Kesalahan::class)->name('kesalahan');
    Route::get('demo', Pages\Demo::class)->name('demo');
    Route::get('kamus', Pages\Kamus::class)->name('kamus');
    Route::get('terjemahan', Pages\Terjemahan::class)->name('terjemahan');
    Route::get('citra/{line}', LineImageController::class)->name('line.image');

    Route::redirect('settings', 'settings/profile');
    Volt::route('settings/profile', 'settings.profile')->name('settings.profile');
    Volt::route('settings/password', 'settings.password')->name('settings.password');
    Volt::route('settings/appearance', 'settings.appearance')->name('settings.appearance');
});

require __DIR__.'/auth.php';
