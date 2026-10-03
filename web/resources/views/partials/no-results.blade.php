<div class="rounded-xl border border-dashed border-zinc-300 p-6 dark:border-zinc-600">
    <flux:heading size="lg">Belum ada hasil yang diimpor</flux:heading>
    <p class="mt-2 max-w-prose text-sm text-zinc-600 dark:text-zinc-300">
        Angka OCR dihitung di repo Python, lalu diimpor ke web. Jalankan dari folder repo OCR:
    </p>
    <pre class="mt-3 overflow-x-auto rounded-lg bg-zinc-100 p-3 font-mono text-xs dark:bg-zinc-800">.venv/Scripts/python scripts/export_results.py
cd web
php artisan aksara:import
php artisan aksara:annotations</pre>
</div>
