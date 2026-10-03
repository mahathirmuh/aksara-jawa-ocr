<#
Menyalakan / mematikan ketiga layanan Aksara OCR Lab sebagai proses mandiri (tetap hidup setelah terminal ditutup).

  powershell -ExecutionPolicy Bypass -File web\services.ps1 start
  powershell -ExecutionPolicy Bypass -File web\services.ps1 status
  powershell -ExecutionPolicy Bypass -File web\services.ps1 stop

Port: Laravel 8010, layanan model OCR 8011, layanan terjemahan 8012 (8000/8001 dipakai proyek lain).
Log: web\storage\logs\service-*.log. PID: web\storage\app\services.json.
#>
param([ValidateSet('start', 'stop', 'status')][string]$Action = 'status')

$web = $PSScriptRoot
$repo = Split-Path $web -Parent
$python = Join-Path $repo '.venv\Scripts\python.exe'
$pidFile = Join-Path $web 'storage\app\services.json'
$logDir = Join-Path $web 'storage\logs'
$env:PYTHONIOENCODING = 'utf-8'

$services = @(
    @{ name = 'laravel'; port = 8010; file = 'php'; dir = $web
       args = @('artisan', 'serve', '--host=127.0.0.1', '--port=8010', '--no-ansi') },
    @{ name = 'ocr'; port = 8011; file = $python; dir = $repo
       args = @('-m', 'uvicorn', 'src.serve:app', '--host', '127.0.0.1', '--port', '8011') },
    @{ name = 'terjemahan'; port = 8012; file = $python; dir = $web
       args = @('-m', 'uvicorn', 'translate_service:app', '--app-dir', 'tools', '--host', '127.0.0.1', '--port', '8012') }
)

function Get-Listener([int]$port) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
}

function Stop-Tree([int]$id) {
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$id" | ForEach-Object { Stop-Tree $_.ProcessId }
    Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
}

# `php artisan serve` = induk + anak `php -S`; kalau hanya anak (pemilik port) yang dimatikan, induk menyalakannya
# lagi. Naik ke induk selama induknya masih bagian dari layanan yang sama.
function Get-ServiceRoot([int]$id) {
    while ($true) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$id"
        $parent = if ($proc) { Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.ParentProcessId)" }
        if ($parent -and $parent.CommandLine -match 'artisan\s+serve|uvicorn') { $id = $parent.ProcessId } else { return $id }
    }
}

switch ($Action) {
    'start' {
        $pids = @{}
        foreach ($s in $services) {
            if (Get-Listener $s.port) { "{0,-11} sudah berjalan di port {1}" -f $s.name, $s.port; continue }
            $p = Start-Process -FilePath $s.file -ArgumentList $s.args -WorkingDirectory $s.dir -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput (Join-Path $logDir "service-$($s.name).log") `
                -RedirectStandardError (Join-Path $logDir "service-$($s.name).err.log")
            $pids[$s.name] = $p.Id
            "{0,-11} dinyalakan (PID {1}, port {2})" -f $s.name, $p.Id, $s.port
        }
        if ($pids.Count) { $pids | ConvertTo-Json | Set-Content $pidFile -Encoding utf8 }
        'Buka http://127.0.0.1:8010 (model terjemahan butuh ~1 menit untuk termuat).'
    }
    'stop' {
        foreach ($s in $services) {
            $l = Get-Listener $s.port
            if ($l) { Stop-Tree (Get-ServiceRoot $l.OwningProcess); "{0,-11} dihentikan" -f $s.name } else { "{0,-11} tidak berjalan" -f $s.name }
        }
        if (Test-Path $pidFile) {
            (Get-Content $pidFile -Raw | ConvertFrom-Json).PSObject.Properties | ForEach-Object { Stop-Tree $_.Value }
            Remove-Item $pidFile
        }
    }
    'status' {
        foreach ($s in $services) {
            $l = Get-Listener $s.port
            "{0,-11} port {1}: {2}" -f $s.name, $s.port, $(if ($l) { "hidup (PID $($l.OwningProcess))" } else { 'mati' })
        }
    }
}
