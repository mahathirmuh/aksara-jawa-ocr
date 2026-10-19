<#
Penjaga rantai Transformer resep penuh (scripts/run_arch_transformer_full.sh). Bila ada proses src.train tetapi log run-nya
(out/checkpoints/<run>/log.jsonl) tidak bertambah selama 25 menit, proses training itu dihentikan; skrip rantai lalu
mencobanya ulang dari checkpoint terakhir. Lahir dari kejadian 2026-10-18 19:18: training macet (semua thread menunggu,
0 CPU) saat memori sistem hampir penuh, dan baru ketahuan 40 menit kemudian. Berhenti sendiri saat rantai selesai atau
gagal total. Catatannya di out/archT_watchdog.log: log rantai dipegang bash yang sedang menulis, jadi PowerShell tidak
bisa menambahinya (Add-Content: "being used by another process"); log rantai hanya dibaca.

  Start-Process powershell -ArgumentList '-ExecutionPolicy Bypass -File scripts\arch_watchdog.ps1' -WindowStyle Hidden
#>
$repo = Split-Path $PSScriptRoot -Parent
$log = Join-Path $repo 'out\archT_chain.log'
$own = Join-Path $repo 'out\archT_watchdog.log'
# 15 menit: jeda normal terpanjang log run ~12 menit (50 langkah + validasi 500 baris). Semula 25; diturunkan sesudah
# macet kedua (2026-10-19 ~00:05, archT_fase7_track langkah 150) yang menghabiskan 25 menit menunggu.
$limitMinutes = 15

function Say([string]$message) {
    Add-Content -Path $own -Value ("penjaga: $message " + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -Encoding utf8
}

function Stop-Tree([int]$id) {
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$id" | ForEach-Object { Stop-Tree $_.ProcessId }
    Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
}

Say 'mulai'
# Laptop tidur (Sleep/Hibernate/Modern Standby) membekukan training DAN penjaga; sesudah bangun log run tampak basi
# padahal training tidak macet (2026-10-19 07:17-08:16: penjaga menghentikan training yang hanya membeku). Jeda antar
# putaran penjaga > 5 menit = laptop baru bangun: umur log dihitung ulang dari saat bangun.
$last = Get-Date
$resumedAt = Get-Date '2000-01-01'
while ($true) {
    Start-Sleep -Seconds 60
    $now = Get-Date
    if (($now - $last).TotalMinutes -gt 5) {
        $resumedAt = $now
        Say ("jeda {0} menit sejak putaran sebelumnya (laptop tidur?); umur log dihitung dari sekarang" -f [int]($now - $last).TotalMinutes)
    }
    $last = $now
    $text = Get-Content $log -Raw -Encoding utf8
    if ($text -match '=== rantai selesai' -or $text -match '(?m)^GAGAL ') { Say 'selesai'; break }
    $procs = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '-m src\.train --run (\S+)' })
    if ($procs.Count -eq 0) { continue }
    $run = [regex]::Match($procs[0].CommandLine, '--run (\S+)').Groups[1].Value
    $runLog = Join-Path $repo "out\checkpoints\$run\log.jsonl"
    if (-not (Test-Path $runLog)) { continue }
    $fresh = (Get-Item $runLog).LastWriteTime
    if ($resumedAt -gt $fresh) { $fresh = $resumedAt }
    $age = $now - $fresh
    if ($age.TotalMinutes -gt $limitMinutes) {
        Say ("log $run tidak bertambah {0} menit; training dihentikan supaya dilanjutkan dari checkpoint terakhir" -f [int]$age.TotalMinutes)
        $procs | ForEach-Object { Stop-Tree $_.ProcessId }
        Start-Sleep -Seconds 120
        $last = Get-Date
    }
}
