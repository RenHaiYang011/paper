# GPU监控脚本
# 使用方法: .\monitor_gpu.ps1

Write-Host "GPU训练监控 - 按Ctrl+C退出" -ForegroundColor Cyan
Write-Host "=" * 60
Write-Host ""

$iteration = 0
while ($true) {
    Clear-Host
    $iteration++
    
    Write-Host "═════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  GPU训练实时监控 (刷新 #$iteration)" -ForegroundColor Yellow
    Write-Host "═════════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    
    # 显示时间
    Write-Host "时间: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host ""
    
    # GPU状态
    Write-Host "【GPU状态】" -ForegroundColor Green
    try {
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | ForEach-Object {
            $fields = $_ -split ','
            $gpuId = $fields[0].Trim()
            $gpuName = $fields[1].Trim()
            $temp = $fields[2].Trim()
            $gpuUtil = $fields[3].Trim()
            $memUtil = $fields[4].Trim()
            $memUsed = $fields[5].Trim()
            $memTotal = $fields[6].Trim()
            
            Write-Host "  GPU $gpuId : $gpuName" -ForegroundColor White
            Write-Host "    温度: ${temp}°C | GPU利用率: ${gpuUtil}% | 内存利用率: ${memUtil}%" -ForegroundColor Cyan
            Write-Host "    显存: ${memUsed}MB / ${memTotal}MB" -ForegroundColor Cyan
        }
    } catch {
        Write-Host "  无法获取GPU信息 (nvidia-smi未找到)" -ForegroundColor Red
    }
    
    Write-Host ""
    
    # 检查训练进程
    Write-Host "【训练进程】" -ForegroundColor Green
    $pythonProcesses = Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.WorkingSet -gt 100MB}
    if ($pythonProcesses) {
        foreach ($proc in $pythonProcesses) {
            $memMB = [math]::Round($proc.WorkingSet / 1MB, 2)
            $cpuPercent = [math]::Round($proc.CPU, 2)
            Write-Host "  PID: $($proc.Id) | 内存: ${memMB}MB | CPU时间: ${cpuPercent}s" -ForegroundColor Cyan
        }
    } else {
        Write-Host "  未检测到训练进程" -ForegroundColor Yellow
    }
    
    Write-Host ""
    
    # 检查日志文件
    Write-Host "【训练日志】" -ForegroundColor Green
    $logDir = "marl_framework\log"
    if (Test-Path $logDir) {
        $latestLog = Get-ChildItem $logDir -Filter "*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latestLog) {
            Write-Host "  最新日志: $($latestLog.Name)" -ForegroundColor Cyan
            Write-Host "  更新时间: $($latestLog.LastWriteTime)" -ForegroundColor Cyan
            
            # 读取最后几行
            $lastLines = Get-Content $latestLog.FullName -Tail 3 -ErrorAction SilentlyContinue
            if ($lastLines) {
                Write-Host "  最新内容:" -ForegroundColor Gray
                foreach ($line in $lastLines) {
                    Write-Host "    $line" -ForegroundColor Gray
                }
            }
        } else {
            Write-Host "  未找到日志文件" -ForegroundColor Yellow
        }
        
        # 检查模型文件
        $models = Get-ChildItem $logDir -Filter "*.pth" -ErrorAction SilentlyContinue
        if ($models) {
            Write-Host ""
            Write-Host "  已保存模型: $($models.Count) 个" -ForegroundColor Cyan
            $bestModel = $models | Where-Object {$_.Name -eq "best_model.pth"}
            if ($bestModel) {
                $sizeMB = [math]::Round($bestModel.Length / 1MB, 2)
                Write-Host "    ✓ best_model.pth (${sizeMB}MB, 更新于 $($bestModel.LastWriteTime.ToString('HH:mm:ss')))" -ForegroundColor Green
            }
        }
    } else {
        Write-Host "  日志目录不存在" -ForegroundColor Yellow
    }
    
    Write-Host ""
    Write-Host "─────────────────────────────────────────────────────────────" -ForegroundColor Gray
    Write-Host "  下次刷新: 5秒后 | 按 Ctrl+C 退出" -ForegroundColor Gray
    Write-Host "─────────────────────────────────────────────────────────────" -ForegroundColor Gray
    
    Start-Sleep -Seconds 5
}
