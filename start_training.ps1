# GPU加速训练 - 快速启动脚本
# 使用方法: .\start_training.ps1

Write-Host "=" -NoNewline
Write-Host ("=" * 58)
Write-Host "GPU加速MARL训练 - 启动脚本"
Write-Host "=" -NoNewline
Write-Host ("=" * 58)
Write-Host ""

# 检查CUDA
Write-Host "[1/5] 检查CUDA环境..." -ForegroundColor Cyan
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: Python或PyTorch未正确安装" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 显示配置
Write-Host "[2/5] 加载训练配置..." -ForegroundColor Cyan
$config = Get-Content marl_framework\params.yaml | Select-String -Pattern "device:|batch_size:|n_agents:|n_episodes:|budget:"
Write-Host $config
Write-Host ""

# 询问是否继续
Write-Host "[3/5] 准备开始训练" -ForegroundColor Cyan
$response = Read-Host "是否开始训练? (y/n)"
if ($response -ne 'y') {
    Write-Host "训练已取消" -ForegroundColor Yellow
    exit 0
}

Write-Host ""

# 清理旧日志(可选)
Write-Host "[4/5] 清理旧日志..." -ForegroundColor Cyan
$cleanLogs = Read-Host "是否清理旧的训练日志? (y/n)"
if ($cleanLogs -eq 'y') {
    Remove-Item marl_framework\log\*.tfevents.* -ErrorAction SilentlyContinue
    Write-Host "✓ 日志已清理" -ForegroundColor Green
} else {
    Write-Host "保留现有日志" -ForegroundColor Yellow
}

Write-Host ""

# 开始训练
Write-Host "[5/5] 启动训练..." -ForegroundColor Cyan
Write-Host "训练日志将保存到: marl_framework\log\" -ForegroundColor Yellow
Write-Host "使用 Ctrl+C 可以中断训练" -ForegroundColor Yellow
Write-Host ""
Write-Host "=" -NoNewline
Write-Host ("=" * 58)
Write-Host ""

cd marl_framework
python main.py

Write-Host ""
Write-Host "=" -NoNewline
Write-Host ("=" * 58)
if ($LASTEXITCODE -eq 0) {
    Write-Host "训练完成!" -ForegroundColor Green
} else {
    Write-Host "训练异常退出 (退出码: $LASTEXITCODE)" -ForegroundColor Red
}
Write-Host "=" -NoNewline
Write-Host ("=" * 58)
Write-Host ""

# 提示查看结果
Write-Host "查看训练结果:" -ForegroundColor Cyan
Write-Host "  TensorBoard: tensorboard --logdir=marl_framework\log"
Write-Host "  最佳模型: marl_framework\log\best_model.pth"
Write-Host ""
