# 🚀 快速验证启动脚本
# Quick Test Validation Script
# 预计运行时间: 1-2小时

# 进入正确目录
Set-Location -Path "E:\code\paper_code\paper\marl_framework"

Write-Host "🚀 MARL框架快速验证模式" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "⚙️  配置信息:" -ForegroundColor Cyan
Write-Host "  - 配置文件: configs/params_quick_test.yaml" -ForegroundColor Yellow
Write-Host "  - Episodes: 100" -ForegroundColor Yellow
Write-Host "  - Budget: 30步" -ForegroundColor Yellow
Write-Host "  - Agents: 2个" -ForegroundColor Yellow
Write-Host "  - Batch Size: 16" -ForegroundColor Yellow
Write-Host "  - 预计时间: 1-2小时" -ForegroundColor Yellow
Write-Host ""
Write-Host "✅ 验证功能:" -ForegroundColor Cyan
Write-Host "  ✓ 多层级奖励架构" -ForegroundColor White
Write-Host "  ✓ 前沿驱动探索" -ForegroundColor White
Write-Host "  ✓ 三重协同机制" -ForegroundColor White
Write-Host "  ✓ 区域优先搜索" -ForegroundColor White
Write-Host "  ✓ 障碍物避障" -ForegroundColor White
Write-Host "  ✓ 目标发现奖励" -ForegroundColor White
Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host ""

# 检查CUDA是否可用
Write-Host "🔍 检查GPU环境..." -ForegroundColor Cyan
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ GPU检测成功: $gpuInfo" -ForegroundColor Green
    } else {
        Write-Host "⚠️  GPU未检测到，将使用CPU训练（速度较慢）" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  无法检测GPU，将使用CPU训练" -ForegroundColor Yellow
}
Write-Host ""

# 确认开始
Write-Host "按任意键开始训练，或按Ctrl+C取消..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
Write-Host ""

# 记录开始时间
$startTime = Get-Date
Write-Host "⏰ 训练开始时间: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Green
Write-Host ""

# 启动训练
Write-Host "🚀 正在启动训练..." -ForegroundColor Green
Write-Host "💡 提示: 可以打开新终端运行以下命令监控训练:" -ForegroundColor Yellow
Write-Host "   Get-Content -Path 'log\training.log' -Wait -Tail 30" -ForegroundColor Gray
Write-Host ""

try {
    # 运行训练
    python main.py --config configs/params_quick_test.yaml
    
    # 记录结束时间
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Host ""
    Write-Host "================================" -ForegroundColor Green
    Write-Host "✅ 训练完成!" -ForegroundColor Green
    Write-Host "⏰ 结束时间: $($endTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Green
    Write-Host "⏱️  总耗时: $($duration.Hours)小时 $($duration.Minutes)分钟 $($duration.Seconds)秒" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 查看结果:" -ForegroundColor Cyan
    Write-Host "  - 日志文件: log\training.log" -ForegroundColor White
    Write-Host "  - 结果图表: res\" -ForegroundColor White
    Write-Host "  - 模型文件: res\checkpoints\" -ForegroundColor White
    Write-Host "================================" -ForegroundColor Green
    
} catch {
    Write-Host ""
    Write-Host "❌ 训练过程出现错误!" -ForegroundColor Red
    Write-Host "错误信息: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 排查建议:" -ForegroundColor Yellow
    Write-Host "  1. 检查Python环境是否正确" -ForegroundColor White
    Write-Host "  2. 查看 log\training.log 获取详细错误" -ForegroundColor White
    Write-Host "  3. 确认所有依赖包已安装" -ForegroundColor White
    Write-Host "  4. 检查CUDA环境（如使用GPU）" -ForegroundColor White
    exit 1
}
