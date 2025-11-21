# 探索率诊断脚本
# 用于理解episode和training step的关系,以及探索率的实际值

param(
    [int]$CurrentTrainingStep = 300,
    [int]$CurrentEpisode = 750
)

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "🔍 探索率诊断分析" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# 读取当前配置
$configFile = "marl_framework\configs\params.yaml"
if (Test-Path $configFile) {
    $config = Get-Content $configFile -Raw
    
    # 提取参数
    $epsMax = 0.3
    $epsMin = 0.05
    $epsAnneal = 100
    
    if ($config -match "eps_max:\s*([\d.]+)") { $epsMax = [double]$matches[1] }
    if ($config -match "eps_min:\s*([\d.]+)") { $epsMin = [double]$matches[1] }
    if ($config -match "eps_anneal_phase:\s*(\d+)") { $epsAnneal = [int]$matches[1] }
    
    Write-Host "📋 当前配置:" -ForegroundColor Green
    Write-Host "  eps_max: $epsMax"
    Write-Host "  eps_min: $epsMin"
    Write-Host "  eps_anneal_phase: $epsAnneal episodes"
    Write-Host ""
    
    # 计算training step和episode的关系
    Write-Host "📊 Training Step vs Episode 关系:" -ForegroundColor Green
    Write-Host "  Budget (时间步): 50"
    Write-Host "  智能体数量: 4"
    Write-Host "  每episode产生样本: 51 × 4 = 204"
    Write-Host "  Batch size: 128"
    Write-Host "  Batch number: 4"
    Write-Host "  每次训练需要样本: 128 × 4 = 512"
    Write-Host "  每次训练需要episodes: 512 / 204 ≈ 2.5"
    Write-Host ""
    Write-Host "  → 所以1个training step ≈ 2.5个episodes" -ForegroundColor Yellow
    Write-Host "  → 100个training steps ≈ 250个episodes" -ForegroundColor Yellow
    Write-Host "  → 300个training steps ≈ 750个episodes" -ForegroundColor Yellow
    Write-Host ""
    
    # 计算不同阶段的探索率
    Write-Host "🎯 探索率随时间变化:" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Episode数 | Training Step | 探索率(eps)" -ForegroundColor Cyan
    Write-Host "  ---------|---------------|-------------" -ForegroundColor Cyan
    
    $milestones = @(1, 10, 25, 50, 100, 150, 200, 300, 500, 750, 1000)
    foreach ($ep in $milestones) {
        $step = [int]($ep / 2.5)
        
        # 计算探索率
        if ($ep -gt $epsAnneal) {
            $eps = $epsMin
        } else {
            $eps = $epsMax - ($ep / $epsAnneal) * ($epsMax - $epsMin)
        }
        
        $eps = [Math]::Max($eps, $epsMin)
        $eps = [Math]::Round($eps, 4)
        
        $color = "White"
        if ($eps -lt 0.1) { $color = "Green" }
        elseif ($eps -lt 0.2) { $color = "Yellow" }
        else { $color = "Red" }
        
        $highlight = if ($ep -eq $CurrentEpisode) { " ← 当前位置" } else { "" }
        Write-Host ("  {0,8} | {1,13} | {2,11}" -f $ep, $step, $eps) -ForegroundColor $color -NoNewline
        if ($highlight) { Write-Host $highlight -ForegroundColor Magenta }
        else { Write-Host "" }
    }
    
    Write-Host ""
    Write-Host "💡 解读:" -ForegroundColor Green
    
    # 计算当前探索率
    $currentEps = if ($CurrentEpisode -gt $epsAnneal) { $epsMin } 
                   else { $epsMax - ($CurrentEpisode / $epsAnneal) * ($epsMax - $epsMin) }
    $currentEps = [Math]::Max($currentEps, $epsMin)
    
    Write-Host "  在第 $CurrentEpisode 个episode (≈ 第 $CurrentTrainingStep 个training step):"
    Write-Host "  探索率 = $('{0:N4}' -f $currentEps)"
    Write-Host ""
    
    if ($currentEps -gt 0.15) {
        Write-Host "  ⚠️  探索率过高 (>15%)!" -ForegroundColor Red
        Write-Host "  → 模型仍在大量随机探索,学习的策略未被充分使用" -ForegroundColor Red
        Write-Host "  → 轨迹看起来会很随机,与初期没有明显区别" -ForegroundColor Red
        Write-Host ""
        Write-Host "  建议:" -ForegroundColor Yellow
        Write-Host "  1. 降低 eps_max 到 0.2-0.3" -ForegroundColor Yellow
        Write-Host "  2. 降低 eps_anneal_phase 到 50-100" -ForegroundColor Yellow
    } elseif ($currentEps -gt 0.08) {
        Write-Host "  ⚠️  探索率偏高 (8-15%)" -ForegroundColor Yellow
        Write-Host "  → 模型正在从探索转向利用,但可能还不够快" -ForegroundColor Yellow
        Write-Host "  → 轨迹应该开始出现一些规律,但仍有随机性" -ForegroundColor Yellow
    } else {
        Write-Host "  ✅ 探索率正常 (<8%)" -ForegroundColor Green
        Write-Host "  → 模型主要使用学习的策略,只有少量探索" -ForegroundColor Green
        Write-Host "  → 轨迹应该表现出明确的学习行为" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "🚀 修复建议:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  当前配置 (修复后):" -ForegroundColor Green
    Write-Host "    eps_max: 0.3           ← 降低初始探索"
    Write-Host "    eps_min: 0.05          ← 保持最小探索"
    Write-Host "    eps_anneal_phase: 100  ← 在100个episodes内完成衰减"
    Write-Host ""
    Write-Host "  效果预测:" -ForegroundColor Yellow
    Write-Host "    • 0-40 steps (0-100 episodes): 探索率从30%降到5%"
    Write-Host "    • 40+ steps (100+ episodes): 保持5%探索率"
    Write-Host "    • 300 steps时: 探索率应该是5%"
    Write-Host ""
    Write-Host "  如果还没效果,尝试:" -ForegroundColor Yellow
    Write-Host "    eps_max: 0.2           ← 更低的初始探索"
    Write-Host "    eps_min: 0.0           ← 完全不探索"
    Write-Host "    eps_anneal_phase: 50   ← 更快衰减"
    Write-Host ""
}

# 检查模型是否真的在学习
Write-Host "=================================" -ForegroundColor Cyan
Write-Host "🔍 学习效果检查" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

$historyFile = "marl_framework\res\training_history.csv"
if (Test-Path $historyFile) {
    $history = Import-Csv $historyFile
    
    if ($history.Count -ge 20) {
        $first10 = $history | Select-Object -First 10 | ForEach-Object { [double]$_.episode_return }
        $last10 = $history | Select-Object -Last 10 | ForEach-Object { [double]$_.episode_return }
        
        $first10Avg = ($first10 | Measure-Object -Average).Average
        $last10Avg = ($last10 | Measure-Object -Average).Average
        $improvement = $last10Avg - $first10Avg
        $improvementPct = if ($first10Avg -ne 0) { ($improvement / [Math]::Abs($first10Avg)) * 100 } else { 0 }
        
        Write-Host "  前10个episode平均回报: $('{0:N2}' -f $first10Avg)"
        Write-Host "  最近10个episode平均回报: $('{0:N2}' -f $last10Avg)"
        Write-Host "  改进: $('{0:N2}' -f $improvement) ($('{0:N1}' -f $improvementPct)%)"
        Write-Host ""
        
        if ($improvement -gt 0 -and $improvementPct -gt 10) {
            Write-Host "  ✅ 模型正在学习! 回报有明显改善" -ForegroundColor Green
        } elseif ($improvement -gt 0) {
            Write-Host "  ⚠️  有改善但不明显,可能需要:" -ForegroundColor Yellow
            Write-Host "     • 提高学习率" -ForegroundColor Yellow
            Write-Host "     • 降低探索率" -ForegroundColor Yellow
            Write-Host "     • 检查奖励函数设计" -ForegroundColor Yellow
        } else {
            Write-Host "  ❌ 回报未改善,可能问题:" -ForegroundColor Red
            Write-Host "     • 探索率过高,策略未被使用" -ForegroundColor Red
            Write-Host "     • 学习率过低,更新太慢" -ForegroundColor Red
            Write-Host "     • 奖励信号不清晰" -ForegroundColor Red
        }
    } else {
        Write-Host "  ⚠️  数据不足,无法评估学习效果" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠️  未找到训练历史文件" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "脚本执行完成" -ForegroundColor Cyan
