# 训练配置文件

## 📁 配置文件说明

### params.yaml (默认/完整配置)
- **用途**: 完整训练,最佳性能
- **Budget**: 14 步
- **Episodes**: 1500
- **Batch size**: 64
- **训练时间**: 40-80 小时
- **推荐场景**: 论文实验、最终模型

### params_balanced.yaml ⭐ 推荐
- **用途**: 平衡配置,性价比最高
- **Budget**: 12 步
- **Episodes**: 1000
- **Batch size**: 48
- **训练时间**: 20-30 小时
- **推荐场景**: 生产部署、日常训练

### params_fast.yaml
- **用途**: 快速测试,验证代码
- **Budget**: 8 步
- **Episodes**: 500
- **Batch size**: 32
- **训练时间**: 10-15 小时
- **推荐场景**: 算法验证、调试

### params_test.yaml
- **用途**: 单元测试、CI/CD
- **Budget**: 4 步
- **Episodes**: 10
- **Batch size**: 8
- **训练时间**: <1 小时

## 🚀 使用方法

```bash
# 使用默认配置
python main.py

# 使用指定配置
CONFIG_FILE_PATH=configs/params_balanced.yaml python main.py

# 使用训练脚本
cd scripts
CONFIG_FILE_PATH=configs/params_balanced.yaml ./train_with_backup.sh exp_name
```

## ⚙️ 配置参数对比

| 参数 | params.yaml | params_balanced.yaml | params_fast.yaml |
|------|-------------|---------------------|------------------|
| budget | 14 | 12 | 8 |
| n_episodes | 1500 | 1000 | 500 |
| batch_size | 64 | 48 | 32 |
| data_passes | 5 | 3 | 3 |
| 训练步数 | ~4,800 | ~3,333 | ~2,083 |
| 预期时间 | 40-80h | 20-30h | 10-15h |
| 模型性能 | 100% | 93-95% | 80-85% |

## 📝 自定义配置

复制现有配置文件并修改:

```bash
cp params_balanced.yaml params_custom.yaml
# 编辑 params_custom.yaml
CONFIG_FILE_PATH=configs/params_custom.yaml python main.py
```

## 🔗 相关文档

- [配置选择指南](../docs/CONFIG_SELECTION_GUIDE.md)
- [训练优化](../docs/TRAINING_OPTIMIZATION.md)
- [GPU使用指南](../docs/GPU_TRAINING_GUIDE.md)
