# V4：当前 V3 算法的独立编号复跑

版本核对显示，11 月 19 日的提交增加了实验队列及训练后评测流程，没有更改 V3 模型算法。V3 模型文件来自 11 月 12 日的提交 `dd4b1dd`。用户提供的旧 V3 日志记录了干净精度 87.91%；这份结果低于无增强 V2 的约 89.06%，不能忽略。

按独立命名运行的要求，V4 使用新的模型入口、配置、日志及结果目录。**V4 复用当前 V3 的相同算法和初始化，属于独立复跑，不是新算法，也不保证精度提高。** 新模型入口继承 V3，不覆盖其前向计算或初始化。已有 V3 的权重、日志及评测结果全部保留。

## 本轮设置

- 只从头训练 V4 一轮：65 epoch、seed 1、batch size 128。
- 训练期 `joint_mask_prob=0.0`，不新增关节遮挡增强；原有旋转、裁剪、交叉熵及优化器设置不变。
- 参数量与当前 V3 相同：1,902,995，比标准组合增加 18,051。
- 训练后复用已完成的无增强 V2 权重，与 V4 在同一批干净/20% 遮挡测试输入上配对推理。V2 不会重新训练。
- 测试仍为每条序列固定 5/25 个关节、全部帧及人物共享，坐标置零，遮挡种子 2026。
- 推理与保存结果审计通过后，才请求 AutoDL 关机；任何步骤失败则保留实例排错。

启动前需保留原标准 Mamba 与无增强 V2 的实验目录：

```text
work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128
work_dir/ntu60/xsub/ctrgcn_mamba_robust_v2_clean_joint_fast128
```

## 云端运行

```bash
cd /root/autodl-tmp/ctr-gcn
git pull --ff-only origin main
source /root/autodl-tmp/venv-ctr-infer/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p logs
python tools/train_v4_autodl.py --check-only
```

配置、已有参考权重及输出路径检查通过后，启动完整后台流程：

```bash
nohup python -u tools/train_v4_autodl.py \
  >> logs/v4_pipeline.log 2>&1 < /dev/null &
echo "V4 完整流程 PID: $!"
tail -f logs/v4_pipeline.log
```

完整流程会自动检查真实 CUDA/AMP 和干净输入梯度，再开始正式训练。`--check-only` 不访问 GPU，不训练，也不关机。无需另挂旧版训练后关机监控，以免在遮挡评测之前关机。`Ctrl+C` 只退出日志查看，后台流程继续运行。

## 日志及输出

| 内容 | 路径 |
| --- | --- |
| 完整流程及关机日志 | `logs/v4_pipeline.log` |
| 训练日志 | `logs/ntu60_xsub_joint_mamba_robust_v4_clean.log` |
| 配对评测日志 | `logs/occlusion20_robust_v4_clean_seed2026.log` |
| 产物审计报告 | `logs/audit_occlusion20_v4_clean_seed2026.json` |
| V4 训练结果 | `work_dir/ntu60/xsub/ctrgcn_mamba_robust_v4_clean_joint_fast128` |
| 干净/20% 评测结果 | `work_dir/ntu60/xsub/occlusion20_robust_v4_clean_seed2026` |

训练目录保存 `best.pt`、配置、逐轮权重、最终预测、`result_summary.json` 和 `pipeline_summary.json`。评测目录保存 `summary.json`、`summary.csv`、分数、掩码及逐样本预测。评测输出的 `baseline` 指已有无增强 V2，`mamba` 指本轮 V4。

查看训练进度：

```bash
tail -f logs/ntu60_xsub_joint_mamba_robust_v4_clean.log
```

下次开机后查看结果：

```bash
cd /root/autodl-tmp/ctr-gcn
cat work_dir/ntu60/xsub/occlusion20_robust_v4_clean_seed2026/summary.csv
```

已有 V4 输出时会拒绝覆盖；旧 V3 目录本身不会阻止这个独立命名的 V4 流程。是否有改善应以此次实际结果为准，不能把重新命名写成算法创新或将旧结果改标为 V4。
