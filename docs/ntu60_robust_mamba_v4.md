# V4：软可见性聚合与一致的状态参数校正

本版针对无额外训练遮挡的 V2（干净 89.06%、20% 遮挡 80.19%）继续研究算法改进，目标是在保持干净识别能力的同时争取将遮挡精度提升至 81%以上。该目标尚待实验验证。

**此前提交 `c2dee7c` 中的 V4 仅给 V3 换了实验编号，没有算法改动。本文所述版本重新实现了模型，基于 V2 保留干净输入路径，不再继承 V3 模型。** 本次使用带 `v4_state` 的全新结果目录，旧 V3 和旧命名版 V4 的权重、日志均保留，不能混用其结果。

## 本次实际改变的计算

1. **软可见性聚合。** 根据当前输入的零坐标启发式估计关节可见性，分别计算普通关节均值 `f_mean` 和可见位置归一化均值 `f_visible`，使用 `f_mean + (1-q) * (f_visible-f_mean)` 形成时序输入。完整观测时保持原均值；缺失位置的图卷积特征仍保留部分贡献，避免 V3 硬剔除可能丢失的邻域语义。空人物与填充帧沿用原处理。零坐标只是可见性启发式，不是真实置信度标注。
2. **无额外可学习参数的一致性门控。** 以当前 token 与最近至多三个历史 token 的均值计算归一化差异 `innovation`，再使用 `r = 1-(1-q)*(0.5+0.5*innovation)`。差异限制在 `[0,1]`，因此 `r` 位于 `[q,(1+q)/2]`；`q=1` 时严格回到 `r=1`。该门控取消了 V2 在完整观测时缺少分类梯度的评分网络，不需要依靠新增训练遮挡学习评分参数。原有局部时序分支仍通过分类损失学习。
3. **写入内容与选择参数使用同一校正特征。** Mamba 卷积得到原特征 `u`，以 `w = r*u + (1-r)*history(u)` 构造写入估计。与 V2/V3 只更改写入输入不同，本版由同一 `w` 重新生成时间步长和输入、输出选择参数（`Δ、B、C`），再执行状态扫描。原 `D*u` 直通和原输出门控保留，完整观测时 `w=u`。
4. **保留 V2 的局部时序残差与初始化顺序。** 模型不新增 Mamba 层或大型分支；软聚合、门控、历史补偿、选择参数校正、局部补偿均提供独立开关。新增参数仅为原局部补偿分支的 **17,921**，总参数为 **1,902,865**，低于新增 0.2M 的限制。

`q=1` 时的恒等路径有助于控制对干净输入的扰动，但真实骨架可能存在天然零坐标，训练后的干净准确率也不能由此获得数学保证。持续五关节缺失时，历史特征具有相同的输入缺失；历史均值仅是替代特征估计，不是恢复的干净关节，也不能仅凭准确率提升认定状态污染已被消除。

## 训练和评测条件保持一致

- NTU60、X-Sub、Joint；只从头训练本版 V4 一轮。
- 训练期 `joint_mask_prob=0.0`，不新增关节遮挡增强；原随机旋转和时间裁剪保留。
- 原交叉熵、SGD、65 epoch、seed 1、batch size 128、学习率及衰减计划、AMP/TF32 设置不变。
- 测试仍采用每条序列固定 5/25 个关节、全部帧及人物共享、三维坐标置零、遮挡种子 2026 的协议。
- 对照为已经训练好的无增强 V2。V2 只参与配对推理，不重新训练。

启动前需保留原标准 Mamba 和无增强 V2 的已完成实验目录：

```text
work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128
work_dir/ntu60/xsub/ctrgcn_mamba_robust_v2_clean_joint_fast128
```

## 云端运行：训练、评测、保存校验、关机

如果旧命名版 V4 已经启动，先确认其进程状态，避免同时运行两个训练任务；不要对模糊进程名执行批量终止。旧训练未运行时，执行：

```bash
cd /root/autodl-tmp/ctr-gcn
git pull --ff-only origin main
source /root/autodl-tmp/venv-ctr-infer/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p logs
python tools/train_v4_autodl.py --check-only
```

检查输出应包含算法标识 `v4_soft_pool_consistent_selective_scan`，用于区分旧命名版 V4。检查通过后启动完整后台流程：

```bash
nohup python -u tools/train_v4_autodl.py \
  >> logs/v4_state_pipeline.log 2>&1 < /dev/null &
echo "V4 状态参数校正实验 PID: $!"
tail -f logs/v4_state_pipeline.log
```

流程会先核对算法标识、配置、参考权重、新结果路径和依赖，再检查真实官方 CUDA/AMP 下的等价路径、缺失输入与梯度。检查通过后从头训练 V4，保存最佳权重，自动进行 V2/V4 干净与20%遮挡配对推理，最后核验掩码、标签、权重哈希及保存分数与指标的一致性。**全部结果保存并通过校验后才请求 AutoDL 关机。** 任一环节失败时保留实例及日志供排错。

`--check-only` 不训练、不访问 GPU，也不关机。无需另外启动旧版训练后关机监控。按 `Ctrl+C` 退出日志查看不会停止后台任务。

## 日志及结果

| 内容 | 路径 |
| --- | --- |
| 完整流程及关机日志 | `logs/v4_state_pipeline.log` |
| V4 训练日志 | `logs/ntu60_xsub_joint_mamba_robust_v4_state.log` |
| 干净/20%配对评测日志 | `logs/occlusion20_robust_v4_state_seed2026.log` |
| 流程状态记录 | `logs/v4_state_pipeline_state.json` |
| V4 训练结果 | `work_dir/ntu60/xsub/ctrgcn_mamba_robust_v4_state_joint_fast128` |
| 配对评测结果 | `work_dir/ntu60/xsub/occlusion20_robust_v4_state_seed2026` |

训练目录保存权重、配置、最终预测及流程摘要，并记录算法标识、源码哈希与源码快照，区分本次实现和旧命名版 V4。评测目录保存 `summary.json`、`summary.csv`、`scores.npz`、`masks.npz` 和 `predictions.csv`。其中 `baseline` 是已有无增强 V2，`mamba` 是本次 V4。

查看训练进度：

```bash
tail -f logs/ntu60_xsub_joint_mamba_robust_v4_state.log
```

下次开机后查看对比结果：

```bash
cat /root/autodl-tmp/ctr-gcn/work_dir/ntu60/xsub/occlusion20_robust_v4_state_seed2026/summary.csv
```

已有本版输出时脚本会拒绝覆盖。旧 V3、旧命名版 V4 的目录不需要删除。本版是否达到干净精度和81%以上遮挡精度目标，以正式训练和配对评测结果为准。
