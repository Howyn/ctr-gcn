# NTU60 五部位遮挡单种子消融

用 NTU RGB+D 60 X-Sub 的 Joint 单流评估可靠性写入门控。三组从同一随机种子初始化，训练数据保留原有时间裁剪和随机旋转，人工遮挡只用于最终测试。

| 训练组 | 关节聚合 | Mamba-1 状态写入 |
| --- | --- | --- |
| Baseline | 关节均值 | 原始写入，系数恒为 1 |
| A | 可学习关节注意力 | 原始写入，系数恒为 1 |
| B | 与 A 相同的关节注意力 | 以输入可见性和注意力权重调节写入 |

三组沿用完整 CTR-GCN 和单层 Mamba-1，使用同一个显式 CUDA scan。训练配方为 seed=1、65 epoch、64 帧、batch=128、SGD、AMP/TF32，继承仓库的 `ctrgcn_mamba_fast128.yaml`。**主表固定取第 65 轮权重**。训练入口仍会记录每轮干净测试表现，评测脚本按固定轮次取权重。

## 门控定义

每个时间步用一个共享线性层给关节特征打分，在 25 个关节上归一化得到注意力 `a[t,j]`。权重零初始化，初始聚合等于均值。`q[t,j]` 从输入坐标是否为零估计，按时间下采样到 Mamba token 尺度。估计时忽略空帧和空人物槽位。

`r[t] = 1 - sum_j(a[t,j] * (1 - q[t,j]))`，取值在 `[0, 1]`。缺失位置承载的注意力越高，越抑制该时间步写入。实现中仅对 Mamba 的 `B_t` 乘以 `r[t]`，状态转移、`D` 直通项和输出门保持原定义。A/B 参数量一致。分类损失训练注意力，门控由该注意力与可见性直接计算。输入完整时门控系数为 1，实际干净精度以实验为准。

## 评测与判定

采用 [RA-GCNv2 作者代码](https://gitee.com/yfsong0709/RA-GCNv2/blob/master/src/dataprocessor.py) 的关节分组。每次遮挡一个部位，对 64 帧测试输入的所有时间、人物、XYZ 通道置零。这是 RA-GCN 部位定义下的 CTR-GCN 统一评测。报告表中的六列条件是干净、左臂、右臂、双手、双腿、躯干，主指标是后五列 Top-1 的等权平均。每列使用完整 16,487 条测试序列。

除三组独立训练的结果外，脚本用 B 的同一权重关闭门控再评测一遍，输出 `b_gate_off`。`A−Baseline` 检查注意力贡献，`B−A` 检查加入门控后的收益，`B−B_gate_off` 检查训练后的 B 是否依赖推理时的写入调节。

- 首轮过线目标是 `B−Baseline` 的遮挡均值至少 +0.30 个百分点、`B−A` 超过 +0.10 个百分点、B 的干净精度降幅控制在 0.10 个百分点内。优先追求 `B−A` 达到 +0.30 个百分点及以上。
- B 干净精度下降达到 0.30 个百分点，或 `B−A` 提升不超过 0.10 个百分点，暂不进入后续扩展实验。干净下降在 0.10 到 0.30 个百分点之间时，记录为待复核。查看五个分项和 `B−B_gate_off`，判断均值提升集中在哪些部位以及是否有门控直接贡献。
- 单种子结果用于判断是否值得扩展。准确率对比支持该测试设置下的性能差异。状态污染及恢复过程需要单独的时序观测实验。

## 启动

在已配置官方 Mamba-1 CUDA 扩展的 Linux 环境，从仓库根目录运行。默认数据为 `data/ntu/NTU60_CS.npz`，输出在 `work_dir/ntu60/xsub/part_ablation_seed1`。可用 `--data-path` 和 `--run-dir` 指定路径。

```bash
python tools/run_part_ablation.py plan
python tools/run_part_ablation.py smoke
python tools/run_part_ablation.py train && python tools/run_part_ablation.py evaluate
```

`train` 自带训练前检查，依次运行 Baseline、A、B，已完整结束且配置一致的分组可以跳过。可用 `train --variant baseline`、`train --variant a` 或 `train --variant b` 分开执行。中断的分组需保留到其他位置后重新训练，评测需要三组完整权重。结果写入 `evaluation/summary.csv` 和 `evaluation/summary.json`，后者含关节编号、权重哈希、配置、参数量和模型差值。

## AutoDL 后台运行和关机

代码推送到 GitHub 的 `main` 后，在 AutoDL SSH 终端执行：

```bash
cd /root/autodl-tmp/ctr-gcn && git pull --ff-only origin main && mkdir -p logs && {
  nohup /root/autodl-tmp/venv-ctr-infer/bin/python -u tools/run_part_ablation.py autodl \
    >> logs/part_ablation_seed1.log 2>&1 < /dev/null &
  echo "PID=$!"
}
```

`autodl` 在同一进程内依次完成检查、训练、评测和落盘校验。确认三组第 65 轮权重哈希及两份汇总表一致后，调用 AutoDL 的 `/usr/bin/shutdown`。中途出错时查看 `logs/part_ablation_seed1.log`。这个流程已经包含关机步骤，若实例还有旧训练任务或关机监控，先确认它们已结束。

```bash
tail -f /root/autodl-tmp/ctr-gcn/logs/part_ablation_seed1.log
```

关机后可在下次开机读取 `/root/autodl-tmp/ctr-gcn/work_dir/ntu60/xsub/part_ablation_seed1/evaluation/summary.csv`。
