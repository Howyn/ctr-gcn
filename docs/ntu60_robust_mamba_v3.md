# V3：可见性归一化聚合与可训练的状态写入残差

只启动一个新训练候选，不自动启动消融、外部模型或额外数据集。
主要参照是无新增训练遮挡的 V2：干净 89.0580%、20% 遮挡 80.1905%。
标准组合为 88.9185% / 79.5536%。本轮目标是维持约 89.06% 的干净精度，
争取将遮挡精度提高到 81% 以上；训练前不能保证达到。

## 本次仅改模型，训练条件保持原样

- 训练不生成关节遮挡，joint_mask_prob=0.0；测试只应用原先约定的 5/25
  关节 Mask，不叠加其他遮挡增强。
- 保留原随机旋转/裁剪、Joint、64 帧、65 epoch、seed=1、batch=128，
  SGD / lr=.1 / warmup=5 / steps=[35,55] / weight_decay=.0004 / AMP/TF32。
- 损失仍为原单次交叉熵，没有辅助损失、蒸馏、预训练或旧权重热启动。
- 测试仍使用原来的固定 5/25 关节、所有帧和人物共享、xyz=0、seed=2026。
- 原 V1、V2、增强版和无增强版的模型、配置、结果均保留。
- 启动器先对照原标准 Mamba 实际保存的 config.yaml，检查一致性后再训练。

## 模型改动与边界

1. **可见性归一化关节聚合**：从当前坐标提取零值启发式可见性，按有效帧
   聚合到主干输出时间分辨率，再对可见关节特征归一化池化。不给模型传评测
   mask、标签或原始干净坐标。完整观测直接使用原 mean，空人物沿用原空槽
   处理。它不恢复真实缺失坐标，也不承诺缺失位置的中间特征一定无用。
2. **干净输入可训练的写入选择器**：V2 的 r=1-(1-q)*sigmoid(score) 在 q=1
   时对 score 的梯度恒为零。V3 在所有输入上计算 g=sigmoid(score(stats)+log(q))，
   用 w=u+rho*(1-g)*(p-u) 修正写入，其中 p 是前至多三个 token 的均值，
   rho=tanh(scale)，scale 从零开始。第一步选择器梯度为零，但 scale 可以更新；
   随后 score 在完整干净输入上也能获得分类梯度。CUDA 检查专门验证这一点。
3. **小评估器**：不再使用 256→32 投影，直接计算六个统计量后接 6→16→1。
   选择器连同残差系数共 130 个参数。原多尺度局部卷积分支保留为独立开关。
4. **保留标准 Mamba 直通**：复用 V2 已验证的官方 CUDA scan 适配器，只修正
   写入输入；delta/A/B/C 的投影规则和 D*u 直通保持。不是再次衰减整个输出。

rho 是有界的**有符号**残差系数。有效写入系数 r=1-rho*(1-g) 在 [0,2] 内，
不应写成置信度概率；rho<0 时是外推校正，不能声称所有情况都抑制状态写入。
零初始化保持初始状态写入不变，不是训练后干净准确率不下降的数学保证。

总参数 **1,902,995**，比标准组合增加 **18,051（0.018051M）**，
比 V2 少 8,191（总模型约 0.43%）。新增模块参数减少约 31.2%，但主干占
大部分开销，不能在实际测速前声称明显加速。

四个独立开关为 use_visibility_pooling、use_reliability、use_history_compensation、
use_local_compensation。历史替代仅在启用状态写入选择器时使用；关闭历史替代
会改为调节当前 u 的写入幅度，不会同时删除独立局部卷积分支。
目前只训练完整 V3；开关用于之后明确提出的消融。

这是一项待验证的实现方案，不宣称“首次提出”或保证论文可接收。部位扫描、
可靠性状态更新均有相关文献，需要在正式论文中明确区别：
[Parts-Mamba](https://arxiv.org/abs/2511.16860)、
[BayesMamba](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1786524/full)。

## 一次启动：检查、训练、20% 遮挡评测、保存校验、关机

新流程由 `tools/train_v3_autodl.py` 顺序执行以下步骤：

1. 对照原标准 Mamba 和无额外遮挡 V2 的实际保存配置，检查训练条件；确认参考权重和数据文件存在，新结果路径未被使用。
2. 检查训练依赖、CUDA、NumPy/PyTorch 转换和关机命令格式，自动运行 V3 的官方 CUDA/AMP 等价性、干净输入梯度及缺失关节前向检查。
3. 从头训练 V3，完成 65 个 epoch，保存并校验最佳权重、最后一轮权重、配置和最终预测文件。
4. 自动选取 V3 最佳权重，与已有的**无额外训练遮挡 V2**进行配对干净/20% 遮挡推理；两个模型共享原先固定的测试掩码。
5. 运行已保存产物审计，核验掩码、标签、权重哈希及预测分数与汇总指标的一致性。全部通过、结果落盘后，才请求 AutoDL 关机。

启动前应已有下面两个实验目录。第一项用于核对原训练配方，第二项还提供本次配对评测的对照权重：

```text
work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128
work_dir/ntu60/xsub/ctrgcn_mamba_robust_v2_clean_joint_fast128
```

云服务器执行：

```bash
cd /root/autodl-tmp/ctr-gcn
git pull --ff-only origin main
source /root/autodl-tmp/venv-ctr-infer/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p logs
python tools/train_v3_autodl.py --check-only
```

`--check-only` 只检查配置、已有参考权重和输出路径，不访问 GPU，不训练也不关机。看到检查通过后，后台启动整个流程：

```bash
nohup python -u tools/train_v3_autodl.py \
  >> logs/v3_pipeline.log 2>&1 < /dev/null &
echo "V3 完整流程 PID: $!"
tail -f logs/v3_pipeline.log
```

退出日志查看可按 `Ctrl+C`，后台流程继续执行。无需另外启动旧版 `train_robust_autodl.sh v3` 或关机监控：它们的单训练关机逻辑会与本次“评测后关机”顺序冲突。

## 日志与结果

| 内容 | 路径 |
| --- | --- |
| 完整流程进度与关机记录 | `logs/v3_pipeline.log` |
| V3 训练日志 | `logs/ntu60_xsub_joint_mamba_robust_v3_clean.log` |
| 干净/20% 配对推理日志 | `logs/occlusion20_robust_v3_clean_seed2026.log` |
| 已保存产物审计报告 | `logs/audit_occlusion20_v3_clean_seed2026.json` |
| V3 训练结果 | `work_dir/ntu60/xsub/ctrgcn_mamba_robust_v3_clean_joint_fast128` |
| 干净/20% 配对推理结果 | `work_dir/ntu60/xsub/occlusion20_robust_v3_clean_seed2026` |

V3 训练目录保存 `best.pt`、原始逐轮权重、配置、预测文件、`result_summary.json` 和 `pipeline_summary.json`。遮挡评测目录保存 `summary.json`、`summary.csv`、`scores.npz`、`masks.npz` 和 `predictions.csv`。本次评测日志中的 `baseline` 指无额外训练遮挡 V2，`mamba` 指新训练的 V3。

查看具体训练进度：

```bash
tail -f logs/ntu60_xsub_joint_mamba_robust_v3_clean.log
```

已有正式目录或评测输出时拒绝覆盖。任一步报错或结果校验失败，流程停止并保留实例供排错；应先查看相关日志，避免重复启动训练。本流程只训练完整 V3 一轮，不自动启动消融或重训参考模型。

## 可选：训练前单独测速

完整流程已包含 CUDA 检查，无需手动重复。若需要先了解当前显卡耗时，可在启动完整流程前单独运行：

```bash
set -o pipefail
python -u main.py \
  --config config/nturgbd-cross-subject/ctrgcn_mamba_robust_v3_clean_fast128.yaml \
  --phase benchmark \
  --work-dir work_dir/ntu60/xsub/benchmark_mamba_robust_v3 \
  --device 0 --seed 1 \
  2>&1 | tee logs/benchmark_mamba_robust_v3.log
```

测速只更新一次性临时模型，不保存权重；正式流程仍从头训练。估算不包含全部初始化、写盘及后续评测时间，不保证两小时内完成。

已保存产物审计不等于在线全部输入路径审计，也不证明历史训练数据来源。
如需进一步人工排查，可另行使用审计工具的在线抽查功能。
V3 的 gate_diagnostics 输出 residual_write_coefficient / signed_residual_scale，
不将有效写入系数误标为 [0,1] 的可靠性概率。
