# V3：可见性归一化聚合与可训练的状态写入残差

只启动一个新训练候选，不自动启动消融、外部模型或额外数据集。
主要参照是无新增训练遮挡的 V2：干净 89.0580%、20% 遮挡 80.1905%。
标准组合为 88.9185% / 79.5536%。目标是超过 V2，争取遮挡达到 81.05%
（相对标准组合 +1.5 个百分点）；训练前不能保证达到。

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

## 云端先检查实际 CUDA 与速度

```bash
cd /root/autodl-tmp/ctr-gcn
git pull --ff-only origin main
source /root/autodl-tmp/venv-ctr-infer/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
set -o pipefail
python -u tools/check_robust_mamba.py --variant v3 \
  2>&1 | tee logs/check_mamba_v3.log
```

必须看到 clean selector gradient 非零、五关节缺失前向通过、最后 PASS。
检查只更新一次性临时模型，不保存或更改已有训练权重。通过后再测速：

```bash
python -u main.py \
  --config config/nturgbd-cross-subject/ctrgcn_mamba_robust_v3_clean_fast128.yaml \
  --phase benchmark \
  --work-dir work_dir/ntu60/xsub/benchmark_mamba_robust_v3 \
  --device 0 --seed 1 \
  2>&1 | tee logs/benchmark_mamba_robust_v3.log
```

以当前显卡的估算时间和显存为准；估算不含初始化/写盘，不保证两小时内结束。
benchmark 不保存权重，正式训练会创建一个新进程，从头开始。

## 正式只训练这一组，训练后保存并关机

```bash
bash tools/train_robust_autodl.sh v3
tail -n 20 logs/autodl_shutdown_mamba_robust_v3_clean.log
tail -f logs/ntu60_xsub_joint_mamba_robust_v3_clean.log
```

确认配置对照 PASS、训练 joint-mask probability=0.0、监控 Armed、Training epoch: 1。
已有正式目录/日志时拒绝覆盖。只有导入阶段失败且没有训练产物，才可在修复
缺包后使用 v3 --retry-startup；它不能用于恢复已经开始的训练。

## 训练后原协议推理

GPU 重新开机、激活虚拟环境并设置线程后：

```bash
python -u tools/evaluate_occlusion.py \
  --baseline-dir work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128 \
  --mamba-dir work_dir/ntu60/xsub/ctrgcn_mamba_robust_v3_clean_joint_fast128 \
  --output-dir work_dir/ntu60/xsub/occlusion20_robust_v3_clean_seed2026 \
  --rate 0.2 --mask-seed 2026 \
  --batch-size 128 --num-worker 2 --device 0 \
  2>&1 | tee logs/occlusion20_robust_v3_clean_seed2026.log
```

预期 same_training_feeder_args=true。与标准组合比较净收益，并与已保存的
V2 无增强结果比较是否进一步改善。需要两两直接复算时评估器也接受 V2
作为 baseline-dir；不自动启动额外推理或其他训练实验。

审计工具同样支持 V3，并验证可见性池化只使用本次主干特征和输入坐标。
V3 的 gate_diagnostics 输出 residual_write_coefficient / signed_residual_scale，
不将有效写入系数误标为 [0,1] 的可靠性概率。
