# 第二版：局部历史补偿状态写入，保留干净输入路径

第一版实测：干净 88.78%、20% 遮挡 80.43%。原版标准组合为 88.92% / 79.55%。
第一版的模型文件、配置、权重目录和推理结果保持原样。第二版是一次新实验，
目标是超过 80.43% 并恢复约 88.92% 的干净精度，不能在训练前保证达到。

## 改动依据与算法

第一版同时缩放状态写入和整个 Mamba 输出，而且使用干净训练，门控没有接触
人工遮挡。前者可能过度削弱可用特征，后者可能限制门控对遮挡的学习；
这是代码层面的改进假设，尚未由第一版的内部状态/门控观测证实为掉点原因。

第二版做如下明确改动：

1. **完整观测时门控恒等**：依旧使用轻量一致性统计量，令
   `r_t = 1 - (1-q_t) * sigmoid(score_t)`，其中 q 是输入可观测关节比例。
   q=1 时 r 精确等于 1；q=.8 时 r 位于 [.8, 1]，不会像第一版被直接限制到
   .8 以下。门控初始化在非饱和区，配合训练遮挡学习缺失条件下的调节。
2. **局部历史替代部分状态写入**：保留 Mamba 标准卷积和 delta/B/C 投影，
   用前 3 个 token 的均值作为局部候选 `p_t`（首时刻没有历史，取当前 token）。
   `w_t = r_t u_t + (1-r_t) p_t`，随后用官方 CUDA 扫描计算
   `h_t = exp(delta_t A) h_(t-1) + delta_t B_t w_t`。
   与仅抑制新信息相比，提供不依赖递归状态的局部替代写入。
3. **保留直通与读出幅度**：官方 scan 的 D/z 在外部重接为
   `(C_t h_t + D u_t) * SiLU(z_t)`，其中 D 的输入仍是原始 u，不是 w。
   不再把整个 Mamba 输出乘一次 r；仅精确关闭为零时触发硬兜底。
4. **多尺度局部残差**：深度因果卷积 kernel=3、dilation=1/2 两分支融合，
   接 256→32→256 瓶颈。残差系数从 0 开始，通过 tanh 限制幅度。

最终为 `tokens + Mamba_readout + tanh(alpha) * local_features`。
不添加新的 Mamba 层，不增大主干，不增加注意力模块。
总参数 1,911,186，比原版增加 **26,242（0.026242M，约 1.39%）**，
构造时仍硬性检查 0.2M 上限。

`use_reliability` 和 `use_local_compensation` 仍独立控制：

| 可靠性 | 局部补偿 | 行为 |
| --- | --- | --- |
| false | false | 原版标准组合，旧参数键和计算路径相同 |
| true | false | 调节当前写入，无局部替代、无外部局部残差 |
| false | true | 标准 Mamba，加独立多尺度局部残差 |
| true | true | 局部历史替代写入，加局部残差 |

局部历史来自当前已观测特征，不是不可获得的干净状态，也不声称恢复了缺失
关节坐标。原始全零关节仍只是缺失启发式，复用第一版的空人物/补零排除规则。
门控不会把不同人物的状态混在一起。主干仍是原来的离线 CTR-GCN，不能声称
整个系统是流式因果模型。

## 本轮训练策略确实改变，需要明确区分归因

每条训练样本以 0.25 概率随机缺失 1..5 个关节，整段和两个人物共享这些关节；
约 75% 样本保留干净输入。训练掩码只使用训练随机数，与测试 seed 2026 独立。
一次样本仍只做一次前向和一次交叉熵，没有双分支教师网络或两倍训练计算。

`joint_mask_prob` 默认 0，不消耗额外随机数，旧训练流程保持原样；
对 test split 或 Bone/Motion 启用此增强会直接报错。测试遮挡仍由现有
evaluate_occlusion.py 生成，完全不变。

遮挡模拟本身是已有训练策略，不能把它包装为本工作的状态空间创新。
参见 [骨架识别中的遮挡训练研究](https://doi.org/10.1145/3581783.3612170)。
V2 相对原先干净训练标准组合的提升（如果出现）是“模型＋训练策略”的总效果。
要单独说明架构贡献，应使用同增强的标准组合或同训练条件消融。

已附可选公平对照配置 `ctrgcn_mamba_maskaug_fast128.yaml`：除原版模型与独立
输出目录外，和 V2 的数据、遮挡增强、优化器、epoch、batch 完全相同。
本次启动命令**只训练 V2 一组**，不会额外启动该对照或批量调参。
关闭 V2 配置的两个模型开关、保留增强，也等价于该标准模型对照。

保持其余设置：从头训练、65 epoch、batch/test batch 128、SGD、lr .1、
warm-up 5、step 35/55、统一 weight decay .0004、AMP、worker 2、seed 1。
不启用失败的 no-weight-decay 校准组，不热启动旧权重。
选择 checkpoint 仍按干净评估最优 epoch，不在遮挡测试集上筛权重。

## 拉取并验证真实 CUDA

```bash
cd /root/autodl-tmp/ctr-gcn
git pull --ff-only origin main
source /root/autodl-tmp/venv-ctr-infer/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
set -o pipefail
python -u tools/check_robust_mamba.py --variant v2 \
  2>&1 | tee logs/check_robust_mamba_v2.log
```

最后必须 PASS。检查参数 1,911,186、额外 26,242、r=1 与标准融合算子的
FP32/AMP 一致性、有限梯度以及真实 SGD 更新。失败时不开始训练。
本地 CPU 参考算子的测试不能代替这项真实 CUDA 验证。

## 测速后启动一组正式训练，结束自动关机

```bash
python -u main.py \
  --config config/nturgbd-cross-subject/ctrgcn_mamba_robust_v2_fast128.yaml \
  --phase benchmark \
  --work-dir work_dir/ntu60/xsub/benchmark_mamba_robust_v2 \
  --device 0 --seed 1 \
  2>&1 | tee logs/benchmark_mamba_robust_v2.log
```

测速使用正式的训练增强和真实反向计算，不保存实验权重。训练时长仍需实测，
两小时目标要为数据加载和保存预留余量。正式训练重新初始化。

```bash
bash tools/train_robust_autodl.sh v2
tail -n 20 logs/autodl_shutdown_mamba_robust_v2.log
tail -f logs/ntu60_xsub_joint_mamba_robust_v2.log
```

确认监控日志出现预检成功和 Armed。后台训练不受 SSH 断开影响。
完成后沿用已验证成功的 AutoDL 关机监控，先校验结果、复制 best.pt、记录
SHA256、保存 result_summary.json 和日志、同步磁盘，再请求关机。
已有 V2 正式目录或日志时启动器拒绝覆盖，不影响第一版的结果。
训练/结果验证失败则保留错误日志、保持实例运行。

## 训练后配对推理

重新开机并激活同一环境后执行：

```bash
python -u tools/evaluate_occlusion.py \
  --baseline-dir work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128 \
  --mamba-dir work_dir/ntu60/xsub/ctrgcn_mamba_robust_v2_joint_fast128 \
  --output-dir work_dir/ntu60/xsub/occlusion20_robust_v2_seed2026 \
  --rate 0.2 --mask-seed 2026 \
  --batch-size 128 --num-worker 2 --device 0 \
  2>&1 | tee logs/occlusion20_robust_v2_seed2026.log
```

推理会提示训练增强不同，在 summary.json 中记录两组 train_feeder_args，
避免把“算法＋训练策略”的差值误写成纯架构贡献。比较目标保留原版
88.92% / 79.55%，也要报告 V1 的 88.78% / 80.43%。不能只挑较弱模型作对照。
无论是否达标都保留本轮结果，不默认增加训练次数或测试遮挡种子。

实现基于 [官方 Mamba-1](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba_simple.py)
和 [selective_scan 接口](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/selective_scan_interface.py)。
状态写入混合、门控与局部补偿是本仓库的实验设计，精度提升不自动证明内部
状态污染减轻；若需要该机制结论，还需单独定义和验证状态偏差。
