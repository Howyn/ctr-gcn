# 可靠性感知状态写入与局部时序补偿

这是新的算法实验，尚无精度或训练时长结论。主对照是原版
`ctrgcn_mamba_fast128.yaml`（干净约 88.92%，20% 遮挡约 79.55%），
不使用校准版的 78.14% 作为主要比较起点。纯 CTR-GCN 的结果继续保留。

## 算法与边界

CTR-GCN 主干、关节池化、Mamba-1 参数大小和分类头不变。64 帧输入经主干后，
每个人物得到 16 个 256 维 token。每个人物分别扫描，不跨样本/人物共享状态。

1. **感知输入可靠性**：256→32 的投影计算相邻余弦距离、归一化一阶差分、
   二阶运动差分和幅度变化，联合关节可观测比例及其变化，共 6 个统计量。
   `6→16→1` 网络输出分数，最终 `r = observed_fraction * sigmoid(score)`。
   差分大不直接等于异常；统计量由分类损失联合学习，没有状态污染标签。
2. **控制状态写入**：沿用官方 Mamba-1 的卷积、参数投影与 CUDA selective scan，
   在扫描前令 `B_t <- r_t B_t`。对应官方实现的递推为
   `h_t = exp(delta_t A) h_(t-1) + delta_t r_t B_t u_t`。
   不修改 delta/A/C/D，也不冻结历史状态；历史仍按标准规律衰减。
3. **局部补偿**：归一化 token 经因果深度时序卷积（核 3）和
   `256→32→256` 投影。该路径不依赖递归状态。
4. **融合**：`output = tokens + r * mamba_with_controlled_writes + alpha * local`。
   `alpha` 从 0 开始学习。`r=0` 时是主干特征加局部时序残差；并不保证错误
   的可靠性估计能自动被纠正。保留路径也不等于训练后干净精度必定不下降。

可观测比例只来自实际输入坐标，不读取测试遮挡 ID、标签或原始完整样本。
xyz 全零只是缺失启发式：原点关节也可能为零，因此它不是传感器置信度。
全空人物、全零帧不作为 100% 遮挡计入比例，人物池化仍沿用原版规则。
时间维通过有效帧加权池化对齐 token 长度，不把补零数量当作遮挡比例。
当前协议只遮挡 5/25 个关节；这不是针对完全消失人物的检测器。
仅靠时间差分难以检测整段固定关节缺失，因此显式加入可观测比例。

关闭两个开关，参数名称、初始主干/Mamba 权重和前向路径与原版一致。
可靠性启用时使用官方 `causal_conv1d_fn + selective_scan_fn` 显式路径，
仍是并行 CUDA 扫描，不使用 Python 逐帧训练循环。它不再使用完整 Mamba
融合快路径，所以实际速度需要单独测量，不能由参数数目推断。
新增局部卷积和统计量是因果运算，但原有 CTR-GCN 主干含非因果时间卷积，
本模型仍是离线序列分类器，不声称是流式因果识别器。

参考：[官方 Mamba-1 实现](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba_simple.py)、
[官方扫描接口](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/selective_scan_interface.py)。
本方案的组合、门控和局部补偿是本仓库的新实验设计，并非官方 Mamba 原有功能。

## 参数与开关

| 新增部分 | 参数数 |
| --- | ---: |
| 一致性投影 | 8,192 |
| 可靠性网络 | 129 |
| 局部深度时序卷积 | 768 |
| 局部降维与升维 | 16,384 |
| 残差系数 | 1 |
| 新增合计 | 25,474 |
| 完整 NTU60 模型 | 1,910,418 |

构造模型时硬性检查新增参数不超过 200,000。模型参数开关互相独立：

| use_reliability | use_local_compensation | 对应实验 |
| --- | --- | --- |
| false | false | 原版标准组合 |
| true | false | 只调节状态写入与 Mamba 输出 |
| false | true | 标准 Mamba 加局部补偿 |
| true | true | 完整方案，当前只训练这一组 |

可靠性关闭时取标准 Mamba 路径，局部分支仍能独立工作。未来消融需各自独立
训练、独立 work_dir，不能只在完整模型推理时关开关就当作训练消融。

## 云端准备与真实 CUDA 检查

沿用已验证正常的推理虚拟环境，不重新安装 PyTorch/Mamba，不重做数据。
如果是新实例，应先确认这个环境也迁移到了相同位置。

```bash
cd /root/autodl-tmp/ctr-gcn
git pull --ff-only origin main
source /root/autodl-tmp/venv-ctr-infer/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
set -o pipefail
python -u tools/check_robust_mamba.py 2>&1 | tee logs/check_robust_mamba.log
```

必须看到最后的 PASS。该检查验证 NumPy 双向转换、CUDA 可用、参数预算、
`r=1` 显式扫描与标准融合 Mamba 的 FP32/AMP 输出一致性，并要求实际完成
两次有限梯度更新。数据同时包括干净和 20% 零关节样本，要求新模块也有更新。
初始 AMP 溢出由 GradScaler 校准；持续失败不能跳过，必要时用
`python tools/check_robust_mamba.py --precision fp32` 定位。
检查只使用一次性随机数据和权重，不启动实验训练，不覆盖已有结果。

## 先测速，再训练

```bash
python -u main.py \
  --config config/nturgbd-cross-subject/ctrgcn_mamba_robust_fast128.yaml \
  --phase benchmark \
  --work-dir work_dir/ntu60/xsub/benchmark_mamba_robust \
  --device 0 --seed 1 \
  2>&1 | tee logs/benchmark_mamba_robust.log
```

测速使用真实数据和训练反向过程，但不会保存实验权重。参考原版实测约
0.299 s/batch；GPU、负载或实例改变时，旧值不能作为严格同机速度对照。
两小时是运行目标，不能承诺。若估计接近两小时，应为加载、日志和保存预留时间。
benchmark 与训练 work_dir 分开，正式训练从头重新初始化。

完整方案配置与原版标准组合相同：干净训练数据、原有随机旋转/crop、65 epoch、
batch/test batch 128、SGD、lr 0.1、warm-up 5、step 35/55、weight decay 0.0004、
AMP、worker 2、seed 1。保持原版统一 weight decay，不启用校准版分组；
不加入遮挡增强、额外损失、预训练或热启动。
因此本轮只能检验该算法在干净训练条件下能否提高遮挡表现，不能预设增益。

## 后台训练、验证保存结果、自动关机

CUDA 检查与测速通过后，在仍激活虚拟环境的终端执行一次：

```bash
bash tools/train_robust_autodl.sh
```

启动器首先检查关机脚本（不关机），并验证依赖。已有正式 work_dir 或同名训练/
监控日志时拒绝覆盖，避免重复运行。随后用当前环境的 Python 后台启动训练和
已修复的 AutoDL 监控。终端退出不会中断 nohup 进程。

```bash
tail -n 20 logs/autodl_shutdown_mamba_robust.log
tail -f logs/ntu60_xsub_joint_mamba_robust.log
```

监控日志应出现预检成功和 `Armed: waiting for training PID ...`。
仅在 65 轮训练结束、日志和最佳/最后 checkpoint 等结果通过验证后，监控才会
复制最佳权重为 `best.pt`，保存控制台日志和 `result_summary.json`，校验 SHA256、
同步磁盘，然后请求关机。使用已在用户实例成功关机的 ENOEXEC/bash 回退修复。
训练或结果验证失败会留下日志并保持实例运行；最终关机状态以 AutoDL 控制台为准。
本启动器不自动执行任何额外遮挡实验。

结果目录：`work_dir/ntu60/xsub/ctrgcn_mamba_robust_joint_fast128`。
保存到云盘不等于异地备份，后续仍可按已有流程下载 work_dir/logs。

## 训练后：与原版标准组合做一次配对推理

重新开机，激活同一环境，继续使用 seed 2026 和原来的固定 5/25 关节协议：

```bash
cd /root/autodl-tmp/ctr-gcn
source /root/autodl-tmp/venv-ctr-infer/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
set -o pipefail
python -u tools/evaluate_occlusion.py \
  --baseline-dir work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128 \
  --mamba-dir work_dir/ntu60/xsub/ctrgcn_mamba_robust_joint_fast128 \
  --output-dir work_dir/ntu60/xsub/occlusion20_robust_seed2026 \
  --rate 0.2 --mask-seed 2026 \
  --batch-size 128 --num-worker 2 --device 0 \
  2>&1 | tee logs/occlusion20_robust_seed2026.log
```

这里输出标签 `baseline` 是原版 CTR-GCN+Mamba，`mamba` 是改进模型；
summary.json 内记录实际模型路径、参数和权重来源。两个模型使用完全相同的
样本、遮挡和预处理，分别报告干净/遮挡精度。选权重仍按已有干净训练评估的
最佳 epoch，不在遮挡测试集上筛权重。

本轮以干净约 88.92% 和遮挡 79.55% 为主要对照；±0.3 不再作为相对纯 CTR-GCN
的前置门槛。精度提升支持指定协议下识别性能改善；若要断言内部状态污染减轻，
还需要定义并观察对应状态，不能把门控分数直接当成已验证的污染真值。

## 本地验证范围

单元测试用测试专用的 CPU 参考扫描验证：单位门控下的输出/梯度一致性、只修改
B 写入项、两开关独立、参数预算、局部初始零残差、门控关闭时的兜底路径、
空人物/补零处理、无未来输入泄漏，以及配置保持原版训练条件。
生产模型没有 CPU 扫描替代品。真实 CUDA/AMP 是否通过，以云端检查输出为准。
