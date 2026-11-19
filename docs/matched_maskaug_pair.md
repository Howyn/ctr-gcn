# 同款遮挡增强下的 Baseline 与 V2 对照

这里的 Baseline 是 **CTR-GCN＋标准 Mamba**，V2 是 **CTR-GCN＋改进 Mamba V2**。两轮均从头训练，使用此前 V2 增强版本的训练方案：每次读取训练样本时，以 25% 的概率随机选择 1～5 个关节，将其整个时间序列、全部人物和三维坐标置零，其余 75% 样本保留原有输入。随机旋转、时间裁剪等原有处理不变。

两轮采用相同的数据、训练与测试预处理、优化器和学习率计划、65 个 epoch、训练及测试 batch size 128，以及种子 1。增强仅用于训练；测试阶段仍使用原有干净测试输入。后续 20% 遮挡推理应继续使用共同的固定测试掩码。

## 配置和输出

| 实验 | 配置 | 结果目录 |
| --- | --- | --- |
| Baseline＋增强 | `config/nturgbd-cross-subject/ctrgcn_mamba_maskaug_pair_fast128.yaml` | `work_dir/ntu60/xsub/ctrgcn_mamba_maskaug_pair_joint_fast128` |
| V2＋增强 | `config/nturgbd-cross-subject/ctrgcn_mamba_robust_v2_maskaug_pair_fast128.yaml` | `work_dir/ntu60/xsub/ctrgcn_mamba_robust_v2_maskaug_pair_joint_fast128` |

配置分别复制此前的标准 Mamba 增强配置和 V2 增强配置，仅更改输出目录。旧实验权重和结果保留。

## 云端启动

在已安装训练依赖与官方 Mamba CUDA 扩展的环境中执行：

```bash
cd /root/autodl-tmp/ctr-gcn
git pull --ff-only origin main
source /root/autodl-tmp/venv-ctr-infer/bin/activate
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p logs
python tools/train_maskaug_pair.py --check-only
nohup python -u tools/train_maskaug_pair.py \
  >> logs/train_maskaug_pair.log 2>&1 < /dev/null &
echo "串行训练进程 PID: $!"
tail -f logs/train_maskaug_pair.log
```

训练顺序为 Baseline＋增强、V2＋增强。第一轮结束后保存并核验结果，再启动第二轮；两轮正常完成且结果保存校验通过后调用 AutoDL 关机命令。无需分别启动旧版单实验关机监控，否则它可能在第一轮结束后请求关机。运行时间取决于实际显卡与服务器负载，两轮合计不能按单轮耗时估算。

若启动检查或训练报错，应先查看日志并处理原因，避免重复启动。后续评估这两个新目录，即可比较相同增强条件下的结构差异；也可结合已有的两组无额外遮挡增强结果，形成“模型结构×训练增强”的对照。

## 日志与保存结果

- 队列进度及关机记录：`logs/train_maskaug_pair.log`。
- 第一轮训练日志：`logs/baseline_maskaug_pair.log`。
- 第二轮训练日志：`logs/v2_maskaug_pair.log`。
- 两轮各自保存 `best.pt`、原始逐轮权重、最终最佳模型预测、`config.yaml`、`training_console.log` 和 `result_summary.json`。
- 队列摘要：`work_dir/ntu60/xsub/maskaug_pair_seed1/pair_summary.json`，含两轮结果及最佳权重 SHA-256。

脚本启动前检查 `main.py` 所需训练依赖（包括 scikit-learn）、官方 Mamba、CUDA 与 NumPy/PyTorch 转换；检查失败时不会启动第一轮。训练失败、最终文件缺失或保存校验失败时停止队列并保留实例，便于排错。

`--check-only` 仅检查配置和输出路径，不访问 GPU，也不关机。正常训练调用会自动完成依赖和关机命令预检查。若已完成第一轮而第二轮尚未创建任何结果，可以用 `--resume` 重新校验并跳过第一轮；该选项不会恢复中途失败的训练，也不会覆盖已有文件。

两轮均使用相同增强分布与种子，模型初始化的随机数消耗不同，因此不保证每次训练取样得到逐位相同的遮挡。测试阶段比较时仍使用相同的固定测试掩码。本队列执行训练及原有干净测试；20% 遮挡配对推理在训练结束后另行运行。
