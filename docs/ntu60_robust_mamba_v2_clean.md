# 回到原计划：仅比较 V2 模块，不增加训练关节遮挡

当前只补这一组，不自动启动标准模型遮挡增强对照、模块消融或其他遮挡比例。
纯 CTR-GCN、标准 Mamba 和原先的 20% 遮挡推理结果均继续使用，无需重跑。

V2 旧实验的 89.40% / 87.46% 使用了 25% 概率、1..5 关节的训练遮挡增强。
它只能标记为“V2 + 训练遮挡增强”，不能当作保持原训练条件的结构改进结果。
训练增强可能贡献很大，但当前数据无法给出占比，也不能断言全部提升来自增强。

新增配置：`config/nturgbd-cross-subject/ctrgcn_mamba_robust_v2_clean_fast128.yaml`。
V2 模型实现与之前相同，只关闭新增训练遮挡，并使用独立输出目录。
原有随机旋转、裁剪不变；seed=1，65 epoch，batch/test batch=128，SGD，
lr=.1，warmup=5，steps=[35,55]，统一 weight decay=.0004，AMP/TF32 不变。
不加载旧权重，从头训练。新增参数量仍为 26,242。

## 启动这一组训练，结束后保存结果并关机

```bash
cd /root/autodl-tmp/ctr-gcn
git pull --ff-only origin main
source /root/autodl-tmp/venv-ctr-infer/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
bash tools/train_robust_autodl.sh v2-clean
```

启动器先读取**已经完成的标准 Mamba 实验实际保存的 config.yaml**，检查
训练/测试输入、seed、优化器、学习率、epoch、batch、AMP 等是否一致，检查
双方训练遮挡概率均为 0；不一致时直接退出，不启动训练。
然后使用既有的关机预检查、结果核验和关机监控。正式实验目录或日志已存在
时拒绝覆盖，避免误重启。V2 CUDA 路径已经验证，本轮无需重新测速。

```bash
tail -n 20 logs/autodl_shutdown_mamba_robust_v2_clean.log
tail -f logs/ntu60_xsub_joint_mamba_robust_v2_clean.log
```

关机监控应显示 `Armed: waiting for training PID ...`。最终保存/校验失败会
保持开机并记录原因，不会把失败实验当成成功结果。

### 更换实例后训练缺包或导入失败

推理环境能运行 Mamba 不代表 main.py 的 sklearn / tensorboardX / torchlight
等训练入口依赖齐全。启动器现在先执行 main.py --help 和 NTU feeder 导入，
通过后才创建训练和关机监控进程。安装训练依赖时保持 NumPy=1.26.4，不安装
旧 requirements.txt，也不更换 PyTorch/CUDA：

```bash
python -m pip install -r requirements-train.txt -i https://mirrors.aliyun.com/pypi/simple
python -m pip install --no-deps --no-build-isolation -e ./torchlight
bash tools/train_robust_autodl.sh v2-clean --retry-startup
```

--retry-startup 只处理有明确 ModuleNotFoundError、未出现 Training epoch、
且实验目录仅含监控锁的失败尝试。旧日志/锁目录会移到 logs/failed_startups，
不删除；仍有训练进程或工作目录含配置/权重等其他文件时拒绝归档或重启。
它会等待旧关机监控释放锁，最多 20 秒；不能用于恢复已开始的训练。

## 训练完成后只做原来的干净 / 20% 双场景推理

重新 GPU 开机、激活同一虚拟环境后运行：

```bash
cd /root/autodl-tmp/ctr-gcn
source /root/autodl-tmp/venv-ctr-infer/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
set -o pipefail
python -u tools/evaluate_occlusion.py \
  --baseline-dir work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128 \
  --mamba-dir work_dir/ntu60/xsub/ctrgcn_mamba_robust_v2_clean_joint_fast128 \
  --output-dir work_dir/ntu60/xsub/occlusion20_robust_v2_clean_seed2026 \
  --rate 0.2 --mask-seed 2026 \
  --batch-size 128 --num-worker 2 --device 0 \
  2>&1 | tee logs/occlusion20_robust_v2_clean_seed2026.log
```

测试仍为每条序列固定缺失 5/25 关节、全部帧和两个人物共享、xyz 置零。
两模型共享相同测试输入与 mask；新结果应记录 `same_training_feeder_args=true`。

新实验与标准组合 88.92% / 79.55% 比较，回答同训练条件下的结构收益；
新实验与旧 V2 增强版 89.40% / 87.46% 比较，初步估计增强在 V2 下的效果。
单次训练差异仍含随机波动，不能把两者差值当成完全确定的因果占比。
先取得这一组结果，再决定下一步；不预设它能维持增强版的 87.46%。
