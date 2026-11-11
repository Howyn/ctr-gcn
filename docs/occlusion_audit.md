# 20% 遮挡结果排查

V2 单次实测干净 89.40%、遮挡 87.46%，标准组合为 88.92% / 79.55%。
增益大不是错误的证明，也不能跳过验证。以下工具只读已有结果与权重，
不训练、不更新 BN、不重选权重、不覆盖原始推理结果。

## 先核对已经保存的证据（无需 GPU）

```bash
cd /root/autodl-tmp/ctr-gcn
git pull --ff-only origin main
source /root/autodl-tmp/venv-ctr-infer/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u tools/audit_occlusion.py \
  --result-dir work_dir/ntu60/xsub/occlusion20_robust_v2_seed2026 \
  --artifacts-only
```

检查保存的 16487 条测试标签与 NPZ 的 y_test 完全一致；x_test / x_train
分别为 (16487,300,150) / (40091,300,150)，但不会加载大型训练数组。
逐条核对五个不同关节、seed/hash、每个模型四组分数的形状和有限性，
从分数独立重算 Top1 并核对 CSV/JSON、样本顺序、标签、mask、权重 SHA256。
输出打乱标签后的精度作为统计负对照。若分数存在最大值并列，沿用原评测的
argsort tie 规则并报告数量；不会把 argmax 的并列规则差别误报成计算错误。

## 在 GPU 上检查模型实际收到的数据

```bash
set -o pipefail
python -u tools/audit_occlusion.py \
  --result-dir work_dir/ntu60/xsub/occlusion20_robust_v2_seed2026 \
  --samples 16 --num-worker 2 --device 0 \
  --report logs/audit_occlusion20_v2.json \
  2>&1 | tee logs/audit_occlusion20_v2.log
```

这是跨整个测试集均匀抽取的 16 个样本的前向抽查，不是再算一次完整准确率。
报告已存在时请换一个报告文件名；原始结果不被覆盖。程序验证：

1. worker=0 与 worker=2 的测试样本、标签、索引逐元素一致。
2. 用独立循环实现 mask，与正式评测的 scatter 实现逐元素比较；选中关节
   的所有坐标、帧、人物为零，保留关节不变，干净输入未被原地修改。
   20% 是关节索引比例，不是非零数值的比例；空人物和原点关节本来可能为零。
3. 挂钩检查模型根输入、data_bn 输入、l10 特征到 temporal_norm 的路径，
   以及局部补偿分支是否收到当前 forward 的 normalized 特征。
   可靠性评估器的 tokens 和可观测性也必须来自本次输入，并输出干净/遮挡
   两种情况下 q/r 的 min/mean/max；空人物的 q=1 不能误读为遮挡未生效。
4. 重新严格加载每个 checkpoint，第一次就做遮挡前向；随后先做干净前向
   再做相同遮挡前向，两次输出应一致，检查是否借用了之前的干净缓存。
5. 改写那些即将被遮挡的坐标，再遮挡后输入必须完全一致；在改写输入的
   一次前向之后，遮挡输出仍应一致，进一步检查跨调用缓存。
6. 输入全零时，同一批的相同输入应产生相同输出。全零输入的预测不必是
   均匀概率，训练得到的偏置/BN 可以给出固定类别；不能要求 logits 也全零。
7. 所有参数、BN buffers 前后完全相同，所有子模块为 eval，无梯度。
8. 抽查 logits 与保存分数的差值、预测一致率；不同 GPU / AMP batch 大小
   可以有舍入误差，logits_close=false 或预测不一致需进一步在原 batch/GPU 下
   复算。它们是报告项，不能因为其他 PASS 就忽略。

mask / 路由 / 参数被修改等检查失败会报错。报告里的 logits_close 为 false
不是自动判定泄露，但也是尚未解释的差异，不能作为完全通过。

## 零值、补偿与泄露的边界

V2 的 observed_joint_fraction 只从本次输入计算关节是否全零，不读取标签、
测试 seed、关节列表或原始干净数组。用可见的缺失标记判断可靠性属于明确的
零填充协议，不等同于获得缺失坐标。不过它依赖人工零值特征，不代表对所有
现实遮挡、噪声坐标都成立；空人物/原点造成的可观测性误判也应记录。

局部历史来自本次已遮挡输入的 backbone/Mamba 特征。图卷积和 BN 后缺失
位置出现非零特征可以是邻域传播和偏置，并不能据此判定恢复了真实坐标。
整个 CTR-GCN 是离线模型，不能把局部历史的因果运算写成整个网络流式因果。

## 仍需人工核对的项目

- Feeder 的 train/test 分别只读 x_train/y_train 与 x_test/y_test；训练时不能
  用测试标签做 loss 或将测试样本加入训练。现有代码符合这条，但一次运行后的
  文件不能证明历史训练过程没有被改动。
- 保留的原始序列 ID / performer 元数据中，NTU60 X-Sub 的训练受试者集合与
  测试集合应无交集。NPZ 仅含四个数据/标签数组时，无法仅凭它恢复受试者身份。
  检查 seq_transformation.py 的 get_indices 与当时生成数据所用的元数据；
  样本数正确并不单独证明不存在重复序列或错误划分。
- 最优 epoch 仍按干净精度选，不反复用本次遮挡测试集挑权重或调参；此项目
  原训练过程确实每轮评测 test 并从中选择最优，这是既有协议，也应如实披露。
- V2 训练时 25% 样本随机遮挡 1..5 关节，而旧标准组合没有。比较说明整体方案
  效果，不能单独证明状态空间模块贡献。待排除实现错误后，公平的算法对照为
  ctrgcn_mamba_maskaug_fast128.yaml；不需要为排查工具重新训练。
- 固定 seed 的 mask 独立于标签；训练 mask 不读取测试 mask，不从文件导入
  seed2026 的掩码。若后续验证跨 seed，应另存结果；不覆盖当前证据。

所有检查通过也只是排除了列出的具体错误，不能作为“没有任何泄露”的证明。
