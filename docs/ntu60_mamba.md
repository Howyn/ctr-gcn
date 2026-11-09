# CTR-GCN + standard Mamba-1: NTU60 X-Sub Joint

This is an explicit combination baseline, rather than a reproduction of a
particular published CTR-GCN/Mamba fusion architecture.

## Model

- Retain all ten CTR-GCN blocks, their temporal convolutions and the classifier.
- The backbone emits `(N, 2, 256, 16, 25)` features for a 64-frame input.
- Average the 25 joints and form a separate `(16, 256)` sequence for each person.
- Apply one pre-LayerNorm residual block: `x + Mamba(LayerNorm(x))`.
- Use the official `mamba_ssm.Mamba` (Mamba-1): `d_model=256`, `d_state=16`,
  `d_conv=4`, `expand=2`, unidirectional, fast CUDA path enabled.
- Average time and people, then use the existing classifier.

No local replacement SSM, bidirectional variant, attention, or extra MLP is used.
Mamba's specialized parameter initialization is retained. The backbone's state
dictionary keys and baseline classifier outputs are preserved by the feature
extraction refactor.

## Experiment settings

`config/nturgbd-cross-subject/ctrgcn_mamba_fast128.yaml` matches the previous
`fast128.yaml` for data, augmentations, 64-frame input, train/test batch size 128,
65 epochs, SGD, momentum 0.9, Nesterov, base LR 0.1, weight decay 0.0004,
five warmup epochs, LR steps 35/55, AMP/TF32, workers 2 and CPU threads 1.
Use seed 1 and train from scratch; do not initialize from the completed baseline
checkpoint. Original SGD parameter grouping is retained for a controlled
comparison. Added Mamba parameters receive the same optimizer settings.

Outputs go to `work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128`, separately from
the 89.64% baseline. Record accuracy and actual runtime; neither an improvement
nor a two-hour runtime can be assumed from the baseline alone.

## AutoDL: pull and install once

```bash
cd /root/autodl-tmp/ctr-gcn
git pull --ff-only origin main
python tools/install_mamba.py
```

The installer targets Linux x86_64, the existing PyTorch 2.5.1 CUDA 12 build,
and supported CPython versions. It detects the actual C++ ABI and installs the
matching official release wheels for `mamba-ssm==2.2.4` and
`causal-conv1d==1.5.0.post8`, with no PyTorch replacement and no local CUDA build.
Runtime dependencies are listed in `requirements-mamba.txt`. Official wheels
are downloaded into the ignored `.wheelhouse/` cache using resumable requests
and retries. The installer checks response lengths, wheel metadata and every
archive member's CRC before passing a local file to pip. It reuses verified
cached files on later runs. If a proxy ignores Range requests, the file is
restarted safely rather than appended twice. No third-party wheel mirror is used.

If ordinary dependencies installed successfully but a GitHub wheel was truncated
(for example, pip reports an invalid wheel before the full download completes),
enable AutoDL's academic proxy in the current terminal and retry only the wheels:

```bash
source /etc/network_turbo
python tools/install_mamba.py --skip-runtime-deps
```

Interrupted `.part` files remain available for the next retry. Fully downloaded
but corrupt files are discarded and fetched again. If GitHub release downloads
still fail, retain the exact error and resolve network access; do not replace
these modules with an unrelated pure-Python package. AutoDL's proxy can slow
ordinary pip mirrors, which is why this recovery command skips those dependencies.

Installation ends by running `tools/check_mamba.py`: the actual hybrid model's
CUDA forward, AMP backward, nonzero Mamba gradients and SGD update must pass.
The script can be rerun independently:

```bash
python tools/check_mamba.py
```

AMP smoke checks allow bounded loss-scale calibration: initial invalid gradients
cause GradScaler to skip the update and reduce its scale, as in training. The
check requires two finite SGD updates, finite parameters and a nonzero gradient
and actual weight change in Mamba. It prints the first invalid gradient names
on overflow. Repeated overflows or non-finite forward values still fail; they
are not silently accepted. The disposable smoke updates use the first warmup
epoch's LR 0.02; this does not change the training config.

If AMP cannot pass, use a separate full-precision diagnostic run:

```bash
python tools/check_mamba.py --precision fp32
```

FP32 passing is diagnostic information, not approval of the AMP training path.
The official CUDA checks still have to pass in the intended precision before
benchmarking. The original training architecture, optimizer and AMP scaler
defaults are unchanged by these smoke-check diagnostics.

## Benchmark before the full run

```bash
python -u main.py \
  --config config/nturgbd-cross-subject/ctrgcn_mamba_fast128.yaml \
  --phase benchmark \
  --work-dir work_dir/ntu60/xsub/benchmark_ctrgcn_mamba128 \
  --seed 1
```

The benchmark includes warmup and real augmented data. The full-run projection
excludes initialization and file-writing overhead. It saves no checkpoint.

## Train and automatically shut down after verified results

Run this block once, after the smoke test and benchmark succeed:

```bash
cd /root/autodl-tmp/ctr-gcn
mkdir -p logs

nohup python -u main.py \
  --config config/nturgbd-cross-subject/ctrgcn_mamba_fast128.yaml \
  --device 0 --seed 1 \
  > logs/ntu60_xsub_joint_mamba_fast128.log 2>&1 < /dev/null &
TRAIN_PID=$!

nohup python -u tools/autodl_shutdown.py \
  --pid "$TRAIN_PID" \
  --config config/nturgbd-cross-subject/ctrgcn_mamba_fast128.yaml \
  --work-dir work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128 \
  --training-log logs/ntu60_xsub_joint_mamba_fast128.log \
  > logs/autodl_shutdown_mamba.log 2>&1 < /dev/null &

echo "Training PID: $TRAIN_PID; shutdown watcher PID: $!"
```

Confirm `Armed: waiting for training PID ...` in `logs/autodl_shutdown_mamba.log`.
The watcher checks complete results before writing `best.pt`,
`result_summary.json` and a console-log copy, syncing, and requesting shutdown.
On failure it records the error and leaves the instance running.

```bash
tail -f logs/ntu60_xsub_joint_mamba_fast128.log
```

## Sources and validation scope

- [Official Mamba-1 module](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba_simple.py)
- [Official Mamba release wheels](https://github.com/state-spaces/mamba/releases/tag/v2.2.4)
- [Official causal-conv1d wheels](https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.5.0.post8)

CPU tests check the backbone regression, person/time layout and gradient flow
with an explicitly isolated test double. They do not validate the official CUDA
kernels; that requires the AutoDL smoke test above. No accuracy for the hybrid
is claimed before full training and evaluation.
