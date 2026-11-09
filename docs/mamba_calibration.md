# One controlled standard-Mamba calibration trial

## Purpose and limit

The first standard combination reached 88.9064% clean Top-1, below the 89.6403%
CTR-GCN reference. The project's 0.3 percentage-point lower tolerance is about
89.34%; 89.4% is a conservative operational target. This trial is an attempt to
improve the training recipe, not a promise that the target will be reached.

## The identified recipe omission

The official Mamba-1 implementation marks `A_log` and `D` parameters with
`_no_weight_decay=True`. The original CTR-GCN trainer ignored this metadata and
applied the same weight decay to every parameter. The original experiment is
preserved and remains reproducible with the old config.

This trial enables `respect_no_weight_decay: True`, exempting only the official
marked parameters. All other parameters retain weight decay 0.0004. The Mamba
block, fusion, initialization, data, augmentations, SGD momentum/Nesterov, LR 0.1,
five warmup epochs, LR steps 35/55, batch size 128, 65 epochs, AMP and seed 1 are
unchanged. No reliability-aware module is introduced at this stage. The initial
gap has not been demonstrated to be caused by weight decay; this trial tests that
specific training-recipe correction.

Source: [official Mamba-1 parameter markers](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/modules/mamba_simple.py).

## Pull and smoke-test on AutoDL

```bash
cd /root/autodl-tmp/ctr-gcn
git pull --ff-only origin main
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python tools/check_mamba.py --respect-no-weight-decay
```

The printed no-weight-decay list should contain `temporal_mamba.A_log` and
`temporal_mamba.D`. Require the AMP backward/SGD PASS before training.

## Train once, from scratch, and attach shutdown monitoring

```bash
mkdir -p logs

nohup python -u main.py \
  --config config/nturgbd-cross-subject/ctrgcn_mamba_calibrated_fast128.yaml \
  --device 0 --seed 1 \
  > logs/ntu60_xsub_joint_mamba_calibrated.log 2>&1 < /dev/null &
TRAIN_PID=$!

nohup python -u tools/autodl_shutdown.py \
  --pid "$TRAIN_PID" \
  --config config/nturgbd-cross-subject/ctrgcn_mamba_calibrated_fast128.yaml \
  --work-dir work_dir/ntu60/xsub/ctrgcn_mamba_calibrated_joint_fast128 \
  --training-log logs/ntu60_xsub_joint_mamba_calibrated.log \
  > logs/autodl_shutdown_mamba_calibrated.log 2>&1 < /dev/null &

echo "Training PID: $TRAIN_PID; shutdown watcher PID: $!"
```

Confirm the watcher is armed, then inspect the first few epochs:

```bash
tail -n 20 logs/autodl_shutdown_mamba_calibrated.log
tail -f logs/ntu60_xsub_joint_mamba_calibrated.log
```

The trial has its own result directory and preserves the old 88.91% run. Record
the outcome whether or not it passes the threshold. No test-set sweep or search
over multiple alternative recipes is launched by these commands. The model's
compute graph is unchanged, but actual runtime still needs to be measured.

### Repair an already-running watcher after an Exec format error

Some AutoDL instances provide `/usr/bin/shutdown` as a shell script without a
shebang. Direct Python execution then raises `ENOEXEC`, although running it from
a shell works. The watcher now falls back to `/bin/bash /usr/bin/shutdown` only
for this format error. Other command failures remain visible and are not treated
as successful shutdowns.

After pulling the fix, this command validates file format and shell syntax only;
it does not execute shutdown or touch the running training process:

```bash
python tools/autodl_shutdown.py --check-shutdown
```

An already-running watcher keeps its old Python code in memory. Replace only
that watcher after the format check passes; do not restart training. Use the
actual training PID from its `Armed` log. The new watcher must again print `Armed`
and a successful invocation preflight. The fallback is unit-tested locally;
actual platform power-off must still be confirmed in the AutoDL console.

## Re-evaluate both scenes after training

A new checkpoint requires a new clean/occluded evaluation; do not combine its
clean score with the old model's 79.55% occluded score. Use the same masking
protocol and seed 2026, with a new result directory:

```bash
python -u tools/evaluate_occlusion.py \
  --mamba-dir work_dir/ntu60/xsub/ctrgcn_mamba_calibrated_joint_fast128 \
  --output-dir work_dir/ntu60/xsub/occlusion20_calibrated_seed2026 \
  --rate 0.2 --mask-seed 2026 \
  --batch-size 128 --num-worker 2 --device 0 \
  2>&1 | tee logs/occlusion20_calibrated_seed2026.log
```

This corrects the standard-combination training recipe. Later architectural
changes to CTR-GCN or Mamba must be named and recorded separately. Better
occluded accuracy demonstrates better predictions under the specified missing
inputs. A more specific claim about internal state contamination requires a
definition and evidence about that state, rather than accuracy alone.
