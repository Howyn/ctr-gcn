# Paired 20% missing-joint inference: CTR-GCN versus standard Mamba

## Goal and fixed protocol

Evaluate the already-trained models without further training or adaptation.
The clean reference runs were 89.64% (CTR-GCN, best epoch 60) and 88.91%
(CTR-GCN + standard Mamba-1, best epoch 38), each from training seed 1.

The agreed synthetic test protocol is:

- Use the entire NTU60 X-Sub Joint test set and each run's saved test preprocessing.
- After the standard center crop and resize to 64 frames, select exactly 5 of
  25 distinct joint indices per sample, uniformly without replacement.
- Keep those indices fixed for every frame and both people in that sample.
- Set all three coordinate channels of those joints to zero.
- Use mask seed 2026. Both models share the exact same input samples and masks.
- Run both clean and occluded inputs in the same evaluation process on the same
  GPU, with both models in eval mode and inference mode; no gradients or updates.

"20%" counts joint indices, including any joints already zero in the input; it
does not promise that exactly 20% of previously nonzero values are removed.
Empty second-person slots remain empty. No raw dataset or checkpoint is edited.
This is synthetic joint dropout, not a claim to reproduce a specific real-camera
occlusion benchmark. Occlusion protocols differ across the literature, so the
definition must accompany the results; see, for example, the different scenarios
in [Cormier et al., WACV Workshops 2024](https://openaccess.thecvf.com/content/WACV2024W/RWS/papers/Cormier_Enhancing_Skeleton-Based_Action_Recognition_in_Real-World_Scenarios_Through_Realistic_Data_WACVW_2024_paper.pdf).

## Check the required artifacts before inference

The source run directories must contain their own `config.yaml` and trained
weights. The script reads each `log.txt` to select the best epoch automatically,
with an exact checkpoint-name match. It also supports `best.pt` when numbered
checkpoints were pruned. It never silently loads incomplete model keys.

```bash
cd /root/autodl-tmp/ctr-gcn
git pull --ff-only origin main
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python tools/evaluate_occlusion.py --check-only
```

Default directories:

```text
work_dir/ntu60/xsub/ctrgcn_joint_fast128
work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128
```

If the baseline exists only on the old instance, copy its completed run directory
to the new instance first. Do not use the Mamba checkpoint as the baseline or
substitute a random-initialized model. Different paths can be selected using
`--baseline-dir`, `--mamba-dir`, `--baseline-weights` and `--mamba-weights`.

## Run paired inference

After the artifact check passes:

```bash
mkdir -p logs
python -u tools/evaluate_occlusion.py \
  --rate 0.2 --mask-seed 2026 \
  --batch-size 128 --num-worker 2 --device 0 \
  2>&1 | tee logs/occlusion20_seed2026.log
```

The script loads the saved models strictly, uses a single test loader for both
models, saves the masks, and runs all four model/input combinations in one pass.
It requires matching saved data preprocessing, Joint modality, and precision
settings. It reports clean accuracy again so it can be compared with the source
training logs; a difference above 0.2 percentage points is flagged for inspection,
not silently treated as an occlusion effect.

## Results

The default output directory is `work_dir/ntu60/xsub/occlusion20_seed2026`:

- `summary.json`: protocol, model checkpoints and SHA-256 hashes, clean references,
  GPU/software versions, current code commit, actual inference time and metrics.
- `summary.csv`: clean Top-1/Top-5, occluded Top-1/Top-5 and Top-1 drop in percentage points.
- `masks.npz`: sample indices, labels and each sample's 0-based masked joint indices.
- `scores.npz`: labels and all four score arrays.
- `predictions.csv`: labels, 1-based masked joint IDs and all four predictions.

Existing completed results are not overwritten; choose a new `--output-dir` for
a rerun or a different mask seed. The arrays are deterministic for the fixed seed
and NumPy version, and the exact mask array hash is recorded. Future comparisons
should reuse the same protocol and seed. For stronger claims about robustness,
repeat with additional mask seeds and report variability.

Compare both occluded accuracy and clean-to-occluded drop; a smaller drop alone
does not establish superiority if the model starts from a lower clean accuracy.
This stage tests the current models' robustness; it does not presuppose that the
standard Mamba combination will be better.
