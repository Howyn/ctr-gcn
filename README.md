# CTR-GCN
This repo is the official implementation for [Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition](https://arxiv.org/abs/2107.12213). The paper is accepted to ICCV2021.

Note: We also provide a simple and strong baseline model, which achieves 83.7% on NTU120 CSub with joint modality only, to facilitate the development of skeleton-based action recognition.

## Architecture of CTR-GC
![image](src/framework.jpg)
# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX

- We provide the dependency file of our experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running `pip install -r requirements.txt `
- Run `pip install -e torchlight` 

# Data Preparation

### Download datasets.

#### There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA

#### NTU RGB+D 60 and 120

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

#### NW-UCLA

1. Download dataset from [here](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0)
2. Move `all_sqe` to `./data/NW-UCLA`

### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```



# Training & Testing

### Training

- Change the config file depending on what you want.

```
# Example: training CTRGCN on NTU RGB+D 120 cross subject with GPU 0
python main.py --config config/nturgbd120-cross-subject/default.yaml --work-dir work_dir/ntu120/csub/ctrgcn --device 0
# Example: training provided baseline on NTU RGB+D 120 cross subject
python main.py --config config/nturgbd120-cross-subject/default.yaml --model model.baseline.Model--work-dir work_dir/ntu120/csub/baseline --device 0
```

- To train model on NTU RGB+D 60/120 with bone or motion modalities, setting `bone` or `vel` arguments in the config file `default.yaml` or in the command line.

```
# Example: training CTRGCN on NTU RGB+D 120 cross subject under bone modality
python main.py --config config/nturgbd120-cross-subject/default.yaml --train_feeder_args bone=True --test_feeder_args bone=True --work-dir work_dir/ntu120/csub/ctrgcn_bone --device 0
```

- To train model on NW-UCLA with bone or motion modalities, you need to modify `data_path` in `train_feeder_args` and `test_feeder_args` to "bone" or "motion" or "bone motion", and run

```
python main.py --config config/ucla/default.yaml --work-dir work_dir/ucla/ctrgcn_xxx --device 0
```

- To train your own model, put model file `your_model.py` under `./model` and run:

```
# Example: training your own model on NTU RGB+D 120 cross subject
python main.py --config config/nturgbd120-cross-subject/default.yaml --model model.your_model.Model --work-dir work_dir/ntu120/csub/your_model --device 0
```

### Optional runtime acceleration (PyTorch 2.5+)

`config/nturgbd-cross-subject/fast.yaml` enables CUDA float16 AMP with gradient
scaling, cuDNN tuning, TF32, pinned transfers and persistent data workers. It
keeps the NTU60 X-Sub Joint model, batch size 64, learning rate and 65-epoch
schedule. These runtime options change numerical behavior and determinism;
record them when comparing experiments. The default config leaves them off.

Before a full run, stop any other training process on the GPU and benchmark:

```bash
python main.py --config config/nturgbd-cross-subject/fast.yaml --phase benchmark --work-dir work_dir/ntu60/xsub/benchmark_fast64
```

The benchmark warms up 10 batches, then times 50 training and 50 test batches
with CUDA synchronization at the timing boundaries. Its full-run projection
includes per-epoch evaluation and a final test, but excludes startup, checkpoint
and logging overhead. It is an estimate, not a runtime guarantee. No checkpoint
is saved; start the actual experiment in a fresh process:

```bash
python main.py --config config/nturgbd-cross-subject/fast.yaml --device 0
```

Use a new work directory for each training run. If data workers fail due to
shared-memory limits, reduce `--num-worker` (zero is also supported). Changing
batch size or shortening training changes the experiment settings and must be
reported separately from this runtime acceleration.

`config/nturgbd-cross-subject/fast128.yaml` is a separate experiment preset with
training/test batch size 128, two workers and one main-process CPU thread. It
retains the 65-epoch schedule and learning rate, but changes training batch size
from the original 64; validate its accuracy separately. Benchmark projections
are hardware-dependent and do not guarantee a two-hour runtime.

See the PyTorch [AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
and [performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

### AutoDL shutdown after an existing training run

After starting the fast128 training command, attach a separate watcher to the
training parent's PID (not a data worker). This does not restart training:

```bash
nohup python -u tools/autodl_shutdown.py --pid TRAINING_PID > logs/autodl_shutdown.log 2>&1 < /dev/null &
```

The watcher validates the PID/config, waits for that process to exit, and checks
the 65-epoch completion summary, best/final checkpoints, final evaluation scores
and logs. It copies the best checkpoint to `best.pt`, verifies its SHA-256, saves
`training_console.log` and `result_summary.json` in the fast128 work directory,
syncs the filesystem and invokes `/usr/bin/shutdown`, as documented by
[AutoDL](https://www.autodl.com/docs/save_money/). If training or verification
fails, it exits without shutting down; inspect `logs/autodl_shutdown.log`.
`--dry-run` performs verification/saving but skips shutdown. Other runs can
specify `--config`, `--work-dir`, `--training-log`, and `--expected-epochs`.
Use `--check-shutdown` to inspect command format and shell syntax without
requesting shutdown. AutoDL scripts that raise `Exec format error` on direct
execution are retried with `/bin/bash`; this does not hide other failures.

### Testing

For the additional **CTR-GCN + standard Mamba-1** NTU60 X-Sub Joint experiment,
see [installation, architecture, benchmark and training commands](docs/ntu60_mamba.md).
Its training settings match `fast128.yaml`, and its results use a separate directory.
The [standard-Mamba calibration trial](docs/mamba_calibration.md) optionally honors
the official state-parameter no-weight-decay markers without changing the model.

For paired clean versus **20% missing-joint inference** with those trained models,
see the [fixed occlusion protocol and evaluation commands](docs/occlusion20.md).

For the new **reliability-controlled Mamba state writes + local temporal residual**,
see [algorithm, parameter budget, CUDA checks, training and AutoDL shutdown](docs/ntu60_robust_mamba.md).
Its two modules have independent switches and add 25,474 parameters. The first
trial keeps the original standard hybrid's clean-only training recipe.

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

- To ensemble the results of different modalities, run 
```
# Example: ensemble four modalities of CTRGCN on NTU RGB+D 120 cross subject
python ensemble.py --datasets ntu120/xsub --joint-dir work_dir/ntu120/csub/ctrgcn --bone-dir work_dir/ntu120/csub/ctrgcn_bone --joint-motion-dir work_dir/ntu120/csub/ctrgcn_motion --bone-motion-dir work_dir/ntu120/csub/ctrgcn_bone_motion
```

### Pretrained Models

- Download pretrained models for producing the final results on NTU RGB+D 60&120 cross subject [[Google Drive]](https://drive.google.com/drive/folders/1C9XUAgnwrGelvl4mGGVZQW6akiapgdnd?usp=sharing).
- Put files to <work_dir> and run **Testing** command to produce the final result.

## Acknowledgements

This repo is based on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch).

Thanks to the original authors for their work!

# Citation

Please cite this work if you find it useful:.

      @inproceedings{chen2021channel,
        title={Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition},
        author={Chen, Yuxin and Zhang, Ziqi and Yuan, Chunfeng and Li, Bing and Deng, Ying and Hu, Weiming},
        booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
        pages={13359--13368},
        year={2021}
      }

# Contact
For any questions, feel free to contact: `chenyuxin2019@ia.ac.cn`
