"""Paired clean/occluded NTU60 evaluation with the same missing joints for both models."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def best_checkpoint(work_dir, explicit=None):
    work_dir = Path(work_dir)
    log = (work_dir / 'log.txt').read_text(encoding='utf-8') if (work_dir / 'log.txt').is_file() else ''
    epochs = re.findall(r'Epoch number:\s*(\d+)', log)
    accuracies = re.findall(r'Best accuracy:\s*([0-9.eE+-]+)', log)
    epoch = int(epochs[-1]) if epochs else None
    reference = float(accuracies[-1]) * 100 if accuracies else None
    if explicit is not None:
        path = Path(explicit)
        explicit_epoch = re.fullmatch(r'runs-(\d+)-\d+\.pt', path.name)
        if explicit_epoch and int(explicit_epoch[1]) != epoch:
            epoch, reference = int(explicit_epoch[1]), None
        elif path.name != 'best.pt' and not explicit_epoch:
            epoch, reference = None, None
    elif epoch is not None:
        matches = list(work_dir.glob('runs-{}-*.pt'.format(epoch)))
        if not matches and (work_dir / 'best.pt').is_file():
            path = work_dir / 'best.pt'
        elif len(matches) != 1:
            raise ValueError('Expected one checkpoint for best epoch {} in {}'.format(epoch, work_dir))
        else:
            path = matches[0]
    elif (work_dir / 'best.pt').is_file():
        path = work_dir / 'best.pt'
    else:
        raise FileNotFoundError('No best checkpoint found in {}. Copy the completed run here first.'.format(work_dir))
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError('Checkpoint is missing or empty: {}'.format(path))
    return path.resolve(), epoch, reference


def read_config(work_dir, expected_model):
    path = Path(work_dir) / 'config.yaml'
    if not path.is_file():
        raise FileNotFoundError('Saved training config is missing: {}'.format(path))
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    allowed = (expected_model,) if isinstance(expected_model, str) else expected_model
    if config['model'] not in allowed:
        raise ValueError('Wrong model in {}: {}'.format(path, config['model']))
    if config['feeder'] != 'feeders.feeder_ntu.Feeder':
        raise ValueError('This evaluation requires the NTU feeder.')
    feeder = config['test_feeder_args']
    if feeder.get('split') != 'test' or feeder.get('bone', False) or feeder.get('vel', False):
        raise ValueError('Evaluation must use the Joint test split.')
    if feeder.get('debug', False) or any(feeder.get(key, False) for key in
                                      ('random_choose', 'random_shift', 'random_move', 'random_rot',
                                       'joint_mask_prob')):
        raise ValueError('Debug mode or random test augmentation is not allowed.')
    if (config['model_args']['num_point'] != 25 or config['model_args']['num_class'] != 60
            or config['model_args']['num_person'] != 2):
        raise ValueError('This protocol is configured for NTU60 with 25 joints.')
    return config


def matching_test_settings(configs):
    reference = dict(configs['baseline']['test_feeder_args'])
    reference['data_path'] = str((ROOT / reference['data_path']).resolve())
    other = dict(configs['mamba']['test_feeder_args'])
    other['data_path'] = str((ROOT / other['data_path']).resolve())
    if reference != other:
        raise ValueError('Both models must use identical test preprocessing and dataset paths.')
    for key in ('amp', 'fast'):
        if configs['baseline'].get(key, False) != configs['mamba'].get(key, False):
            raise ValueError('Both saved configs must use the same {} setting.'.format(key))
    return reference


def matching_training_inputs(configs):
    """Report augmentation differences rather than attribute them to architecture."""
    settings = []
    for config in configs.values():
        arguments = dict(config['train_feeder_args'])
        arguments.setdefault('joint_mask_prob', 0.0)
        arguments.setdefault('joint_mask_max', 5)
        arguments['data_path'] = str((ROOT / arguments['data_path']).resolve())
        settings.append(arguments)
    return all(arguments == settings[0] for arguments in settings[1:])


def make_joint_masks(count, num_joints=25, rate=0.2, seed=2026):
    raw = rate * num_joints
    missing = round(raw)
    if count < 1 or not 0 <= rate <= 1 or not math.isclose(raw, missing, abs_tol=1e-8):
        raise ValueError('Rate must specify an exact integer number of joints, and count must be positive.')
    if seed < 0:
        raise ValueError('Mask seed must be nonnegative.')
    rng = np.random.default_rng(seed)
    if missing == 0:
        return np.empty((count, 0), dtype=np.uint8)
    # A separate random ranking per sample gives exactly k distinct joint IDs.
    rankings = np.argsort(rng.random((count, num_joints)), axis=1)
    return np.sort(rankings[:, :missing], axis=1).astype(np.uint8)


def mask_inputs(data, joint_indices):
    """Zero all coordinates, times and people for each sample's selected joints."""
    if data.ndim != 5 or joint_indices.ndim != 2 or data.shape[0] != joint_indices.shape[0]:
        raise ValueError('Expected data B,C,T,V,M and joint IDs B,K.')
    mask = torch.zeros((data.shape[0], data.shape[3]), dtype=torch.bool, device=data.device)
    mask.scatter_(1, joint_indices.to(device=data.device, dtype=torch.long), True)
    return data.masked_fill(mask[:, None, None, :, None], 0)


def metrics(scores, labels):
    # Match the original NTU feeder's top-k tie handling.
    ranking = scores.argsort(axis=1)
    prediction = ranking[:, -1]
    return dict(top1_percent=float(np.mean(prediction == labels) * 100),
                top5_percent=float(np.mean(np.any(ranking[:, -5:] == labels[:, None], axis=1)) * 100))


@torch.inference_mode()
def evaluate_pair(models, loader, joint_masks, labels, device, amp):
    count = len(labels)
    classes = 60
    scores = {name: {kind: np.empty((count, classes), dtype=np.float32)
                     for kind in ('clean', 'occluded')} for name in models}
    visited = np.zeros(count, dtype=bool)
    for model in models.values():
        model.eval()
    for data, batch_labels, indices in loader:
        indices = np.asarray(indices)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= count):
            raise ValueError('Invalid test sample indices.')
        if len(np.unique(indices)) != len(indices) or visited[indices].any():
            raise ValueError('Repeated test sample indices.')
        if not np.array_equal(np.asarray(batch_labels), labels[indices]):
            raise ValueError('Test labels and sample indices are misaligned.')
        data = data.to(device, dtype=torch.float32, non_blocking=device.type == 'cuda')
        masked = mask_inputs(data, torch.from_numpy(joint_masks[indices].astype(np.int64)))
        for name, model in models.items():
            for kind, inputs in (('clean', data), ('occluded', masked)):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model(inputs)
                array = logits.float().cpu().numpy()
                if array.shape != (len(indices), classes) or not np.isfinite(array).all():
                    raise ValueError('{} {} returned invalid scores.'.format(name, kind))
                scores[name][kind][indices] = array
        visited[indices] = True
    if not visited.all():
        raise ValueError('Not all test samples were evaluated.')
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, default=Path('work_dir/ntu60/xsub/ctrgcn_joint_fast128'))
    parser.add_argument('--mamba-dir', type=Path, default=Path('work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128'))
    parser.add_argument('--baseline-weights', type=Path)
    parser.add_argument('--mamba-weights', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('work_dir/ntu60/xsub/occlusion20_seed2026'))
    parser.add_argument('--rate', type=float, default=0.2)
    parser.add_argument('--mask-seed', type=int, default=2026)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-worker', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--check-only', action='store_true', help='check saved configs/checkpoints without loading data or CUDA')
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_worker < 0:
        parser.error('Batch size must be positive; worker count must be nonnegative.')
    make_joint_masks(1, rate=args.rate, seed=args.mask_seed)
    work_dirs = dict(baseline=(ROOT / args.baseline_dir).resolve(), mamba=(ROOT / args.mamba_dir).resolve())
    configs = {name: read_config(directory, ('model.ctrgcn.Model', 'model.ctrgcn_mamba.Model',
                                           'model.ctrgcn_mamba_robust_v2.Model',
                                           'model.ctrgcn_mamba_robust_v3.Model')
                                 if name == 'baseline' else ('model.ctrgcn_mamba.Model',
                                                            'model.ctrgcn_mamba_robust.Model',
                                                            'model.ctrgcn_mamba_robust_v2.Model',
                                                            'model.ctrgcn_mamba_robust_v3.Model'))
               for name, directory in work_dirs.items()}
    feeder_args = matching_test_settings(configs)
    same_training_inputs = matching_training_inputs(configs)
    if not same_training_inputs:
        print('NOTE: training input/augmentation settings differ; this comparison includes '
              'training-recipe changes and cannot isolate the architecture effect.', flush=True)
    data_path = Path(feeder_args['data_path'])
    if not data_path.is_file():
        raise FileNotFoundError('Prepared dataset is missing: {}'.format(data_path))
    provenance = {}
    for name, directory in work_dirs.items():
        override = args.baseline_weights if name == 'baseline' else args.mamba_weights
        path, epoch, reference = best_checkpoint(directory, ROOT / override if override else None)
        provenance[name] = dict(checkpoint=str(path), best_epoch=epoch,
                                training_best_top1_percent=reference,
                                model=configs[name]['model'], model_args=configs[name]['model_args'],
                                train_feeder_args=configs[name]['train_feeder_args'],
                                saved_config=str(directory / 'config.yaml'))
        print('{}: best epoch {}, reference {}, checkpoint {}'.format(name, epoch, reference, path), flush=True)
    print('Dataset:', data_path, flush=True)
    if args.check_only:
        print('PASS: both saved configs, dataset path and best checkpoint files are present.', flush=True)
        return
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse accidental replacement of a completed result set.
    if (output_dir / 'summary.json').exists():
        raise FileExistsError('Results already exist. Choose a new --output-dir: {}'.format(output_dir))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')
    device = torch.device('cuda', args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(1)
    fast = configs['baseline'].get('fast', False)
    amp = configs['baseline'].get('amp', False)
    torch.backends.cudnn.deterministic = not fast
    torch.backends.cudnn.benchmark = fast
    torch.backends.cuda.matmul.allow_tf32 = fast
    torch.backends.cudnn.allow_tf32 = fast
    models = {}
    for name, config in configs.items():
        module_name, class_name = config['model'].rsplit('.', 1)
        model = getattr(importlib.import_module(module_name), class_name)(**config['model_args'])
        weights = torch.load(provenance[name]['checkpoint'], map_location='cpu', weights_only=True)
        weights = {key.removeprefix('module.'): value for key, value in weights.items()}
        model.load_state_dict(weights, strict=True)
        models[name] = model.to(device).eval()
        provenance[name]['checkpoint_sha256'] = file_digest(provenance[name]['checkpoint'])
        provenance[name]['parameters'] = sum(parameter.numel() for parameter in model.parameters())
    from feeders.feeder_ntu import Feeder
    dataset = Feeder(**feeder_args)
    labels = np.asarray(dataset.label, dtype=np.int64)
    joint_masks = make_joint_masks(len(labels), rate=args.rate, seed=args.mask_seed)
    np.savez_compressed(output_dir / 'masks.npz', sample_indices=np.arange(len(labels)),
                        labels=labels, masked_joint_indices=joint_masks,
                        seed=args.mask_seed, rate=args.rate)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         num_workers=args.num_worker, shuffle=False,
                                         drop_last=False, pin_memory=True)
    start = time.perf_counter()
    from tqdm import tqdm
    scores = evaluate_pair(models, tqdm(loader, desc='Paired clean / occluded'),
                           joint_masks, labels, device, amp)
    elapsed = time.perf_counter() - start
    np.savez_compressed(output_dir / 'scores.npz', labels=labels,
                        **{'{}_{}'.format(name, kind): value for name, kinds in scores.items()
                           for kind, value in kinds.items()})
    rows = []
    for name in models:
        clean = metrics(scores[name]['clean'], labels)
        occluded = metrics(scores[name]['occluded'], labels)
        drop = clean['top1_percent'] - occluded['top1_percent']
        rows.append(dict(model=name, clean_top1_percent=clean['top1_percent'],
                         occluded_top1_percent=occluded['top1_percent'], drop_pp=drop,
                         clean_top5_percent=clean['top5_percent'],
                         occluded_top5_percent=occluded['top5_percent']))
        reference = provenance[name]['training_best_top1_percent']
        if reference is not None:
            difference = clean['top1_percent'] - reference
            provenance[name]['clean_reference_difference_pp'] = difference
            if abs(difference) > 0.2:
                print('WARNING: {} clean accuracy differs from its training log by {:.3f} pp; '
                      'check this before interpreting occlusion results.'.format(name, difference), flush=True)
        print('RESULT {}: clean {:.2f}%, occluded {:.2f}%, drop {:.2f} pp'.format(
            name, clean['top1_percent'], occluded['top1_percent'], drop), flush=True)
    with (output_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        keys = [(name, kind) for name in models for kind in ('clean', 'occluded')]
        predictions = [scores[name][kind].argsort(axis=1)[:, -1] for name, kind in keys]
        writer.writerow(['sample_index', 'label', 'masked_joints_1based'] + ['{}_{}'.format(*key) for key in keys])
        for index, label in enumerate(labels):
            writer.writerow([index, int(label), ' '.join(str(int(joint)+1) for joint in joint_masks[index])]
                            + [int(prediction[index]) for prediction in predictions])
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    summary = dict(status='complete', created_at_utc=datetime.now(timezone.utc).isoformat(),
                   protocol='per_sample_fixed_joints_zero_after_test_crop_resize',
                   rate=args.rate, missing_joints=int(joint_masks.shape[1]), mask_seed=args.mask_seed,
                   same_joints_for_all_frames_and_people=True,
                   mask_sha256=hashlib.sha256(joint_masks.tobytes()).hexdigest(),
                   samples=len(labels), data_path=str(data_path),
                   data_size_bytes=data_path.stat().st_size,
                   test_feeder_args=feeder_args,
                   same_training_feeder_args=same_training_inputs,
                   batch_size=args.batch_size, amp=amp, fast=fast,
                   gpu=torch.cuda.get_device_name(device), pytorch=torch.__version__, numpy=np.__version__,
                   git_commit=revision.stdout.strip(), elapsed_seconds=elapsed,
                   checkpoints=provenance, results=rows)
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print('Saved paired inference results:', output_dir, flush=True)


if __name__ == '__main__':
    main()
