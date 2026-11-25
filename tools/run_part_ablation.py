"""Run the single-seed NTU60 X-Sub joint-only part-occlusion ablation."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TEMPLATE = ROOT / 'config/nturgbd-cross-subject/ctrgcn_mamba_fast128.yaml'
VARIANTS = {'baseline': (False, False), 'a': (True, False), 'b': (True, True)}
# NTU joint IDs, 1-based. Source: RA-GCNv2/src/dataprocessor.py, Occlusion_part.
PARTS = {
    'left_arm': (5, 6, 7, 8, 22, 23),
    'right_arm': (9, 10, 11, 12, 24, 25),
    'both_hands': (22, 23, 24, 25),
    'both_legs': (13, 14, 15, 16, 17, 18, 19, 20),
    'trunk': (1, 2, 3, 4, 21),
}
TRAIN_COUNT, TEST_COUNT = 40091, 16487
PROTOCOL = 'ra_gcn_body_parts_zero_after_ctr_gcn_test_crop_resize'
ACCURACY_COLUMNS = ('clean', *PARTS, 'occluded_mean')


def training_config(variant, directory, data_path):
    config = yaml.safe_load(TEMPLATE.read_text(encoding='utf-8'))
    attention, gate = VARIANTS[variant]
    config.update(model='model.ctrgcn_mamba_part_ablation.Model',
                  work_dir=str(directory), seed=1, optimizer='SGD', start_epoch=0,
                  save_interval=1, respect_no_weight_decay=False)
    config['model_args'].update(use_joint_attention=attention, use_write_gate=gate)
    for key in ('train_feeder_args', 'test_feeder_args'):
        config[key].update(data_path=str(data_path), joint_mask_prob=0.0)
    return config


def check_data(path):
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(f'Prepared NTU60 X-Sub archive is missing: {path}')
    with np.load(path, allow_pickle=False) as archive:
        for split, count in (('train', TRAIN_COUNT), ('test', TEST_COUNT)):
            if f'x_{split}' not in archive.files:
                raise ValueError(f'Missing x_{split} in {path}')
            labels = archive[f'y_{split}']
            if labels.shape != (count, 60) or not ((labels > 0).sum(axis=1) == 1).all():
                raise ValueError(f'Unexpected X-Sub {split} labels in {path}')


def final_checkpoint(config):
    """Use the predeclared final epoch and verify the training run finished."""
    directory = Path(config['work_dir'])
    saved = yaml.safe_load((directory / 'config.yaml').read_text(encoding='utf-8'))
    for key, value in {**config, 'phase': 'train', 'weights': None}.items():
        if saved.get(key) != value:
            raise ValueError(f'Configuration mismatch for {directory.name}: {key}')
    log = (directory / 'log.txt').read_text(encoding='utf-8')
    if 'Best accuracy:' not in log or 'Epoch number:' not in log:
        raise ValueError(f'Training did not finish in {directory}')
    matches = list(directory.glob(f"runs-{config['num_epoch']}-*.pt"))
    if len(matches) != 1 or not matches[0].stat().st_size:
        raise ValueError(f'Expected exactly one nonempty final-epoch checkpoint in {directory}')
    return matches[0]


def smoke(config):
    import torch

    from model.ctrgcn_mamba_part_ablation import Model, write_reliability
    from model.ctrgcn_mamba_robust import reliable_mamba
    from model.ctrgcn_mamba_robust_v3 import joint_observability
    from tools.check_mamba import run_backward_check

    if not torch.cuda.is_available():
        raise RuntimeError('The official Mamba-1 CUDA kernels require a Linux CUDA environment.')
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    model = Model(**config['model_args']).cuda()
    frames = config['train_feeder_args']['window_size']
    clean = torch.randn(2, 3, frames, 25, 2, device='cuda')
    occluded = clean.clone()
    occluded[:, :, :, [i - 1 for i in PARTS['left_arm']], :] = 0
    attention = torch.full((2, 2, frames // 4, 25), 1 / 25, device='cuda')
    torch.testing.assert_close(write_reliability(attention, joint_observability(clean, frames // 4)),
                               torch.ones_like(attention[..., :1]), atol=0, rtol=0)
    torch.testing.assert_close(write_reliability(attention, joint_observability(occluded, frames // 4)),
                               torch.full_like(attention[..., :1], 19 / 25))
    with torch.no_grad():
        tokens = torch.randn(4, frames // 4, model.fc.in_features, device='cuda')
        unit = torch.ones_like(tokens[..., :1])
        for amp in (False, True):
            with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
                native = model.temporal_mamba(tokens)
                explicit = reliable_mamba(model.temporal_mamba, tokens, unit)
            tolerance = 5e-3 if amp else 1e-4
            torch.testing.assert_close(explicit, native, atol=tolerance, rtol=tolerance)
        model.eval()
        torch.testing.assert_close(model(clean), model(clean, disable_write_gate=True))
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9,
                                nesterov=True, weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['amp'])
    labels = torch.tensor([0, 1], device='cuda')
    run_backward_check(model, optimizer, scaler, clean, labels, config['amp'])
    if not model.joint_score.weight.detach().ne(0).any().item():
        raise RuntimeError('The joint attention weights did not update.')
    print('PASS: native Mamba parity, observed joint gate, attention and CUDA training step.', flush=True)


def train(configs):
    for name, config in configs.items():
        directory = Path(config['work_dir'])
        if directory.exists():
            try:
                checkpoint = final_checkpoint(config)
            except (OSError, ValueError) as error:
                raise RuntimeError(f'{directory} already exists and is incomplete or incompatible. '
                                   'Preserve it separately before rerunning this variant.') from error
            print(f'{name}: using completed {checkpoint}', flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=False)
        snapshot = directory / 'launch.yaml'
        snapshot.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        subprocess.run([sys.executable, '-u', str(ROOT / 'main.py'), '--config', str(snapshot)],
                       cwd=ROOT, check=True)
        print(f'{name}: completed {final_checkpoint(config)}', flush=True)


def accuracy_row(correct, count):
    result = {condition: 100 * hits / count for condition, hits in correct.items()}
    result['occluded_mean'] = sum(result[part] for part in PARTS) / len(PARTS)
    return result


def paired_summary(results):
    baseline, a, b, gate_off = (results[name] for name in ('baseline', 'a', 'b', 'b_gate_off'))
    return {
        'b_vs_baseline_clean_pp': b['clean'] - baseline['clean'],
        'a_vs_baseline_occluded_pp': a['occluded_mean'] - baseline['occluded_mean'],
        'b_vs_a_occluded_pp': b['occluded_mean'] - a['occluded_mean'],
        'b_vs_baseline_occluded_pp': b['occluded_mean'] - baseline['occluded_mean'],
        'b_vs_own_gate_off_occluded_pp': b['occluded_mean'] - gate_off['occluded_mean'],
    }


def evaluate(configs, output_dir):
    import numpy as np
    import torch
    from tqdm import tqdm

    from feeders.feeder_ntu import Feeder
    from model.ctrgcn_mamba_part_ablation import Model
    from tools.evaluate_occlusion import file_digest, mask_inputs

    checkpoints = {name: final_checkpoint(config) for name, config in configs.items()}
    if output_dir.exists():
        raise FileExistsError(f'Results already exist: {output_dir}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for Mamba inference.')
    device = torch.device('cuda', 0)
    torch.cuda.set_device(device)
    common = configs['baseline']
    torch.set_num_threads(common['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = common['fast']
    torch.backends.cudnn.allow_tf32 = common['fast']
    torch.backends.cudnn.benchmark = common['fast']
    dataset = Feeder(**common['test_feeder_args'])
    if len(dataset.data) != TEST_COUNT or len(dataset) != TEST_COUNT:
        raise ValueError('Expected the full prepared NTU60 X-Sub test split.')
    loader = torch.utils.data.DataLoader(dataset, batch_size=common['test_batch_size'],
                                         num_workers=common['num_worker'], pin_memory=True)
    parts = {part: torch.tensor(joints, device=device) - 1 for part, joints in PARTS.items()}
    results, provenance = {}, {}
    for name, config in configs.items():
        model = Model(**config['model_args']).to(device).eval()
        weights = torch.load(checkpoints[name], map_location='cpu', weights_only=True)
        model.load_state_dict(weights, strict=True)
        conditions = ('clean', *PARTS)
        counts = {name: dict.fromkeys(conditions, 0)}
        if name == 'b':
            counts['b_gate_off'] = dict.fromkeys(conditions, 0)
        offset = 0
        with torch.inference_mode():
            for data, labels, indices in tqdm(loader, desc=name):
                batch = len(labels)
                if not np.array_equal(indices.numpy(), np.arange(offset, offset + batch)):
                    raise ValueError('Evaluation samples must appear exactly once in dataset order.')
                offset += batch
                data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                batch_hits = {}
                for condition in conditions:
                    inputs = data if condition == 'clean' else mask_inputs(data, parts[condition].expand(batch, -1))
                    for row in counts:
                        with torch.autocast('cuda', dtype=torch.float16, enabled=common['amp']):
                            logits = model(inputs, disable_write_gate=row == 'b_gate_off')
                        hits = (logits.argmax(dim=1) == labels).sum()
                        batch_hits[row, condition] = torch.stack((hits, logits.isfinite().all()))
                for (row, condition), (hits, finite) in zip(batch_hits, torch.stack(tuple(batch_hits.values())).tolist()):
                    if not finite:
                        raise ValueError(f'Non-finite logits: {row} / {condition}')
                    counts[row][condition] += hits
        if offset != TEST_COUNT:
            raise ValueError(f'{name}: evaluated {offset} of {TEST_COUNT} samples.')
        results.update({row: accuracy_row(hits, offset) for row, hits in counts.items()})
        provenance[name] = dict(checkpoint=str(checkpoints[name]), sha256=file_digest(checkpoints[name]),
                                parameters=sum(p.numel() for p in model.parameters()), config=config)
        del model

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    comparison = paired_summary(results)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as staging:
        staged = Path(staging)
        with (staged / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['model', *ACCURACY_COLUMNS])
            writer.writeheader()
            writer.writerows(dict(model=name, **row) for name, row in results.items())
        (staged / 'summary.json').write_text(json.dumps({
            'samples': TEST_COUNT, 'checkpoint_rule': 'fixed_final_epoch',
            'protocol': PROTOCOL,
            'body_part_joint_ids_1based': PARTS, 'metric': 'top1_percent',
            'gpu': torch.cuda.get_device_name(device), 'torch': str(torch.__version__),
            'models': provenance, 'accuracy': results, 'differences_pp': comparison,
        }, indent=2) + '\n', encoding='utf-8')
        output_dir.mkdir(exist_ok=False)
        for filename in ('summary.csv', 'summary.json'):
            (staged / filename).replace(output_dir / filename)
    print(json.dumps(comparison, indent=2), flush=True)
    print(f'Results: {output_dir}', flush=True)


def verify_evaluation(configs, output_dir):
    from tools.autodl_shutdown import digest

    summary = json.loads((output_dir / 'summary.json').read_text(encoding='utf-8'))
    if (summary['samples'] != TEST_COUNT or summary['protocol'] != PROTOCOL
            or summary['checkpoint_rule'] != 'fixed_final_epoch'
            or summary['body_part_joint_ids_1based'] != {part: list(ids) for part, ids in PARTS.items()}):
        raise ValueError('Evaluation protocol or sample count did not match the run.')
    expected_rows = set(configs) | {'b_gate_off'}
    if set(summary['accuracy']) != expected_rows or set(summary['models']) != set(configs):
        raise ValueError('Evaluation is missing one or more model variants.')
    with (output_dir / 'summary.csv').open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['model', *ACCURACY_COLUMNS]:
            raise ValueError('Evaluation CSV headers did not match the evaluation protocol.')
        rows = list(reader)
    if len(rows) != len(expected_rows) or {row['model'] for row in rows} != expected_rows:
        raise ValueError('Evaluation CSV does not contain exactly one row per variant.')
    for entry in rows:
        name = entry['model']
        values = {key: float(entry[key]) for key in ACCURACY_COLUMNS}
        mean = sum(values[key] for key in PARTS) / len(PARTS)
        if (values != summary['accuracy'][name] or not all(0 <= v <= 100 for v in values.values())
                or not math.isclose(values['occluded_mean'], mean)):
            raise ValueError(f'Evaluation results are invalid or disagree for {name}.')
    for name, config in configs.items():
        checkpoint = final_checkpoint(config)
        model = summary['models'][name]
        if (model['config'] != config or Path(model['checkpoint']) != checkpoint
                or model['sha256'] != digest(checkpoint)):
            raise ValueError(f'Evaluation does not match the final {name} checkpoint.')


def run_autodl(configs, run_dir, data_path):
    """Run the complete experiment and request power-off after verification."""
    if not sys.platform.startswith('linux'):
        raise RuntimeError('The autodl action must run inside the Linux AutoDL instance.')
    import fcntl
    from tools.autodl_shutdown import check_shutdown_command, request_shutdown

    shutdown = Path('/usr/bin/shutdown')
    check_shutdown_command(shutdown)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / 'autodl.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output_dir = run_dir / 'evaluation'
        if not output_dir.exists():
            check_data(data_path)
            smoke(configs['b'])
            train(configs)
            evaluate(configs, output_dir)
        verify_evaluation(configs, output_dir)
        print(f'Verified final checkpoints and results in {output_dir}. Requesting AutoDL shutdown.', flush=True)
        os.sync()
        request_shutdown(shutdown)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'smoke', 'train', 'evaluate', 'autodl'))
    parser.add_argument('--data-path', type=Path, default=Path('data/ntu/NTU60_CS.npz'))
    parser.add_argument('--run-dir', type=Path, default=Path('work_dir/ntu60/xsub/part_ablation_seed1'))
    parser.add_argument('--variant', choices=VARIANTS, help='train only this variant')
    args = parser.parse_args()
    if args.variant and args.action != 'train':
        parser.error('--variant is only valid with train')
    run_dir = (ROOT / args.run_dir).resolve()
    data_path = (ROOT / args.data_path).resolve()
    configs = {name: training_config(name, run_dir / name, data_path) for name in VARIANTS}
    if args.action == 'plan':
        print(yaml.safe_dump(configs, sort_keys=False), end='')
    elif args.action == 'smoke':
        smoke(configs['b'])
        check_data(data_path)
        print(f'PASS: prepared X-Sub archive {data_path}', flush=True)
    elif args.action == 'train':
        check_data(data_path)
        smoke(configs['b'])
        train({args.variant: configs[args.variant]} if args.variant else configs)
    elif args.action == 'evaluate':
        check_data(data_path)
        evaluate(configs, run_dir / 'evaluation')
    else:
        run_autodl(configs, run_dir, data_path)


if __name__ == '__main__':
    main()
