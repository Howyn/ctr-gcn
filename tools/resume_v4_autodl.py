"""Continue an interrupted V4 from saved weights, evaluate, audit, then shut down.

Legacy checkpoints contain model weights only. Momentum, AMP scaler and random
states cannot be recovered; this is explicitly recorded as weights-only recovery.
The interrupted directory is never overwritten or deleted.
"""

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_v3_autodl as pipeline
from tools.autodl_shutdown import atomic_copy, digest, persist_results, verify_results
from tools.train_maskaug_pair import run_training, write_json

PATTERN = re.compile(r'runs-(\d+)-(\d+)\.pt')


def checkpoint_state(path):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint is not a model state dictionary.')
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise ValueError('Checkpoint has unexpected non-tensor values.')
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError('Checkpoint contains non-finite weights.')
    if state.get('fc.weight', torch.empty(0)).shape != (60, 256):
        raise ValueError('Checkpoint is not the expected NTU60 model.')
    return state


def historical_accuracy(path):
    """Recover full precision clean accuracy from the saved confusion matrix."""
    with path.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 61 or any(len(row) != 60 for row in rows):
        raise ValueError('Incomplete NTU60 confusion matrix: ' + str(path))
    matrix = [[int(value) for value in row] for row in rows[1:]]
    if any(value < 0 for row in matrix for value in row) or sum(map(sum, matrix)) != 16487:
        raise ValueError('Confusion matrix does not cover the complete test split.')
    return sum(matrix[i][i] for i in range(60)) / 16487


def select_checkpoints(source, final_epoch=65):
    candidates = sorted(((int(match[1]), int(match[2]), path)
                         for path in source.glob('runs-*.pt')
                         if (match := PATTERN.fullmatch(path.name))), reverse=True)
    rejected, latest = [], None
    for epoch, step, path in candidates:
        if not 1 <= epoch <= final_epoch:
            continue
        try:
            state = checkpoint_state(path)
            shapes = {key: tuple(value.shape) for key, value in state.items()}
            del state
        except (MemoryError, KeyboardInterrupt):
            raise
        except Exception as error:
            rejected.append(dict(path=str(path), reason=str(error)))
            continue
        latest = dict(epoch=epoch, step=step, path=path, sha256=digest(path), shapes=shapes)
        break
    if latest is None:
        raise ValueError('No readable, finite checkpoint; do not restart from an empty/corrupt file.')
    # Every fully evaluated epoch has a confusion matrix. Select by clean data,
    # never by the subsequent occlusion test. The chosen candidate is re-evaluated
    # on the live clean test split by main.py before continuing.
    evaluated = []
    for epoch, step, path in candidates:
        csv_path = source / ('epoch{}_test_each_class_acc.csv'.format(epoch))
        if epoch <= latest['epoch'] and csv_path.is_file():
            try:
                evaluated.append((historical_accuracy(csv_path), -epoch, step, path))
            except (ValueError, OSError):
                continue
    previous_best = None
    for accuracy, negative_epoch, step, path in sorted(evaluated, reverse=True):
        try:
            state = checkpoint_state(path)
            if {key: tuple(value.shape) for key, value in state.items()} != latest['shapes']:
                raise ValueError('Historical checkpoint shapes differ from the latest checkpoint.')
            del state
        except (MemoryError, KeyboardInterrupt):
            raise
        except Exception as error:
            rejected.append(dict(path=str(path), reason=str(error)))
            continue
        previous_best = dict(epoch=-negative_epoch, path=path, accuracy=accuracy, sha256=digest(path))
        break
    if previous_best is None:
        raise ValueError('No complete prior clean evaluation/checkpoint pair; inspect the original run.')
    latest.pop('shapes')
    return latest, previous_best, rejected


def build_resume_plan(repository=ROOT, workers=2):
    repository = Path(repository).resolve()
    plan = pipeline.build_plan(repository, variant='v4')
    source = plan['job']['work']
    old_config = yaml.safe_load((source / 'config.yaml').read_text(encoding='utf-8'))
    wanted = plan['job']['config']
    for key, value in wanted.items():
        if key != 'work_dir' and old_config.get(key) != value:
            raise ValueError('Interrupted training recipe differs: ' + key)
    if old_config.get('weights') is not None or old_config.get('start_epoch', 0) != 0:
        raise ValueError('Expected the original interrupted fresh V4 run.')
    archived = source / plan['algorithm_source'].name
    if digest(archived) != digest(plan['algorithm_source']):
        raise ValueError('V4 model source changed since the interrupted run.')
    latest, previous, rejected = select_checkpoints(source)
    if latest['epoch'] >= 65:
        raise ValueError('Epoch 65 weights already exist; inspect completion instead of restarting training.')
    number = 1
    while True:
        tag = 'v4_state_resume{}'.format(number)
        work = source.with_name(source.name + '_resume{}'.format(number))
        log = repository / 'logs' / ('ntu60_xsub_joint_mamba_robust_' + tag + '.log')
        output = repository / 'work_dir/ntu60/xsub' / ('occlusion20_robust_' + tag + '_seed2026')
        reserved = [work, log, output]
        reserved.extend(repository / 'logs' / name for name in (
            tag + '_pipeline_state.json', tag + '_pipeline_config.yaml',
            'occlusion20_robust_' + tag + '_seed2026.log',
            'audit_occlusion20_' + tag + '_seed2026.log',
            'audit_occlusion20_' + tag + '_seed2026.json'))
        if not any(path.exists() for path in reserved):
            break
        number += 1
    plan = deepcopy(plan)
    plan['recovery_config_sha256'] = digest(source / 'config.yaml')
    plan['label'] = 'V4 weights-only resume{}'.format(number)
    config = plan['job']['config']
    config.update(work_dir=str(work), weights=str(latest['path']), start_epoch=latest['epoch'],
                  prior_best_weights=str(previous['path']), num_worker=workers)
    plan['job'].update(work=work, log=log)
    plan.update(output_dir=output, source_dir=source, recovery=latest, previous_best=previous,
                rejected_checkpoints=rejected)
    names = dict(evaluation_log='occlusion20_robust_' + tag + '_seed2026.log',
                 audit_log='audit_occlusion20_' + tag + '_seed2026.log',
                 audit_report='audit_occlusion20_' + tag + '_seed2026.json',
                 state_path=tag + '_pipeline_state.json', snapshot=tag + '_pipeline_config.yaml')
    for key, name in names.items():
        plan[key] = repository / 'logs' / name
    return plan


def execute_resume(plan, repository=ROOT, shutdown=True):
    repository = Path(repository)
    job = plan['job']
    for path in (job['work'], job['log'], plan['snapshot'], plan['state_path'], plan['output_dir'],
                 plan['evaluation_log'], plan['audit_log'], plan['audit_report']):
        if path.exists():
            raise ValueError('Refusing to overwrite recovery output: ' + str(path))
    if shutil.disk_usage(repository).free < 512 * 1024 ** 2:
        raise ValueError('Less than 512 MiB disk space remains; free disk space before continuing.')
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    state = dict(status='resuming', algorithm_id=plan['algorithm_id'],
                 created_at_utc=datetime.now(timezone.utc).isoformat(), results_verified=False,
                 recovery_mode='weights_only_optimizer_scaler_rng_not_restored',
                 source_dir=str(plan['source_dir']), resume_checkpoint=str(plan['recovery']['path']),
                 source_config_sha256=plan['recovery_config_sha256'],
                 resume_checkpoint_sha256=plan['recovery']['sha256'],
                 completed_training_epochs=plan['recovery']['epoch'],
                 previous_best_checkpoint=str(plan['previous_best']['path']),
                 previous_best_sha256=plan['previous_best']['sha256'],
                 rejected_checkpoints=plan['rejected_checkpoints'],
                 model_source_sha256=digest(plan['algorithm_source']),
                 training_num_worker=job['config']['num_worker'],
                 reference_checkpoint=str(plan['reference_weights']),
                 reference_sha256=plan['reference_sha256'])

    def save():
        write_json(plan['state_path'], state)
        if job['work'].is_dir():
            write_json(job['work'] / 'pipeline_summary.json', state)

    try:
        save()
        for path, checksum in ((plan['recovery']['path'], state['resume_checkpoint_sha256']),
                               (plan['previous_best']['path'], state['previous_best_sha256'])):
            if digest(path) != checksum:
                raise ValueError('Source checkpoint changed before recovery.')
        plan['snapshot'].write_text(yaml.safe_dump(job['config']), encoding='utf-8')
        run_training(job, plan['snapshot'], repository, environment)
        saved = yaml.safe_load((job['work'] / 'config.yaml').read_text(encoding='utf-8'))
        for key, value in job['config'].items():
            if saved.get(key) != value:
                raise ValueError('Saved recovery configuration differs: ' + key)
        summary = verify_results(job['work'], job['log'], 65)
        summary.update(recovery_mode=state['recovery_mode'], source_dir=state['source_dir'],
                       resumed_after_epoch=plan['recovery']['epoch'])
        result = persist_results(summary, job['work'], job['log'])
        archived = job['work'] / plan['algorithm_source'].name
        if digest(archived) != state['model_source_sha256'] or digest(plan['algorithm_source']) != state['model_source_sha256']:
            raise ValueError('V4 model source changed during recovery.')
        atomic_copy(archived, job['work'] / 'model_source_snapshot.py')
        state.update(status='evaluating', training_summary=str(result))
        save()
        checkpoint = Path(summary['best_checkpoint'])
        pipeline.run_evaluation(plan, checkpoint, repository, environment)
        state.update(status='auditing')
        save()
        evaluation = pipeline.verify_evaluation(plan, checkpoint, repository, environment)
        state.update(status='complete', results_verified=True, evaluation=evaluation)
        save()
        for row in evaluation['results']:
            print('{}: clean {:.4f}%, 20% occluded {:.4f}%'.format(
                row['model'], row['clean_top1_percent'], row['occluded_top1_percent']), flush=True)
        getattr(os, 'sync', lambda: None)()
        if shutdown:
            state['status'] = 'shutdown_requested'
            save()
            getattr(os, 'sync', lambda: None)()
            print('Recovery + clean/20% inference saved and audited. Requesting AutoDL shutdown.', flush=True)
            pipeline.request_shutdown(Path('/usr/bin/shutdown'))
        return state
    except BaseException as error:
        state.update(status='failed', error=str(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--num-worker', type=int, default=2)
    args = parser.parse_args()
    if args.num_worker < 0:
        parser.error('Worker count must be nonnegative.')
    pipeline.require_idle_training(ROOT)
    plan = build_resume_plan(workers=args.num_worker)
    latest = plan['recovery']
    print('Resume checkpoint:', latest['path'], flush=True)
    print('Continue epochs {}..65; LR uses original absolute epoch schedule.'.format(latest['epoch'] + 1), flush=True)
    print('Previous clean best retained:', plan['previous_best']['path'], flush=True)
    print('WARNING: old files contain weights only; optimizer/scaler/RNG states restart.', flush=True)
    print('New training output:', plan['job']['work'], flush=True)
    print('Training log:', plan['job']['log'], flush=True)
    print('Paired clean/20% output:', plan['output_dir'], flush=True)
    for rejected in plan['rejected_checkpoints']:
        print('Skipped invalid checkpoint:', rejected['path'], rejected['reason'], flush=True)
    if args.check_only:
        print('PASS: checkpoints and recipe checked; no training, evaluation or shutdown requested.')
        return
    if not sys.platform.startswith('linux'):
        parser.error('Run recovery inside the Linux AutoDL instance.')
    with pipeline.controller_lock(ROOT, 'v4'):
        pipeline.require_idle_training(ROOT)
        # Catch missing sklearn/CUDA/NumPy/Mamba and power-off support before training.
        pipeline.training_preflight([plan['job']], ROOT)
        for filename in ('evaluate_occlusion.py', 'audit_occlusion.py'):
            pipeline.subprocess.run([sys.executable, str(ROOT / 'tools' / filename), '--help'],
                                    cwd=ROOT, check=True, stdout=pipeline.subprocess.DEVNULL)
        execute_resume(plan)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('FAILED: {}. No shutdown requested after failure; inspect the recovery log.'.format(error), file=sys.stderr)
        sys.exit(1)
