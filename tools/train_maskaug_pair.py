"""Train standard Mamba + prior V2 masking, then V2 + identical recipe on AutoDL.

One controller owns both child processes. It verifies and saves each result,
and requests shutdown only after BOTH runs complete successfully.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.autodl_shutdown import (
    check_shutdown_command, digest, persist_results, request_shutdown, verify_results,
)

CONFIG_DIR = Path('config/nturgbd-cross-subject')
PAIR_DIR = Path('work_dir/ntu60/xsub/maskaug_pair_seed1')
RUNS = (
    ('baseline', 'ctrgcn_mamba_maskaug_pair_fast128.yaml', 'baseline_maskaug_pair'),
    ('v2', 'ctrgcn_mamba_robust_v2_maskaug_pair_fast128.yaml', 'v2_maskaug_pair'),
)


def validate_pair(baseline, v2, original):
    """Only architecture/output paths may differ from the previous V2 recipe."""
    expected = deepcopy(original)
    if (expected['model'] != 'model.ctrgcn_mamba_robust_v2.Model'
            or expected['train_feeder_args'].get('joint_mask_prob') != 0.25
            or expected['train_feeder_args'].get('joint_mask_max') != 5
            or expected['test_feeder_args'].get('joint_mask_prob', 0) != 0):
        raise ValueError('Original V2 augmentation recipe changed; review it before training.')
    if (expected.get('weights') is not None or expected.get('start_epoch', 0) != 0
            or expected.get('phase', 'train') != 'train'
            or expected.get('seed', 1) != 1 or expected.get('num_epoch') != 65):
        raise ValueError('Expected the previous 65-epoch, seed-1, from-scratch V2 recipe.')
    for name, candidate in (('v2', v2), ('baseline', baseline)):
        wanted = deepcopy(expected)
        wanted['work_dir'] = candidate['work_dir']
        if name == 'baseline':
            wanted['model'] = 'model.ctrgcn_mamba.Model'
            for key in ('use_reliability', 'use_local_compensation', 'history_window'):
                wanted['model_args'].pop(key)
        if wanted != candidate:
            raise ValueError(name + ' differs from the previous V2 recipe beyond architecture/work_dir.')
    if baseline['work_dir'] == v2['work_dir']:
        raise ValueError('The two runs must have separate output directories.')


def load_jobs(repository=ROOT):
    repository = Path(repository).resolve()
    jobs = []
    for role, filename, log_name in RUNS:
        config_path = repository / CONFIG_DIR / filename
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        work = (repository / config['work_dir']).resolve()
        if not work.is_relative_to(repository / 'work_dir') or work == repository / 'work_dir':
            raise ValueError('Output directory must be inside this repository/work_dir.')
        jobs.append(dict(role=role, config_path=config_path, config=config, work=work,
                         log=repository / 'logs' / (log_name + '.log')))
    original = yaml.safe_load((repository / CONFIG_DIR / 'ctrgcn_mamba_robust_v2_fast128.yaml')
                              .read_text(encoding='utf-8'))
    validate_pair(jobs[0]['config'], jobs[1]['config'], original)
    if jobs[0]['work'] == jobs[1]['work']:
        raise ValueError('Output directories resolve to the same path.')
    old_paths = [(repository / original['work_dir']).resolve(),
                 (repository / 'work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128').resolve()]
    for job in jobs:
        if job['work'] in old_paths:
            raise ValueError('Paired experiments must preserve the previous trained models.')
    return jobs


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def plan_for(jobs):
    return [dict(role=job['role'], config=str(job['config_path']),
                 config_sha256=digest(job['config_path']), work_dir=str(job['work']),
                 training_log=str(job['log'])) for job in jobs]


def verified_run(job):
    saved = yaml.safe_load((job['work'] / 'config.yaml').read_text(encoding='utf-8'))
    for key, value in job['config'].items():
        if key != 'work_dir' and saved.get(key) != value:
            raise ValueError('Saved run config changed: {} / {}'.format(job['role'], key))
    if (saved.get('seed') != 1 or saved.get('device') != [0]
            or saved.get('phase') != 'train' or saved.get('weights') is not None
            or saved.get('start_epoch') != 0):
        raise ValueError('Saved run is not the requested seed-1 fresh GPU-0 training.')
    return verify_results(job['work'], job['log'], job['config']['num_epoch'])


def check_outputs(jobs, queue_dir, resume):
    state_path = queue_dir / 'pair_summary.json'
    if not resume:
        for path in [queue_dir] + [path for job in jobs for path in (job['work'], job['log'])]:
            if path.exists():
                raise ValueError('Refusing to overwrite existing results/logs: {}. '
                                 'Use --resume only for already-complete verified stages.'.format(path))
        return None
    state = json.loads(state_path.read_text(encoding='utf-8'))
    if state['plan'] != plan_for(jobs):
        raise ValueError('Resume plan/config hashes differ from the saved pair.')
    previous_missing = False
    for job in jobs:
        exists = job['work'].exists() or job['log'].exists()
        if exists:
            if previous_missing:
                raise ValueError('Cannot resume out-of-order stages.')
            summary = verified_run(job)
            recorded = state.get('results', {}).get(job['role'])
            if recorded and digest(Path(summary['checkpoint'])) != recorded['checkpoint_sha256']:
                raise ValueError('Completed checkpoint hash changed; refusing resume.')
        else:
            previous_missing = True
    return state


def run_training(job, snapshot, repository, environment):
    # Exclusive create prevents truncating a previous training log.
    with job['log'].open('x', encoding='utf-8') as output:
        command = [sys.executable, '-u', str(repository / 'main.py'), '--config', str(snapshot),
                   '--work-dir', str(job['work']), '--device', '0', '--seed', '1']
        result = subprocess.run(command, cwd=repository, env=environment,
                                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('{} training exited with code {}; inspect {}'.format(
            job['role'], result.returncode, job['log']))


def execute_pair(jobs, queue_dir, repository=ROOT, resume=False, shutdown=True):
    """Return codes AND saved artifacts gate the next run and final shutdown."""
    queue_dir, repository = Path(queue_dir), Path(repository)
    state = check_outputs(jobs, queue_dir, resume)
    queue_dir.mkdir(parents=True, exist_ok=resume)
    state = state or dict(status='ready', plan=plan_for(jobs), results={},
                          created_at_utc=datetime.now(timezone.utc).isoformat())
    state.pop('error', None)
    state.pop('shutdown_completed', None)
    state_path = queue_dir / 'pair_summary.json'
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    try:
        for number, job in enumerate(jobs, 1):
            state.update(status='running', current_stage=job['role'])
            write_json(state_path, state)
            if resume and (job['work'].exists() or job['log'].exists()):
                print('Rechecking completed stage:', job['role'], flush=True)
            else:
                snapshot = queue_dir / (job['role'] + '_config.yaml')
                snapshot.write_text(yaml.safe_dump(job['config']), encoding='utf-8')
                print('START {}/2: {}; log: {}'.format(number, job['role'], job['log']), flush=True)
                run_training(job, snapshot, repository, environment)
            summary = verified_run(job)
            output = persist_results(summary, job['work'], job['log'])
            state['results'][job['role']] = dict(summary, summary_file=str(output))
            write_json(state_path, state)
            print('SAVED {}/2: {} best {:.4f}% -> {}'.format(
                number, job['role'], summary['best_test_accuracy_percent'], output), flush=True)
        state.update(status='results_verified', current_stage=None)
        write_json(state_path, state)
        getattr(os, 'sync', lambda: None)()
        if shutdown:
            print('BOTH experiments saved and verified. Requesting AutoDL shutdown.', flush=True)
            state['status'] = 'shutdown_requested'
            write_json(state_path, state)
            getattr(os, 'sync', lambda: None)()
            request_shutdown(Path('/usr/bin/shutdown'))
            print('Shutdown command returned success; power-off requested.', flush=True)
        else:
            print('Both results verified; shutdown disabled for this invocation.', flush=True)
        return state
    except BaseException as error:
        state.update(status='failed', error=str(error), shutdown_completed=False)
        write_json(state_path, state)
        print('FAILED: {}; no further training/shutdown will be requested.'.format(error), flush=True)
        raise


@contextmanager
def controller_lock(repository):
    import fcntl
    logs = repository / 'logs'
    logs.mkdir(exist_ok=True)
    with (logs / 'maskaug_pair.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('A paired-training controller is already active.') from error
        yield


def preflight(jobs, repository):
    # Check BOTH training stacks before paying for the first 65 epochs.
    for job in jobs:
        dataset = (repository / job['config']['train_feeder_args']['data_path']).resolve()
        if not dataset.is_file() or dataset.stat().st_size == 0:
            raise ValueError('Training dataset missing/empty: ' + str(dataset))
    mode = check_shutdown_command(Path('/usr/bin/shutdown'))
    print('Shutdown invocation preflight passed; mode:', mode, flush=True)
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    subprocess.run([sys.executable, str(repository / 'main.py'), '--help'],
                   cwd=repository, env=environment, check=True, stdout=subprocess.DEVNULL)
    script = (
        'import numpy as np, torch, mamba_ssm, causal_conv1d; '
        'from feeders.feeder_ntu import Feeder; '
        'from model.ctrgcn_mamba import Model; '
        'from model.ctrgcn_mamba_robust_v2 import Model as V2; '
        'assert torch.cuda.is_available(), "CUDA is unavailable"; '
        'assert np.array_equal(torch.from_numpy(np.arange(3)).numpy(), np.arange(3)); '
        'print("PASS: training imports, CUDA, NumPy bridge and both model imports; GPU:", '
        'torch.cuda.get_device_name(0))'
    )
    subprocess.run([sys.executable, '-c', script], cwd=repository, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='check paired configs/paths only; no training/shutdown')
    parser.add_argument('--resume', action='store_true', help='reverify/skip completed stages; never resume partial training')
    args = parser.parse_args()
    jobs = load_jobs()
    queue_dir = ROOT / PAIR_DIR
    print('PASS: standard Mamba and V2 share the previous V2 augmentation/training recipe.', flush=True)
    print('Train masking: probability .25, 1..5/25 joints, fixed across time/people; test unchanged.', flush=True)
    if args.check_only:
        check_outputs(jobs, queue_dir, args.resume)
        for job in jobs:
            print('{}: {} -> {}'.format(job['role'], job['config_path'], job['work']))
        print('Configuration/path checks passed. No training or shutdown was requested.')
        return
    if not sys.platform.startswith('linux'):
        parser.error('Launch training inside the Linux AutoDL instance; --check-only works locally.')
    with controller_lock(ROOT):
        check_outputs(jobs, queue_dir, args.resume)
        preflight(jobs, ROOT)
        execute_pair(jobs, queue_dir, ROOT, resume=args.resume)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('STOPPED: {}. Inspect the pair/training logs; shutdown was not confirmed.'.format(error),
              file=sys.stderr, flush=True)
        sys.exit(1)
