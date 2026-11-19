"""Run a clean V3/V4 experiment, paired inference, artifact checks, then power off.

The comparison uses the existing clean-trained V2 checkpoint. No new training
masks, extra training runs or independent training-only shutdown watcher.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.autodl_shutdown import (atomic_copy, digest, persist_results, process_identity,
                                  request_shutdown, required_file)
from tools.check_clean_comparison import check_comparison
from tools.train_maskaug_pair import preflight as training_preflight
from tools.train_maskaug_pair import run_training, verified_run, write_json

CONFIG = Path('config/nturgbd-cross-subject/ctrgcn_mamba_robust_v3_clean_fast128.yaml')
REFERENCE = Path('work_dir/ntu60/xsub/ctrgcn_mamba_robust_v2_clean_joint_fast128')
STANDARD = Path('work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128')
OUTPUT = Path('work_dir/ntu60/xsub/occlusion20_robust_v3_clean_seed2026')
V4_ALGORITHM_ID = 'v4_soft_pool_consistent_selective_scan'


def require_idle_training(repository, proc_root=Path('/proc')):
    """Refuse overlapping old/new training rather than stopping unknown PIDs."""
    repository = Path(repository).resolve()
    if not proc_root.is_dir():
        return
    active = []
    for process in proc_root.iterdir():
        if not process.name.isdigit() or process_identity(int(process.name), proc_root) is None:
            continue
        try:
            arguments = (process / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            cwd = (process / 'cwd').resolve(strict=True)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if cwd == repository and any((cwd / value).resolve() == repository / 'main.py'
                                     for value in arguments[1:] if not value.startswith('-')):
            active.append(process.name)
    if active:
        raise ValueError('Training is already active (PID {}); do not launch an overlapping run.'.format(', '.join(active)))


def best_reference(directory):
    text = required_file(directory / 'log.txt').read_text(encoding='utf-8')
    epochs = re.findall(r'Epoch number:\s*(\d+)', text)
    if not epochs:
        raise ValueError('Reference V2 has no completed best-epoch summary.')
    checkpoints = list(directory.glob('runs-{}-*.pt'.format(int(epochs[-1]))))
    if len(checkpoints) == 1:
        return required_file(checkpoints[0]).resolve()
    if not checkpoints:
        return required_file(directory / 'best.pt').resolve()
    raise ValueError('Reference V2 has ambiguous best-epoch checkpoints.')


def build_plan(repository=ROOT, variant='v3'):
    if variant not in ('v3', 'v4'):
        raise ValueError('Supported experiment identifiers: v3, v4.')
    repository = Path(repository).resolve()
    config_path = repository / str(CONFIG).replace('v3', variant)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    standard = yaml.safe_load((repository / STANDARD / 'config.yaml').read_text(encoding='utf-8'))
    reference_dir = repository / REFERENCE
    reference_config = yaml.safe_load((reference_dir / 'config.yaml').read_text(encoding='utf-8'))
    candidate_model = 'model.ctrgcn_mamba_robust_{}.Model'.format(variant)
    label = variant.upper()
    run_tag = 'v4_state' if variant == 'v4' else 'v3_clean'
    log_tag = 'v4_state' if variant == 'v4' else 'v3'
    if (config['model'] != candidate_model
            or reference_config['model'] != 'model.ctrgcn_mamba_robust_v2.Model'):
        raise ValueError('Expected {} and the existing clean-trained V2 reference.'.format(label))
    check_comparison(standard, config)
    check_comparison(standard, reference_config)
    if variant == 'v4' and config['model_args'].get('algorithm_id') != V4_ALGORITHM_ID:
        raise ValueError('V4 must use the new consistent-parameter algorithm, not the name-only alias.')
    if (config.get('num_epoch') != 65 or config.get('seed', 1) != 1
            or config.get('save_epoch') != 0):
        raise ValueError('Expected the original 65-epoch, seed-1 recipe with every epoch saved.')
    work = (repository / config['work_dir']).resolve()
    if (not work.is_relative_to(repository / 'work_dir') or work == repository / 'work_dir'
            or work in (reference_dir.resolve(), (repository / STANDARD).resolve())):
        raise ValueError('{} must use a new work directory inside this repository/work_dir.'.format(label))
    reference_weights = best_reference(reference_dir)
    logs = repository / 'logs'
    return dict(
        variant=variant, label=label, candidate_model=candidate_model,
        architecture_equivalent_to='model.ctrgcn_mamba_robust_v3.Model' if variant == 'v3' else None,
        algorithm_id=V4_ALGORITHM_ID if variant == 'v4' else 'v3_residual_write_selector',
        algorithm_source=ROOT / 'model/ctrgcn_mamba_robust_{}.py'.format(variant),
        job=dict(role=variant, config_path=config_path, config=config, work=work,
                 log=logs / 'ntu60_xsub_joint_mamba_robust_{}.log'.format(run_tag)),
        reference_dir=reference_dir, reference_weights=reference_weights,
        reference_sha256=digest(reference_weights), reference_config=reference_config,
        output_dir=repository / ('work_dir/ntu60/xsub/occlusion20_robust_{}_seed2026'.format(run_tag)),
        evaluation_log=logs / 'occlusion20_robust_{}_seed2026.log'.format(run_tag),
        audit_log=logs / 'audit_occlusion20_{}_seed2026.log'.format(run_tag),
        audit_report=logs / 'audit_occlusion20_{}_seed2026.json'.format(run_tag),
        state_path=logs / '{}_pipeline_state.json'.format(log_tag),
        snapshot=logs / '{}_pipeline_config.yaml'.format(log_tag),
        cuda_log=logs / 'check_mamba_{}_pipeline.log'.format(log_tag),
        legacy_watcher_log=logs / 'autodl_shutdown_mamba_robust_{}.log'.format(run_tag),
    )


def check_fresh(plan):
    paths = [plan['job']['work'], plan['job']['log']]
    paths += [plan[key] for key in ('output_dir', 'evaluation_log', 'audit_log', 'audit_report',
                                   'state_path', 'snapshot', 'legacy_watcher_log')]
    for path in paths:
        if path.exists():
            raise ValueError('Existing {} run/log/evaluation found; refusing overwrite: {}'.format(plan['label'], path))


def logged_command(command, log, repository, environment, exclusive=True):
    with log.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        subprocess.run(command, cwd=repository, env=environment, stdin=subprocess.DEVNULL,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def preflight(plan, repository):
    training_preflight([plan['job']], repository)
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # Import evaluation/audit dependencies NOW, before spending the training time.
    for filename in ('evaluate_occlusion.py', 'audit_occlusion.py'):
        subprocess.run([sys.executable, str(repository / 'tools' / filename), '--help'],
                       cwd=repository, env=environment, check=True, stdout=subprocess.DEVNULL)
    print('Checking {} CUDA/AMP, clean-path gradients and masked forward; log:'.format(plan['label']),
          plan['cuda_log'], flush=True)
    logged_command([sys.executable, '-u', str(repository / 'tools/check_robust_mamba.py'),
                    '--variant', plan['variant']], plan['cuda_log'], repository, environment, exclusive=False)
    print('PASS: {} CUDA check. Training will use fresh, disposable-check-independent weights.'.format(plan['label']), flush=True)


def run_evaluation(plan, checkpoint, repository, environment):
    command = [sys.executable, '-u', str(repository / 'tools/evaluate_occlusion.py'),
               '--baseline-dir', str(plan['reference_dir']), '--mamba-dir', str(plan['job']['work']),
               '--baseline-weights', str(plan['reference_weights']), '--mamba-weights', str(checkpoint),
               '--output-dir', str(plan['output_dir']), '--rate', '0.2', '--mask-seed', '2026',
               '--batch-size', '128', '--num-worker', '2', '--device', '0']
    logged_command(command, plan['evaluation_log'], repository, environment)


def verify_evaluation(plan, checkpoint, repository, environment):
    output = plan['output_dir']
    for name in ('summary.json', 'summary.csv', 'predictions.csv', 'scores.npz', 'masks.npz'):
        required_file(output / name)
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    expected = dict(status='complete', protocol='per_sample_fixed_joints_zero_after_test_crop_resize',
                    rate=0.2, missing_joints=5, mask_seed=2026, samples=16487,
                    same_joints_for_all_frames_and_people=True, same_training_feeder_args=True)
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError('Evaluation protocol/completion mismatch: ' + key)
    for role, path, model, checksum in (
            ('baseline', plan['reference_weights'], 'model.ctrgcn_mamba_robust_v2.Model',
             plan['reference_sha256']),
            ('mamba', checkpoint, plan['candidate_model'], digest(checkpoint))):
        information = summary['checkpoints'][role]
        if (Path(information['checkpoint']).resolve() != Path(path).resolve()
                or information['checkpoint_sha256'] != checksum or digest(path) != checksum
                or information['model'] != model):
            raise ValueError('Evaluation used a different checkpoint/model: ' + role)
    if plan['variant'] == 'v4':
        candidate = summary['checkpoints']['mamba']
        if (candidate.get('model_args', {}).get('algorithm_id') != V4_ALGORITHM_ID
                or candidate.get('parameters') != 1902865):
            raise ValueError('Evaluation does not identify the actual V4 algorithm/parameter count.')
    rows = {row['model']: row for row in summary['results']}
    if len(summary['results']) != 2 or set(rows) != {'baseline', 'mamba'}:
        raise ValueError('Expected exactly the V2 and {} evaluation rows.'.format(plan['label']))
    for row in rows.values():
        clean, masked = row['clean_top1_percent'], row['occluded_top1_percent']
        if (not all(math.isfinite(value) and 0 <= value <= 100 for value in (clean, masked))
                or not math.isfinite(row['drop_pp'])
                or abs(clean - masked - row['drop_pp']) > 1e-8):
            raise ValueError('Invalid evaluation accuracy/drop.')
    # Independently recompute masks, labels, checkpoint hashes and metrics from
    # the saved score arrays. This path allocates no skeleton array or live model.
    command = [sys.executable, '-u', str(repository / 'tools/audit_occlusion.py'),
               '--result-dir', str(output), '--artifacts-only', '--report', str(plan['audit_report'])]
    logged_command(command, plan['audit_log'], repository, environment)
    report = json.loads(required_file(plan['audit_report']).read_text(encoding='utf-8'))
    if (report.get('scope') != 'saved_artifacts_only' or report.get('samples') != 16487
            or report.get('mask_seed') != 2026 or report.get('missing_joints') != 5
            or report.get('same_training_inputs') is not True):
        raise ValueError('Artifact audit report does not match the requested comparison.')
    for role, row in rows.items():
        for kind, key in (('clean', 'clean_top1_percent'), ('occluded', 'occluded_top1_percent')):
            audited = report['metrics'][role + '_' + kind]['top1_percent']
            if not math.isfinite(audited) or abs(audited - row[key]) > 1e-8:
                raise ValueError('Audited accuracy differs from evaluation summary.')
    return dict(summary_file=str(output / 'summary.json'),
                summary_sha256=digest(output / 'summary.json'), audit_report=str(plan['audit_report']),
                audit_sha256=digest(plan['audit_report']), results=summary['results'])


def execute_pipeline(plan, repository=ROOT, shutdown=True):
    repository = Path(repository)
    check_fresh(plan)
    job = plan['job']
    state = dict(status='training', created_at_utc=datetime.now(timezone.utc).isoformat(),
                 experiment=plan['variant'], architecture_equivalent_to=plan['architecture_equivalent_to'],
                 algorithm_id=plan['algorithm_id'], model_source_sha256=digest(plan['algorithm_source']),
                 config=str(job['config_path']), config_sha256=digest(job['config_path']),
                 reference_checkpoint=str(plan['reference_weights']),
                 reference_sha256=plan['reference_sha256'], results_verified=False)
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')

    def save_state():
        write_json(plan['state_path'], state)
        if job['work'].is_dir():
            write_json(job['work'] / 'pipeline_summary.json', state)

    try:
        save_state()
        plan['snapshot'].write_text(yaml.safe_dump(job['config']), encoding='utf-8')
        print('START training {}, 65 epochs, training joint-mask probability=0.0; log:'.format(plan['label']), job['log'], flush=True)
        run_training(job, plan['snapshot'], repository, environment)
        training = verified_run(job)
        training_output = persist_results(training, job['work'], job['log'])
        if plan['variant'] == 'v4':
            archived_source = required_file(job['work'] / plan['algorithm_source'].name)
            if (digest(archived_source) != state['model_source_sha256']
                    or digest(plan['algorithm_source']) != state['model_source_sha256']):
                raise ValueError('Model source changed during training; refusing mismatched evaluation.')
            atomic_copy(archived_source, job['work'] / 'model_source_snapshot.py')
        state.update(status='evaluating', training_summary=str(training_output))
        save_state()
        checkpoint = Path(training['best_checkpoint'])
        print('SAVED training. START paired V2/{} clean + 20% inference; log:'.format(plan['label']),
              plan['evaluation_log'], flush=True)
        run_evaluation(plan, checkpoint, repository, environment)
        state['status'] = 'auditing'
        save_state()
        evaluation = verify_evaluation(plan, checkpoint, repository, environment)
        state.update(status='complete', results_verified=True, evaluation=evaluation)
        save_state()
        for row in evaluation['results']:
            name = 'V2 reference' if row['model'] == 'baseline' else plan['label'] + ' run'
            print('{}: clean {:.4f}%, 20% occluded {:.4f}%'.format(
                name, row['clean_top1_percent'], row['occluded_top1_percent']), flush=True)
        print('SAVED AND VERIFIED training + inference + artifact audit.', flush=True)
        getattr(os, 'sync', lambda: None)()
        if shutdown:
            state['status'] = 'shutdown_requested'
            save_state()
            getattr(os, 'sync', lambda: None)()
            print('Results on disk. Requesting AutoDL shutdown.', flush=True)
            request_shutdown(Path('/usr/bin/shutdown'))
            print('Shutdown command returned success; power-off requested.', flush=True)
        return state
    except BaseException as error:
        state.update(status='failed', error=str(error))
        save_state()
        print('STOPPED: {}; no further shutdown request will be made. Inspect logs.'.format(error), flush=True)
        raise


@contextmanager
def controller_lock(repository, variant='v3'):
    import fcntl
    (repository / 'logs').mkdir(exist_ok=True)
    with (repository / 'logs' / '{}_pipeline.lock'.format(variant)).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('A {} training/evaluation controller is already active.'.format(variant.upper())) from error
        yield


def main(variant='v3'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='check configs/reference/paths; no CUDA, training or shutdown')
    args = parser.parse_args()
    require_idle_training(ROOT)
    plan = build_plan(variant=variant)
    check_fresh(plan)
    if variant == 'v4':
        print('ALGORITHM:', plan['algorithm_id'], flush=True)
        print('V4 uses soft visibility pooling and consistent state/parameter correction; extra parameters: 17921.', flush=True)
    print('PASS: {} and existing V2 match the original clean-training recipe; no added training masking.'.format(plan['label']), flush=True)
    print('Reference V2 checkpoint:', plan['reference_weights'], flush=True)
    print('{} output:'.format(plan['label']), plan['job']['work'], flush=True)
    print('20% paired inference output:', plan['output_dir'], flush=True)
    if args.check_only:
        print('Configuration/reference/path checks passed. No training or shutdown requested.')
        return
    if not sys.platform.startswith('linux'):
        parser.error('Launch this pipeline inside the Linux AutoDL instance.')
    with controller_lock(ROOT, variant):
        require_idle_training(ROOT)
        check_fresh(plan)
        preflight(plan, ROOT)
        execute_pipeline(plan, ROOT)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('FAILED: {}. Training/evaluation/shutdown was not completed; inspect pipeline logs.'.format(error),
              file=sys.stderr, flush=True)
        sys.exit(1)
