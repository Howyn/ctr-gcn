"""Verify a completed CTR-GCN run, persist its results, then shut down AutoDL.

Attach this watcher to an already-running training PID. No training is launched
or stopped here. Only the Python standard library is needed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


def process_identity(pid, proc_root=Path('/proc')):
    """Return the kernel start time, treating exited/zombie processes as gone."""
    try:
        stat = (proc_root / str(pid) / 'stat').read_text()
    except FileNotFoundError:
        return None
    # comm can contain spaces or parentheses; fields after its final ')' are fixed.
    fields = stat[stat.rfind(')') + 2:].split()
    return None if fields[0] == 'Z' else fields[19]


def option_value(arguments, name):
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(name + '='):
            return value.split('=', 1)[1]
    return None


def validate_training_process(pid, repository, config, work_dir):
    identity = process_identity(pid)
    if identity is None:
        raise ValueError('Training PID is not running; no shutdown has been armed.')
    process = Path('/proc') / str(pid)
    cwd = (process / 'cwd').resolve(strict=True)
    arguments = (process / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    if cwd != repository or not any(
            (cwd / value).resolve() == repository / 'main.py'
            for value in arguments[1:] if not value.startswith('-')):
        raise ValueError('PID is not running this repository\'s main.py.')
    if option_value(arguments, '--phase') not in (None, 'train'):
        raise ValueError('PID is not a training run.')
    config_argument = option_value(arguments, '--config')
    if config_argument is None or (cwd / config_argument).resolve() != config:
        raise ValueError('Training config does not match the watched config.')
    work_argument = option_value(arguments, '--work-dir')
    if work_argument is not None and (cwd / work_argument).resolve() != work_dir:
        raise ValueError('Training work directory does not match the watched directory.')
    if process_identity(pid) != identity:
        raise ValueError('Training process changed while attaching the watcher.')
    return identity


def required_file(path, allow_empty=False):
    if not path.is_file() or (not allow_empty and path.stat().st_size == 0):
        raise ValueError('Missing or empty result file: {}'.format(path))
    return path


def last_match(pattern, text, description):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        raise ValueError('Completion summary is missing: {}'.format(description))
    return matches[-1]


def verify_results(work_dir, training_log, expected_epochs):
    """The non-child process's final summary and artifacts prove completion."""
    required_file(training_log)
    console = training_log.read_text(encoding='utf-8', errors='replace')
    if 'Traceback (most recent call last)' in console:
        raise ValueError('Training log contains a traceback; leaving the instance running.')
    log_path = required_file(work_dir / 'log.txt')
    log = log_path.read_text(encoding='utf-8', errors='replace')
    config = required_file(work_dir / 'config.yaml').read_text(encoding='utf-8')
    configured_epochs = int(last_match(r'^num_epoch:\s*(\d+)\s*$', config, 'num_epoch'))
    configured_work = last_match(r'^work_dir:\s*(.+)\s*$', config, 'work_dir').strip().strip("'\"")
    repository = Path(__file__).resolve().parents[1]
    if (repository / configured_work).resolve() != work_dir.resolve():
        raise ValueError('Saved config points to a different work directory.')
    if configured_epochs != expected_epochs:
        raise ValueError('Saved config does not request the expected epoch count.')
    epochs = [int(value) for value in re.findall(r'Training epoch: (\d+)\b', log)]
    if not epochs or max(epochs) != expected_epochs:
        raise ValueError('The requested final training epoch was not reached.')
    accuracy = float(last_match(r'Best accuracy:\s*(\S+)', log, 'best accuracy'))
    best_epoch = int(last_match(r'Epoch number:\s*(\d+)', log, 'best epoch'))
    last_match(r'Model total number of params:\s*(\d+)', log, 'model size')
    last_match(r'seed:\s*(\d+)', log, 'final summary seed')
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError('Invalid final accuracy.')
    if not 1 <= best_epoch <= expected_epochs:
        raise ValueError('Invalid best epoch.')
    best_weights = list(work_dir.glob('runs-{}-*.pt'.format(best_epoch)))
    if len(best_weights) != 1:
        raise ValueError('Expected exactly one checkpoint for the best epoch.')
    best_weights = required_file(best_weights[0])
    final_weights = list(work_dir.glob('runs-{}-*.pt'.format(expected_epochs)))
    if len(final_weights) != 1:
        raise ValueError('Final epoch checkpoint is missing or ambiguous.')
    artifacts = [
        required_file(work_dir / 'config.yaml'), log_path, best_weights,
        required_file(final_weights[0]),
        required_file(work_dir / 'epoch1_test_score.pkl'),
        required_file(work_dir / 'epoch1_test_each_class_acc.csv'),
        required_file(best_weights.with_name(best_weights.stem + '_right.txt')),
        required_file(best_weights.with_name(best_weights.stem + '_wrong.txt'), allow_empty=True),
    ]
    return {
        'status': 'results_verified',
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'completed_epochs': expected_epochs,
        'best_epoch': best_epoch,
        'best_test_accuracy': accuracy,
        'best_test_accuracy_percent': accuracy * 100,
        'checkpoint': str(best_weights.resolve()),
        'training_console_log': str(training_log.resolve()),
        'artifacts': [str(path.resolve()) for path in artifacts],
    }


def atomic_copy(source, destination):
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with source.open('rb') as stream:
                shutil.copyfileobj(stream, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        except BaseException:
            temporary.close()
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def persist_results(summary, work_dir, training_log):
    source = Path(summary['checkpoint'])
    target = work_dir / 'best.pt'
    atomic_copy(source, target)
    checksum = digest(source)
    if digest(target) != checksum:
        raise ValueError('Best checkpoint copy failed checksum verification.')
    atomic_copy(training_log, work_dir / 'training_console.log')
    summary.update(best_checkpoint=str(target.resolve()), checkpoint_sha256=checksum)
    output = work_dir / 'result_summary.json'
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=work_dir,
                                     delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            json.dump(summary, temporary, indent=2, ensure_ascii=False)
            temporary.write('\n')
            temporary.flush()
            os.fsync(temporary.fileno())
        except BaseException:
            temporary.close()
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True, help='PID of the running training parent')
    parser.add_argument('--config', default='config/nturgbd-cross-subject/fast128.yaml')
    parser.add_argument('--work-dir', default='work_dir/ntu60/xsub/ctrgcn_joint_fast128')
    parser.add_argument('--training-log', default='logs/ntu60_xsub_joint_fast128.log')
    parser.add_argument('--expected-epochs', type=int, default=65)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--dry-run', action='store_true', help='verify/save results without shutting down')
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('This watcher must run inside the Linux AutoDL instance.')
    if args.pid < 1 or args.expected_epochs < 1 or args.poll_seconds <= 0:
        parser.error('PID, expected epochs and poll interval must be positive.')
    repository = Path(__file__).resolve().parents[1]
    config = (repository / args.config).resolve()
    work_dir = (repository / args.work_dir).resolve()
    training_log = (repository / args.training_log).resolve()
    shutdown = Path('/usr/bin/shutdown')
    if not args.dry_run and not os.access(shutdown, os.X_OK):
        raise ValueError('AutoDL /usr/bin/shutdown is unavailable; watcher not armed.')
    identity = validate_training_process(args.pid, repository, config, work_dir)
    # Lock the exact result directory so duplicate watchers cannot race.
    import fcntl
    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / 'shutdown_watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('Armed: waiting for training PID {} (start {}), results: {}'.format(
            args.pid, identity, work_dir), flush=True)
        while process_identity(args.pid) == identity:
            time.sleep(args.poll_seconds)
        print('Training process exited. Verifying final results.', flush=True)
        summary = verify_results(work_dir, training_log, args.expected_epochs)
        summary['watched_pid'] = args.pid
        summary['watched_process_start'] = identity
        output = persist_results(summary, work_dir, training_log)
        print('Saved and verified: {}'.format(output), flush=True)
        os.sync()
        if args.dry_run:
            print('Dry run complete; shutdown was not requested.', flush=True)
            return
        print('Results are on disk. Requesting AutoDL shutdown.', flush=True)
        os.sync()
        subprocess.run([str(shutdown)], check=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('FAILED: {}\nShutdown was not completed; inspect the logs.'.format(error),
              file=sys.stderr, flush=True)
        sys.exit(1)
