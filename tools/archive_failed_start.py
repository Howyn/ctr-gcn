"""Preserve a confirmed import-only startup failure before retrying that run."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.autodl_shutdown import WATCH_LOCK_NAME, process_identity


@contextmanager
def watcher_lock(work):
    if not work.exists():
        yield
        return
    import fcntl
    with (work / WATCH_LOCK_NAME).open('a') as stream:
        deadline = time.monotonic() + 20
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ValueError('Previous shutdown watcher is still active; nothing was archived.')
                time.sleep(0.25)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def archive_failed_start(repository, work, training_log, watcher_log):
    repository = Path(repository).resolve()
    work, training_log, watcher_log = map(lambda path: Path(path).resolve(), (work, training_log, watcher_log))
    for path, parent in ((work, repository / 'work_dir'), (training_log, repository / 'logs'),
                         (watcher_log, repository / 'logs')):
        if path == parent or not path.is_relative_to(parent):
            raise ValueError('Refusing to move a path outside the intended run/log directories.')
    if not any(path.exists() for path in (work, training_log, watcher_log)):
        print('No previous startup artifacts to archive.')
        return None
    if not training_log.is_file():
        raise ValueError('No startup log proves an import failure; refusing to archive.')
    text = training_log.read_text(encoding='utf-8', errors='replace')
    if 'ModuleNotFoundError:' not in text or 'Traceback (most recent call last)' not in text:
        raise ValueError('Only a confirmed missing-module startup failure can be archived.')
    if 'Training epoch:' in text or 'Best accuracy:' in text:
        raise ValueError('Training started or completed; refusing to archive this run.')
    if work.exists() and (not work.is_dir() or any(path.name != WATCH_LOCK_NAME for path in work.iterdir())):
        raise ValueError('Run directory contains files beyond a startup watcher lock; refusing to archive.')
    watch = watcher_log.read_text(encoding='utf-8', errors='replace') if watcher_log.is_file() else ''
    for pid, start in re.findall(r'waiting for training PID (\d+) \(start (\d+)\)', watch):
        if process_identity(int(pid)) == start:
            raise ValueError('The previous training process is still running; refusing to retry.')
    with watcher_lock(work):
        # Recheck after acquiring the old watcher's lock, before any move.
        if work.exists() and any(path.name != WATCH_LOCK_NAME for path in work.iterdir()):
            raise ValueError('Run artifacts changed while waiting; nothing was archived.')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        destination = repository / 'logs' / 'failed_startups' / (work.name + '-' + stamp)
        destination.mkdir(parents=True, exist_ok=False)
        for path in (work, training_log, watcher_log):
            if path.exists():
                path.rename(destination / path.name)
    print('Preserved import-only startup failure:', destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--training-log', type=Path, required=True)
    parser.add_argument('--watcher-log', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run startup recovery on the Linux AutoDL instance.')
    archive_failed_start(ROOT, args.work_dir, args.training_log, args.watcher_log)


if __name__ == '__main__':
    main()
