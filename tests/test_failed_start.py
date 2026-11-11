from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import archive_failed_start as recovery


class FailedStartTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.work = self.root / 'work_dir' / 'v2_clean'
        self.work.mkdir(parents=True)
        (self.work / recovery.WATCH_LOCK_NAME).touch()
        (self.root / 'logs').mkdir()
        self.training = self.root / 'logs' / 'train.log'
        self.watcher = self.root / 'logs' / 'watch.log'
        self.training.write_text("Traceback (most recent call last):\nModuleNotFoundError: No module named 'sklearn'\n")
        self.watcher.write_text('Armed: waiting for training PID 123 (start 456)\n')

    def recover(self):
        with patch.object(recovery, 'watcher_lock', side_effect=lambda work: nullcontext()), \
                patch.object(recovery, 'process_identity', return_value=None):
            return recovery.archive_failed_start(self.root, self.work, self.training, self.watcher)

    def test_import_only_failure_is_preserved_without_deleting_logs(self):
        text = self.training.read_text()
        archived = self.recover()
        self.assertTrue(archived.is_relative_to(self.root / 'logs' / 'failed_startups'))
        self.assertEqual((archived / 'train.log').read_text(), text)
        self.assertTrue((archived / 'watch.log').is_file())
        self.assertTrue((archived / self.work.name / recovery.WATCH_LOCK_NAME).is_file())
        self.assertFalse(self.work.exists())
        self.assertFalse(self.training.exists())

    def test_started_training_or_any_run_artifact_is_never_archived(self):
        original = self.training.read_text()
        self.training.write_text(original + 'Training epoch: 1\n')
        with self.assertRaisesRegex(ValueError, 'Training started'):
            self.recover()
        self.training.write_text(original)
        (self.work / 'weights.pt').write_bytes(b'keep checkpoint')
        with self.assertRaisesRegex(ValueError, 'beyond a startup'):
            self.recover()
        self.assertEqual((self.work / 'weights.pt').read_bytes(), b'keep checkpoint')

    def test_running_process_is_never_retried(self):
        with patch.object(recovery, 'process_identity', return_value='456'):
            with self.assertRaisesRegex(ValueError, 'still running'):
                recovery.archive_failed_start(self.root, self.work, self.training, self.watcher)
        self.assertTrue(self.training.exists())

    def test_unproven_failure_and_outside_paths_are_rejected(self):
        self.training.write_text('Something failed\n')
        with self.assertRaisesRegex(ValueError, 'confirmed missing-module'):
            self.recover()
        with self.assertRaisesRegex(ValueError, 'outside the intended'):
            recovery.archive_failed_start(self.root, self.root, self.training, self.watcher)


if __name__ == '__main__':
    unittest.main()
