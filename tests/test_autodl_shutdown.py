import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    'autodl_shutdown', Path(__file__).resolve().parents[1] / 'tools' / 'autodl_shutdown.py')
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


class ResultVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.console = self.work / 'console.log'
        self.console.write_text('Training finished normally\n', encoding='utf-8')
        (self.work / 'config.yaml').write_text(
            'num_epoch: 65\nwork_dir: {}\n'.format(self.work), encoding='utf-8')
        (self.work / 'log.txt').write_text(
            'Training epoch: 1\nTraining epoch: 65\nBest accuracy: 0.9\n'
            'Epoch number: 60\nModel total number of params: 1446672\nseed: 1\n',
            encoding='utf-8')
        for name in ('runs-60-18780.pt', 'runs-65-20345.pt', 'epoch1_test_score.pkl',
                     'epoch1_test_each_class_acc.csv', 'runs-60-18780_right.txt'):
            (self.work / name).write_bytes(b'saved result')
        (self.work / 'runs-60-18780_wrong.txt').touch()

    def verify(self):
        return watcher.verify_results(self.work, self.console, 65)

    def test_completed_run_is_copied_and_summarized(self):
        summary = self.verify()
        output = watcher.persist_results(summary, self.work, self.console)
        saved = json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(saved['completed_epochs'], 65)
        self.assertEqual(saved['best_epoch'], 60)
        self.assertEqual(saved['best_test_accuracy_percent'], 90.0)
        self.assertEqual(watcher.digest(self.work / 'best.pt'), saved['checkpoint_sha256'])
        self.assertEqual((self.work / 'training_console.log').read_bytes(), self.console.read_bytes())

    def test_missing_final_checkpoint_blocks_completion(self):
        (self.work / 'runs-65-20345.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'Final epoch checkpoint'):
            self.verify()

    def test_unfinished_final_summary_blocks_completion(self):
        path = self.work / 'log.txt'
        path.write_text(path.read_text().replace('seed: 1\n', ''), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'final summary seed'):
            self.verify()

    def test_traceback_blocks_shutdown_even_if_old_results_exist(self):
        self.console.write_text('Traceback (most recent call last):\nRuntimeError\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'traceback'):
            self.verify()

    def test_nonfinite_accuracy_is_rejected(self):
        path = self.work / 'log.txt'
        path.write_text(path.read_text().replace('0.9', 'nan'), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Invalid final accuracy'):
            self.verify()

    def test_missing_final_scores_blocks_completion(self):
        (self.work / 'epoch1_test_score.pkl').unlink()
        with self.assertRaisesRegex(ValueError, 'epoch1_test_score'):
            self.verify()

    def test_mismatched_saved_directory_is_rejected(self):
        (self.work / 'config.yaml').write_text('num_epoch: 65\nwork_dir: elsewhere\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'different work directory'):
            self.verify()


class ProcessIdentityTests(unittest.TestCase):
    def test_identity_handles_spaces_and_parentheses_and_zombies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = root / '123'
            process.mkdir()
            fields = ['R'] + ['0'] * 18 + ['999']
            stat = process / 'stat'
            stat.write_text('123 (python worker (train)) ' + ' '.join(fields))
            self.assertEqual(watcher.process_identity(123, root), '999')
            fields[19] = '1000'
            stat.write_text('123 (python) ' + ' '.join(fields))
            self.assertEqual(watcher.process_identity(123, root), '1000')
            fields[0] = 'Z'
            stat.write_text('123 (python) ' + ' '.join(fields))
            self.assertIsNone(watcher.process_identity(123, root))
            self.assertIsNone(watcher.process_identity(456, root))


if __name__ == '__main__':
    unittest.main()
