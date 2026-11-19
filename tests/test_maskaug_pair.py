"""CPU-only checks of matched recipes and the two-stage shutdown gate."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from tools import train_maskaug_pair as pair


class MatchedRecipeTests(unittest.TestCase):
    def setUp(self):
        jobs = pair.load_jobs()
        self.baseline, self.v2 = [deepcopy(job['config']) for job in jobs]
        self.original = yaml.safe_load((pair.ROOT / pair.CONFIG_DIR /
                                       'ctrgcn_mamba_robust_v2_fast128.yaml').read_text(encoding='utf-8'))

    def test_pair_preserves_previous_v2_recipe_and_augmentation(self):
        pair.validate_pair(self.baseline, self.v2, self.original)
        for config in (self.baseline, self.v2):
            self.assertEqual(config['train_feeder_args']['joint_mask_prob'], 0.25)
            self.assertEqual(config['train_feeder_args']['joint_mask_max'], 5)
            self.assertEqual(config['test_feeder_args'], self.original['test_feeder_args'])
        expected_v2 = deepcopy(self.original)
        expected_v2['work_dir'] = self.v2['work_dir']
        self.assertEqual(expected_v2, self.v2)

    def test_augmentation_changes_are_rejected_on_either_model(self):
        for role in ('baseline', 'v2'):
            for key, value in (('joint_mask_prob', 0.5), ('joint_mask_max', 10),
                               ('random_rot', False), ('split', 'test')):
                baseline, v2 = deepcopy(self.baseline), deepcopy(self.v2)
                changed = baseline if role == 'baseline' else v2
                changed['train_feeder_args'][key] = value
                with self.subTest(role=role, key=key), self.assertRaises(ValueError):
                    pair.validate_pair(baseline, v2, self.original)

    def test_training_changes_and_warm_starts_are_rejected(self):
        for role in ('baseline', 'v2'):
            for key, value in (('seed', 2), ('optimizer', 'Adam'), ('base_lr', 0.01),
                               ('weight_decay', 0), ('step', [30, 50]), ('batch_size', 64),
                               ('num_epoch', 60), ('respect_no_weight_decay', True),
                               ('weights', 'previous.pt'), ('start_epoch', 1), ('phase', 'test')):
                baseline, v2 = deepcopy(self.baseline), deepcopy(self.v2)
                changed = baseline if role == 'baseline' else v2
                changed[key] = value
                with self.subTest(role=role, key=key), self.assertRaises(ValueError):
                    pair.validate_pair(baseline, v2, self.original)

    def test_both_candidates_cannot_change_augmentation_together(self):
        baseline, v2 = deepcopy(self.baseline), deepcopy(self.v2)
        for config in (baseline, v2):
            config['train_feeder_args']['joint_mask_prob'] = 0.5
        with self.assertRaises(ValueError):
            pair.validate_pair(baseline, v2, self.original)

    def test_reference_recipe_itself_must_have_original_masking(self):
        original = deepcopy(self.original)
        original['train_feeder_args']['joint_mask_prob'] = 0.0
        with self.assertRaisesRegex(ValueError, 'Original V2 augmentation recipe changed'):
            pair.validate_pair(self.baseline, self.v2, original)

    def test_test_masking_shared_backbone_and_v2_flags_cannot_change(self):
        mutations = (
            lambda config: config['test_feeder_args'].update(joint_mask_prob=0.25),
            lambda config: config['model_args'].update(mamba_d_state=32),
            lambda config: config['model_args'].update(use_reliability=False),
        )
        for mutate in mutations:
            candidate = deepcopy(self.v2)
            mutate(candidate)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                pair.validate_pair(self.baseline, candidate, self.original)

    def test_same_directory_is_rejected(self):
        self.baseline['work_dir'] = self.v2['work_dir']
        with self.assertRaisesRegex(ValueError, 'separate output directories'):
            pair.validate_pair(self.baseline, self.v2, self.original)


class PairExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repository = Path(self.temporary.name)
        (self.repository / 'logs').mkdir()
        self.queue = self.repository / 'queue'
        self.jobs = pair.load_jobs()
        for job in self.jobs:
            job['work'] = self.repository / 'work_dir' / job['role']
            job['log'] = self.repository / 'logs' / (job['role'] + '.log')
            job['config'] = deepcopy(job['config'])
            job['config']['work_dir'] = str(job['work'])
        self.events = []

    def fake_training(self, job, snapshot, repository, environment):
        """Produce the real completion verifier's files without model/GPU work."""
        self.events.append('train:' + job['role'])
        self.assertEqual(repository, self.repository)
        self.assertEqual(environment['OMP_NUM_THREADS'], '1')
        self.assertEqual(environment['MKL_NUM_THREADS'], '1')
        self.assertEqual(yaml.safe_load(snapshot.read_text(encoding='utf-8')), job['config'])
        job['work'].mkdir(parents=True)
        saved = dict(job['config'], seed=1, device=[0], phase='train', weights=None, start_epoch=0)
        (job['work'] / 'config.yaml').write_text(yaml.safe_dump(saved), encoding='utf-8')
        job['log'].write_text('Training finished normally\n', encoding='utf-8')
        (job['work'] / 'log.txt').write_text(
            'Training epoch: 1\nTraining epoch: 65\nBest accuracy: 0.89\n'
            'Epoch number: 60\nModel total number of params: 1884944\nseed: 1\n',
            encoding='utf-8')
        for name in ('runs-60-18780.pt', 'runs-65-20345.pt', 'epoch1_test_score.pkl',
                     'epoch1_test_each_class_acc.csv', 'runs-60-18780_right.txt'):
            (job['work'] / name).write_bytes(b'fixture result')
        (job['work'] / 'runs-60-18780_wrong.txt').touch()

    def execute(self, **kwargs):
        return pair.execute_pair(self.jobs, self.queue, repository=self.repository, **kwargs)

    def verify_and_record(self, job):
        self.events.append('verify:' + job['role'])
        return self.real_verify(job)

    def persist_and_record(self, summary, work, log):
        self.events.append('persist:' + work.name)
        return self.real_persist(summary, work, log)

    def test_real_verification_and_persistence_precede_single_shutdown(self):
        self.real_verify, self.real_persist = pair.verified_run, pair.persist_results

        def shutdown(path):
            self.events.append('shutdown')
            self.assertEqual(path, Path('/usr/bin/shutdown'))
            state = json.loads((self.queue / 'pair_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(set(state['results']), {'baseline', 'v2'})
            for job in self.jobs:
                result = json.loads((job['work'] / 'result_summary.json').read_text(encoding='utf-8'))
                self.assertEqual(result['checkpoint_sha256'], pair.digest(job['work'] / 'best.pt'))

        with patch.object(pair, 'run_training', side_effect=self.fake_training), \
                patch.object(pair, 'verified_run', side_effect=self.verify_and_record), \
                patch.object(pair, 'persist_results', side_effect=self.persist_and_record), \
                patch.object(pair, 'request_shutdown', side_effect=shutdown) as request:
            self.execute()
        self.assertEqual(self.events, ['train:baseline', 'verify:baseline', 'persist:baseline',
                                       'train:v2', 'verify:v2', 'persist:v2', 'shutdown'])
        request.assert_called_once()

    def test_baseline_process_failure_blocks_v2_and_shutdown(self):
        with patch.object(pair, 'run_training', side_effect=RuntimeError('baseline exited with code 1')) as run, \
                patch.object(pair, 'verified_run') as verify, \
                patch.object(pair, 'persist_results') as persist, \
                patch.object(pair, 'request_shutdown') as request:
            with self.assertRaisesRegex(RuntimeError, 'exited with code 1'):
                self.execute()
        self.assertEqual(run.call_count, 1)
        verify.assert_not_called()
        persist.assert_not_called()
        request.assert_not_called()
        self.assertEqual(json.loads((self.queue / 'pair_summary.json').read_text())['status'], 'failed')

    def test_missing_baseline_artifact_blocks_v2_and_shutdown(self):
        def incomplete(job, *args):
            self.fake_training(job, *args)
            (job['work'] / 'runs-65-20345.pt').unlink()

        with patch.object(pair, 'run_training', side_effect=incomplete) as run, \
                patch.object(pair, 'request_shutdown') as request:
            with self.assertRaisesRegex(ValueError, 'Final epoch checkpoint'):
                self.execute()
        self.assertEqual(run.call_count, 1)
        request.assert_not_called()

    def test_v2_process_failure_keeps_baseline_result_without_shutdown(self):
        def fail_second(job, *args):
            if job['role'] == 'v2':
                raise RuntimeError('v2 exited with code 1')
            self.fake_training(job, *args)

        with patch.object(pair, 'run_training', side_effect=fail_second) as run, \
                patch.object(pair, 'request_shutdown') as request:
            with self.assertRaisesRegex(RuntimeError, 'v2 exited'):
                self.execute()
        self.assertEqual(run.call_count, 2)
        self.assertTrue((self.jobs[0]['work'] / 'result_summary.json').is_file())
        request.assert_not_called()

    def test_persistence_failure_blocks_next_stage_and_shutdown(self):
        with patch.object(pair, 'run_training', side_effect=self.fake_training) as run, \
                patch.object(pair, 'persist_results', side_effect=OSError('disk full')), \
                patch.object(pair, 'request_shutdown') as request:
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.execute()
        self.assertEqual(run.call_count, 1)
        request.assert_not_called()

    def test_existing_second_output_prevents_starting_either_model(self):
        for existing in (self.jobs[1]['work'], self.jobs[1]['log']):
            if existing == self.jobs[1]['work']:
                existing.mkdir(parents=True)
            else:
                existing.touch()
            with self.subTest(path=existing), patch.object(pair, 'run_training') as run, \
                    patch.object(pair, 'request_shutdown') as request:
                with self.assertRaisesRegex(ValueError, 'Refusing to overwrite'):
                    self.execute()
                run.assert_not_called()
                request.assert_not_called()
            existing.rmdir() if existing.is_dir() else existing.unlink()

    def test_shutdown_false_preserves_both_results_without_power_request(self):
        with patch.object(pair, 'run_training', side_effect=self.fake_training), \
                patch.object(pair, 'request_shutdown') as request:
            state = self.execute(shutdown=False)
        self.assertEqual(state['status'], 'results_verified')
        self.assertEqual(set(state['results']), {'baseline', 'v2'})
        request.assert_not_called()

    def test_saved_seed_or_augmentation_mismatch_rejects_completed_artifacts(self):
        job = self.jobs[0]
        snapshot = self.repository / 'snapshot.yaml'
        snapshot.write_text(yaml.safe_dump(job['config']), encoding='utf-8')
        self.fake_training(job, snapshot, self.repository,
                           {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        path = job['work'] / 'config.yaml'
        original = yaml.safe_load(path.read_text(encoding='utf-8'))
        for key, value in (('seed', 2), ('weights', 'old.pt'), ('start_epoch', 1), ('phase', 'test')):
            changed = dict(original, **{key: value})
            path.write_text(yaml.safe_dump(changed), encoding='utf-8')
            with self.subTest(key=key), self.assertRaises(ValueError):
                pair.verified_run(job)
        changed = deepcopy(original)
        changed['train_feeder_args']['joint_mask_prob'] = 0
        path.write_text(yaml.safe_dump(changed), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Saved run config changed'):
            pair.verified_run(job)

    def test_training_uses_current_interpreter_and_nonzero_return_is_failure(self):
        snapshot = self.repository / 'snapshot.yaml'
        environment = {'OMP_NUM_THREADS': '1'}
        with patch.object(pair.subprocess, 'run', return_value=subprocess.CompletedProcess([], 3)) as run:
            with self.assertRaisesRegex(RuntimeError, 'exited with code 3'):
                pair.run_training(self.jobs[0], snapshot, self.repository, environment)
        command = run.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, '-u', str(self.repository / 'main.py')])
        self.assertIn(str(snapshot), command)
        self.assertEqual(run.call_args.kwargs['stdin'], subprocess.DEVNULL)
        self.assertEqual(run.call_args.kwargs['cwd'], self.repository)

    def test_resume_rechecks_completed_baseline_and_runs_only_unstarted_v2(self):
        def stop_between_stages(job, *args):
            if job['role'] == 'v2':
                raise RuntimeError('controller interrupted before v2 launch')
            self.fake_training(job, *args)

        with patch.object(pair, 'run_training', side_effect=stop_between_stages), \
                patch.object(pair, 'request_shutdown'):
            with self.assertRaises(RuntimeError):
                self.execute()
        self.events.clear()
        with patch.object(pair, 'run_training', side_effect=self.fake_training), \
                patch.object(pair, 'request_shutdown') as shutdown:
            self.execute(resume=True)
        self.assertEqual(self.events, ['train:v2'])
        shutdown.assert_called_once()

    def test_resume_rejects_changed_checkpoint_before_any_training_or_shutdown(self):
        with patch.object(pair, 'run_training', side_effect=self.fake_training):
            self.execute(shutdown=False)
        (self.jobs[0]['work'] / 'runs-60-18780.pt').write_bytes(b'changed checkpoint')
        with patch.object(pair, 'run_training') as run, patch.object(pair, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(ValueError, 'checkpoint hash changed'):
                self.execute(resume=True)
        run.assert_not_called()
        shutdown.assert_not_called()

    def test_resume_rejects_partial_stage_instead_of_overwriting_it(self):
        def interrupted(job, *args):
            self.fake_training(job, *args)
            (job['work'] / 'runs-65-20345.pt').unlink()
            raise RuntimeError('interrupted partial run')

        with patch.object(pair, 'run_training', side_effect=interrupted):
            with self.assertRaises(RuntimeError):
                self.execute()
        with patch.object(pair, 'run_training') as run, patch.object(pair, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(ValueError, 'Final epoch checkpoint'):
                self.execute(resume=True)
        run.assert_not_called()
        shutdown.assert_not_called()


if __name__ == '__main__':
    unittest.main()
