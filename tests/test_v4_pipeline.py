"""Check real V4 identity, old-run isolation and the audited shutdown sequence."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import yaml

from tools import train_v3_autodl as pipeline
from test_v3_pipeline import PipelineFixture as BaseFixture


class V4Fixture(BaseFixture):
    def setUp(self):
        super().setUp()
        self.v3_plan = self.plan
        self.v4_relative = Path(str(pipeline.CONFIG).replace('v3', 'v4'))
        source = pipeline.ROOT / self.v4_relative
        config = yaml.safe_load(source.read_text(encoding='utf-8'))
        target = self.repository / self.v4_relative
        target.write_text(yaml.safe_dump(config), encoding='utf-8')
        self.plan = pipeline.build_plan(self.repository, variant='v4')

    def fake_training(self, *arguments):
        super().fake_training(*arguments)
        source = self.plan['algorithm_source']
        shutil.copy2(source, self.plan['job']['work'] / source.name)
        log = self.plan['job']['work'] / 'log.txt'
        log.write_text(log.read_text().replace('1902995', '1902865'), encoding='utf-8')

    def evaluation_summary(self, checkpoint):
        summary = super().evaluation_summary(checkpoint)
        summary['checkpoints']['mamba']['model'] = 'model.ctrgcn_mamba_robust_v4.Model'
        summary['checkpoints']['mamba']['model_args'] = self.plan['job']['config']['model_args']
        summary['checkpoints']['mamba']['parameters'] = 1902865
        return summary


class V4VersionTests(V4Fixture):
    def test_config_keeps_training_recipe_and_declares_the_actual_new_algorithm(self):
        original = deepcopy(self.configs['candidate'])
        candidate = self.plan['job']['config']
        original.update(model='model.ctrgcn_mamba_robust_v4.Model', work_dir=candidate['work_dir'])
        original['model_args'].update(algorithm_id=pipeline.V4_ALGORITHM_ID, use_parameter_correction=True)
        self.assertEqual(original, candidate)
        self.assertEqual(candidate['train_feeder_args']['joint_mask_prob'], 0)

    def test_model_has_new_implementation_instead_of_the_v3_alias(self):
        path = pipeline.ROOT / 'model/ctrgcn_mamba_robust_v4.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
                   and node.module == 'model.ctrgcn_mamba_robust_v2']
        self.assertEqual(len(imports), 1)
        imported = next(item for item in imports[0].names if item.name == 'Model')
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Model']
        self.assertEqual(len(classes), 1)
        self.assertEqual(len(classes[0].bases), 1)
        self.assertEqual(classes[0].bases[0].id, imported.asname or imported.name)
        methods = {node.name for node in classes[0].body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({'__init__', 'forward'}.issubset(methods))
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertIn('consistent_mamba', functions)

    def test_v4_paths_and_identity_are_explicit(self):
        self.assertEqual(self.plan['variant'], 'v4')
        self.assertEqual(self.plan['job']['role'], 'v4')
        self.assertEqual(self.plan['candidate_model'], 'model.ctrgcn_mamba_robust_v4.Model')
        self.assertIsNone(self.plan['architecture_equivalent_to'])
        self.assertEqual(self.plan['algorithm_id'], pipeline.V4_ALGORITHM_ID)
        for path in [self.plan['job']['work'], self.plan['job']['log']] + [self.plan[key] for key in
                     ('output_dir', 'evaluation_log', 'audit_log', 'audit_report', 'state_path',
                      'snapshot', 'cuda_log', 'legacy_watcher_log')]:
            self.assertIn('v4', path.name)
            self.assertNotIn('v3', path.name)

    def test_existing_v3_artifacts_do_not_block_or_get_modified_by_v4(self):
        old_paths = [self.v3_plan['job']['work'], self.v3_plan['job']['log']]
        old_paths += [self.v3_plan[key] for key in ('output_dir', 'evaluation_log', 'audit_log',
                      'audit_report', 'state_path', 'snapshot', 'legacy_watcher_log')]
        old_paths += [self.repository / 'work_dir/ntu60/xsub/ctrgcn_mamba_robust_v4_clean_joint_fast128',
                      self.repository / 'work_dir/ntu60/xsub/occlusion20_robust_v4_clean_seed2026',
                      self.repository / 'logs/v4_pipeline_state.json',
                      self.repository / 'logs/autodl_shutdown_mamba_robust_v4_clean.log']
        for path in old_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path in (self.v3_plan['job']['work'], self.v3_plan['output_dir']) or path.parent.name == 'xsub':
                path.mkdir()
                (path / 'preserved.txt').write_text('previous V3 result', encoding='utf-8')
            else:
                path.write_text('previous V3 result', encoding='utf-8')
        pipeline.check_fresh(self.plan)
        for path in old_paths:
            content = path / 'preserved.txt' if path.is_dir() else path
            self.assertEqual(content.read_text(encoding='utf-8'), 'previous V3 result')

    def test_existing_v4_output_still_prevents_training(self):
        self.plan['output_dir'].mkdir(parents=True)
        with patch.object(pipeline, 'run_training') as train, \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(ValueError, 'refusing overwrite'):
                self.execute()
        train.assert_not_called()
        shutdown.assert_not_called()

    def test_name_only_v4_configuration_is_rejected(self):
        config = deepcopy(self.plan['job']['config'])
        config['model_args'].pop('algorithm_id')
        self.plan['job']['config_path'].write_text(yaml.safe_dump(config), encoding='utf-8')
        with self.assertRaises(ValueError):
            pipeline.build_plan(self.repository, variant='v4')


class V4ExecutionTests(V4Fixture):
    def test_full_v4_run_uses_new_identity_and_shuts_down_only_after_audit(self):
        def shutdown(path):
            self.events.append('shutdown')
            state = self.state()
            self.assertEqual(state['experiment'], 'v4')
            self.assertIsNone(state['architecture_equivalent_to'])
            self.assertEqual(state['algorithm_id'], pipeline.V4_ALGORITHM_ID)
            self.assertEqual(state['model_source_sha256'], pipeline.digest(
                self.plan['job']['work'] / 'model_source_snapshot.py'))
            self.assertTrue(state['results_verified'])
            self.assertTrue((self.plan['job']['work'] / 'best.pt').is_file())
            self.assertTrue(self.plan['audit_report'].is_file())

        with patch.object(pipeline, 'run_training', side_effect=self.fake_training), \
                patch.object(pipeline, 'run_evaluation', side_effect=self.fake_evaluation), \
                patch.object(pipeline, 'logged_command', side_effect=self.fake_audit), \
                patch.object(pipeline, 'request_shutdown', side_effect=shutdown) as request:
            self.execute()
        self.assertEqual(self.events, ['train', 'evaluate', 'audit', 'shutdown'])
        request.assert_called_once_with(Path('/usr/bin/shutdown'))

    def test_v3_model_identity_is_not_accepted_as_v4_evaluation(self):
        def wrong_identity(*arguments):
            self.fake_evaluation(*arguments)
            path = self.plan['output_dir'] / 'summary.json'
            summary = json.loads(path.read_text(encoding='utf-8'))
            summary['checkpoints']['mamba']['model'] = 'model.ctrgcn_mamba_robust_v3.Model'
            path.write_text(json.dumps(summary), encoding='utf-8')

        with patch.object(pipeline, 'run_training', side_effect=self.fake_training), \
                patch.object(pipeline, 'run_evaluation', side_effect=wrong_identity), \
                patch.object(pipeline, 'logged_command') as audit, \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(ValueError, 'different checkpoint/model'):
                self.execute()
        audit.assert_not_called()
        shutdown.assert_not_called()

    def test_archived_model_source_mismatch_blocks_evaluation(self):
        def changed_archive(*arguments):
            self.fake_training(*arguments)
            saved = self.plan['job']['work'] / self.plan['algorithm_source'].name
            saved.write_text(saved.read_text(encoding='utf-8') + '\n# different training source\n', encoding='utf-8')
        with patch.object(pipeline, 'run_training', side_effect=changed_archive), \
                patch.object(pipeline, 'run_evaluation') as evaluate, \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(ValueError, 'Model source changed'):
                self.execute()
        evaluate.assert_not_called()
        shutdown.assert_not_called()

    def test_old_alias_parameter_count_is_not_accepted(self):
        def wrong_count(*arguments):
            self.fake_evaluation(*arguments)
            saved = self.plan['output_dir'] / 'summary.json'
            summary = json.loads(saved.read_text(encoding='utf-8'))
            summary['checkpoints']['mamba']['parameters'] = 1902995
            saved.write_text(json.dumps(summary), encoding='utf-8')
        with patch.object(pipeline, 'run_training', side_effect=self.fake_training), \
                patch.object(pipeline, 'run_evaluation', side_effect=wrong_count), \
                patch.object(pipeline, 'logged_command') as audit, \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            with self.assertRaisesRegex(ValueError, 'actual V4 algorithm/parameter count'):
                self.execute()
        audit.assert_not_called()
        shutdown.assert_not_called()

    def test_v4_preflight_checks_v4_cuda_variant(self):
        with patch.object(pipeline, 'training_preflight'), \
                patch.object(pipeline.subprocess, 'run'), \
                patch.object(pipeline, 'logged_command') as logged:
            pipeline.preflight(self.plan, self.repository)
        command = logged.call_args.args[0]
        self.assertEqual(command[command.index('--variant') + 1], 'v4')
        self.assertEqual(logged.call_args.args[1], self.plan['cuda_log'])

    def test_v4_check_only_does_not_train_or_request_shutdown(self):
        with patch.object(pipeline.sys, 'argv', ['train_v4_autodl.py', '--check-only']), \
                patch.object(pipeline, 'build_plan', return_value=self.plan) as build, \
                patch.object(pipeline, 'preflight') as preflight, \
                patch.object(pipeline, 'execute_pipeline') as execute, \
                patch.object(pipeline, 'request_shutdown') as shutdown:
            pipeline.main(variant='v4')
        build.assert_called_once_with(variant='v4')
        preflight.assert_not_called()
        execute.assert_not_called()
        shutdown.assert_not_called()

    def test_evaluation_keeps_the_shared_fixed_five_joint_protocol(self):
        checkpoint = self.plan['job']['work'] / 'best.pt'
        with patch.object(pipeline, 'logged_command') as logged:
            pipeline.run_evaluation(self.plan, checkpoint, self.repository, {})
        command = logged.call_args.args[0]
        for option, expected in (('--mamba-dir', str(self.plan['job']['work'])),
                                 ('--mamba-weights', str(checkpoint)),
                                 ('--baseline-dir', str(self.plan['reference_dir'])),
                                 ('--rate', '0.2'), ('--mask-seed', '2026')):
            self.assertEqual(command[command.index(option) + 1], expected)


class ExistingTrainingTests(unittest.TestCase):
    def test_an_existing_main_process_blocks_new_launch(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            process = proc / '123'
            repository = process / 'cwd'
            repository.mkdir(parents=True)
            fields = ['R'] + ['0'] * 18 + ['777']
            (process / 'stat').write_text('123 (python) ' + ' '.join(fields))
            (process / 'cmdline').write_bytes(b'python\0-u\0main.py\0--config\0old.yaml\0')
            with self.assertRaisesRegex(ValueError, 'Training is already active'):
                pipeline.require_idle_training(repository, proc)
            fields[0] = 'Z'
            (process / 'stat').write_text('123 (python) ' + ' '.join(fields))
            pipeline.require_idle_training(repository, proc)


if __name__ == '__main__':
    unittest.main()
