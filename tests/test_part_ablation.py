import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
import yaml

from model.ctrgcn_mamba import Model as StandardHybrid
from model.ctrgcn_mamba_part_ablation import Model
from test_robust_mamba import cpu_official_operators
from tools.autodl_shutdown import digest
from tools.evaluate_occlusion import mask_inputs
from tools.run_part_ablation import (ACCURACY_COLUMNS, PARTS, PROTOCOL, TEST_COUNT, VARIANTS,
                                     accuracy_row, final_checkpoint, training_config, verify_evaluation)


class PartAblationTests(unittest.TestCase):
    def test_mask_affects_only_the_requested_joints_for_all_people_and_frames(self):
        data = torch.ones(2, 3, 8, 25, 2)
        self.assertEqual(tuple(map(len, PARTS.values())), (6, 6, 4, 8, 5))
        for part, joints in PARTS.items():
            with self.subTest(part=part):
                indices = (torch.tensor(joints) - 1).expand(2, -1)
                masked = mask_inputs(data, indices)
                actual = torch.where((masked == 0).all(dim=(0, 1, 2, 4)))[0] + 1
                self.assertEqual(actual.tolist(), list(joints))
                self.assertEqual(int((masked == 0).sum()), 2 * 3 * 8 * 2 * len(joints))
        self.assertTrue((data == 1).all())

    def test_matched_models_keep_native_baseline_and_isolate_gate(self):
        torch.set_num_threads(1)
        clean = torch.randn(2, 3, 8, 25, 2)
        masked = mask_inputs(clean, (torch.tensor(PARTS['left_arm']) - 1).expand(2, -1))
        args = training_config('b', Path('b'), Path('NTU60_CS.npz'))['model_args']
        native_args = {key: value for key, value in args.items() if key not in
                       ('use_joint_attention', 'use_write_gate')}
        with cpu_official_operators():
            torch.manual_seed(1)
            native = StandardHybrid(**native_args).eval()
            expected_rng = torch.get_rng_state()
            models = {}
            for group, (attention, gate) in VARIANTS.items():
                torch.manual_seed(1)
                models[group] = Model(**native_args, use_joint_attention=attention, use_write_gate=gate).eval()
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            baseline, a, b = models.values()
            b.load_state_dict(a.state_dict(), strict=True)
            with torch.no_grad():
                torch.testing.assert_close(baseline(clean), native(clean))
                torch.testing.assert_close(a(clean), baseline(clean))
                torch.testing.assert_close(b(clean), a(clean))
                torch.testing.assert_close(b(masked, disable_write_gate=True), a(masked))
                self.assertGreater((b(masked) - a(masked)).abs().max().item(), 1e-6)
            b.train()
            b(clean).square().mean().backward()
            self.assertGreater(b.joint_score.weight.grad.abs().max().item(), 0)

    def test_all_runs_share_recipe_and_use_predeclared_final_epoch(self):
        configs = {name: training_config(name, Path(name), Path('NTU60_CS.npz')) for name in VARIANTS}
        common = [{key: value for key, value in config.items() if key not in ('model_args', 'work_dir')}
                  for config in configs.values()]
        self.assertEqual(common[0], common[1])
        self.assertEqual(common[1], common[2])
        self.assertEqual(common[0]['train_feeder_args']['joint_mask_prob'], 0.0)
        self.assertEqual(common[0]['seed'], 1)
        self.assertEqual(common[0]['num_epoch'], 65)
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            config = training_config('a', run_dir, Path('NTU60_CS.npz'))
            (run_dir / 'config.yaml').write_text(yaml.safe_dump({**config, 'phase': 'train', 'weights': None}))
            (run_dir / 'log.txt').write_text('Best accuracy: 0.88\nEpoch number: 63\n')
            for epoch in (63, 65):
                (run_dir / f'runs-{epoch}-5.pt').write_bytes(b'checkpoint')
            self.assertEqual(final_checkpoint(config).name, 'runs-65-5.pt')
        hits = {'clean': 90, **dict(zip(PARTS, (70, 60, 80, 90, 50)))}
        self.assertEqual(accuracy_row(hits, 100)['occluded_mean'], 70)

    def test_poweroff_verification_rejects_changed_weights_and_disagreeing_results(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs, models = {}, {}
            for name in VARIANTS:
                directory = root / name
                directory.mkdir()
                config = training_config(name, directory, root / 'NTU60_CS.npz')
                configs[name] = config
                (directory / 'config.yaml').write_text(yaml.safe_dump({**config, 'phase': 'train', 'weights': None}))
                (directory / 'log.txt').write_text('Best accuracy: 0.88\nEpoch number: 63\n')
                checkpoint = directory / 'runs-65-5.pt'
                checkpoint.write_bytes(b'checkpoint')
                models[name] = dict(config=config, checkpoint=str(checkpoint), sha256=digest(checkpoint))
            results = {name: dict.fromkeys(ACCURACY_COLUMNS, 70.0) for name in (*VARIANTS, 'b_gate_off')}
            summary = dict(samples=TEST_COUNT, protocol=PROTOCOL, body_part_joint_ids_1based=PARTS,
                           checkpoint_rule='fixed_final_epoch', models=models, accuracy=results)
            json_path = root / 'summary.json'
            json_path.write_text(json.dumps(summary), encoding='utf-8')
            with (root / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['model', *ACCURACY_COLUMNS])
                writer.writeheader()
                writer.writerows(dict(model=name, **result) for name, result in results.items())
            verify_evaluation(configs, root)
            results['b']['clean'] = 71.0
            json_path.write_text(json.dumps(summary), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'invalid or disagree'):
                verify_evaluation(configs, root)
            results['b']['clean'] = 70.0
            json_path.write_text(json.dumps(summary), encoding='utf-8')
            (root / 'b' / 'runs-65-5.pt').write_bytes(b'different checkpoint')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                verify_evaluation(configs, root)


if __name__ == '__main__':
    unittest.main()
