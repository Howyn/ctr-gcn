"""Validate the no-added-occlusion V2 run against the actual saved standard run."""

import argparse
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(optimizer='SGD', base_lr=0.1, weight_decay=0.0004, nesterov=True,
                lr_decay_rate=0.1, warm_up_epoch=5, num_epoch=65, batch_size=128,
                test_batch_size=128, num_worker=2, amp=True, fast=True,
                cpu_threads=1, respect_no_weight_decay=False, seed=1)


def input_settings(arguments):
    result = deepcopy(arguments)
    result.setdefault('joint_mask_prob', 0.0)
    result.setdefault('joint_mask_max', 5)
    result['data_path'] = str((ROOT / result['data_path']).resolve())
    return result


def check_comparison(reference, candidate):
    if reference['model'] != 'model.ctrgcn_mamba.Model':
        raise ValueError('Reference must be the completed original standard Mamba run.')
    if candidate['model'] != 'model.ctrgcn_mamba_robust_v2.Model':
        raise ValueError('Candidate must use the unchanged V2 architecture.')
    if reference['feeder'] != candidate['feeder']:
        raise ValueError('Feeder differs from original standard run.')
    for key, split in (('train_feeder_args', 'train'), ('test_feeder_args', 'test')):
        original, new = input_settings(reference[key]), input_settings(candidate[key])
        if original['split'] != split or new['split'] != split:
            raise ValueError('Incorrect split in ' + key)
        if original['joint_mask_prob'] != 0 or new['joint_mask_prob'] != 0:
            raise ValueError('Added joint masking must be disabled for both runs.')
        if original != new:
            raise ValueError('Input/preprocessing settings differ in ' + key)
    for key, default in DEFAULTS.items():
        if reference.get(key, default) != candidate.get(key, default):
            raise ValueError('Training setting differs from original run: ' + key)
    if reference.get('step', [20, 40, 60]) != candidate.get('step', [20, 40, 60]):
        raise ValueError('Learning-rate schedule differs from original run.')
    if candidate.get('weights') is not None or candidate.get('start_epoch', 0) != 0:
        raise ValueError('This comparison must train from scratch, not resume or fine-tune.')
    if candidate.get('phase', 'train') != 'train':
        raise ValueError('Candidate phase must be train.')
    shared = dict(candidate['model_args'])
    expected_extra = dict(use_reliability=True, use_local_compensation=True, history_window=3)
    for key, expected in expected_extra.items():
        if shared.pop(key, None) != expected:
            raise ValueError('V2 model setting changed: ' + key)
    if shared != reference['model_args']:
        raise ValueError('Shared backbone/Mamba model arguments differ.')
    if (ROOT / candidate['work_dir']).resolve() == (ROOT / reference['work_dir']).resolve():
        raise ValueError('Candidate must use a separate work directory.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path,
                        default=ROOT / 'work_dir/ntu60/xsub/ctrgcn_mamba_joint_fast128/config.yaml')
    parser.add_argument('--candidate', type=Path,
                        default=ROOT / 'config/nturgbd-cross-subject/ctrgcn_mamba_robust_v2_clean_fast128.yaml')
    args = parser.parse_args()
    reference = yaml.safe_load(args.reference.read_text(encoding='utf-8'))
    candidate = yaml.safe_load(args.candidate.read_text(encoding='utf-8'))
    check_comparison(reference, candidate)
    print('PASS: compared against the original run\'s saved config: {}'.format(args.reference))
    print('Training joint-mask probability: 0.0; original rotation/cropping unchanged.')
    print('Same data, test preprocessing, optimizer, LR schedule, epochs, batch sizes and seed.')
    print('Train V2 from scratch; work directory: {}'.format(candidate['work_dir']))


if __name__ == '__main__':
    main()
