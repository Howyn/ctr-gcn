"""Read-only checks of saved occlusion results and a small live CUDA replay.

No training, optimizer, checkpoint writes or changes to the evaluated models.
Artifact checks alone do not prove the historical training split was correct.
"""

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_occlusion as evaluation


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def independent_mask(data, joints):
    """Reference implementation deliberately independent of mask_inputs/scatter."""
    result = data.clone()
    for row, selected in enumerate(joints):
        result[row, :, :, [int(joint) for joint in selected], :] = 0
    return result


def check_mask(data, masked, joints):
    require(torch.equal(masked, independent_mask(data, joints)),
            'Mask has wrong axes, changed retained joints, or failed to zero selected joints.')
    for row, selected in enumerate(joints):
        require(torch.count_nonzero(masked[row, :, :, list(map(int, selected)), :]).item() == 0,
                'A selected joint still has nonzero coordinates.')
    removed = int(torch.count_nonzero(data).item() - torch.count_nonzero(masked).item())
    require(removed > 0, 'No previously nonzero coordinates were removed in this batch.')
    return dict(removed_nonzero_coordinates=removed,
                removed_fraction_of_nonzero_coordinates=removed / torch.count_nonzero(data).item())


def npz_shape(path, key):
    # Read only the NPY header; do not allocate the multi-GB training array.
    with zipfile.ZipFile(path) as archive, archive.open(key + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        return reader(stream)[0]


def audit_artifacts(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    count, missing = summary['samples'], summary['missing_joints']
    require(summary['protocol'] == 'per_sample_fixed_joints_zero_after_test_crop_resize',
            'Unexpected occlusion protocol.')
    require(missing == 5 and summary['rate'] == 0.2, 'This audit expects 5/25 missing joints.')
    with np.load(directory / 'masks.npz') as archive:
        joints, labels = archive['masked_joint_indices'], archive['labels']
        np.testing.assert_array_equal(archive['sample_indices'], np.arange(count))
        require(int(archive['seed']) == summary['mask_seed'], 'Mask seed differs from summary.')
        require(float(archive['rate']) == summary['rate'], 'Mask rate differs from summary.')
    require(joints.shape == (count, 5), 'Saved masks do not have exactly five joint IDs per sample.')
    require(np.all((joints >= 0) & (joints < 25)), 'Joint ID out of range.')
    require(np.all(np.diff(np.sort(joints.astype(int), axis=1), axis=1) > 0), 'Duplicate joint IDs.')
    np.testing.assert_array_equal(joints, evaluation.make_joint_masks(count, seed=summary['mask_seed']))
    require(hashlib.sha256(joints.tobytes()).hexdigest() == summary['mask_sha256'],
            'Saved mask hash differs from summary.')
    configs = {}
    for name, information in summary['checkpoints'].items():
        config = evaluation.read_config(Path(information['saved_config']).parent, information['model'])
        require(config['model_args'] == information['model_args'], 'Model arguments changed since evaluation.')
        if 'train_feeder_args' in information:
            require(config['train_feeder_args'] == information['train_feeder_args'],
                    'Training input settings changed since evaluation.')
        require(config['train_feeder_args']['split'] == 'train', 'Training config used a non-training split.')
        require(evaluation.file_digest(information['checkpoint']) == information['checkpoint_sha256'],
                'Checkpoint content has changed since evaluation.')
        configs[name] = config
    feeder_args = evaluation.matching_test_settings(configs)
    require(feeder_args == summary['test_feeder_args'], 'Test settings changed since evaluation.')
    if 'data_size_bytes' in summary:
        require(Path(feeder_args['data_path']).stat().st_size == summary['data_size_bytes'],
                'Dataset file size changed since evaluation.')
    require(count == 16487 and npz_shape(feeder_args['data_path'], 'x_test') == (16487, 300, 150),
            'Unexpected NTU60 X-Sub test count or shape.')
    require(npz_shape(feeder_args['data_path'], 'x_train') == (40091, 300, 150),
            'Unexpected NTU60 X-Sub training count or shape.')
    with np.load(feeder_args['data_path']) as archive:
        one_hot = archive['y_test']
    require(one_hot.shape == (count, 60) and np.all((one_hot == 0) | (one_hot == 1))
            and np.all(one_hot.sum(axis=1) == 1), 'Invalid test one-hot labels.')
    np.testing.assert_array_equal(labels, one_hot.argmax(axis=1))
    with np.load(directory / 'scores.npz') as archive:
        np.testing.assert_array_equal(archive['labels'], labels)
        scores = {key: archive[key] for key in ('baseline_clean', 'baseline_occluded',
                                               'mamba_clean', 'mamba_occluded')}
    report = dict(samples=count, mask_seed=summary['mask_seed'], missing_joints=missing,
                  same_training_inputs=evaluation.matching_training_inputs(configs), metrics={})
    predictions = {}
    shuffled = np.random.default_rng(9182).permutation(labels)
    for key, values in scores.items():
        require(values.shape == (count, 60) and np.isfinite(values).all(), 'Invalid score array: ' + key)
        # Use argmax independently of evaluate_occlusion.metrics. Preserve its
        # documented argsort tie rule only for actual tied maximum scores.
        tied = (values == values.max(axis=1, keepdims=True)).sum(axis=1) > 1
        prediction = values.argmax(axis=1)
        prediction[tied] = values[tied].argsort(axis=1)[:, -1]
        predictions[key] = prediction
        report['metrics'][key] = dict(top1_percent=100 * float(np.mean(prediction == labels)),
                                     tied_maxima=int(tied.sum()),
                                     shuffled_label_top1_percent=100 * float(np.mean(prediction == shuffled)))
    for row in summary['results']:
        clean = report['metrics'][row['model'] + '_clean']['top1_percent']
        occluded = report['metrics'][row['model'] + '_occluded']['top1_percent']
        for actual, saved in ((clean, row['clean_top1_percent']), (occluded, row['occluded_top1_percent']),
                              (clean - occluded, row['drop_pp'])):
            require(abs(actual - saved) < 1e-9, 'Summary accuracy/drop disagrees with saved scores.')
    with (directory / 'predictions.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == count, 'Prediction CSV has a wrong number of samples.')
    for index, row in enumerate(rows):
        require(int(row['sample_index']) == index and int(row['label']) == int(labels[index]),
                'Prediction CSV labels or order differ.')
        require(list(map(int, row['masked_joints_1based'].split())) == (joints[index].astype(int) + 1).tolist(),
                'Prediction CSV masks differ.')
        for key in scores:
            require(int(row[key]) == int(predictions[key][index]), 'Prediction CSV disagrees with scores.')
    print('PASS artifacts: masks, test labels, checkpoint hashes, all scores and CSV/JSON metrics.', flush=True)
    return summary, configs, feeder_args, joints, labels, scores, report


@torch.inference_mode()
def checked_forward(model, data, amp=False, diagnostics=None):
    """Inspect the actual model/backbone/local-branch boundaries on this call."""
    original = data.clone()
    captured, handles = {}, []

    def root_hook(module, arguments):
        require(len(arguments) == 1 and torch.equal(arguments[0], original),
                'Model did not receive exactly the intended skeleton input.')
        require(not torch.is_grad_enabled() and not any(m.training for m in model.modules()),
                'Gradients or a training-mode submodule are enabled during inference.')
        captured['root'] = True

    def bn_hook(module, arguments):
        batch, channels, frames, joints, people = original.shape
        expected = original.permute(0, 4, 3, 1, 2).contiguous().view(batch, people * joints * channels, frames)
        require(torch.equal(arguments[0], expected), 'Backbone received coordinates different from masked input.')
        captured['backbone'] = True

    def features_hook(module, arguments, output):
        captured['tokens'] = output.mean(dim=-1).permute(0, 2, 1).contiguous().clone()

    def norm_hook(module, arguments, output):
        require(torch.equal(arguments[0], captured['tokens']), 'Temporal branch bypassed current backbone features.')
        captured['normalized'] = output.clone()

    def local_hook(module, arguments):
        require(torch.equal(arguments[0], captured['normalized']), 'Local branch bypassed current normalized features.')
        captured['local'] = True

    def reliability_hook(module, arguments, output):
        from model.ctrgcn_mamba_robust import observed_joint_fraction
        require(torch.equal(arguments[0], captured['tokens']), 'Reliability estimator bypassed current features.')
        expected = observed_joint_fraction(original, captured['tokens'].shape[1])
        require(torch.equal(arguments[1], expected), 'Reliability estimator used missingness from another input.')
        captured['observed'], captured['reliability'] = arguments[1].clone(), output.clone()

    handles.append(model.register_forward_pre_hook(root_hook))
    handles.append(model.data_bn.register_forward_pre_hook(bn_hook))
    handles.append(model.l10.register_forward_hook(features_hook))
    handles.append(model.temporal_norm.register_forward_hook(norm_hook))
    if getattr(model, 'use_local_compensation', False):
        handles.append(model.temporal_local.register_forward_pre_hook(local_hook))
    if getattr(model, 'use_reliability', False):
        handles.append(model.temporal_reliability.register_forward_hook(reliability_hook))
    try:
        with torch.autocast(device_type=data.device.type, dtype=torch.float16, enabled=amp):
            result = model(data)
        require(captured.get('root') and captured.get('backbone') and 'normalized' in captured,
                'Expected model/backbone/temporal hooks were not called.')
        if getattr(model, 'use_local_compensation', False):
            require(captured.get('local'), 'Enabled local branch was not called.')
        if getattr(model, 'use_reliability', False):
            require('reliability' in captured, 'Enabled reliability estimator was not called.')
            if diagnostics is not None:
                for key in ('observed', 'reliability'):
                    value = captured[key].float()
                    diagnostics[key] = dict(min=float(value.min()), mean=float(value.mean()), max=float(value.max()))
        require(torch.equal(data, original), 'A model mutated its input tensor in place.')
        require(result.shape == (len(data), 60) and torch.isfinite(result).all(), 'Invalid live logits.')
        return result.float().cpu()
    finally:
        for handle in handles:
            handle.remove()


def audit_live(summary, configs, feeder_args, joints, labels, scores, samples, workers, device):
    from feeders.feeder_ntu import Feeder

    require(torch.cuda.is_available(), 'CUDA unavailable; use --artifacts-only for saved-result checks.')
    torch.set_num_threads(1)
    torch.manual_seed(1)
    torch.cuda.set_device(device)
    amp, fast = summary['amp'], summary['fast']
    torch.backends.cudnn.deterministic = not fast
    torch.backends.cudnn.benchmark = fast
    torch.backends.cuda.matmul.allow_tf32 = fast
    torch.backends.cudnn.allow_tf32 = fast
    dataset = Feeder(**feeder_args)
    require(dataset.split == 'test' and dataset.joint_mask_prob == 0 and len(dataset) == len(labels),
            'Live feeder is not the complete, augmentation-free test set.')
    np.testing.assert_array_equal(dataset.label, labels)
    selection = np.linspace(0, len(labels) - 1, min(samples, len(labels)), dtype=int).tolist()
    subset = torch.utils.data.Subset(dataset, selection)
    options = dict(dataset=subset, batch_size=len(selection), shuffle=False, drop_last=False)
    first = next(iter(torch.utils.data.DataLoader(**options, num_workers=0)))
    second = next(iter(torch.utils.data.DataLoader(**options, num_workers=workers)))
    for a, b in zip(first, second):
        require(torch.equal(a, b), 'Test preprocessing/order changes with DataLoader worker count.')
    data, batch_labels, indices = first
    np.testing.assert_array_equal(indices.numpy(), selection)
    np.testing.assert_array_equal(batch_labels.numpy(), labels[selection])
    data = data.to(device, dtype=torch.float32)
    untouched = data.clone()
    selected_joints = joints[selection]
    masked = evaluation.mask_inputs(data, torch.from_numpy(selected_joints.astype(np.int64)))
    report = dict(sample_indices=selection, mask=check_mask(data, masked, selected_joints), models={})
    require(torch.equal(data, untouched), 'Mask helper changed the clean input in place.')
    # Different hidden values must produce exactly the same observable input.
    altered = data.clone()
    for row, selected in enumerate(selected_joints):
        altered[row, :, :, list(map(int, selected)), :] = 0.123
    counterfactual = independent_mask(altered, selected_joints)
    require(torch.equal(counterfactual, masked), 'Hidden coordinates survived masking.')
    for name, config in configs.items():
        module, cls = config['model'].rsplit('.', 1)
        model = getattr(importlib.import_module(module), cls)(**config['model_args'])
        state = torch.load(summary['checkpoints'][name]['checkpoint'], map_location='cpu', weights_only=True)
        model.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()}, strict=True)
        model.to(device).eval()
        snapshot = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        # The first call on a newly loaded model is occluded: no clean pass has
        # occurred from which a branch or a recurrent cache could borrow data.
        gate_diagnostics = dict(clean={}, occluded={})
        occluded_first = checked_forward(model, masked, amp, gate_diagnostics['occluded'])
        clean = checked_forward(model, data, amp, gate_diagnostics['clean'])
        occluded_after_clean = checked_forward(model, masked, amp)
        checked_forward(model, altered, amp)
        counterfactual_result = checked_forward(model, counterfactual, amp)
        for actual in (occluded_after_clean, counterfactual_result):
            torch.testing.assert_close(actual, occluded_first, rtol=1e-3, atol=1e-3,
                                       msg='Masked output depends on an earlier clean/altered forward pass.')
        zero = checked_forward(model, torch.zeros_like(data), amp)
        torch.testing.assert_close(zero, zero[:1].expand_as(zero), rtol=1e-3, atol=1e-3,
                                   msg='Identical all-zero inputs produced different per-sample outputs.')
        model_report = dict(occluded_first_matches_after_clean=True, hidden_value_counterfactual_equal=True,
                            zero_input_predicted_classes=zero.argmax(1).unique().tolist(),
                            gate_diagnostics=gate_diagnostics, replay={})
        for kind, value in (('clean', clean), ('occluded', occluded_first)):
            saved = torch.from_numpy(scores[name + '_' + kind][selection])
            # Smaller batches/different GPUs can change AMP roundoff. Report
            # drift instead of declaring every differing logit a data leak.
            model_report['replay'][kind] = dict(max_abs_logit_difference=float((value - saved).abs().max()),
                prediction_agreement_percent=100 * float((value.argmax(1) == saved.argmax(1)).float().mean()),
                logits_close=bool(torch.allclose(value, saved, rtol=2e-2, atol=3e-2)))
        for key, value in model.state_dict().items():
            require(torch.equal(value.detach().cpu(), snapshot[key]), 'Parameter/BN buffer changed: ' + key)
        require(all(parameter.grad is None for parameter in model.parameters()), 'Inference created gradients.')
        report['models'][name] = model_report
        print('PASS live {}: exact masked input, branch routing, order independence, zero-input control, '
              'unchanged weights/BN.'.format(name), flush=True)
        del model
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path,
                        default=ROOT / 'work_dir/ntu60/xsub/occlusion20_robust_v2_seed2026')
    parser.add_argument('--artifacts-only', action='store_true', help='no GPU or skeleton array allocation')
    parser.add_argument('--samples', type=int, default=16, help='live spot-check count, not a full accuracy rerun')
    parser.add_argument('--num-worker', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--report', type=Path, help='write a separate audit JSON; original results stay untouched')
    args = parser.parse_args()
    if args.samples < 2 or args.num_worker < 0:
        parser.error('Use at least 2 samples and a nonnegative worker count.')
    summary, configs, feeder, joints, labels, scores, report = audit_artifacts(args.result_dir)
    report['scope'] = 'saved_artifacts_only' if args.artifacts_only else 'saved_artifacts_and_live_spot_checks'
    if not args.artifacts_only:
        report['live'] = audit_live(summary, configs, feeder, joints, labels, scores, args.samples,
                                    args.num_worker, torch.device('cuda', args.device))
    report['limitations'] = ['Not proof of historical training-data provenance or subject disjointness.',
                             'Not a matched-augmentation architecture ablation or multiple-seed experiment.',
                             'Live replay is a spot check, not a full independent accuracy rerun.']
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if args.report:
        # Never allow an audit report to replace an existing result or checkpoint.
        target = args.report.resolve()
        require(not target.exists(), 'Audit report already exists; choose a new --report path.')
        require(target.suffix.lower() == '.json', 'Audit report must have a .json suffix.')
        require(args.result_dir.resolve() not in target.parents, 'Write the audit report outside original results.')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    print('DONE: listed checks passed. Review replay drift and stated limits; this is not a blanket no-leak proof.')


if __name__ == '__main__':
    main()
