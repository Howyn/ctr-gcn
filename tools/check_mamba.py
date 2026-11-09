"""Check the actual official CUDA Mamba block and hybrid model before training."""

import argparse
import math
from pathlib import Path
import sys


def run_backward_check(model, optimizer, scaler, data, labels, amp,
                       max_attempts=20, required_steps=2):
    """Require finite, real updates; let GradScaler calibrate on early overflow."""
    import torch

    if required_steps < 1 or max_attempts < required_steps:
        raise ValueError('Require at least one successful step and enough attempts.')
    original_mamba = model.temporal_mamba.in_proj.weight.detach().clone()
    successful = 0
    overflows = 0
    last_bad = []
    for attempt in range(1, max_attempts + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=data.device.type, dtype=torch.float16, enabled=amp):
            logits = model(data)
            loss = torch.nn.functional.cross_entropy(logits, labels)
        if logits.shape != (data.shape[0], model.fc.out_features):
            raise RuntimeError('Hybrid forward pass returned an invalid logits shape.')
        if not torch.isfinite(logits).all().item() or not torch.isfinite(loss).item():
            raise RuntimeError('Non-finite forward logits/loss. Loss scaling cannot fix this; '
                               'run --precision fp32 to isolate the precision problem.')
        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [(name, parameter.grad) for name, parameter in model.named_parameters()
                     if parameter.grad is not None]
        if not gradients:
            raise RuntimeError('No model gradients were produced.')
        last_bad = [name for name, gradient in gradients if not torch.isfinite(gradient).all().item()]
        if last_bad:
            if not amp:
                raise RuntimeError('FP32 backward produced non-finite gradients: {}'.format(last_bad[:8]))
            # unscale_ recorded the invalid gradients: GradScaler must skip this step.
            scaler.step(optimizer)
            scaler.update()
            new_scale = scaler.get_scale()
            if new_scale >= old_scale:
                raise RuntimeError('AMP did not reduce its scale after invalid gradients.')
            overflows += 1
            print('AMP attempt {}: overflow; update skipped; scale {} -> {}; '
                  'first invalid gradients: {}'.format(attempt, old_scale, new_scale, last_bad[:5]),
                  flush=True)
            continue
        mamba_gradient = model.temporal_mamba.in_proj.weight.grad
        if mamba_gradient is None or mamba_gradient.abs().max().item() == 0:
            raise RuntimeError('The Mamba branch did not receive a nonzero gradient.')
        scaler.step(optimizer)
        scaler.update()
        bad_weights = [name for name, parameter in model.named_parameters()
                       if not torch.isfinite(parameter).all().item()]
        if bad_weights:
            raise RuntimeError('Optimizer update produced non-finite parameters: {}'.format(bad_weights[:8]))
        successful += 1
        print('{} attempt {}: finite loss {:.6f}; verified update {}/{}; scale {}'.format(
            'AMP' if amp else 'FP32', attempt, loss.item(), successful, required_steps,
            scaler.get_scale()), flush=True)
        if successful >= required_steps:
            if torch.equal(original_mamba, model.temporal_mamba.in_proj.weight.detach()):
                raise RuntimeError('Mamba weights did not change despite the reported updates.')
            return dict(attempts=attempt, successful_steps=successful,
                        overflow_attempts=overflows, final_scale=scaler.get_scale())
    raise RuntimeError('Only {} finite updates in {} attempts; last invalid gradients: {}. '
                       'Run --precision fp32 for diagnosis; AMP validation has not passed.'.format(
                           successful, max_attempts, last_bad[:8]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', choices=('amp', 'fp32'), default='amp')
    parser.add_argument('--init-scale', type=float, default=65536.0)
    parser.add_argument('--max-attempts', type=int, default=20)
    parser.add_argument('--respect-no-weight-decay', action='store_true',
                        help='smoke-test the calibrated optimizer grouping')
    args = parser.parse_args()
    if not math.isfinite(args.init_scale) or args.init_scale <= 0 or args.max_attempts < 2:
        parser.error('Initial scale must be positive and finite; max attempts must be at least 2.')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    import mamba_ssm
    import causal_conv1d
    from model.ctrgcn_mamba import Model
    from model.optimizer import optimizer_parameters

    if not torch.cuda.is_available():
        raise SystemExit('A CUDA GPU is required for the official Mamba kernels.')
    print('PyTorch:', torch.__version__, 'GPU:', torch.cuda.get_device_name(0), flush=True)
    print('mamba_ssm:', mamba_ssm.__version__, 'source:', mamba_ssm.__file__, flush=True)
    print('causal_conv1d:', causal_conv1d.__version__, flush=True)
    torch.set_num_threads(1)
    torch.manual_seed(1)
    model = Model(num_class=60, num_point=25, num_person=2,
                  graph='graph.ntu_rgb_d.Graph', graph_args={'labeling_mode': 'spatial'}).cuda()
    model.train()
    # Use the first warmup epoch's actual LR (0.1 / 5) for these disposable updates.
    parameters, exempt_names = optimizer_parameters(model, 0.0004, args.respect_no_weight_decay)
    print('No-weight-decay parameters:', exempt_names, flush=True)
    optimizer = torch.optim.SGD(parameters, lr=0.02, momentum=0.9, nesterov=True,
                                weight_decay=0.0004)
    amp = args.precision == 'amp'
    scaler = torch.amp.GradScaler('cuda', init_scale=args.init_scale, enabled=amp)
    data = torch.randn(2, 3, 64, 25, 2, device='cuda')
    labels = torch.tensor([0, 1], device='cuda')
    result = run_backward_check(model, optimizer, scaler, data, labels, amp,
                                max_attempts=args.max_attempts)
    torch.cuda.synchronize()
    print('Parameters:', sum(parameter.numel() for parameter in model.parameters()))
    print('Verified steps:', result)
    print('PASS: official Mamba-1 hybrid CUDA forward, {} backward and SGD step.'.format(
        'AMP' if amp else 'FP32'))


if __name__ == '__main__':
    main()
