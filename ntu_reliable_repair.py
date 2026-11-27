#!/usr/bin/env python3
"""NTU60 X-Sub supervised occlusion pilot. Run from the CTR-GCN checkout."""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from tempfile import NamedTemporaryFile

import numpy as np

TRAIN_VARIANTS = ("teacher", "a", "b", "c", "bc")
VARIANTS = (*TRAIN_VARIANTS, "interp", "b_off", "bc_off")
PARTS = {"left_arm": (5, 6, 7, 8, 22, 23), "right_arm": (9, 10, 11, 12, 24, 25),
         "both_hands": (22, 23, 24, 25), "both_legs": (13, 14, 15, 16, 17, 18, 19, 20),
         "trunk": (1, 2, 3, 4, 21)}
TRAIN_COUNT, TEST_COUNT = 40091, 16487
TEST_CONDITIONS = ("clean", "burst20", "random20", *PARTS)
TRIAL_CONDITIONS = ("clean", "burst20", "random20")
PROTOCOL_VERSION = 1


def stratified_holdout(labels, seed, fraction=0.1):
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(np.flatnonzero(labels == label)) for label in np.unique(labels)]
    indices = np.concatenate([group[:max(1, round(fraction * len(group)))] for group in groups])
    return np.sort(indices)


def make_train_mask(batch, frames, rng):
    # Independent RNG, shared across all variants. Each clean view gets one
    # corrupted copy, with equal probability of burst or persistent masking.
    joints = np.argsort(rng.random((batch, 25)), axis=1)[:, :5]
    burst = rng.random(batch) < 0.5
    length = max(1, round(frames * 0.2))
    start = rng.integers(0, frames - length + 1, size=batch)
    time = np.arange(frames)[None, :]
    windows = ~burst[:, None] | ((time >= start[:, None]) & (time < start[:, None] + length))
    return np.any(np.arange(25)[None, :, None] == joints[:, None, :], axis=-1)[:, None, :] & windows[:, :, None]


def make_eval_mask(joints, frames, condition):
    if condition == "clean":
        return None
    if condition not in TEST_CONDITIONS:
        raise ValueError(f"Unknown evaluation condition: {condition}")
    if condition in PARTS:
        joints = np.broadcast_to(np.array(PARTS[condition], dtype=np.int64) - 1, (len(joints), len(PARTS[condition])))
    mask = np.zeros((len(joints), frames, 25), dtype=bool)
    if condition == "burst20":
        length = max(1, round(frames * 0.2))
        start = (frames - length) // 2
        span = slice(start, start + length)
    else:
        span = slice(None)
    mask[np.arange(len(joints))[:, None], span, joints] = True
    return mask


def write_json(path, obj):
    with NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as out:
        temporary = Path(out.name)
        json.dump(obj, out, indent=2, ensure_ascii=False)
        out.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path, rows):
    with NamedTemporaryFile(mode="w", dir=path.parent, newline="", encoding="utf-8", delete=False) as out:
        temporary = Path(out.name)
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def config_for(args):
    import yaml

    path = args.config.resolve(strict=True)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("model") != "model.ctrgcn_mamba.Model" or config.get("feeder") != "feeders.feeder_ntu.Feeder":
        raise ValueError("Use the standard CTR-GCN+Mamba NTU configuration.")
    if (config["model_args"].get("num_class") != 60 or config["model_args"].get("num_point") != 25
            or config["model_args"].get("num_person") != 2):
        raise ValueError("Expected the NTU60, 25-joint, two-person model.")
    train, test = (dict(config[key]) for key in ("train_feeder_args", "test_feeder_args"))
    for tag, settings in (("train", train), ("test", test)):
        if settings.get("split") != tag or any(settings.get(k, False) for k in ("bone", "vel", "debug", "normalization")):
            raise ValueError("Expected normalized source coordinates from the original NPZ, Joint only and no feeder normalization.")
        if settings.get("joint_mask_prob", 0):
            raise ValueError("Choose the original clean config. This runner generates the paired mask after cropping.")
    if train.get("window_size") != test.get("window_size") or test.get("window_size", -1) < 5:
        raise ValueError("Training and evaluation need the same positive window size.")
    if len(test.get("p_interval", [])) != 1 or any(test.get(k, False) for k in ("random_rot", "random_shift", "random_move")):
        raise ValueError("Validation and test must use deterministic cropping and no rotation.")
    data = Path(args.data or train["data_path"]).resolve(strict=True)
    if args.data is None and data != Path(test["data_path"]).resolve(strict=True):
        raise ValueError("The original training and test config must point to the same NTU X-Sub NPZ.")
    for settings in (train, test):
        settings["data_path"] = str(data)
        settings["joint_mask_prob"] = 0.0
    config.update(train_feeder_args=train, test_feeder_args=test)
    sig = {"path": str(data), "size": data.stat().st_size, "mtime_ns": data.stat().st_mtime_ns}
    return config, data, sig


def load_training_labels(data):
    with np.load(data, allow_pickle=False) as archive:
        if not {"x_train", "x_test", "y_train", "y_test"} <= set(archive.files):
            raise ValueError("Expected the prepared X-Sub NPZ archive.")
        y = archive["y_train"]
    if y.shape != (TRAIN_COUNT, 60) or not np.all((y > 0).sum(axis=1) == 1):
        raise ValueError("Unexpected labels in the prepared NTU60 X-Sub train split.")
    return y.argmax(axis=1)


def choose_validation(args, labels):
    if args.val_indices:
        values = np.load(args.val_indices.resolve(strict=True), allow_pickle=False)
        if (values.ndim != 1 or values.dtype.kind not in "iu" or not len(values)
                or np.any(values < 0) or np.any(values >= len(labels))
                or len(np.unique(values)) != len(values)):
            raise ValueError("--val-indices must be a unique 0-based integer list in a .npy file.")
        validation = np.sort(values.astype(np.int64))
        mode = "supplied_train_indices"
    else:
        validation = stratified_holdout(labels, args.split_seed)
        mode = "stratified_train_samples"
    training = np.setdiff1d(np.arange(len(labels)), validation)
    if (len(np.unique(labels[training])) != 60 or len(np.unique(labels[validation])) != 60
            or len(training) < args.batch):
        raise ValueError("Both subsets need all 60 classes and the training split must fit a full batch.")
    return training, validation, mode


def cpu_self_test():
    labels = np.repeat(np.arange(60), 8)
    val = stratified_holdout(labels, 12)
    assert len(val) == 60 and np.array_equal(labels[val], np.arange(60))
    assert np.array_equal(val, stratified_holdout(labels, 12))
    joints = np.tile(np.arange(5), (3, 1))
    burst = make_eval_mask(joints, 64, "burst20")
    assert not burst[:, :25].any() and not burst[:, 38:].any()
    assert np.all(burst[:, 25:38].sum(axis=-1) == 5)
    assert np.all(make_eval_mask(joints, 64, "random20").sum(axis=-1) == 5)
    for name, ids in PARTS.items():
        assert np.array_equal(np.flatnonzero(make_eval_mask(joints, 64, name)[0, 0]) + 1, np.sort(ids))
    mask = make_train_mask(128, 64, np.random.default_rng(0))
    assert set(mask.sum(axis=1).max(axis=-1)) == {13, 64}
    assert set(mask.sum(axis=-1).ravel()) == {0, 5}
    assert np.array_equal(mask, make_train_mask(128, 64, np.random.default_rng(0)))
    assert not np.array_equal(mask, make_train_mask(128, 64, np.random.default_rng(1)))
    assert epoch_lr(0.05, 5, 65) == 0.05 and epoch_lr(0.05, 65, 65) < 0.001
    print("PASS: deterministic split, 5-joint masks, 13-frame burst and all five part masks.", flush=True)


def init_torch():
    global torch, nn, F
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % 2**32)
    random.seed(seed)


def model_input(data, synthetic_mask):
    # Infer active actor slots from the corrupted input. The evaluator only
    # supplies the simulated missing positions, never their original values.
    if synthetic_mask is None:
        return data, None
    mask = torch.as_tensor(synthetic_mask, device=data.device)[:, :, :, None]
    masked = data.masked_fill(mask[:, None], 0)
    active = masked.ne(0).any(dim=1).any(dim=2)
    return masked, mask & active[:, :, None, :]


def repair_prior(data, missing, neighbors, radius=16):
    visible = data.ne(0).any(dim=1).permute(0, 3, 2, 1)
    # Pose workspace B,M,V,T,C. Gather only genuinely visible endpoints.
    x = data.permute(0, 4, 3, 2, 1)
    steps = x.shape[-2]
    time = torch.arange(steps, device=x.device)
    left = torch.where(visible, time, -steps).cummax(dim=-1).values
    right = torch.where(visible.flip((-1,)), time.flip((0,)), 2 * steps).cummin(dim=-1).values.flip((-1,))

    def at(index):
        return x.gather(-2, index.clamp(0, steps - 1)[..., None].expand(*index.shape, 3))

    both = (time - left <= radius) & (right - time <= radius) & (right > left)
    ratio = (time - left).float().div((right - left).clamp_min(1)).clamp(0, 1)[..., None]
    temporal = torch.lerp(at(left), at(right), ratio)
    raw = x * visible[..., None]
    degree = torch.einsum("jk,bmkt->bmjt", neighbors, visible.float())
    spatial = torch.einsum("jk,bmktc->bmjtc", neighbors, raw) / degree.clamp_min(1)[..., None]
    # Long gaps start from a conservative spatial estimate. An unsupported
    # joint stays at the original input; padded actor slots never get filled.
    prior = torch.where(both[..., None], temporal, 0.25 * spatial)
    target = missing.permute(0, 3, 2, 1)
    supported = target & (both | (degree > 0))
    return x, prior, supported, both, degree


def apply_repair(data, missing, neighbors, refiner=None, interpolation_only=False):
    if missing is None:
        return data, None
    with torch.autocast(device_type=data.device.type, enabled=False):
        x, prior, support, both, degree = repair_prior(data.float(), missing, neighbors.float())
        if interpolation_only:
            support = support & both
        if refiner is not None:
            features = torch.cat((x, prior, missing.permute(0, 3, 2, 1)[..., None].float(),
                                  both[..., None].float(), degree[..., None].clamp(max=4) / 4), dim=-1)
            n, m, v, t, c = features.shape
            correction = refiner(features.reshape(n * m * v, t, c).transpose(1, 2))
            prior = prior + correction.transpose(1, 2).reshape(n, m, v, t, 3)
        repaired = torch.where(support[..., None], prior, x)
    return repaired.permute(0, 4, 3, 2, 1).contiguous(), support.permute(0, 3, 2, 1)


def make_model(config, variant, device):
    from model.ctrgcn_mamba import Model

    base = Model(**config["model_args"]).to(device)
    if not hasattr(base, "temporal_mamba"):
        raise ValueError("Expected the existing baseline with its temporal_mamba module.")
    refiner = None
    if variant in ("b", "bc"):
        refiner = nn.Sequential(nn.Conv1d(9, 24, 5, padding=2), nn.SiLU(),
                                nn.Conv1d(24, 24, 3, padding=2, dilation=2), nn.SiLU(),
                                nn.Conv1d(24, 3, 1)).to(device)
        nn.init.zeros_(refiner[-1].weight)
        nn.init.zeros_(refiner[-1].bias)
    # Use the baseline graph's one- and two-edge anatomical neighborhood.
    graph = torch.as_tensor(base.graph.A, dtype=torch.float32, device=device)
    if graph.ndim != 3 or graph.shape[-2:] != (25, 25):
        raise ValueError("Unexpected CTR-GCN skeleton adjacency.")
    edge = graph.ne(0).any(dim=0)
    edge = (edge | edge.T).float()
    edge.fill_diagonal_(0)
    neighbors = (edge + 0.5 * ((edge @ edge) > 0).float() * (edge == 0).float())
    neighbors.fill_diagonal_(0)
    return base, refiner, neighbors


def run_model(base, inputs, with_tokens=False):
    if not with_tokens:
        return base(inputs), None
    tokens = []
    handle = base.temporal_mamba.register_forward_hook(lambda _m, _a, result: tokens.append(result))
    try:
        scores = base(inputs)
    finally:
        handle.remove()
    if len(tokens) != 1 or tokens[0].ndim != 3:
        raise ValueError("Expected one Mamba output with (batch * persons, time, channels).")
    return scores, tokens[0]


def masked_average(value, mask):
    mask = mask.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def repair_loss(repaired, clean, support):
    valid = clean.ne(0).any(dim=1)
    target = support & valid
    pos = masked_average(F.smooth_l1_loss(repaired, clean, reduction="none", beta=0.05), target[:, None])
    delta = repaired[:, :, 1:] - repaired[:, :, :-1]
    truth = clean[:, :, 1:] - clean[:, :, :-1]
    affected = (target[:, 1:] | target[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    vel = masked_average(F.smooth_l1_loss(delta, truth, reduction="none", beta=0.05), affected[:, None])
    return pos, vel


def align_loss(tokens, reference, missing):
    affected = missing.any(dim=2).permute(0, 2, 1).reshape(-1, 1, missing.shape[1])
    if tokens.shape != reference.shape or tokens.shape[0] != affected.shape[0]:
        raise ValueError("Mamba tokens do not align with batch/person order.")
    # A downsampled token is weighted by the fraction of its input bin that
    # was corrupted; silent second-person slots are outside the mask.
    weight = F.adaptive_avg_pool1d(affected.float(), tokens.shape[1]).squeeze(1)
    distance = 1 - F.cosine_similarity(tokens.float(), reference.detach().float(), dim=-1, eps=1e-6)
    return (distance * weight).sum() / weight.sum().clamp_min(1)


def make_loader(data, indices, args, seed, train=False):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(data, indices.tolist()) if indices is not None else data,
        batch_size=args.batch, shuffle=train, drop_last=train, num_workers=args.workers,
        generator=generator, worker_init_fn=worker_seed, pin_memory=True)


def batch_on_gpu(batch, device):
    data, labels, indices = batch
    return (data.to(device, dtype=torch.float32, non_blocking=True),
            labels.to(device, dtype=torch.long, non_blocking=True), indices)


def compute_loss(base, refiner, neighbors, teacher, clean, labels, requested, args):
    half = args.amp and clean.is_cuda
    with torch.autocast(device_type=clean.device.type, dtype=torch.float16, enabled=half):
        if requested is None:
            prediction, _ = run_model(base, clean)
            loss = F.cross_entropy(prediction, labels)
            return loss, (loss.detach(),)
        masked, missing = model_input(clean, requested)
        repaired, support = apply_repair(masked, missing, neighbors, refiner) if refiner is not None else (masked, None)
        mixed = torch.cat((clean, repaired), dim=0)
        score, tokens = run_model(base, mixed, with_tokens=teacher is not None)
        classification = F.cross_entropy(score, labels.repeat(2))
        loss = classification
        pos = vel = alignment = classification.new_zeros(())
        if refiner is not None:
            pos, vel = repair_loss(repaired.float(), clean.float(), support)
            loss = loss + args.recon * pos + args.velocity * vel
        if teacher is not None:
            with torch.no_grad():
                _, reference = run_model(teacher, clean, with_tokens=True)
            alignment = align_loss(tokens[tokens.shape[0] // 2:], reference, missing)
            loss = loss + args.align * alignment
    return loss, tuple(value.detach() for value in (classification, pos, vel, alignment))


def atomic_save(path, payload):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_training_checkpoint(path, base, refiner, identity, epoch, metrics):
    atomic_save(path, {"base": base.state_dict(), "refiner": None if refiner is None else refiner.state_dict(),
                       "identity": identity, "epoch": epoch, "validation": metrics})


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def restore_trained(path, base, refiner, identity, variant):
    item = load_checkpoint(path)
    if item["identity"] != {**identity, "variant": variant}:
        raise ValueError(f"Checkpoint settings do not match this run: {path}")
    base.load_state_dict(item["base"], strict=True)
    if refiner is not None:
        refiner.load_state_dict(item["refiner"], strict=True)
    return item


def evaluate(base, refiner, neighbors, variant, data, indices, args, device, masks, conditions):
    loader = make_loader(data, indices, args, args.mask_seed, train=False)
    base.eval()
    if refiner is not None:
        refiner.eval()
    hits = torch.zeros((len(conditions), 2), dtype=torch.long, device=device)
    count = 0
    with torch.inference_mode():
        for batch in loader:
            inputs, labels, ids = batch_on_gpu(batch, device)
            for item, kind in enumerate(conditions):
                masked, missing = model_input(inputs, make_eval_mask(masks[np.asarray(ids)], inputs.shape[2], kind))
                if variant in ("b", "bc", "interp"):
                    masked, _ = apply_repair(masked, missing, neighbors, refiner, interpolation_only=variant == "interp")
                with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                    scores = base(masked)
                if not torch.isfinite(scores).all().item():
                    raise ValueError(f"Non-finite evaluation logits: {variant}/{kind}.")
                # Match the original repo's NumPy tie policy.
                order = torch.from_numpy(scores.float().cpu().numpy().argsort(axis=1)[:, -5:].copy()).to(device)
                correct = order == labels[:, None]
                hits[item] += torch.stack((correct[:, -1].sum(), correct.any(dim=1).sum()))
            count += len(labels)
    expected = len(indices) if indices is not None else len(data)
    if count != expected:
        raise ValueError(f"Evaluation count mismatch: {count} of {expected}.")
    return {kind: {"top1": 100 * a / count, "top5": 100 * b / count}
            for kind, (a, b) in zip(conditions, hits.cpu().tolist())}


def trial_identity(args, config, sig, validation):
    return {"protocol": PROTOCOL_VERSION, "dataset": sig, "model_args": config["model_args"],
            "script_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in map(Path, ("model/ctrgcn.py", "model/ctrgcn_mamba.py",
                                                     "feeders/feeder_ntu.py", "feeders/tools.py"))},
            "train_feeder": config["train_feeder_args"], "test_feeder": config["test_feeder_args"],
            "val_hash": hashlib.sha256(validation.astype("<i8", copy=False).tobytes()).hexdigest(),
            "seed": args.seed, "split_seed": args.split_seed, "mask_seed": args.mask_seed,
            "epochs": args.epochs, "batch": args.batch, "workers": args.workers, "lr": args.lr,
            "recon": args.recon, "velocity": args.velocity, "align": args.align, "amp": args.amp}


def epoch_lr(lr, epoch, total):
    warmup = min(5, max(1, round(total * 0.08)))
    if epoch <= warmup:
        return lr * epoch / warmup
    steps = (max(warmup + 1, round(total * 35 / 65)), max(warmup + 2, round(total * 55 / 65)))
    return lr * 0.1 ** sum(epoch > step for step in steps)


def train_variant(args, config, train_set, val_set, indices, val_indices, masks, identity, variant, device):
    directory = args.out.resolve() / variant
    directory.mkdir(exist_ok=args.resume)
    identity = {**identity, "variant": variant}
    settings = directory / "settings.json"
    if settings.exists() and json.loads(settings.read_text(encoding="utf-8")) != identity:
        raise ValueError(f"Resume settings do not match: {directory}")
    write_json(settings, identity)
    if (directory / "complete.json").exists():
        print(f"Already complete: {variant}", flush=True)
        return
    teacher = None
    if variant in ("c", "bc"):
        teacher, _, _ = make_model(config, "teacher", device)
        restore_trained(args.out.resolve() / "teacher" / "best.pt", teacher, None,
                        {k: v for k, v in identity.items() if k != "variant"}, "teacher")
        teacher.eval().requires_grad_(False)
    seed_all(args.seed)
    base, refiner, neighbors = make_model(config, variant, device)
    parameters = list(base.parameters()) + ([] if refiner is None else list(refiner.parameters()))
    optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0004)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp, init_scale=4096)
    best = float("-inf")
    history = []
    start = 1
    last = directory / "last.pt"
    if last.is_file():
        state = restore_trained(last, base, refiner, {k: v for k, v in identity.items() if k != "variant"}, variant)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start = state["epoch"] + 1
        best = state["best"]
        history = state["history"]
        if state["best_epoch"] == state["epoch"]:
            # Recover an interruption between the atomic latest and best saves.
            save_training_checkpoint(directory / "best.pt", base, refiner, identity, state["epoch"], state["validation"])
        elif not (directory / "best.pt").is_file():
            raise FileNotFoundError(directory / "best.pt")
        write_json(directory / "history.json", history)
        best_epoch = state["best_epoch"]
        del state
    for epoch in range(start, args.epochs + 1):
        seed_all(args.seed + epoch * 1009)
        lr = epoch_lr(args.lr, epoch, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        rng = np.random.default_rng(np.random.SeedSequence([args.mask_seed, epoch]))
        loader = make_loader(train_set, indices, args, args.seed + epoch * 619, train=True)
        base.train()
        if refiner is not None:
            refiner.train()
        totals = None
        successful = 0
        for index, batch in enumerate(loader):
            clean, labels, _ = batch_on_gpu(batch, device)
            requested = None if variant == "teacher" else make_train_mask(len(labels), clean.shape[2], rng)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = compute_loss(base, refiner, neighbors, teacher, clean, labels, requested, args)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite training loss: {variant}, epoch={epoch}, batch={index}.")
            scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(parameters, 5.0)
            if not args.amp and not torch.isfinite(grad_norm).item():
                raise RuntimeError(f"Non-finite FP32 gradients: {variant}, epoch={epoch}, batch={index}.")
            scaler.step(optimizer)
            scaler.update()
            successful += scaler.get_scale() >= scale
            values = torch.stack(terms)
            totals = values if totals is None else totals + values
            if (index + 1) % 50 == 0 or index + 1 == len(loader):
                print(f"{variant} {epoch}/{args.epochs} step {index+1}/{len(loader)} "
                      f"loss {loss.item():.4f}", flush=True)
        if not successful:
            raise RuntimeError(f"{variant} epoch {epoch} ran without a successful optimizer step.")
        metrics = evaluate(base, refiner, neighbors, variant, val_set, val_indices, args, device, masks, TRIAL_CONDITIONS)
        losses = (totals / len(loader)).cpu().tolist()
        record = {"epoch": epoch, "lr": lr, "loss_terms": losses, "successful_steps": successful, **metrics}
        print(json.dumps({"variant": variant, **record}), flush=True)
        history.append(record)
        score = metrics["clean" if variant == "teacher" else "burst20"]["top1"]
        improved = score > best
        if improved:
            best, best_epoch = score, epoch
        atomic_save(last, {"base": base.state_dict(), "refiner": None if refiner is None else refiner.state_dict(),
                           "identity": identity, "epoch": epoch, "validation": metrics, "history": history,
                           "best": best, "best_epoch": best_epoch, "optimizer": optimizer.state_dict(),
                           "scaler": scaler.state_dict()})
        if improved:
            save_training_checkpoint(directory / "best.pt", base, refiner, identity, epoch, metrics)
        write_json(directory / "history.json", history)
    write_json(directory / "complete.json", {"best_selection_top1": best, "epochs": args.epochs, "variant": variant})


def checkpoint_row(variant, epoch, metrics):
    return {"model": variant, "selected_epoch": epoch,
            **{f"{kind}_{metric}": value for kind, values in metrics.items() for metric, value in values.items()}}


def write_validation_table(output):
    rows = []
    for variant in ("teacher", "a", "b", "c", "bc"):
        run = output / variant
        if (run / "complete.json").is_file():
            item = load_checkpoint(run / "best.pt")
            rows.append(checkpoint_row(variant, item["epoch"], item["validation"]))
    write_csv(output / "validation.csv", rows)


def doctor(args, config, device):
    cpu_self_test()
    # GPU smoke test with an actual small NTU-shaped input and all branches.
    seed_all(args.seed)
    x = torch.randn(2, 3, config["test_feeder_args"]["window_size"], 25, 2, device=device) * 0.25
    x[1, ..., 1] = 0
    labels = torch.tensor([0, 1], device=device)
    base, refiner, graph = make_model(config, "bc", device)
    teacher, _, _ = make_model(config, "teacher", device)
    teacher.load_state_dict(base.state_dict())
    teacher.eval().requires_grad_(False)
    base.eval()
    ids = np.tile(np.arange(5), (2, 1))
    requested = make_eval_mask(ids, x.shape[2], "burst20")
    masked, missing = model_input(x, requested)
    repaired, supported = apply_repair(masked, missing, graph, refiner)
    assert not supported[1, ..., 1].any().item()
    assert torch.equal(repaired[~missing[:, None].expand_as(x)], masked[~missing[:, None].expand_as(x)])
    assert torch.isfinite(repaired).all().item() and supported.any().item()
    assert torch.equal(apply_repair(x, None, graph, refiner)[0], x)
    ramp = (1.0 + torch.arange(x.shape[2], device=device) / 100)[None, None, :, None, None]
    ramp = ramp.expand_as(x).contiguous()
    ramp_masked, ramp_missing = model_input(ramp, requested)
    if ramp.shape[2] * 0.2 <= 15:
        ramp_repaired, ramp_support = apply_repair(ramp_masked, ramp_missing, graph, interpolation_only=True)
        assert torch.equal(ramp_support, ramp_missing)
        torch.testing.assert_close(ramp_repaired, ramp)
    with torch.no_grad():
        original = base(x)
        hooked, tokens = run_model(base, x, with_tokens=True)
        torch.testing.assert_close(original, hooked)
        assert tokens.shape[0] == 4
    base.train()
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp, init_scale=128)
    first = refiner[-1].weight.detach().clone()
    for _ in range(4):
        base.zero_grad(set_to_none=True)
        refiner.zero_grad(set_to_none=True)
        loss, terms = compute_loss(base, refiner, graph, teacher, x, labels, requested, args)
        assert torch.isfinite(loss).item() and all(torch.isfinite(v).item() for v in terms)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if not torch.equal(first, refiner[-1].weight):
            break
    else:
        raise RuntimeError("GPU smoke did not update the repair head.")
    assert all(parameter.grad is None for parameter in teacher.parameters())
    print("PASS: official Mamba forward/backward, masks, clean and padded slots, repair update, frozen teacher.", flush=True)


def run_probe(args, config, device):
    from feeders.feeder_ntu import Feeder
    from tools.evaluate_occlusion import best_checkpoint, make_joint_masks, read_config

    source = args.historical_dir.resolve(strict=True)
    saved = read_config(source, "model.ctrgcn_mamba.Model")
    expected = dict(config["test_feeder_args"])
    actual = dict(saved["test_feeder_args"])
    for values in (actual, expected):
        values.setdefault("joint_mask_prob", 0.0)
        values["data_path"] = str(Path(values["data_path"]).resolve())
    if actual != expected or saved["model_args"] != config["model_args"]:
        raise ValueError("Historical checkpoint preprocessing or model settings differ from the requested configuration.")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "historical_probe.csv"
    if path.exists():
        raise FileExistsError(path)
    checkpoint, epoch, _ = best_checkpoint(source)
    base, _, neighbors = make_model(config, "a", device)
    state = load_checkpoint(checkpoint)
    base.load_state_dict({name.removeprefix("module."): val for name, val in state.items()}, strict=True)
    test_set = Feeder(**expected)
    if len(test_set) != TEST_COUNT:
        raise ValueError("Unexpected number of official NTU60 X-Sub test samples.")
    masks = make_joint_masks(len(test_set), rate=0.2, seed=args.mask_seed).astype(np.int64)
    print(f"Historical diagnostic only: {checkpoint}", flush=True)
    rows = []
    for variant in ("a", "interp"):
        metrics = evaluate(base, None, neighbors, variant, test_set, None, args, device, masks, TRIAL_CONDITIONS)
        row = checkpoint_row(variant, epoch, metrics)
        rows.append(row)
        print(json.dumps(row), flush=True)
    write_csv(path, rows)
    print(f"Saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("self-check", "doctor", "probe", "train", "test"))
    parser.add_argument("--repo", type=Path, default=Path("/root/autodl-tmp/ctr-gcn"))
    parser.add_argument("--config", type=Path, default=Path("config/nturgbd-cross-subject/ctrgcn_mamba_fast128.yaml"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--historical-dir", type=Path,
                        default=Path("work_dir/ntu60/xsub/ctrgcn_mamba_maskaug_pair_joint_fast128"))
    parser.add_argument("--out", type=Path, default=Path("work_dir/ntu60/xsub/reliable_repair_pilot_s1"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["a", "b"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=32, help="matched source clips; training may process twice this count")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=31415)
    parser.add_argument("--mask-seed", type=int, default=2026)
    parser.add_argument("--val-indices", type=Path, help="optional 0-based validation indices within the official training set")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--recon", type=float, default=0.5)
    parser.add_argument("--velocity", type=float, default=0.1)
    parser.add_argument("--align", type=float, default=0.1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true", help="resume a training run from its last completed epoch")
    args = parser.parse_args()
    if args.action == "self-check":
        cpu_self_test()
        return
    if (min(args.epochs, args.batch) < 1 or min(args.workers, args.seed, args.split_seed, args.mask_seed) < 0
            or not all(np.isfinite(value) for value in (args.lr, args.recon, args.velocity, args.align))
            or args.lr <= 0 or min(args.recon, args.velocity, args.align) < 0
            or len(set(args.variants)) != len(args.variants)):
        parser.error("Invalid training settings or repeated variants.")
    if args.action == "train" and not set(args.variants) <= set(TRAIN_VARIANTS):
        parser.error("interp, b_off and bc_off are inference controls.")
    if args.resume and args.action != "train":
        parser.error("--resume only applies to training.")
    os.chdir(args.repo.resolve(strict=True))
    sys.path.insert(0, str(Path.cwd()))
    config, data, sig = config_for(args)
    init_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("Use the CUDA AutoDL environment with official Mamba installed.")
    torch.set_num_threads(1)
    device = torch.device("cuda", 0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.action == "probe":
        run_probe(args, config, device)
        return
    labels = load_training_labels(data)
    training, validation, mode = choose_validation(args, labels)
    identity = trial_identity(args, config, sig, validation)
    if args.action == "train":
        for variant in args.variants:
            if (args.out / variant).exists() and not args.resume:
                raise FileExistsError(f"Existing run directory: {args.out / variant}. Use --resume or a fresh --out.")
        if {"c", "bc"}.intersection(args.variants):
            teacher_first = "teacher" in args.variants and args.variants.index("teacher") < min(
                args.variants.index(variant) for variant in args.variants if variant in ("c", "bc"))
            if not teacher_first and not (args.out / "teacher" / "complete.json").is_file():
                raise FileNotFoundError("Train the clean reference first: --variants teacher.")
    if args.action == "doctor":
        print(f"Prepared NTU60: {data}; train={len(training)}, held-out={len(validation)} ({mode})", flush=True)
        doctor(args, config, device)
        return
    from feeders.feeder_ntu import Feeder
    from tools.evaluate_occlusion import make_joint_masks
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("a", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / "split.json"
        split_info = {"mode": mode, "train_count": len(training), "validation_count": len(validation),
                      "identity": identity, "validation_indices": validation.tolist()}
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != split_info:
                raise ValueError("Existing split/settings differ; use a fresh --out directory.")
        elif args.action == "train":
            write_json(manifest, split_info)
        else:
            raise FileNotFoundError(manifest)
        run_experiment(args, config, labels, training, validation, identity, output, device, Feeder, make_joint_masks)
    finally:
        lock.close()


def run_experiment(args, config, labels, training, validation, identity, output, device, Feeder, make_joint_masks):
    if args.action == "train":
        train_set = Feeder(**config["train_feeder_args"])
        if not np.array_equal(np.asarray(train_set.label), labels):
            raise ValueError("Feeder training samples are out of order.")
        # Only the item transforms differ. Share the already loaded X_train array.
        val_set = copy.copy(train_set)
        val_set.random_rot = False
        val_set.p_interval = config["test_feeder_args"]["p_interval"]
        masks = make_joint_masks(len(labels), rate=0.2, seed=args.mask_seed).astype(np.int64)
        for variant in args.variants:
            train_variant(args, config, train_set, val_set, training, validation, masks, identity, variant, device)
            write_validation_table(output)
    else:
        test_set = Feeder(**config["test_feeder_args"])
        if len(test_set) != TEST_COUNT:
            raise ValueError("Unexpected number of official NTU60 X-Sub test samples.")
        masks = make_joint_masks(len(test_set), rate=0.2, seed=args.mask_seed).astype(np.int64)
        table = output / ("test_" + "_".join(args.variants) + ".csv")
        if table.exists():
            raise FileExistsError(table)
        rows = []
        for variant in args.variants:
            actual = {"interp": "a", "b_off": "b", "bc_off": "bc"}.get(variant, variant)
            if not (output / actual / "complete.json").is_file():
                raise FileNotFoundError(f"Training has not finished for {actual}.")
            base, refiner, neighbors = make_model(config, actual, device)
            item = restore_trained(output / actual / "best.pt", base, refiner, identity, actual)
            metrics = evaluate(base, refiner, neighbors, variant, test_set, None, args, device, masks, TEST_CONDITIONS)
            row = checkpoint_row(variant, item["epoch"], metrics)
            row["parts_mean_top1"] = sum(metrics[part]["top1"] for part in PARTS) / len(PARTS)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del base, refiner, neighbors, item
        write_csv(table, rows)
        print(f"Saved {table}", flush=True)


if __name__ == "__main__":
    main()
