"""Parameter grouping for optional upstream Mamba no-weight-decay markers."""


def optimizer_parameters(model, weight_decay, respect_no_weight_decay=False):
    """Preserve the original recipe by default; optionally honor parameter tags."""
    if not respect_no_weight_decay:
        return list(model.parameters()), []
    decay, no_decay, names = [], [], []
    for name, parameter in model.named_parameters():
        if getattr(parameter, '_no_weight_decay', False):
            no_decay.append(parameter)
            names.append(name)
        else:
            decay.append(parameter)
    groups = []
    if decay:
        groups.append(dict(params=decay, weight_decay=weight_decay))
    if no_decay:
        groups.append(dict(params=no_decay, weight_decay=0.0))
    return groups, names
