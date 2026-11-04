"""FP16-intermediate loss scaling and narrowly scoped resume compatibility.

Scaling affects the backward execution only. Gradients are restored to their
original units before clipping, AdamW, EMA or logging; forward/storage are intact.
"""
import copy
import math
import torch

EXECUTION_KEYS = ('micro_batch_size', 'gradient_accumulation_steps',
                  'gradient_loss_scale', 'runtime_overlay', 'runtime_overlay_sha256')

def loss_scale(profile):
    value = float(profile['musa'].get('gradient_loss_scale', 1.))
    if not math.isfinite(value) or value < 1 or value > 2**24 or math.frexp(value)[0] != .5:
        raise ValueError('gradient_loss_scale must be an exact power of two in [1, 2**24]')
    return value

@torch.no_grad()
def unscale_gradients(model, scale):
    if scale == 1: return
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            if parameter.grad.dtype != torch.float32:
                raise TypeError('This backend requires FP32 trainable master gradients')
            parameter.grad.div_(scale)

def resume_profiles_compatible(saved, current):
    """Same experiment and assets; only batch splitting/numeric execution may vary."""
    saved, current = copy.deepcopy(saved), copy.deepcopy(current)
    for profile in (saved, current):
        train, runtime = profile['train'], profile['runtime']
        if train['batch_size'] != 64 or train['fsdp_devices'] != 4:
            return False
        if runtime['cuda_visible_devices'] != [4, 5, 6, 7]:
            return False
        if profile['musa']['micro_batch_size'] * profile['musa']['gradient_accumulation_steps'] * 4 != 64:
            return False
        loss_scale(profile)
        for key in EXECUTION_KEYS: profile['musa'].pop(key, None)
    return saved == current
