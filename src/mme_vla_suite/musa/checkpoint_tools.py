"""Read a verified four-rank checkpoint into the same unsharded model."""
from pathlib import Path
import hashlib
import json
import torch
from mme_vla_suite.musa.model import PercMemPi05Torch
from mme_vla_suite.musa.prepare_assets import load_tree

def original_name(name):
    return name.replace('_fsdp_wrapped_module.', '')

def load_unsharded(profile, checkpoint, device):
    checkpoint = Path(checkpoint)
    complete = json.loads((checkpoint / 'complete.json').read_text())
    metadata = json.loads((checkpoint / 'metadata.json').read_text())
    if metadata['world_size'] != 4 or complete['step'] != metadata['step']:
        raise ValueError('Incomplete or unexpected checkpoint')
    root = Path(profile['musa']['root'])
    exported = json.loads((root / 'assets/export_complete.json').read_text())
    base = load_tree(profile['musa']['base_export'])
    memory = load_tree(profile['musa']['memory_initial'])
    model = PercMemPi05Torch(base, memory, exported['memory_config'])
    del base, memory
    states = [torch.load(checkpoint / ('rank%d.pt' % rank), map_location='cpu',
                         mmap=True, weights_only=False) for rank in range(4)]
    for rank, state in enumerate(states):
        if state['step'] != metadata['step'] or state['rank'] != rank:
            raise ValueError('Rank/step mismatch')
    shards = [{original_name(k): v for k, v in state['parameters'].items()} for state in states]
    frozen = {original_name(k): v for k, v in torch.load(checkpoint / 'frozen.pt',
               map_location='cpu', mmap=True, weights_only=False).items()}
    copied = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                value = frozen[name]
                if value.shape != parameter.shape:
                    raise ValueError('Frozen parameter shape differs: ' + name)
                parameter.copy_(value)
                continue
            flat = parameter.reshape(-1)
            offset = 0
            for shard in shards:
                value = shard[name].reshape(-1)
                flat[offset:offset + value.numel()].copy_(value)
                offset += value.numel()
            if offset != parameter.numel():
                raise ValueError('Incomplete original parameter: ' + name)
            copied += offset
    expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert copied == expected == 2971264033
    digest = hashlib.sha256()
    for path in sorted(model.vision.names):
        value = model.vision.p(path).detach().float().numpy()
        digest.update(path.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode()); digest.update(value.tobytes())
    if digest.hexdigest() != metadata['frozen_hashes']['vision']:
        raise ValueError('Frozen vision hash differs')
    del states, shards, frozen
    model.to(device)
    return model, {'checkpoint': str(checkpoint), 'step': metadata['step'],
                   'trainable_elements_restored': copied,
                   'total_parameters': sum(p.numel() for p in model.parameters()),
                   'vision_hash': digest.hexdigest(), 'world_size_consolidated': 4}
