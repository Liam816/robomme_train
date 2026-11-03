"""Check original JAX FP16 gradient behavior at production memory dimensions.

All perturbations are local diagnostic model copies, never experiment weights.
"""
import dataclasses
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import numpy as np
import torch, torch_musa
from flax import nnx

from mme_vla_suite.musa.model import Memory, flatten
from mme_vla_suite.musa.prepare_assets import load_tree
from mme_vla_suite.musa.runtime import load_profile
from mme_vla_suite.musa.validate_e2e import ReferenceMemory
from mme_vla_suite.models.representation.perc_memory_e2e import replay_memory
from mme_vla_suite.models.representation.perc_memory_v1 import PercMemConfig
from openpi.shared.nnx_utils import PathRegex


def main():
    profile = load_profile()
    root = Path(profile['musa']['root'])
    metadata = json.loads((root/'assets/export_complete.json').read_text())
    cfg = PercMemConfig(**metadata['memory_config'])
    initial = load_tree(profile['musa']['memory_initial'])
    reference = ReferenceMemory(cfg)
    state = nnx.state(reference, nnx.Param)
    state.replace_by_pure_dict(jax.tree.map(jnp.asarray, initial))
    nnx.update(reference, state)
    cache = Path(profile['data']['dataset_path'])
    source = json.loads((cache/'meta/source_index.json').read_text())
    episode = next(e for e in source['episodes'] if e['length']>=401)
    name = f"episode_{episode['episode_index']:06d}"
    native = np.array(np.load(cache/f'native/{name}.npy',mmap_mode='r')[:41])[None]
    ticks = np.load(cache/f'keyframes/{name}.npz')['tick'][:41][None]
    mask = np.ones((1,41),bool)
    now = np.array([400],np.int32)
    x, t, v, z = [jnp.asarray(a) for a in (native,ticks,mask,now)]
    rng = np.random.default_rng(123)
    coefficient = rng.normal(size=(1,1024)).astype(np.float32)
    coefficient /= np.linalg.norm(coefficient)
    perturbation = rng.normal(size=(2048,2048)).astype(np.float32)
    torch.musa.set_device(0)
    device = torch.device('musa',0)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.mudnn.allow_tf32=False;torch.backends.musa.matmul.allow_tf32=False
    port = Memory(flatten(initial),dataclasses.asdict(cfg)).to(device)
    arguments = [torch.from_numpy(a.copy()).to(device) for a in (native,ticks,mask,now)]
    result = {'scope':'controlled linear condition objective; production dimensions, real 41-frame history',
              'coefficient_norm':float(np.linalg.norm(coefficient)),'cases':{}}
    for scale in [2e-8,.03]:
        weights = perturbation*scale
        reference.mem_reader.output_final.kernel.value = jnp.asarray(weights)
        with torch.no_grad():port.p('mem_reader/output_final/kernel').copy_(torch.from_numpy(weights).to(device))
        for quantized in [True,False]:
            def objective(module):
                features = module.mem_feature_adapter(x.astype(jnp.float32))
                if quantized:features=features.astype(jnp.float16)
                selected, tick, valid, _ = replay_memory(module.mem_scorer,features,t,v,z,cfg)
                condition=module.mem_cond_adapter(module.mem_reader(selected.astype(jnp.float32),tick,valid,z))
                return (condition*jnp.asarray(coefficient)).sum()
            value, gradient = nnx.value_and_grad(objective,argnums=nnx.DiffState(0,PathRegex('mem_scorer/.*')))(reference)
            norm = sum(float(np.square(np.asarray(a,dtype=np.float64)).sum()) for a in jax.tree.leaves(gradient.to_pure_dict()))**.5
            label=f'{scale:g}_'+('fp16' if quantized else 'fp32_diagnostic')
            entry={'jax_scorer_grad_norm':norm,'loss':float(value)}
            if quantized:
                port.zero_grad(set_to_none=True)
                loss=(port(*arguments)*torch.from_numpy(coefficient).to(device)).sum()
                loss.backward()
                port_norm=sum(float(p.grad.float().square().sum().cpu()) for p in port.scorer.parameters() if p.grad is not None)**.5
                entry.update(musa_scorer_grad_norm=port_norm,musa_loss=float(loss.detach().cpu()))
            result['cases'][label]=entry
            (root/'precision_diagnostic.json').write_text(json.dumps(result,indent=2))
            print(label,entry,flush=True)
    small=result['cases']['2e-08_fp16']
    assert small['jax_scorer_grad_norm']==small['musa_scorer_grad_norm']==0
    assert result['cases']['2e-08_fp32_diagnostic']['jax_scorer_grad_norm']>0
    large=result['cases']['0.03_fp16']
    assert large['jax_scorer_grad_norm']>0 and large['musa_scorer_grad_norm']>0
    result['passed']=True
    result['conclusion']='Original FP16 selection gradients can underflow with tiny reader output weights; a nonzero scorer gradient is not a valid first-three-warmup-step gate. Original FP16 experiment is preserved.'
    (root/'precision_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
