"""Numerical migration checks against the repository's unmodified JAX modules."""
from __future__ import annotations
import os,json,math
from pathlib import Path
os.environ["JAX_PLATFORMS"]="cpu"
import jax,jax.numpy as jnp,numpy as np,optax,torch
from flax import nnx,traverse_util
from mme_vla_suite.models.representation.perc_memory_v1 import PercMemConfig,MemReader
from mme_vla_suite.models.integration.percmem_pi05 import _identity_pad,_COND_KERNEL_INIT
from mme_vla_suite.models.integration.history_gemma import HistoryBlock
from openpi.models.gemma import Config,RMSNorm
from openpi.models import siglip
from mme_vla_suite.musa.model import Memory,GemmaBlock,Vision,Tree
from mme_vla_suite.musa.train import lr_at

torch.set_num_threads(4)
torch.set_float32_matmul_precision("highest")
ROOT=Path(os.environ.get("ROBOMME_MUSA_ROOT","/data/heping/robomme_musa"))
report={}
def flat(tree): return traverse_util.flatten_dict(tree,sep="/")
def compare(name,actual,expected,atol=2e-5,rtol=2e-4):
    a=np.asarray(actual,dtype=np.float32);b=np.asarray(expected,dtype=np.float32)
    error=float(np.max(np.abs(a-b)))
    good=bool(np.allclose(a,b,atol=atol,rtol=rtol))
    report[name]={"passed":good,"max_abs_error":error,"atol":atol,"rtol":rtol}
    if not good:raise AssertionError(name+": "+str(report[name]))
class JaxMemory(nnx.Module):
    def __init__(self,cfg):
        rngs=nnx.Rngs(42)
        self.mem_feature_adapter=nnx.Linear(cfg.native_dim,cfg.feature_dim,kernel_init=_identity_pad,rngs=rngs)
        self.mem_reader=MemReader(cfg,rngs=rngs)
        self.mem_cond_adapter=nnx.Linear(cfg.output_dim,1024,use_bias=False,kernel_init=_COND_KERNEL_INIT,rngs=rngs)
    def __call__(self,x,ticks,valid,now):
        x=self.mem_feature_adapter(x.astype(jnp.float32)).astype(jnp.float16)
        return self.mem_cond_adapter(self.mem_reader(x.astype(jnp.float32),ticks,valid,now))

cfg=PercMemConfig()
memory=JaxMemory(cfg)
rng=np.random.default_rng(42)
x=rng.normal(size=(2,3,64,2048)).astype(np.float32)
ticks=np.array([[0,10,20],[0,10,20]],np.int32)
valid=np.array([[True,True,True],[False,False,False]])
now=np.array([21,21],np.int32)
target=rng.normal(size=(2,1024)).astype(np.float32)
pt=Memory(flat(nnx.state(memory,nnx.Param).to_pure_dict()),__import__("dataclasses").asdict(cfg))
tx,tt,tv,tn=[torch.from_numpy(z) for z in [x,ticks,valid,now]]
first=pt(tx,tt,tv,tn)
assert torch.equal(first,torch.zeros_like(first))
loss=(first-torch.from_numpy(target)).square().mean();loss.backward()
assert pt.p("mem_reader/output_final/kernel").grad.norm()>0
assert torch.count_nonzero(pt.p("mem_reader/proj/kernel").grad)==0
report["zero_init_and_first_gradient"]={"passed":True,"reader_final_grad_norm":float(pt.p("mem_reader/output_final/kernel").grad.norm())}
# Non-zero output projection exposes all reader, age, attention and adapter paths.
memory.mem_reader.output_final.kernel.value=jnp.asarray(rng.normal(0,0.005,size=(2048,2048)).astype(np.float32))
pt=Memory(flat(nnx.state(memory,nnx.Param).to_pure_dict()),__import__("dataclasses").asdict(cfg))
expected=memory(jnp.asarray(x),jnp.asarray(ticks),jnp.asarray(valid),jnp.asarray(now))
actual=pt(tx,tt,tv,tn)
compare("memory_forward",actual.detach().numpy(),expected,atol=2e-4,rtol=3e-3)
assert np.all(np.asarray(expected)[1]==0) and torch.equal(actual[1],torch.zeros_like(actual[1]))
def objective(module):
    result=module(jnp.asarray(x),jnp.asarray(ticks),jnp.asarray(valid),jnp.asarray(now))
    return jnp.mean((result-jnp.asarray(target))**2)
jl,jg=nnx.value_and_grad(objective)(memory)
tl=(actual-torch.from_numpy(target)).square().mean();tl.backward()
compare("memory_loss",tl.detach().numpy(),jl,atol=1e-5)
jf=flat(jg.to_pure_dict())
for key in ["mem_reader/output_final/kernel","mem_reader/age_in/kernel","mem_cond_adapter/kernel","mem_feature_adapter/kernel"]:
    compare("memory_gradient/"+key,pt.p(key).grad.numpy(),jf[key],atol=2e-5,rtol=8e-3)

configs=(Config(width=32,depth=1,mlp_dim=64,num_heads=2,num_kv_heads=1,head_dim=8),
         Config(width=16,depth=1,mlp_dim=32,num_heads=2,num_kv_heads=1,head_dim=8))
block=HistoryBlock(configs=configs,integration_type=None)
prefix=rng.normal(size=(2,4,32)).astype(np.float32)
suffix=rng.normal(size=(2,3,16)).astype(np.float32)
cond=rng.normal(size=(2,16)).astype(np.float32)
positions=np.broadcast_to(np.arange(7,dtype=np.int32),(2,7))
mask=np.ones((2,1,7,7),bool);mask[:,:,:4,4:]=False
args=([jnp.asarray(prefix),jnp.asarray(suffix)],None,jnp.asarray(positions),jnp.asarray(mask),[None,jnp.asarray(cond)],None,None)
variables=block.init(jax.random.key(42),*args)
bp=flat(variables["params"])
for k in bp:
    if "Dense_0" in k:bp[k]=jnp.asarray(rng.normal(0,0.01,size=bp[k].shape).astype(np.float32))
params=traverse_util.unflatten_dict(bp,sep="/")
tb=GemmaBlock({k:np.asarray(v) for k,v in bp.items()})
jout=block.apply({"params":params},*args)[0]
tout=tb(torch.tensor(prefix),torch.tensor(suffix),torch.tensor(cond),torch.tensor(positions),torch.tensor(mask[:,0]))
for i in range(2):compare("gemma_block_forward_"+str(i),tout[i].detach().numpy(),jout[i])
def jloss(p):
    out=block.apply({"params":p},*args)[0]
    return sum(jnp.mean(z*z) for z in out)
jgrad=flat(jax.grad(jloss)(params))
sum(z.square().mean() for z in tout).backward()
for key in bp:compare("gemma_gradient/"+key,tb.p(key).grad.numpy(),jgrad[key],atol=3e-5,rtol=4e-3)

# In adaptive RMSNorm, (1 + scale) is evaluated in the activation dtype
# before multiplying the FP32 normalized inputs.
rms=RMSNorm()
rx=jnp.asarray(rng.normal(size=(2,3,16)).astype(np.float32),dtype=jnp.bfloat16)
rc=jnp.asarray(rng.normal(size=(2,16)).astype(np.float32))
rp=rms.init(jax.random.key(9),rx,rc)['params']
rp['Dense_0']['kernel']=jnp.asarray(rng.normal(0,.08,size=rp['Dense_0']['kernel'].shape).astype(np.float32))
rj,rg=rms.apply({'params':rp},rx,rc)
rt=GemmaBlock({'rms/'+k:np.asarray(v) for k,v in flat(rp).items()})
rv,gate=rt.rms(torch.tensor(np.asarray(rx,dtype=np.float32)).bfloat16(),'rms',torch.tensor(np.asarray(rc)))
compare('adarms_bf16_rounding',rv.detach().float().numpy(),rj,atol=.008,rtol=.008)
compare('adarms_bf16_gate',gate.detach().float().numpy(),rg,atol=.008,rtol=.008)

vision=siglip._Module(num_classes=24,patch_size=(4,4),width=32,depth=2,mlp_dim=64,num_heads=4,pool_type="none",scan=True,dtype_mm="float32")
image=rng.normal(size=(2,16,16,3)).astype(np.float32)
vp=vision.init(jax.random.key(1),jnp.asarray(image),train=False)["params"]
vp["head"]["kernel"]=jnp.asarray(rng.normal(0,0.02,size=vp["head"]["kernel"].shape).astype(np.float32))
vp=jax.tree.map(lambda z:z.astype(jnp.bfloat16).astype(jnp.float32),vp)
vt=Vision({k:np.asarray(v) for k,v in flat(vp).items()})
vo=vision.apply({"params":vp},jnp.asarray(image),train=False)[0]
compare("vision_forward",vt(torch.tensor(image),torch.float32).numpy(),vo,atol=3e-5,rtol=3e-4)

# NNX's action output projection promotes the BF16 activation and FP32
# parameters to FP32. This boundary must not become a BF16 dense operation.
linear=nnx.Linear(16,8,rngs=nnx.Rngs(7))
linear_state={k:np.asarray(v) for k,v in flat(nnx.state(linear,nnx.Param).to_pure_dict()).items()}
input32=rng.normal(size=(2,3,16)).astype(np.float32)
jinput=jnp.asarray(input32,dtype=jnp.bfloat16)
expected=linear(jinput)
actual=Tree({'out/'+k:v for k,v in linear_state.items()}).linear(torch.tensor(input32).bfloat16(),'out',torch.float32)
assert expected.dtype==jnp.float32 and actual.dtype==torch.float32
compare("action_output_fp32_promotion",actual.detach().numpy(),expected,atol=1e-6,rtol=1e-5)

# Exact schedule, clipping and AdamW/EMA semantics on fixed gradients.
params0=rng.normal(size=(11,7)).astype(np.float32)
tp=torch.nn.Parameter(torch.tensor(params0));ema=tp.detach().clone()
opt=torch.optim.AdamW([tp],lr=0.0,betas=(0.9,0.95),eps=1e-8,weight_decay=0,foreach=False)
jp=jnp.asarray(params0);je=jp
schedule=optax.warmup_cosine_decay_schedule(5e-5/10001,5e-5,10000,80000,5e-5)
points=[0,1,9999,10000,20000,79999]
compare("learning_rate_schedule",[lr_at({"learning_rate":{"value":5e-5,"warmup_steps":10000}},i) for i in points],[float(schedule(i)) for i in points],atol=1e-11,rtol=1e-6)
validation_lr=5e-4
txo=optax.chain(optax.clip_by_global_norm(1.0),optax.adamw(validation_lr,b1=.9,b2=.95,eps=1e-8,weight_decay=0))
state=txo.init(jp)
for step in range(3):
    grad=rng.normal(size=jp.shape).astype(np.float32)
    tp.grad=torch.tensor(grad);norm=tp.grad.float().square().sum().sqrt()
    if norm>1:tp.grad.div_(norm)
    opt.param_groups[0]["lr"]=validation_lr;opt.step();ema.mul_(.999).add_(tp.detach(),alpha=.001)
    update,state=txo.update(jnp.asarray(grad),state,jp);jp=optax.apply_updates(jp,update);je=.999*je+.001*jp
compare("adamw_update",tp.detach().numpy(),jp,atol=2e-7,rtol=1e-6)
compare("ema_update",ema.numpy(),je,atol=3e-7,rtol=1e-6)
delta=float(np.max(np.abs(tp.detach().numpy()-params0)))
assert delta>1e-4
report["optimizer_nontrivial_update"]={"passed":True,"controlled_diagnostic_lr":validation_lr,"max_parameter_delta":delta}
report["passed"]=True
(ROOT/"numerical_cpu_validation.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
