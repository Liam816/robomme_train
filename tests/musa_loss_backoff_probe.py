import os,json,copy
import torch,torch_musa
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from mme_vla_suite.musa.numerics import backward_with_backoff
from mme_vla_suite.musa.train import exact_grad_norm
rank=int(os.environ['LOCAL_RANK'])
torch.musa.set_device(rank)
device=torch.device('musa',rank)
dist.init_process_group('mccl')
class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.weight=torch.nn.Parameter(torch.tensor([.25,.5,.75,1.]))
  self.dropout=torch.nn.Dropout(.5)
 def forward(self):
  return self.dropout(self.weight.half().float()).sum()
a=FSDP(Tiny().to(device),device_id=device,use_orig_params=True)
b=FSDP(Tiny().to(device),device_id=device,use_orig_params=True)
oa=torch.optim.AdamW(a.parameters(),lr=5e-5)
ob=torch.optim.AdamW(b.parameters(),lr=5e-5)
torch.manual_seed(42+rank)
cpu_rng=torch.get_rng_state();gpu_rng=torch.musa.get_rng_state(device)
observed=[]
def backward(scale):
 loss=a();observed.append(loss.detach().clone());(loss*scale).backward();return loss.detach()
loss,norms,scale,retries=backward_with_backoff(a,backward,lambda:exact_grad_norm(a,device,1.),2**20,device)
end_rng=torch.musa.get_rng_state(device)
torch.set_rng_state(cpu_rng);torch.musa.set_rng_state(gpu_rng,device)
reference=b();reference.backward();reference_norms=exact_grad_norm(b,device,1.)
assert torch.equal(end_rng,torch.musa.get_rng_state(device))
assert retries>0 and norms==reference_norms
assert torch.equal(loss,reference.detach()) and all(torch.equal(loss,x) for x in observed)
for x,y in zip(a.parameters(),b.parameters()):assert torch.equal(x.grad,y.grad)
oa.step();ob.step()
for x,y in zip(a.parameters(),b.parameters()):
 assert torch.equal(x,y)
 for key in ('step','exp_avg','exp_avg_sq'):assert torch.equal(oa.state[x][key],ob.state[y][key])
print(json.dumps({'rank':rank,'passed':True,'scale':scale,'retries':retries,'loss':float(loss),'grad_norm':norms[0],'rng_and_optimizer_identical':True}),flush=True)
dist.barrier();dist.destroy_process_group()
