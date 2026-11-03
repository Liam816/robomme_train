"""Four-card MUSA P0 training: same model, transforms, global batch, loss and optimizer."""
from __future__ import annotations
import argparse,contextlib,hashlib,json,os,time
from datetime import datetime
from pathlib import Path
os.environ["JAX_PLATFORMS"]="cpu"
import numpy as np
import torch,torch_musa
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import yaml
from mme_vla_suite.musa.model import PercMemPi05Torch
from mme_vla_suite.musa.prepare_assets import load_tree
from mme_vla_suite.musa.runtime import load_profile
from mme_vla_suite.musa.data import TrainingDataset,GlobalBatchSampler,collate,global_random,prepare_batch,validate_global_sampler

def atomic(path,value):
    temp=path.with_name(path.name+".tmp");temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def module_hash(module):
    h=hashlib.sha256()
    for name in sorted(module.names):
        value=module.p(name).detach().float().cpu().numpy()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()
def check_profile(profile):
    assert profile["runtime"]["cuda_visible_devices"]==[4,5,6,7]
    assert os.environ.get("MUSA_VISIBLE_DEVICES")=="4,5,6,7"
    assert os.environ.get("CUDA_VISIBLE_DEVICES")=="4,5,6,7"
    assert profile["train"]["fsdp_devices"]==4 and profile["train"]["batch_size"]==64
    assert profile["freeze"]=="vision_only"
    assert profile["model"]["pi05"] and profile["model"]["dtype"]=="bfloat16"
    assert profile["model"]["paligemma_variant"]=="gemma_2b" and profile["model"]["action_expert_variant"]=="gemma_300m"
    assert profile["model"]["action_horizon"]==20 and profile["model"]["action_dim"]==32
    assert profile["model"]["max_token_len"]==64 and not profile["model"]["discrete_state_input"]
    assert not profile["train"]["resume"] and not profile["train"]["overwrite"]
    assert not profile["train"]["wandb_enabled"]
    assert profile["learning_rate"]["schedule"]=="warmup_then_constant"
    assert profile["musa"]["micro_batch_size"]*profile["musa"]["gradient_accumulation_steps"]*4==64
def lr_at(profile,step):
    lr=profile["learning_rate"];warmup=lr["warmup_steps"];initial=lr["value"]/(warmup+1)
    return initial+(lr["value"]-initial)*min(step/warmup,1.0)
def exact_grad_norm(model,device,clip):
    squares=torch.zeros((4,),device=device,dtype=torch.float32)
    for name,p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            value=p.grad.float().square().sum()
            squares[0]+=value
            if '.memory.scorer.' in name:squares[1]+=value
            if 'memory.mem_reader__' in name:squares[2]+=value
            if 'memory.mem_feature_adapter__' in name:squares[3]+=value
    dist.all_reduce(squares,op=dist.ReduceOp.SUM)
    norms=squares.sqrt();norm=norms[0]
    if not torch.isfinite(norm):raise FloatingPointError("Non-finite gradient norm")
    if norm>clip:
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:p.grad.div_(norm/clip)
    return [float(x) for x in norms.detach().cpu()]
def save_checkpoint(model,optimizer,ema,path,step,profile,frozen_hashes):
    rank=dist.get_rank()
    if rank==0:path.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    state={"format":"robomme_musa_fsdp_local_v1","step":step,"rank":rank,"world_size":dist.get_world_size(),
           "parameters":{n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
           "optimizer":optimizer.state_dict(),"ema":{n:t.cpu() for n,t in ema.items()},
           "torch_rng":torch.get_rng_state(),"musa_rng":torch.musa.get_rng_state(),
           "profile":profile,"jax_random_seed":profile["train"]["seed"],"sampler_consumed_global_batches":step}
    # Move optimizer moments to CPU before serialization; immutable files never alias live device state.
    def cpu(x):
        if isinstance(x,torch.Tensor):return x.detach().cpu()
        if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
        if isinstance(x,list):return [cpu(v) for v in x]
        if isinstance(x,tuple):return tuple(cpu(v) for v in x)
        return x
    state["optimizer"]=cpu(state["optimizer"])
    temp=path/f"rank{rank}.pt.tmp";torch.save(state,temp);temp.replace(path/f"rank{rank}.pt")
    if rank==0:
        frozen={n:p.detach().cpu() for n,p in model.named_parameters() if not p.requires_grad}
        torch.save(frozen,path/"frozen.pt")
        atomic(path/"metadata.json",{"format":state["format"],"step":step,"world_size":4,
              "frozen_hashes":frozen_hashes,"wrapping":"each Gemma block plus root, use_orig_params=True; vision/scorer ignored",
              "profile":profile,"resume_scope":"same world size, parameter ordering and wrapper layout; original global-batch sampler"})
    dist.barrier()
    if rank==0:atomic(path/"complete.json",{"step":step,"files":[f"rank{i}.pt" for i in range(4)]+["frozen.pt","metadata.json"]})
    dist.barrier()
def restore_local(model,optimizer,ema,path):
    state=torch.load(path/f"rank{dist.get_rank()}.pt",map_location="cpu",weights_only=False)
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    if parameters.keys()!=state["parameters"].keys():raise ValueError("Checkpoint local parameter keys changed")
    with torch.no_grad():
        for n,p in parameters.items():
            if p.shape!=state["parameters"][n].shape:raise ValueError("Checkpoint shard shape changed: "+n)
            p.copy_(state["parameters"][n])
        for n in ema:ema[n].copy_(state["ema"][n])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"]);torch.musa.set_rng_state(state["musa_rng"])
    return state["step"]
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--steps",type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from",type=Path)
    args=parser.parse_args()
    profile=load_profile(args.config);check_profile(profile)
    rank=int(os.environ["LOCAL_RANK"])
    allowed=os.sched_getaffinity(0)
    affinity=sorted(set(map(int,os.environ["ROBOMME_CPU_POOL"].split(",")))&allowed) if os.environ.get("ROBOMME_CPU_POOL") else sorted(allowed)
    if len(affinity)>=32:os.sched_setaffinity(0,affinity[rank*8:rank*8+8])
    torch.set_num_threads(2);torch.set_float32_matmul_precision("highest")
    torch.backends.mudnn.allow_tf32=False
    torch.backends.musa.matmul.allow_tf32=False
    torch.musa.set_device(rank);device=torch.device("musa",rank)
    dist.init_process_group("mccl")
    if dist.get_world_size()!=4 or torch.musa.device_count()!=4:raise RuntimeError("Exactly four selected MUSA GPUs required")
    torch.manual_seed(profile["train"]["seed"])
    root=Path(profile["musa"]["root"])
    validation=json.loads((root/"numerical_cpu_validation.json").read_text())
    if not validation.get("passed"):raise RuntimeError("Numerical migration validation must pass")
    validation=json.loads((root/"production_validation.json").read_text())
    if not validation.get("passed"):raise RuntimeError("Production-size numerical/vision checks must pass")
    if args.steps is not None and args.steps<1:raise ValueError("At least one optimizer update is required")
    if args.resume_from:
        if not (args.resume_from/"complete.json").exists():raise ValueError("Checkpoint is incomplete")
        saved_meta=json.loads((args.resume_from/"metadata.json").read_text())
        if saved_meta["world_size"]!=4 or saved_meta["profile"]!=profile:
            raise ValueError("Resume requires the same profile and four-rank wrapper layout")
    stamp=[args.run_name or datetime.now().strftime("smoke_%Y%m%d_%H%M%S") if rank==0 else None]
    dist.broadcast_object_list(stamp,src=0,device=device)
    run=Path(profile["train"]["checkpoint_base_dir"])/profile["train"]["exp_name"]/stamp[0]
    if rank==0:
        run.mkdir(parents=True,exist_ok=False)
        atomic(run/"profile.json",profile)
        atomic(run/"status.json",{"stage":"loading","steps":0,"physical_gpus":[4,5,6,7]})
    dist.barrier()
    if not validate_global_sampler():raise AssertionError("Global sample ordering changed")
    base=load_tree(profile["musa"]["base_export"],verify=(rank==0))
    memory=load_tree(profile["musa"]["memory_initial"])
    meta=json.loads((root/"assets/export_complete.json").read_text())
    is_e2e=meta["memory_config"]["stage"]=="e2e"
    if is_e2e:
        if profile['model']['history_config']!='perceptual-learned-adarms-v1-e2e.yaml':raise ValueError('E2E profile required')
        if not json.loads((root/'e2e_validation_cpu.json').read_text()).get('passed'):raise ValueError('E2E forward/gradient validation required')
    native=json.loads((Path(profile["data"]["dataset_path"])/"meta/native_complete.json").read_text())
    if native["vision_id"]!=meta["vision_id"]:raise ValueError("Native cache and current frozen vision weights differ")
    model=PercMemPi05Torch(base,memory,meta["memory_config"])
    del base,memory
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in model.parameters())
    if total!=meta["base_parameters"]+meta["memory_parameters"]:raise ValueError("Port changed the model parameter count")
    if trainable!=total-meta["vision_parameters"]-(0 if is_e2e else meta["scorer_parameters"]):raise ValueError("Port changed the freeze partition")
    model.to(device)
    frozen={}
    if rank==0:
        frozen={"vision":module_hash(model.vision)}
        if not is_e2e:frozen['scorer']=module_hash(model.memory.scorer)
        if frozen["vision"]!=meta["vision_id"]:raise ValueError("Backbone/cache vision identity mismatch")
        if args.resume_from and frozen!=saved_meta["frozen_hashes"]:raise ValueError("Resume frozen weights differ from checkpoint")
    ignored=[model.vision] if is_e2e else [model.vision,model.memory.scorer]
    for i,block in enumerate(model.blocks):
        model.blocks[i]=FSDP(block,device_id=device,use_orig_params=True)
    model=FSDP(model,device_id=device,use_orig_params=True,ignored_modules=ignored)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
          lr=lr_at(profile,0),betas=(profile["optimizer"]["b1"],profile["optimizer"]["b2"]),
          eps=profile["optimizer"]["eps"],weight_decay=profile["optimizer"]["weight_decay"],foreach=False)
    ema={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    start_step=restore_local(model,optimizer,ema,args.resume_from) if args.resume_from else 0
    if args.steps is None:args.steps=profile['train']['num_train_steps']-start_step
    if start_step+args.steps>profile["train"]["num_train_steps"]:raise ValueError("Update budget exceeds the original training schedule")
    selected={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad and "mem_reader__output_final__" in n}
    scorer_initial={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad and '.memory.scorer.' in n} if is_e2e else {}
    dataset=TrainingDataset(profile)
    micro=profile["musa"]["micro_batch_size"];accum=profile["musa"]["gradient_accumulation_steps"]
    sampler=GlobalBatchSampler(len(dataset),rank,4,64,micro,profile["train"]["seed"],start_step=start_step)
    workers=profile["train"]["num_workers"]//4
    kwargs={"multiprocessing_context":"spawn","prefetch_factor":1,"persistent_workers":True} if workers else {}
    loader=torch.utils.data.DataLoader(dataset,batch_sampler=sampler,collate_fn=collate,num_workers=workers,**kwargs)
    iterator=iter(loader);records=[]
    model.train()
    for step in range(start_step,start_step+args.steps):
        started=time.monotonic();optimizer.zero_grad(set_to_none=True)
        lr=lr_at(profile,step)
        for group in optimizer.param_groups:group["lr"]=lr
        keys,noise,times=global_random(profile["train"]["seed"],step,64)
        loss_total=torch.zeros((),device=device)
        for m in range(accum):
            raw=next(iterator);begin=rank*16+m*micro;end=begin+micro
            batch=prepare_batch(raw,keys[begin:end],device)
            draw=torch.from_numpy(noise[begin:end].copy()).to(device)
            flow=torch.from_numpy(times[begin:end].copy()).to(device)
            sync=contextlib.nullcontext() if m==accum-1 else model.no_sync()
            with sync:
                loss=model(batch,draw,flow)
                if not torch.isfinite(loss):raise FloatingPointError("Non-finite loss")
                (loss/accum).backward()
            loss_total+=loss.detach()/accum
        grad_norm,scorer_grad,reader_grad,feature_grad=exact_grad_norm(model,device,profile["optimizer"]["clip_gradient_norm"])
        optimizer.step()
        with torch.no_grad():
            decay=profile["train"]["ema_decay"]
            for name,p in model.named_parameters():
                if p.requires_grad:ema[name].mul_(decay).add_(p.detach(),alpha=1-decay)
        dist.all_reduce(loss_total);loss_total/=4
        delta=torch.zeros((2,),device=device)
        for n,p in model.named_parameters():
            if n in selected:delta[0]+=(p.detach()-selected[n]).square().sum()
            if n in scorer_initial:delta[1]+=(p.detach()-scorer_initial[n]).square().sum()
        dist.all_reduce(delta)
        record={"step":step+1,"loss":float(loss_total.cpu()),"grad_norm":grad_norm,"learning_rate":lr,
                "reader_final_parameter_delta_l2":float(delta[0].sqrt().cpu()),"seconds":time.monotonic()-started}
        if is_e2e:record.update(scorer_grad_norm=scorer_grad,reader_grad_norm=reader_grad,feature_adapter_grad_norm=feature_grad,
                              scorer_parameter_delta_l2=float(delta[1].sqrt().cpu()))
        records.append(record)
        if rank==0:
            print(json.dumps(record),flush=True)
            with (run/'metrics.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
            atomic(run/"status.json",{"stage":"training","steps":step+1,"target_steps":start_step+args.steps,"total_parameters":total,
                   "trainable_parameters":trainable,"records":records[-20:],"physical_gpus":[4,5,6,7]})
        # Match the upstream zero-based save trigger. Directory names count
        # completed optimizer updates, preventing an off-by-one resume schedule.
        if step>start_step and step%profile["train"]["save_interval"]==0 and step+1<start_step+args.steps:
            save_checkpoint(model,optimizer,ema,run/f"step_{step+1:06d}",step+1,profile,frozen)
    if records[-1]["reader_final_parameter_delta_l2"]<=0:raise AssertionError("Reader output parameters did not update")
    if is_e2e and args.steps>=3 and not any(r['scorer_grad_norm']>0 for r in records[1:]):raise AssertionError('E2E scorer received no flow-matching gradient')
    model.eval()
    with torch.no_grad():reference_loss=model(batch,draw,flow).detach().clone()
    end_step=start_step+args.steps
    checkpoint_path=run/f"step_{end_step:06d}"
    save_checkpoint(model,optimizer,ema,checkpoint_path,end_step,profile,frozen)
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in selected:p.add_(1)
    restored_step=restore_local(model,optimizer,ema,checkpoint_path)
    with torch.no_grad():restored_loss=model(batch,draw,flow).detach()
    if not torch.equal(reference_loss,restored_loss):raise AssertionError("Checkpoint reload changed fixed-input loss")
    if rank==0:
        after={"vision":module_hash(model.module.vision)}
        if not is_e2e:after['scorer']=module_hash(model.module.memory.scorer)
        if frozen!=after:raise AssertionError("Frozen weights changed")
        result={"stage":"complete","steps":end_step,"invocation_updates":args.steps,"start_step":start_step,"records":records[-20:],"checkpoint":str(checkpoint_path),
                "checkpoint_reload_fixed_input_identical":True,"frozen_hashes_unchanged":True,
                "total_parameters":total,"trainable_parameters":trainable,"physical_gpus":[4,5,6,7],
                "global_batch":64,"micro_batch_per_rank":micro,"accumulation":accum}
        atomic(run/"status.json",result);print(json.dumps(result),flush=True)
    dist.barrier();del iterator,loader
    dist.destroy_process_group()

if __name__=="__main__":main()
