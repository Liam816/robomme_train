"""Check the actual pretrained production model and history encoder against JAX."""
from __future__ import annotations
import dataclasses,hashlib,json,os,time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
os.environ["JAX_PLATFORMS"]="cpu"
import jax,jax.numpy as jnp
from flax import nnx
import ml_dtypes
import numpy as np
import torch,torch_musa
import yaml
from mme_vla_suite.musa.model import PercMemPi05Torch,Vision,pool_tokens
from mme_vla_suite.musa.prepare_assets import load_tree
from mme_vla_suite.musa.data import TrainingDataset,collate,prepare_batch,global_random,validate_global_sampler
from mme_vla_suite.training.lerobot_framesamp import VideoReader
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.shared.percmem_io import FrozenHistoryEncoder,canonical_vision_params
from openpi.models.model import Observation,preprocess_observation
from openpi.shared.image_tools import resize_with_pad
import openpi.shared.array_typing as at

ROOT=Path("/data/heping/robomme_musa")
REPO=ROOT/"repo"
os.chdir(REPO)
predict_reference=nnx.jit(lambda module,observation,noisy,flow_t:module.predict_velocity(observation,noisy,flow_t))
report={"checked_at":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),"physical_gpu":4}
def save():
    (ROOT/"production_validation.json").write_text(json.dumps(report,indent=2))
def metrics(a,b,*,atol,rtol,cosine=None,relative_l2=None):
    a=np.asarray(a,dtype=np.float32);b=np.asarray(b,dtype=np.float32)
    aa=a.reshape(-1).astype(np.float64);bb=b.reshape(-1).astype(np.float64)
    c=float(np.dot(aa,bb)/max(np.linalg.norm(aa)*np.linalg.norm(bb),1e-30))
    rel=float(np.linalg.norm(aa-bb)/max(np.linalg.norm(bb),1e-20))
    close=bool(np.allclose(a,b,atol=atol,rtol=rtol))
    good=bool(np.isfinite(a).all() and np.isfinite(b).all())
    good=good and (close if relative_l2 is None else rel<=relative_l2)
    if cosine is not None:good=good and c>=cosine
    return {"passed":good,"max_abs_error":float(np.max(np.abs(a-b))),"relative_l2":rel,"cosine":c,
            "atol":atol,"rtol":rtol,"required_cosine":cosine,"required_relative_l2":relative_l2}
def require(name,value):
    report[name]=value;save()
    if not value["passed"]:raise AssertionError(name+": "+str(value))
def main():
    if os.environ.get("MUSA_VISIBLE_DEVICES")!="4":raise RuntimeError("Production comparison uses physical GPU 4 only")
    allowed=os.sched_getaffinity(0)
    available=sorted(set(map(int,os.environ["ROBOMME_CPU_POOL"].split(",")))&allowed) if os.environ.get("ROBOMME_CPU_POOL") else sorted(allowed)
    os.sched_setaffinity(0,available[:8])
    torch.set_num_threads(4);torch.set_float32_matmul_precision("highest")
    torch.backends.mudnn.allow_tf32=False
    torch.backends.musa.matmul.allow_tf32=False
    torch.musa.set_device(0);device=torch.device("musa",0)
    profile=yaml.safe_load((REPO/"configs/train/robomme_percmem_v1_teacher_musa.yaml").read_text())
    base=load_tree(ROOT/"assets/base_export",verify=True)
    memory=load_tree(ROOT/"assets/memory_initial",verify=True)
    metadata=json.loads((ROOT/"assets/export_complete.json").read_text())
    source=json.loads((ROOT/"data/percmem_cache/meta/source_index.json").read_text())
    episode=source["episodes"][0]
    reader=VideoReader(source)
    frames=[image for i,image in enumerate(reader.frames(episode,count=21)) if i in [0,10,20]]
    wrist=next(reader.frames(episode,"wrist_image",20,1))
    rgb=np.stack(frames)
    original_encoder=FrozenHistoryEncoder(base["PaliGemma"]["img"])
    normalized=jnp.asarray(rgb.astype(np.float32)/255*2-1)
    resized=resize_with_pad(normalized,224,224)
    with at.disable_typechecking():
        expected_tokens=np.asarray(original_encoder.image(resized,train=False)[0])
    encoder=Vision({k:v for k,v in __import__("mme_vla_suite.musa.model",fromlist=["flatten"]).flatten(base["PaliGemma"]["img"]).items()}).to(device=device,dtype=torch.float32).eval()
    with torch.no_grad():
        actual_tokens=encoder(torch.from_numpy(np.asarray(resized).copy()).to(device),torch.float32).cpu()
    require("vision",metrics(actual_tokens.numpy(),expected_tokens,atol=1e-3,rtol=2e-4,cosine=.9999))
    actual_native=pool_tokens(actual_tokens,8).to(torch.float16).numpy()
    with at.disable_typechecking():expected_native=np.asarray(original_encoder(jnp.asarray(rgb)))
    require("native_fp16_cache",metrics(actual_native,expected_native,atol=.03125,rtol=2e-3,cosine=.9999))
    del original_encoder,encoder
    torch.musa.empty_cache()
    adapter=TrainingDataset(profile,require_complete=False)
    row=episode["dataset_from_index"]+20
    arrays=adapter.dataset.arrays
    action_rows=np.minimum(row+np.arange(20),episode["dataset_to_index"]-1)
    feats=np.zeros((320,64,2048),np.float16);feats[:3]=actual_native
    ticks=np.zeros(320,np.int32);ticks[:3]=[0,10,20]
    mask=np.zeros(320,bool);mask[:3]=True
    raw={"image":frames[-1],"wrist_image":wrist,"state":np.array(arrays["state"][row]),
         "actions":np.array(arrays["actions"][action_rows]),
         "prompt":source["tasks"][str(int(arrays["task_index"][row]))].lower(),
         "simple_subgoal":"","grounded_subgoal":"","perc_mem_feats":feats,
         "perc_mem_frame_tick":ticks,"perc_mem_frame_mask":mask,"perc_mem_now_tick":np.int32(20)}
    sample=adapter.transform_raw(raw)
    batch_numpy=collate([sample])
    keys,noise,times=global_random(42,0,64)
    batch=prepare_batch(batch_numpy,keys[:1],device,augment=False)
    obs=HistAugObservation(images={"base_0_rgb":jnp.asarray(batch["front"].cpu().numpy()),"left_wrist_0_rgb":jnp.asarray(batch["wrist"].cpu().numpy())},
        image_masks={"base_0_rgb":jnp.ones((1,),bool),"left_wrist_0_rgb":jnp.ones((1,),bool)},
        state=jnp.asarray(batch_numpy["state"]),tokenized_prompt=jnp.asarray(batch_numpy["tokens"]),
        tokenized_prompt_mask=jnp.asarray(batch_numpy["token_mask"]),perc_mem_feats=jnp.asarray(batch_numpy["perc_mem_feats"]),
        perc_mem_frame_tick=jnp.asarray(batch_numpy["perc_mem_frame_tick"]),perc_mem_frame_mask=jnp.asarray(batch_numpy["perc_mem_frame_mask"]),
        perc_mem_now_tick=jnp.asarray(batch_numpy["perc_mem_now_tick"]))
    base={**base,"PaliGemma":{**base["PaliGemma"],"img":jax.tree.map(lambda x:np.asarray(x).astype(ml_dtypes.bfloat16),base["PaliGemma"]["img"])}}
    combined={**base,**memory}
    pt=PercMemPi05Torch(base,memory,metadata["memory_config"]).to(device).eval()
    draw=torch.from_numpy(noise[:1].copy()).to(device)
    flow=torch.from_numpy(times[:1].copy()).to(device)
    actions=batch["actions"].cpu().numpy()
    noisy=np.asarray(times[:1,None,None]*noise[:1]+(1-times[:1,None,None])*actions)
    for precision in ["float32","bfloat16"]:
        config=HistoryPi0Config(**{**profile["model"],"dtype":precision})
        with at.disable_typechecking():
            reference=config.load(combined)
            reference.eval()
            expected=np.asarray(predict_reference(reference,obs,jnp.asarray(noisy),jnp.asarray(times[:1])))
        pt.compute_dtype=torch.float32 if precision=="float32" else torch.bfloat16
        with torch.no_grad():actual=pt.predict_velocity(batch,torch.from_numpy(noisy.copy()).to(device),flow).cpu().numpy()
        if precision=="float32":
            value=metrics(actual,expected,atol=2e-3,rtol=2e-3,cosine=.99999)
        else:
            value=metrics(actual,expected,atol=.03,rtol=.03,cosine=.999,relative_l2=.04)
        require("production_velocity_"+precision,value)
        expected_loss=float(np.mean((expected-(noise[:1]-actions))**2))
        actual_loss=float(np.mean((actual-(noise[:1]-actions))**2))
        require("flow_loss_"+precision,{"passed":abs(actual_loss-expected_loss)<=max(1e-5,abs(expected_loss)*(.003 if precision=="float32" else .04)),
                "jax":expected_loss,"musa":actual_loss})
        if precision=="float32":
            rng=np.random.default_rng(123)
            perturb=rng.normal(0,.001,size=(2048,2048)).astype(np.float32)
            reference.mem_reader.output_final.kernel.value=jnp.asarray(perturb)
            with torch.no_grad():pt.memory.p("mem_reader/output_final/kernel").copy_(torch.from_numpy(perturb).to(device))
            with at.disable_typechecking():changed=np.asarray(predict_reference(reference,obs,jnp.asarray(noisy),jnp.asarray(times[:1])))
            with torch.no_grad():actual_changed=pt.predict_velocity(batch,torch.from_numpy(noisy.copy()).to(device),flow).cpu().numpy()
            require("nonzero_memory_integration",metrics(actual_changed,changed,atol=3e-3,rtol=3e-3,cosine=.99999))
            if np.max(np.abs(changed-expected))<1e-4:raise AssertionError("Memory integration test did not affect action velocity")
            with torch.no_grad():pt.memory.p("mem_reader/output_final/kernel").zero_()
        del reference
        jax.clear_caches()
    # The global-batch keys and exact augmax operations are retained across microbatches.
    im={"base_0_rgb":jnp.repeat(jnp.asarray(batch["front"].cpu().numpy()),64,axis=0),
        "left_wrist_0_rgb":jnp.repeat(jnp.asarray(batch["wrist"].cpu().numpy()),64,axis=0)}
    observation=Observation(images=im,image_masks={k:jnp.ones((64,),bool) for k in im},state=jnp.zeros((64,32)))
    train_rng,_=jax.random.split(jax.random.key(42));prep_key,_,_=jax.random.split(jax.random.fold_in(train_rng,0),3)
    with at.disable_typechecking():augmented=preprocess_observation(prep_key,observation,train=True)
    sub={k:np.repeat(v,2,axis=0) for k,v in batch_numpy.items()}
    prepared=prepare_batch(sub,keys[30:32],torch.device("cpu"),augment=True)
    require("augmentation_front",metrics(prepared["front"].numpy(),np.asarray(augmented.images["base_0_rgb"])[30:32],atol=3e-6,rtol=3e-6))
    require("augmentation_wrist",metrics(prepared["wrist"].numpy(),np.asarray(augmented.images["left_wrist_0_rgb"])[30:32],atol=3e-6,rtol=3e-6))
    require("global_sampler",{"passed":validate_global_sampler(multi_epoch=True)})
    report["total_parameters"]=sum(p.numel() for p in pt.parameters())
    report["trainable_parameters"]=sum(p.numel() for p in pt.parameters() if p.requires_grad)
    assert report["total_parameters"]==metadata["base_parameters"]+metadata["memory_parameters"]
    assert report["trainable_parameters"]==report["total_parameters"]-metadata["vision_parameters"]-metadata["scorer_parameters"]
    report["passed"]=True
    reader.close();save();print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
