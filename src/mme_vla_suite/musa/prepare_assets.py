"""Prepare validated P0 assets using the existing JAX CPU definitions."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ["JAX_PLATFORMS"]="cpu"
os.environ.setdefault("XLA_FLAGS","--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4")
import jax
import numpy as np
from flax import nnx,traverse_util
from omegaconf import OmegaConf
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0Config
from mme_vla_suite.models.representation.perc_memory_v1 import PercMemConfig
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.shared.percmem_io import canonical_vision_params,tree_hash
from mme_vla_suite.training.lerobot_framesamp import prepare_index
from mme_vla_suite.training.lerobot_percmem import prepare_cache
from openpi.models.model import restore_params
from openpi.shared import normalize

ROOT=Path(os.environ.get("ROBOMME_MUSA_ROOT","/data/heping/robomme_musa"))
REPO=Path(os.environ.get("ROBOMME_MUSA_REPO", str(Path(__file__).resolve().parents[3])))
PROFILE=Path(os.environ.get("ROBOMME_MUSA_CONFIG",str(REPO/"configs/train/robomme_percmem_v1_teacher_musa.yaml")))
os.chdir(REPO)

def save_tree(root,tree):
    root.mkdir(parents=True,exist_ok=True)
    manifest={}
    for key,value in traverse_util.flatten_dict(tree,sep="/").items():
        if value is None: continue
        array=np.asarray(value).astype(np.float32)
        name=hashlib.sha256(key.encode()).hexdigest()+".npy"
        path=root/name
        np.save(path,array)
        manifest[key]={"file":name,"shape":list(array.shape),"dtype":"float32","sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2))

def load_tree(root,verify=False):
    root=Path(root);manifest=json.loads((root/"manifest.json").read_text())
    flat={}
    for key,meta in manifest.items():
        path=root/meta["file"]
        if verify and hashlib.sha256(path.read_bytes()).hexdigest()!=meta["sha256"]:raise ValueError("Export hash mismatch: "+key)
        array=np.load(path,mmap_mode="r")
        if list(array.shape)!=meta["shape"] or str(array.dtype)!=meta["dtype"]:raise ValueError("Export shape mismatch: "+key)
        flat[key]=array
    return traverse_util.unflatten_dict(flat,sep="/")

def export():
    if not (ROOT/"assets/download_complete.json").exists():raise RuntimeError("Official weights not yet verified")
    profile=OmegaConf.load(PROFILE)
    base=restore_params(profile.weights.params_path,restore_type=np.ndarray)
    config=HistoryPi0Config(**OmegaConf.to_container(profile.model))
    train_rng,init_rng=jax.random.split(jax.random.key(profile.train.seed))
    _,model_rng=jax.random.split(init_rng)
    # Construct using the original seed and original module order, then preserve only new memory leaves.
    with __import__("openpi.shared.array_typing",fromlist=["disable_typechecking"]).disable_typechecking():
        original=config.create(model_rng)
    state=nnx.state(original,nnx.Param).to_pure_dict()
    memory={k:v for k,v in state.items() if k.startswith("mem_")}
    reference={k:v for k,v in state.items() if not k.startswith("mem_")}
    expected={k:v for k,v in traverse_util.flatten_dict(reference,sep="/").items() if v is not None}
    actual=traverse_util.flatten_dict(base,sep="/")
    if expected.keys()!=actual.keys():raise ValueError("Base checkpoint key mismatch")
    if any(np.shape(expected[k])!=np.shape(actual[k]) for k in expected):raise ValueError("Base checkpoint shape mismatch")
    jax.block_until_ready(memory)
    save_tree(ROOT/"assets/base_export",base)
    save_tree(ROOT/"assets/memory_initial",memory)
    vision=canonical_vision_params(base["PaliGemma"]["img"])
    report={"source":str(profile.weights.params_path),"base_leaves":len(actual),
            "base_parameters":sum(v.size for v in actual.values()),"vision_id":tree_hash(vision),
            "vision_parameters":sum(v.size for v in traverse_util.flatten_dict(vision).values()),
            "scorer_parameters":sum(np.size(v) for v in traverse_util.flatten_dict(memory["mem_scorer"]).values() if v is not None),
            "memory_parameters":sum(np.size(v) for v in traverse_util.flatten_dict(memory).values() if v is not None),
            "memory_config":OmegaConf.to_container(get_history_config(profile.model.history_config).perc_mem),
            "seed":int(profile.train.seed),"initialization":"original HistoryPi0Config.create; original PRNG splitting"}
    (ROOT/"assets/export_complete.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

def index():
    profile=OmegaConf.load(PROFILE)
    history=get_history_config(profile.model.history_config)
    meta=prepare_index(ROOT/"data/lerobot_view",ROOT/"data/index")
    cache=prepare_cache(ROOT/"data/index",Path(profile.data.dataset_path),PercMemConfig.from_history(history))
    print(json.dumps(cache,indent=2),flush=True)

def norm():
    cache=ROOT/"data/percmem_cache"
    source=json.loads((cache/"meta/source_index.json").read_text())
    arrays={k:np.load(cache/f"index/{k}.npy",mmap_mode="r") for k in ["state","actions","episode_index"]}
    samples=np.load(cache/"index/sample_indices.npy",mmap_mode="r")
    ends={e["episode_index"]:e["dataset_to_index"]-1 for e in source["episodes"]}
    statistics={k:normalize.RunningStats() for k in ["state","actions"]}
    for start in range(0,len(samples),128):
        rows=samples[start:start+128]
        state=np.asarray(arrays["state"][rows]).copy()
        last=np.array([ends[int(arrays["episode_index"][r])] for r in rows])
        indices=np.minimum(rows[:,None]+np.arange(20)[None],last[:,None])
        actions=np.asarray(arrays["actions"][indices]).copy()
        actions[...,:7]-=state[:,None,:7]
        statistics["state"].update(state)
        statistics["actions"].update(actions)
    result={k:s.get_statistics() for k,s in statistics.items()}
    target=ROOT/"assets/norm/mme_vla_suite/robomme"
    normalize.save(target,result)
    (target/"provenance.json").write_text(json.dumps({"source":source["source"],"execution_samples":len(samples),"action_horizon":20,
        "transform":"DeltaActions(make_bool_mask(7,-1)); episode-clipped horizon; original RunningStats","quantile_bins":5000},indent=2))
    print("NORMALIZATION_COMPLETE",target,flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["export","index","norm"])
    args=parser.parse_args()
    globals()[args.stage]()
