"""Four-way native FP16 8x8 cache extraction using canonical frozen FP32 SigLIP."""
from __future__ import annotations
import json,os,time
from pathlib import Path
os.environ["JAX_PLATFORMS"]="cpu"
import jax.numpy as jnp
import numpy as np
import torch,torch_musa
import torch.distributed as dist
from mme_vla_suite.musa.model import Vision,pool_tokens
from mme_vla_suite.musa.prepare_assets import load_tree
from mme_vla_suite.training.lerobot_framesamp import VideoReader,write_json
from mme_vla_suite.training.lerobot_percmem import validate_cache
from openpi.shared.image_tools import resize_with_pad

ROOT=Path("/data/heping/robomme_musa")
CACHE=ROOT/"data/percmem_cache"
def main():
    if os.environ.get("MUSA_VISIBLE_DEVICES")!="4,5,6,7":raise RuntimeError("Only physical GPU 4-7 may be used")
    rank=int(os.environ["LOCAL_RANK"]);world=int(os.environ["WORLD_SIZE"])
    if world!=4:raise RuntimeError("Four cache shards required")
    allowed=os.sched_getaffinity(0)
    affinity=sorted(set(map(int,os.environ["ROBOMME_CPU_POOL"].split(",")))&allowed) if os.environ.get("ROBOMME_CPU_POOL") else sorted(allowed)
    if len(affinity)>=32:os.sched_setaffinity(0,affinity[rank*8:rank*8+8])
    torch.set_num_threads(2);torch.set_float32_matmul_precision("highest")
    torch.backends.mudnn.allow_tf32=False
    torch.backends.musa.matmul.allow_tf32=False
    torch.musa.set_device(rank);device=torch.device("musa",rank)
    dist.init_process_group("mccl")
    check=json.loads((ROOT/"production_validation.json").read_text())
    if not check.get("vision",{}).get("passed"):raise RuntimeError("Production SigLIP numerical comparison required")
    report=json.loads((ROOT/"assets/export_complete.json").read_text())
    meta=validate_cache(CACHE,require_complete=False)
    source=json.loads((CACHE/"meta/source_index.json").read_text())
    export=json.loads((ROOT/"assets/base_export/manifest.json").read_text())
    arrays={k.removeprefix("PaliGemma/img/"):np.load(ROOT/"assets/base_export"/v["file"],mmap_mode="r")
            for k,v in export.items() if k.startswith("PaliGemma/img/")}
    encoder=Vision(arrays).to(device=device,dtype=torch.float32).eval();del arrays
    reader=VideoReader(source,max_open=2)
    selected=[e for e in source["episodes"] if e["episode_index"]%world==rank]
    started=time.monotonic();encoded=0
    try:
        for e in selected:
            name=f"episode_{e['episode_index']:06d}"
            out=CACHE/f"native/{name}.npy";record=out.with_suffix(".json")
            plan=np.load(CACHE/f"keyframes/{name}.npz")
            rows=plan["row"]
            if record.exists():
                old=json.loads(record.read_text())
                data=np.load(out,mmap_mode="r")
                if old["vision_id"]!=report["vision_id"] or old["cache_id"]!=meta["cache_id"] or data.shape!=(len(rows),64,2048) or data.dtype!=np.float16:raise ValueError("Incompatible existing native feature")
                continue
            target=out.with_name(out.name+".incomplete")
            mmap=np.lib.format.open_memmap(target,mode="w+",dtype=np.float16,shape=(len(rows),64,2048))
            batch=[];offset=0;wanted=set(map(int,rows))
            def encode(images):
                normalized=np.stack(images).astype(np.float32)/255*2-1
                resized=np.asarray(resize_with_pad(jnp.asarray(normalized),224,224))
                with torch.no_grad():
                    tokens=encoder(torch.from_numpy(resized.copy()).to(device),torch.float32)
                    result=pool_tokens(tokens,8).to(torch.float16).cpu().numpy()
                if not np.isfinite(result).all():raise FloatingPointError("Non-finite native history features")
                return result
            for i,image in enumerate(reader.frames(e)):
                if i not in wanted:continue
                batch.append(image)
                if len(batch)==16 or offset+len(batch)==len(rows):
                    valid=len(batch)
                    while len(batch)<16:batch.append(batch[-1])
                    result=encode(batch)[:valid]
                    mmap[offset:offset+valid]=result
                    offset+=valid;batch.clear()
            if offset!=len(rows):raise ValueError("Episode keyframe extraction incomplete")
            mmap.flush();del mmap;target.replace(out)
            write_json(record,{"episode_index":e["episode_index"],"vision_id":report["vision_id"],"cache_id":meta["cache_id"],
                         "shape":[len(rows),64,2048],"storage":"float16","compute":"torch_musa_fp32_canonical_bf16_weights"})
            encoded+=len(rows)
            print(json.dumps({"rank":rank,"episode":e["episode_index"],"keyframes":len(rows),"encoded":encoded,"seconds":time.monotonic()-started}),flush=True)
    finally:reader.close()
    dist.barrier()
    if rank==0:
        for e in source["episodes"]:
            name=f"episode_{e['episode_index']:06d}"
            data=np.load(CACHE/f"native/{name}.npy",mmap_mode="r")
            record=json.loads((CACHE/f"native/{name}.json").read_text())
            n=len(np.load(CACHE/f"keyframes/{name}.npz")["row"])
            if data.shape!=(n,64,2048) or data.dtype!=np.float16 or record["vision_id"]!=report["vision_id"] or record["cache_id"]!=meta["cache_id"]:raise ValueError("Whole-cache verification failed")
        write_json(CACHE/"meta/native_complete.json",{"cache_id":meta["cache_id"],"vision_id":report["vision_id"],"episodes":1600,"keyframes":meta["keyframes"],"backend":"MUSA FP32; production numerical parity validated"})
        validate_cache(CACHE)
        print("FULL_NATIVE_CACHE_VALIDATED",flush=True)
    dist.barrier();dist.destroy_process_group()
if __name__=="__main__":main()
