"""Original data transforms and original global-batch JAX randomness on CPU."""
from __future__ import annotations
import os
os.environ["JAX_PLATFORMS"]="cpu"
from pathlib import Path
import json
import augmax
import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece
import torch
from omegaconf import OmegaConf
from openpi import transforms
from openpi.models.model import ModelType
from openpi.shared import normalize
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.policies.robomme_policy import RoboMMEInputs
from mme_vla_suite.training.config import PaligemmaTokenizer
from mme_vla_suite.training.lerobot_percmem import LeRobotPercMemDataset

class TrainingDataset:
    def __init__(self,profile,*,require_complete=True):
        self.profile=profile
        history=get_history_config(profile["model"]["history_config"])
        self.dataset=LeRobotPercMemDataset(profile["data"]["dataset_path"],None,history,profile["model"]["action_horizon"],require_complete=require_complete)
        self.inputs=RoboMMEInputs(model_type=ModelType.PI05)
        self.delta=transforms.DeltaActions(transforms.make_bool_mask(7,-1))
        stats=normalize.load(Path(profile["data"]["norm_assets_dir"])/profile["data"]["norm_asset_id"])
        self.norm=transforms.Normalize(stats,use_quantiles=True)
        self.resize=transforms.ResizeImages(224,224)
        self.pad=transforms.PadStatesAndActions(32)
        self.tokenizer=PaligemmaTokenizer.__new__(PaligemmaTokenizer)
        self.tokenizer._max_len=64
        self.tokenizer._tokenizer=sentencepiece.SentencePieceProcessor(model_file=str(Path(profile["musa"]["root"])/"assets/paligemma_tokenizer.model"))
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        return self.transform_raw(self.dataset[index])
    def transform_raw(self,raw):
        raw.update({"observation/state":raw["state"],"observation/image":raw["image"],"observation/wrist_image":raw["wrist_image"]})
        data=self.resize(self.norm(self.delta(self.inputs(raw))))
        tokens,mask=self.tokenizer.tokenize(data["prompt"],state=None)
        data=self.pad(data)
        return {"front":data["image"]["base_0_rgb"],"wrist":data["image"]["left_wrist_0_rgb"],
                "state":data["state"],"actions":data["actions"],"tokens":tokens,"token_mask":mask,
                **{k:data[k] for k in ["perc_mem_feats","perc_mem_frame_tick","perc_mem_frame_mask","perc_mem_now_tick"]}}

def collate(samples):
    return {key:np.stack([sample[key] for sample in samples]) for key in samples[0]}

class GlobalBatchSampler:
    """Match the single-process TorchDataLoader permutation, split each batch across ranks."""
    def __init__(self,length,rank,world,global_batch,micro_batch,seed,start_step=0):
        if global_batch%(world*micro_batch):raise ValueError("Batch must divide world*micro")
        self.length,self.rank,self.world,self.global_batch,self.micro,self.seed=length,rank,world,global_batch,micro_batch,seed
        self.start_step=start_step
    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed)
        epoch=0;step=0
        while True:
            # The upstream loader keeps its workers alive: only its first iterator
            # consumes a base seed. Subsequent epochs reset that same iterator.
            if epoch==0:torch.empty((),dtype=torch.int64).random_(generator=generator)
            permutation=torch.randperm(self.length,generator=generator).tolist()
            count=self.length//self.global_batch
            local=self.global_batch//self.world
            for b in range(count):
                if step>=self.start_step:
                    begin=b*self.global_batch+self.rank*local
                    rows=permutation[begin:begin+local]
                    for offset in range(0,local,self.micro):yield rows[offset:offset+self.micro]
                step+=1
            # RandomSampler generates this empty remainder permutation at epoch end.
            torch.randperm(self.length,generator=generator)
            epoch+=1

_front=augmax.Chain(augmax.RandomCrop(212,212),augmax.Resize(224,224),augmax.Rotate((-5,5)),augmax.ColorJitter(brightness=.3,contrast=.4,saturation=.5))
_wrist=augmax.Chain(augmax.ColorJitter(brightness=.3,contrast=.4,saturation=.5))
_front_augment=jax.jit(jax.vmap(_front))
_wrist_augment=jax.jit(jax.vmap(_wrist))

def global_random(seed,step,batch_size=64):
    train_rng,_=jax.random.split(jax.random.key(seed))
    folded=jax.random.fold_in(train_rng,step)
    prep,noise,time=jax.random.split(folded,3)
    return (jax.random.split(prep,batch_size),
            np.asarray(jax.random.normal(noise,(batch_size,20,32))),
            np.asarray(jax.random.beta(time,1.5,1,(batch_size,))*.999+.001))

def prepare_batch(raw,keys,device,*,augment=True):
    values={}
    for name,fn in [("front",_front_augment),("wrist",_wrist_augment)]:
        image=raw[name].astype(np.float32)/255.0*2-1
        if augment:
            image=np.asarray(fn(keys,jnp.asarray(image)/2+.5))*2-1
        values[name]=torch.from_numpy(np.asarray(image).copy()).to(device)
    for name in raw:
        if name not in values:values[name]=torch.from_numpy(np.asarray(raw[name]).copy()).to(device)
    return values

def validate_global_sampler(*,multi_epoch=False):
    # Compare three actual global batches against the upstream DataLoader behavior.
    class Indices:
        def __len__(self):return 476857
        def __getitem__(self,i):return i
    reference=torch.utils.data.DataLoader(Indices(),batch_size=64,shuffle=True,drop_last=True,num_workers=0,generator=torch.Generator().manual_seed(42))
    samples=[iter(GlobalBatchSampler(476857,r,4,64,2,42)) for r in range(4)]
    it=iter(reference)
    for _ in range(3):
        expected=next(it).tolist()
        actual=[]
        for r in range(4):
            for _ in range(8):actual.extend(next(samples[r]))
        if actual!=expected:raise AssertionError("Global sampler does not match upstream shuffle")
    if multi_epoch:
        reference=torch.utils.data.DataLoader(range(131),batch_size=64,shuffle=True,drop_last=True,
            num_workers=1,persistent_workers=True,multiprocessing_context="spawn",
            generator=torch.Generator().manual_seed(42))
        samples=[iter(GlobalBatchSampler(131,r,4,64,2,42)) for r in range(4)]
        for epoch in range(3):
            for expected in reference:
                actual=[]
                for rank in range(4):
                    for _ in range(8):actual.extend(next(samples[rank]))
                if actual!=expected.tolist():raise AssertionError("Persistent-worker epoch ordering changed")
        del reference
    return True
