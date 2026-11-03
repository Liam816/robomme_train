"""P0 pi0.5/perceptual-memory port with the original JAX parameter layout.

No architecture conversion: every backbone leaf is loaded by its original path.
Trainable tensors remain FP32; frozen SigLIP storage is BF16, as in train.py.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def flatten(tree, prefix=""):
    result = {}
    for k,v in tree.items():
        if v is None: continue
        path = prefix + str(k)
        if isinstance(v,dict): result.update(flatten(v,path+"/"))
        else: result[path]=v
    return result


class Tree(nn.Module):
    def __init__(self, arrays, *, trainable=True, frozen_bf16=False):
        super().__init__()
        arrays={k:v for k,v in arrays.items() if v is not None}
        self.names = {p:p.replace("/","__") for p in arrays}
        for p,v in arrays.items():
            value=torch.as_tensor(np.array(v, dtype=np.float32, copy=True))
            if frozen_bf16: value=value.to(torch.bfloat16)
            self.register_parameter(self.names[p],nn.Parameter(value,requires_grad=trainable))
    def p(self,path): return getattr(self,self.names[path])
    def linear(self,x,path,dtype=None):
        dtype=x.dtype if dtype is None else dtype
        kernel=self.p(path+"/kernel").to(dtype)
        bias=self.p(path+"/bias").to(dtype) if path+"/bias" in self.names else None
        result=F.linear(x.to(dtype),kernel.T,None)
        return result if bias is None else result+bias
    def ln(self,x,path,eps=1e-5,dtype=None):
        dtype=x.dtype if dtype is None else dtype
        xf=x.float()
        mean=xf.mean(-1,keepdim=True)
        var=(xf.square().mean(-1,keepdim=True)-mean.square()).clamp_min(0)
        result=(xf-mean)*torch.rsqrt(var+eps)
        return (result*self.p(path+"/scale").float()+self.p(path+"/bias").float()).to(dtype)


def pool_tokens(x,grid):
    side=math.isqrt(x.shape[-2]); stride=side//grid
    if side*side!=x.shape[-2] or side%grid: raise ValueError("Invalid square token grid")
    return x.float().reshape(*x.shape[:-2],grid,stride,grid,stride,x.shape[-1]).mean((-4,-2)).reshape(*x.shape[:-2],grid*grid,x.shape[-1])


class Vision(Tree):
    def __init__(self,arrays):
        super().__init__(arrays,trainable=False,frozen_bf16=True)
        self.depth=self.p("Transformer/encoderblock/LayerNorm_0/scale").shape[0]
    def forward(self,image,precision=torch.bfloat16):
        # Input is NHWC, normalized and resized by the unchanged JAX pipeline.
        w=self.p("embedding/kernel").float().permute(3,2,0,1)
        x=F.conv2d(image.float().permute(0,3,1,2),w,self.p("embedding/bias").float(),stride=w.shape[-1])
        x=x.flatten(2).transpose(1,2)
        x=(x+self.p("pos_embedding").float()).to(precision)
        root="Transformer/encoderblock/"
        for i in range(self.depth):
            def p(name): return self.p(root+name)[i].to(precision)
            def ln(z,name):
                zf=z.float(); mean=zf.mean(-1,keepdim=True)
                var=(zf.square().mean(-1,keepdim=True)-mean.square()).clamp_min(0)
                return ((zf-mean)*torch.rsqrt(var+1e-6)*p(name+"/scale").float()+p(name+"/bias").float()).to(precision)
            y=ln(x,"LayerNorm_0")
            q,k,v=[torch.einsum("btd,dhc->bthc",y,p("MultiHeadDotProductAttention_0/"+name+"/kernel"))+
                     p("MultiHeadDotProductAttention_0/"+name+"/bias") for name in ("query","key","value")]
            q=(q/math.sqrt(q.shape[-1])).to(precision)
            logits=torch.einsum("bthd,bshd->bhts",q,k).to(precision)
            exponents=(logits-logits.max(-1,keepdim=True).values).exp()
            weights=(exponents/exponents.sum(-1,keepdim=True)).to(precision)
            y=torch.einsum("bhts,bshd->bthd",weights,v)
            y=torch.einsum("bthd,hdc->btc",y,p("MultiHeadDotProductAttention_0/out/kernel"))+p("MultiHeadDotProductAttention_0/out/bias")
            x=(x+y).to(precision)
            y=ln(x,"LayerNorm_1")
            y=F.gelu(y@p("MlpBlock_0/Dense_0/kernel")+p("MlpBlock_0/Dense_0/bias"),approximate="tanh")
            y=y@p("MlpBlock_0/Dense_1/kernel")+p("MlpBlock_0/Dense_1/bias")
            x=(x+y).to(precision)
        x=self.ln(x,"Transformer/encoder_norm",eps=1e-6,dtype=precision)
        return self.linear(x,"head",precision)


def rope(x,positions):
    exponent=torch.arange(0,x.shape[-1],2,device=x.device,dtype=torch.float32)/x.shape[-1]
    timescale=10000.0**exponent
    # JAX splits the channels into halves, then rotates [first_half, second_half].
    radians=positions.float()[...,None,None]/timescale[None,None,None,:]
    sin,cos=radians.sin(),radians.cos()
    a,b=x.float().chunk(2,dim=-1)
    return torch.cat((a*cos-b*sin,b*cos+a*sin),dim=-1).to(x.dtype)


class GemmaBlock(Tree):
    def rms(self,x,path,cond):
        normalized=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)
        if cond is None:
            return (normalized*(1+self.p(path+"/scale").float())).to(x.dtype),None
        modulation=self.linear(cond,path+"/Dense_0",x.dtype)[:,None,:]
        scale,shift,gate=modulation.chunk(3,-1)
        return (normalized*(1+scale).float()+shift.float()).to(x.dtype),gate
    def forward(self,prefix,suffix,cond,positions,mask):
        xs=[prefix,suffix]; normed=[]; gates=[]; qkvs=[]
        for i,x in enumerate(xs):
            tag="" if i==0 else "_1"
            z,g=self.rms(x,"pre_attention_norm"+tag,None if i==0 else cond)
            normed.append(z);gates.append(g)
            q=torch.einsum("btd,ndh->btnh",z,self.p("attn/q_einsum"+tag+"/w").to(z.dtype))
            kv=torch.einsum("btd,ckdh->cbtkh",z,self.p("attn/kv_einsum"+tag+"/w").to(z.dtype))
            qkvs.append((q,kv[0],kv[1]))
        q,k,v=[torch.cat([t[i] for t in qkvs],dim=1) for i in range(3)]
        q=(rope(q,positions)*(q.shape[-1]**-0.5)).to(q.dtype)
        k=rope(k,positions)
        logits=torch.einsum("bthd,bskd->bhts",q.float(),k.float())
        weights=torch.where(mask[:,None,:,:],logits,torch.tensor(-2.3819763e38,device=q.device)).softmax(-1).to(v.dtype)
        encoded=torch.einsum("bhts,bskd->bthd",weights,v)
        output=[];start=0
        for i,x in enumerate(xs):
            tag="" if i==0 else "_1"
            end=start+x.shape[1]
            update=torch.einsum("btnh,nhd->btd",encoded[:,start:end],self.p("attn/attn_vec_einsum"+tag+"/w").to(x.dtype))
            x=(x+update if gates[i] is None else x+update*gates[i]).to(x.dtype)
            z,gate=self.rms(x,"pre_ffw_norm"+tag,None if i==0 else cond)
            gating=self.p("mlp"+tag+"/gating_einsum").to(z.dtype)
            y=F.gelu(z@gating[0],approximate="tanh")*(z@gating[1])
            y=y@self.p("mlp"+tag+"/linear").to(z.dtype)
            output.append((x+y if gate is None else x+y*gate).to(x.dtype))
            start=end
        return output[0],output[1]


class Memory(Tree):
    def __init__(self,arrays,cfg):
        super().__init__({k:v for k,v in arrays.items() if not k.startswith("mem_scorer/")})
        self.scorer=Tree({k.removeprefix("mem_scorer/"):v for k,v in arrays.items() if k.startswith("mem_scorer/")},trainable=False)
        self.cfg=cfg
    def attention(self,q,m,valid,path,heads):
        b,nq,dim=q.shape; dh=dim//heads
        q,k,v=[self.linear(x,path+"/"+name).reshape(b,-1,heads,dh) for x,name in [(q,"q"),(m,"k"),(m,"v")]]
        logits=torch.einsum("bqhd,bkhd->bhqk",q,k)/math.sqrt(dh)
        weights=torch.where(valid[:,None,None,:],logits,-1e30).softmax(-1)
        weights=weights*valid[:,None,None,:]
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-20)
        z=torch.einsum("bhqk,bkhd->bqhd",weights,v).reshape(b,nq,dim)
        return self.linear(z,path+"/out"),weights
    def forward(self,native,ticks,valid,now,return_attention=False):
        c=self.cfg;b,n=native.shape[:2]
        projected=self.linear(native.float(),"mem_feature_adapter").to(torch.float16)
        m=pool_tokens(projected,c["read_grid"])
        root="mem_reader/"
        m=self.ln(self.linear(self.ln(m,root+"input_norm"),root+"proj"),root+"proj_norm")
        ages=(now[:,None]-ticks).float()/c["clock_hz"]
        freq=torch.exp(-math.log(c["sinusoid_base"])*torch.arange(c["age_dim"]//2,device=m.device).float()/(c["age_dim"]//2))
        phase=ages[...,None]*freq
        age=torch.cat((phase.sin(),phase.cos()),-1)
        age=self.linear(F.gelu(self.linear(age,root+"age_in"),approximate="none"),root+"age_out")
        m=m+self.p(root+"position/embedding")[None,None]+age[:,:,None]
        m=m.reshape(b,n*c["read_grid"]**2,c["reader_dim"])
        mask=valid.repeat_interleave(c["read_grid"]**2,dim=1)
        z=self.p(root+"query").expand(b,-1,-1);attentions=[]
        for i in range(c["reader_layers"]):
            p=root+f"layers/layer_{i}/"
            update,weights=self.attention(self.ln(z,p+"cross_norm"),m,mask,p+"cross",c["reader_heads"])
            z=z+update
            normed=self.ln(z,p+"self_norm")
            update,_=self.attention(normed,normed,torch.ones(z.shape[:2],device=z.device,dtype=torch.bool),p+"self_attn",c["reader_heads"])
            z=z+update
            y=self.linear(F.gelu(self.linear(self.ln(z,p+"ffn_norm"),p+"ffn/up"),approximate="none"),p+"ffn/down")
            z=z+y
            if return_attention: attentions.append(weights)
        result=self.linear(F.silu(self.linear(self.ln(z,root+"output_norm").reshape(b,-1),root+"output_in")),root+"output_final")
        result=torch.where(valid.any(1)[:,None],result,0)
        cond=self.linear(result,"mem_cond_adapter")
        return (cond,attentions) if return_attention else cond


class PercMemPi05Torch(nn.Module):
    def __init__(self,base,memory,cfg):
        super().__init__()
        self.cfg=cfg
        if cfg["stage"]!="teacher" or cfg["input_space"]!="native" or cfg["memory_mode"]!="normal":
            raise ValueError("This backend implements the requested P0 native full-history teacher profile")
        flat=flatten(base)
        vision={k.removeprefix("PaliGemma/img/"):v for k,v in flat.items() if k.startswith("PaliGemma/img/")}
        self.vision=Vision(vision)
        llm={k.removeprefix("PaliGemma/llm/"):v for k,v in flat.items() if k.startswith("PaliGemma/llm/")}
        self.embedding=nn.Parameter(torch.tensor(np.array(llm["embedder/input_embedding"],dtype=np.float32)))
        depth=llm["layers/pre_attention_norm/scale"].shape[0]
        self.blocks=nn.ModuleList([GemmaBlock({k.removeprefix("layers/"):v[i] for k,v in llm.items() if k.startswith("layers/")}) for i in range(depth)])
        self.final=GemmaBlock({k:v for k,v in llm.items() if k.startswith("final_norm")})
        self.projections=Tree({k:v for k,v in flat.items() if not k.startswith("PaliGemma/")})
        self.memory=Memory(flatten(memory),cfg)
        self.activation_checkpointing=True
        self.compute_dtype=torch.bfloat16
    def predict_velocity(self,batch,noisy,time,*,memory_cond=None,prefix_embeddings=None):
        dtype=self.compute_dtype
        if prefix_embeddings is None:
            with torch.no_grad():
                images=[self.vision(batch[key],dtype) for key in ["front","wrist"]]
            lang=(F.embedding(batch["tokens"].long(),self.embedding)*math.sqrt(self.embedding.shape[-1])).to(dtype)
            prefix=torch.cat([*images,lang],1)
        else:
            prefix=prefix_embeddings
        image_token_count=prefix.shape[1]-batch["tokens"].shape[1]
        suffix=self.projections.linear(noisy.float(),"action_in_proj").to(dtype)
        width=suffix.shape[-1]
        fraction=torch.linspace(0,1,width//2,device=time.device)
        period=4e-3*(4.0/4e-3)**fraction
        phase=time.float()[:,None]*((2*math.pi)/period)[None]
        time_emb=torch.cat((phase.sin(),phase.cos()),-1)
        cond=F.silu(self.projections.linear(F.silu(self.projections.linear(time_emb,"time_mlp_in")),"time_mlp_out"))
        if memory_cond is None:
            memory_cond=self.memory(batch["perc_mem_feats"],batch["perc_mem_frame_tick"],batch["perc_mem_frame_mask"],batch["perc_mem_now_tick"])
        cond=cond+memory_cond
        valid=torch.cat([torch.ones(prefix.shape[0],image_token_count,device=prefix.device,dtype=torch.bool),batch["token_mask"],torch.ones(suffix.shape[:2],device=suffix.device,dtype=torch.bool)],1)
        positions=valid.long().cumsum(1)-1
        split=prefix.shape[1]
        causal=torch.ones(valid.shape[1],valid.shape[1],device=prefix.device,dtype=torch.bool)
        causal[:split,split:]=False
        mask=valid[:,:,None]&valid[:,None,:]&causal[None]
        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                prefix,suffix=checkpoint(block,prefix,suffix,cond,positions,mask,use_reentrant=False)
            else: prefix,suffix=block(prefix,suffix,cond,positions,mask)
        suffix,_=self.final.rms(suffix,"final_norm_1",cond)
        # The original NNX output Linear has dtype=None and FP32 parameters,
        # so its BF16 suffix input is promoted to FP32 for the projection.
        return self.projections.linear(suffix,"action_out_proj",torch.float32)
    def forward(self,batch,noise=None,time=None):
        actions=batch["actions"].float()
        if noise is None: noise=torch.randn_like(actions)
        if time is None:
            # Beta(1.5,1) inverse CDF, same distribution without unsupported MUSA gamma.
            time=torch.rand(actions.shape[0],device=actions.device).pow(1/1.5)*0.999+0.001
        noisy=time[:,None,None]*noise+(1-time[:,None,None])*actions
        return (self.predict_velocity(batch,noisy,time)-(noise-actions)).square().mean()


    @torch.no_grad()
    def sample_actions(self,batch,*,num_steps=10,noise=None):
        if self.training:raise RuntimeError("Call eval() before deterministic action sampling")
        delta=self.memory(batch["perc_mem_feats"],batch["perc_mem_frame_tick"],batch["perc_mem_frame_mask"],batch["perc_mem_now_tick"])
        images=[self.vision(batch[key],self.compute_dtype) for key in ["front","wrist"]]
        lang=(F.embedding(batch["tokens"].long(),self.embedding)*math.sqrt(self.embedding.shape[-1])).to(self.compute_dtype)
        prefix=torch.cat([*images,lang],1)
        actions=torch.randn((prefix.shape[0],20,32),device=prefix.device) if noise is None else noise
        dt=-1.0/num_steps
        for i in range(num_steps):
            t=torch.full((prefix.shape[0],),1.0+i*dt,device=prefix.device,dtype=torch.float32)
            actions=actions+dt*self.predict_velocity(batch,actions,t,memory_cond=delta,prefix_embeddings=prefix)
        return actions
