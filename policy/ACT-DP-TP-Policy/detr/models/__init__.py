# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import build_seg as build_vae_seg
from .detr_vae_nfp import build as build_vae_nfp
from .detr_vae import build_cnnmlp as build_cnnmlp
from .detr_vae import build_dino as build_dino
from .detr_vae import build_jpeg as build_jpeg
from .detr_vae import build_jpeg_diffusion as build_jpeg_diffusion
from .detr_vae import build_jpeg_diffusion_seperate as build_jpeg_diffusion_seperate
from .detr_vae import build_nf_diffusion_seperate as build_nf_diffusion_seperate
from .detr_vae import build_diffusion as build_diffusion
from .detr_vae import build_diffusion_tp as build_diffusion_tp
from .detr_vae import build_diffusion_tp_with_dual_visual_token as build_diffusion_tp_with_dual_visual_token
from .detr_vae import build_diffusion_pp as build_diffusion_pp
from .detr_vae import build_diffusion_tactile as build_diffusion_tactile

def build_ACT_model(args):
    return build_vae(args)

def build_CNNMLP_model(args):
    return build_cnnmlp(args)

def build_ACTDiffusion_model(args):
    return build_diffusion(args)

def build_ACTDiffusion_tactile_model(args):
    return build_diffusion_tactile(args)

def build_ACTDiffusion_tp_model(args):
    if args.diffusion_timestep_type  == 'vis_cat': # HARDCODE whether use tokenizer feature for decoder & action prediction
        print('Using dual visual token for decoder and action prediction')
        return build_diffusion_tp_with_dual_visual_token(args)
    else:
        return build_diffusion_tp(args)

def build_ACTDiffusion_pp_model(args):
    return build_diffusion_pp(args)
    
# discard
def build_ACT_NF_model(args):
    return build_vae_nfp(args)

def build_ACT_Seg_model(args):
    return build_vae_seg(args)

def build_ACT_dino_model(args):
    return build_dino(args)

def build_ACT_jpeg_model(args):
    return build_jpeg(args)

def build_ACT_jpeg_diffusion_model(args):
    return build_jpeg_diffusion(args)

def build_ACT_jpeg_diffusion_seperate_model(args):
    return build_jpeg_diffusion_seperate(args)

def build_nf_diffusion_seperate_model(args):
    return build_nf_diffusion_seperate(args)

