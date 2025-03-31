# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
from .vision_transformer import Block, get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_v2
from .mr_mg.policy.model.vision_transformer import vit_base

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(14, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent
        
        # settings for next frame prediction
        self.patch_size = 16
        # self.image_size = 224
        # self.img_h, self.img_w = 128, 160
        self.img_h, self.img_w = 224, 224
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # self.n_patch = (self.image_size//self.patch_size)**2
        self.k = 1 # number of next frames
        self.n_patch = (self.img_h//self.patch_size)*(self.img_w//self.patch_size)*(self.k)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.n_patch, hidden_dim), requires_grad=False)  # (1, n_patch, h)
        self.patch_embed = nn.Embedding(self.n_patch, hidden_dim)
        self.decoder_embed = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        decoder_depth = 2 # hardcode
        self.decoder_blocks = nn.ModuleList([
            Block(hidden_dim, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_pred = nn.Linear(hidden_dim, self.patch_size**2 * 3, bias=True) # decoder to patch

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.image_size//self.patch_size), cls_token=False)
        decoder_pos_embed = get_2d_sincos_pos_embed_v2(self.decoder_pos_embed.shape[-1], (self.img_h//self.patch_size, self.img_w//self.patch_size))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0).repeat(1,self.k,1))

        # fwd_params = sum(p.numel() for p in self.decoder_blocks.parameters() if p.requires_grad)
        

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            if is_training:
                next_frame_images = image[:,1:]
                image = image[:,:1]
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED?
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            query_embed = torch.cat([self.query_embed.weight, self.patch_embed.weight], axis=0)
            hs = self.transformer(src, None, query_embed, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
            # hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs[:,:self.num_queries])
        is_pad_hat = self.is_pad_head(hs[:,:self.num_queries])
        
        # next frame prediction
        mask_token = self.mask_token
        mask_tokens = mask_token.repeat(bs, self.n_patch, 1)
        mask_tokens = mask_tokens + self.decoder_pos_embed.repeat(bs, 1, 1)
        
        obs_pred = self.decoder_embed(hs[:,self.num_queries:])
        obs_pred_ = torch.cat([obs_pred, mask_tokens], dim=1)
        for blk in self.decoder_blocks:
            obs_pred_ = blk(obs_pred_)
        obs_pred_ = self.decoder_norm(obs_pred_)
        obs_preds = self.decoder_pred(obs_pred_)
        obs_preds = obs_preds[:,self.n_patch:]
        
        if is_training:
            # next_frame_images = image[:,1:]
            next_frame_images = nn.functional.interpolate(next_frame_images.reshape(bs, self.k*3, 224, 224), size=(self.img_h, self.img_w))
            p = self.patch_size
            h_p = self.img_h // p 
            w_p = self.img_w // p 
            obs_targets = next_frame_images.reshape(shape=(bs, self.k, 3, h_p, p, w_p, p))
            obs_targets = obs_targets.permute(0,1,3,5,4,6,2)
            obs_targets = obs_targets.reshape(shape=(bs, h_p*w_p*self.k, (p**2)*3))
        else:
            obs_targets = torch.zeros_like(obs_preds)
                    
        return a_hat, is_pad_hat, [mu, logvar], [obs_preds, obs_targets]


class DETRVAE_MAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # self.model_mae = vits.__dict__['vit_base'](patch_size=16, num_classes=0)
        self.model_mae = vit_base(patch_size=16, num_classes=0)
        mae_ckpt = 'checkpoints/pretrained/mae_pretrain_vit_base.pth'
        checkpoint = torch.load(mae_ckpt, map_location='cpu')
        self.model_mae.load_state_dict(checkpoint['model'], strict=True)
        print('Load MAE pretrained model')
        # for name, p in self.model_mae.named_parameters():
        #     p.requires_grad = False

        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(14, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent
        
        # settings for next frame prediction
        self.patch_size = 16
        self.img_h, self.img_w = 224, 224
        self.n_patch = (self.img_h//self.patch_size)*(self.img_w//self.patch_size)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.n_patch, hidden_dim), requires_grad=False)  # (1, n_patch, h)
        self.patch_embed = nn.Embedding(self.n_patch, hidden_dim)
        self.decoder_embed = nn.Linear(hidden_dim, hidden_dim, bias=True)
        
        decoder_depth = 2 # hardcode
        self.decoder_blocks = nn.ModuleList([
            Block(hidden_dim, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(decoder_depth)])

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_pred = nn.Linear(hidden_dim, self.patch_size**2 * 3, bias=True) # decoder to patch

        decoder_pos_embed = get_2d_sincos_pos_embed_v2(self.decoder_pos_embed.shape[-1], (self.img_h//self.patch_size, self.img_w//self.patch_size))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))


    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            if is_training:
                next_frame_images = image[:,1:]
                image = image[:,:1]
            for cam_id, cam_name in enumerate(self.camera_names):
                # features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
                # features = features[0] # take the last layer feature
                # pos = pos[0]
                # all_cam_features.append(self.input_proj(features))
                # all_cam_pos.append(pos)
                
                obs_embedings, patch_embedings, pos_mae = self.model_mae(image[:,cam_id])
                
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            # src = torch.cat(all_cam_features, axis=3)
            # pos = torch.cat(all_cam_pos, axis=3)
            query_embed = torch.cat([self.query_embed.weight, self.patch_embed.weight], axis=0)
            hs = self.transformer(patch_embedings, None, query_embed, pos_mae[0,1:], latent_input, proprio_input, self.additional_pos_embed.weight)[0]
            # hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs[:,:self.num_queries])
        is_pad_hat = self.is_pad_head(hs[:,:self.num_queries])
        
        # next frame prediction
        mask_token = self.mask_token
        mask_tokens = mask_token.repeat(bs, self.n_patch, 1)
        mask_tokens = mask_tokens + self.decoder_pos_embed.repeat(bs, 1, 1)
                
        obs_pred = self.decoder_embed(hs[:,self.num_queries:])
        obs_pred_ = torch.cat([obs_pred, mask_tokens], dim=1)
        for blk in self.decoder_blocks:
            obs_pred_ = blk(obs_pred_)
        obs_pred_ = self.decoder_norm(obs_pred_)
        obs_preds = self.decoder_pred(obs_pred_)
        obs_preds = obs_preds[:,self.n_patch:]
        
        if is_training:
            # next_frame_images = image[:,1:]
            # next_frame_images = nn.functional.interpolate(next_frame_images[:,0], size=(self.img_h, self.img_w))
            next_frame_images = next_frame_images[:,0]
            p = self.patch_size
            h_p = self.img_h // p 
            w_p = self.img_w // p 
            obs_targets = next_frame_images.reshape(shape=(bs, 3, h_p, p, w_p, p))
            obs_targets = obs_targets.permute(0,2,4,3,5,1)
            obs_targets = obs_targets.reshape(shape=(bs, h_p*w_p, (p**2)*3))
        else:
            obs_targets = torch.zeros_like(obs_preds)
                    
                    
        return a_hat, is_pad_hat, [mu, logvar], [obs_preds, obs_targets]


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    if not args.mae:
        model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            num_queries=args.num_queries,
            camera_names=args.camera_names,
        )
    else:
        model = DETRVAE_MAE(
            backbones,
            transformer,
            encoder,
            state_dim=state_dim,
            num_queries=args.num_queries,
            camera_names=args.camera_names,
        )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

