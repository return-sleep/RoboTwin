# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import *
from .vision_transformer import (
    Block,
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_v2,
)
from einops import rearrange
import numpy as np

import IPython

e = IPython.embed


class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=10.0):
        """
        Args:
            input_dim (int): input dimension.
            mapping_size (int): Fourier Features output dimension.
            scale (float): scale factor for frequencies.
        """
        super(FourierFeatureMapping, self).__init__()
        self.B = torch.randn((mapping_size, input_dim)) * scale

    def forward(self, x):
        """
        Args:
            x (Tensor): [batch_size, input_dim]
        Returns:
            Tensor: Fourier Features [batch_size, mapping_size * 2]
        """
        x_proj = 2 * torch.pi * x @ self.B.T  # [batch_size, mapping_size]
        return torch.cat(
            [torch.sin(x_proj), torch.cos(x_proj)], dim=-1
        )  # Concatenate sin and cos


class MLPWithFourierFeatures(nn.Module):
    def __init__(self, input_dim, mapping_size, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): input dimension.
            mapping_size (int): Fourier Features output dimension.
            hidden_dim (int): MLP layer layer dimension.
            output_dim (int): MLP output dimension.
        """
        super(MLPWithFourierFeatures, self).__init__()
        self.fourier_mapping = FourierFeatureMapping(input_dim, mapping_size)
        self.mlp = nn.Sequential(
            nn.Linear(mapping_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor):  [batch_size, input_dim]
        Returns:
            Tensor: MLP
        """
        x_mapped = self.fourier_mapping(x)  # Fourier Features Mapping
        output = self.mlp(x_mapped)  #  MLP
        return output


def get_spatial_temporal_positional_encoding(height, width, seq_len, dim):
    """
    Generate spatial-temporal positional encoding for video data.

    Args:
        height (int): Height of the video frames.
        width (int): Width of the video frames.
        seq_len (int): Number of frames in the video.
        dim (int): Embedding dimension.
        device (str): Device to store the tensor.

    Returns:
        torch.Tensor: Positional encoding with shape (seq_len, height, width, dim).
    """
    assert dim % 2 == 0, "Embedding dimension (dim) must be even."

    spatial_dim = dim // 2
    temporal_dim = dim - spatial_dim

    # Temporal encoding
    temporal_encoding = get_sinusoid_encoding_table(seq_len, temporal_dim)[
        0
    ]  # Shape: (seq_len, temporal_dim)

    # Spatial encoding
    position_h = torch.arange(height, dtype=torch.float32).unsqueeze(1)  # (height, 1)
    position_w = torch.arange(width, dtype=torch.float32).unsqueeze(1)  # (1, width)
    div_term_h = torch.exp(
        -torch.arange(0, spatial_dim, 2, dtype=torch.float32)
        * (math.log(10000.0) / spatial_dim)
    ).unsqueeze(0)
    div_term_w = torch.exp(
        -torch.arange(0, spatial_dim, 2, dtype=torch.float32)
        * (math.log(10000.0) / spatial_dim)
    ).unsqueeze(0)

    spatial_encoding_h = torch.sin(
        position_h * div_term_h
    )  # (height, spatial_dim // 2)
    spatial_encoding_w = torch.cos(position_w * div_term_w)  # (width, spatial_dim // 2)

    # print(spatial_encoding_h.shape, spatial_encoding_w.shape)
    # Combine H and W spatial encodings
    spatial_encoding_h = spatial_encoding_h.unsqueeze(1).expand(
        -1, width, -1
    )  # (height,  width£¬ spatial_dim // 2)
    spatial_encoding_w = spatial_encoding_w.unsqueeze(0).expand(
        height, -1, -1
    )  # (height, width , spatial_dim // 2)
    spatial_encoding = torch.cat(
        [spatial_encoding_h, spatial_encoding_w], dim=-1
    )  # (height, width , spatial_dim)
    spatial_encoding = spatial_encoding.unsqueeze(0).repeat(
        seq_len, 1, 1, 1
    )  # (seq_len, height, width , spatial_dim)

    # Combine spatial and temporal
    temporal_encoding = (
        temporal_encoding.unsqueeze(1).unsqueeze(1).repeat(1, height, width, 1)
    )  # (seq_len, height, width, temporal_dim)
    # print(spatial_encoding.shape, temporal_encoding.shape)
    pos_encoding = torch.cat(
        [spatial_encoding, temporal_encoding], dim=-1
    )  # Combine spatial and temporal

    return pos_encoding  # (seq_len, height, width, dim)


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    # print(sinusoid_table.shape)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    if not isinstance(pos, np.ndarray):
        pos = np.array(pos, dtype=np.float64)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_nd_sincos_pos_embed_from_grid(embed_dim, grid_sizes):  # useful
    """
    embed_dim: output dimension for each position
    grid_sizes: the grids sizes in each dimension (K,). for example [T, C, P] n_his, n_view,n_patch
    out: (grid_sizes[0], ..., grid_sizes[K-1], D) for example (T, C, P, D)
    """
    num_sizes = len(grid_sizes)
    # For grid size of 1, we do not need to add any positional embedding
    num_valid_sizes = len([x for x in grid_sizes if x > 1])
    emb = np.zeros(grid_sizes + (embed_dim,))
    # Uniformly divide the embedding dimension for each grid size
    dim_for_each_grid = embed_dim // num_valid_sizes
    # To make it even
    if dim_for_each_grid % 2 != 0:
        dim_for_each_grid -= 1
    valid_size_idx = 0
    for size_idx in range(num_sizes):
        grid_size = grid_sizes[size_idx]
        if grid_size <= 1:
            continue
        pos = np.arange(grid_size)
        posemb_shape = [1] * len(grid_sizes) + [dim_for_each_grid]
        posemb_shape[size_idx] = -1
        emb[
            ...,
            valid_size_idx
            * dim_for_each_grid : (valid_size_idx + 1)
            * dim_for_each_grid,
        ] += get_1d_sincos_pos_embed_from_grid(dim_for_each_grid, pos).reshape(
            posemb_shape
        )
        valid_size_idx += 1
    return emb


class DETRVAE(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(
        self, backbones, transformer, encoder, state_dim, num_queries, camera_names
    ):
        """Initializes the model.
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
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            14, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for proprio and latent

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                qpos.device
            )  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
                features = features[0]  # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
            )[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self.transformer(
                transformer_input, None, self.query_embed.weight, self.pos.weight
            )[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


class DETRVAE_Denoise(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        num_queries,
        camera_names,
        disable_vae_latent,
    ):
        """Initializes the model.
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
        self.disable_vae_latent = disable_vae_latent
        hidden_dim = transformer.d_model  # 512
        # self.action_head = nn.Linear(hidden_dim, state_dim)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)  # proprioception
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            14, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )  # [CLS], qpos, a_seq

        # decoder extra parameters vae latent to decoder token space
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for proprio and latent

    # src, mask, query_embed, pos_embed, latent_input=None, proprio_input=None, additional_pos_embed=None, noisy_actions = None, denoise_steps=None
    def forward(
        self,
        qpos,
        image,
        env_state,
        actions=None,
        is_pad=None,
        denoise_steps=None,
        is_training=True,
    ):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim noisy action
        denoise_step: int, the step of denoise
        """
        # is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence add a paramers = use_latent = True
        if is_training and self.disable_vae_latent == False:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            # vae encoder
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                qpos.device
            )  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(
                latent_sample
            )  # get latent input for decoder TODO do we need this?
        else:  # if dismiss latent ,just set to zero when training or val
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            # latent_sample = torch.randn([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
                features = features[0]  # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)  # B D H W*4
            pos = torch.cat(all_cam_pos, axis=3)  # B D H W*4 src_pos_emb
            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
                actions,
                denoise_steps,
            )[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self.transformer(
                transformer_input,
                None,
                self.query_embed.weight,
                self.pos.weight,
                denoise_steps,
            )[0]
        a_hat = self.action_head(hs)  # predict action or noise output of module
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


class DETRVAE_Denoise_Token_Prediction(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        num_queries,
        camera_names,
        history_step,
        predict_frame,
        image_downsample_rate,
        temporal_downsample_rate,
        disable_vae_latent,
        disable_resnet,
        patch_size=5,
        token_pe_type="learned",
        image_height=480,
        image_width=640,
        tokenizer_model_temporal_rate=8,
        tokenizer_model_spatial_rate=16,
        resize_rate=1,
    ):
        """Initializes the model.
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
        self.future_camera_names = [camera_names[0]] # TODO attention
        self.transformer = transformer
        self.transformer_share_decoder = transformer.share_decoder
        self.predict_only_last = transformer.predict_only_last
        self.encoder = encoder
        self.disable_vae_latent = disable_vae_latent
        self.disable_resnet = disable_resnet
        hidden_dim = transformer.d_model  # 512
        self.hidden_dim = hidden_dim
        token_dim = transformer.token_dim
        # Action head state_dim = action_dim
        # self.action_head = nn.Linear(hidden_dim, state_dim) # replace MLP?
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.token_head = nn.Sequential(
            nn.Linear(
                hidden_dim, hidden_dim
            ),  # Hardcode patch size * path size * patch dim
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim * patch_size * patch_size),
        )

        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # for decoder as PE
        self.diffusion_timestep_type = transformer.diffusion_timestep_type
        if backbones is not None and disable_resnet == False:
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels * (history_step + 1),
                hidden_dim,
                kernel_size=1,
            )  # = MLP c h w -> c' h' w'  frame stack
            self.backbones = nn.ModuleList(backbones)  # N encoders for N view
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)  # proprioception
        elif backbones is not None and disable_resnet == True:
            self.pos_encoder = backbones[0][1]  # for 2D PositionEmbedding
            self.backbones = None  # HardCDcode
            # Hardcode divide the latent feature representation into non-overlapping patches
            self.input_proj_token = nn.Conv2d(
                token_dim, hidden_dim, kernel_size=5, stride=5, bias=False
            )  # Hardcode deal with image token pa
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)  # proprioception
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            14, hidden_dim
        )  # project action to embedding TODO action dim = 14/16
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + history_step + num_queries, hidden_dim),
        )  # [CLS], qpos, a_seq

        # decoder extra parameters vae latent to decoder token space
        self.history_step = history_step
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2 + history_step, hidden_dim
        )  # learned position embedding for proprio and latent and denpoe
        self.denoise_step_pos_embed = nn.Embedding(
            1, hidden_dim
        )  # learned position embedding for denoise step
        # setting for token prediction
        if self.predict_only_last:
            self.num_temporal_token = 1
        else:
            # print('predict frame', predict_frame, 'temporal_downsample_rate', temporal_downsample_rate, 'tokenizer_model_temporal_rate', tokenizer_model_temporal_rate)
            self.num_temporal_token = math.ceil(
                predict_frame
                // temporal_downsample_rate
                / tokenizer_model_temporal_rate
            )
        self.patch_size = patch_size
        self.image_h = (
            image_height
            // image_downsample_rate
            // tokenizer_model_spatial_rate
            // resize_rate
            // self.patch_size
        )  #
        self.image_w = (
            image_width
            // image_downsample_rate
            // tokenizer_model_spatial_rate
            // resize_rate
            // self.patch_size
        )  #
        self.num_pred_token_per_timestep = (
            self.image_h * self.image_w * len(self.future_camera_names)
        ) # 
        self.token_shape = (
            self.num_temporal_token,
            self.image_h,
            self.image_w,
            self.patch_size,
        )
        if token_pe_type == "learned":
            self.query_embed_token = nn.Embedding(
                self.num_temporal_token * self.num_pred_token_per_timestep, hidden_dim
            )  # for decoder as PE TODO replace with temporal spatial PE
        print(
            "predict token shape",  # B T' N_view D*P*P H' W'
            (
                self.num_pred_token_per_timestep,  # H' W' N_view
                self.num_temporal_token,  # T'
                self.image_h,
                self.image_w,
                self.patch_size,
            ),
        )
        query_embed_token_fixed = get_nd_sincos_pos_embed_from_grid(
            hidden_dim,
            (self.num_temporal_token, len(self.future_camera_names), self.image_h, self.image_w),
        )
        self.query_embed_token_fixed = (
            torch.from_numpy(query_embed_token_fixed).view(-1, hidden_dim).float()
        )
        self.token_pe_type = token_pe_type

    def forward(
        self,
        qpos,
        current_image,
        env_state,
        actions,
        is_pad,
        noisy_actions,
        noise_tokens,
        is_tokens_pad,
        denoise_steps,
    ):
        # qpos: batch, 1+ history, qpos_dim
        # current_image: 1. batch, T', num_cam, 3, height, width - image_norm T' = 1+ history
        #                2. batch, T', num_cam*6/16, height', width' - image_token
        # env_state: None
        # actions: batch, seq, action_dim clean action for vae encoder
        # is_pad: batch, seq, for vae encoder
        # noisy_actions: batch, seq, action_dim, noisy action for denoise
        # noise_tokens: batch, seq, num_cam, 6/16, height', width', noise token for denoise
        # is_tokens_pad: batch, seq
        # denoise_steps: int, the step of denoise batch
        is_training = actions is not None  # train or val
        bs = qpos.shape[0]
        is_actions_pad = is_pad

        ### Obtain latent z from action sequence
        mu = logvar = None
        latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
            qpos.device
        )
        latent_input = self.latent_out_proj(
            latent_sample
        )  # TODO maybe add tacile input

        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos)
        # Image observation features and position embeddings
        image = current_image[
            0
        ]  # shape: batch, T', num_cam, 3, height, width image_norm
        image = image.view(
            -1, *image.shape[2:]
        )  #  frame stack shape: batch*T', num_cam, 3, height, width
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](
                image[:, cam_id]
            )  # shape: batch*T', C, H, W
            features = features[0]  # take the last layer feature
            features = features.view(
                bs, -1, *features.shape[-2:]
            )  # shape: batch, T'*C, H, W
            all_cam_features.append(self.input_proj(features))
        src = torch.stack(all_cam_features, axis=-3)  # shape: batch,D,N_view, H, W
        pos = get_nd_sincos_pos_embed_from_grid(
            self.hidden_dim, src.shape[2:]
        )  #  N_view, H, W, D numpy 
        pos = (
            torch.from_numpy(pos).to(src.device).unsqueeze(0).float()
        )  # 1 N_view, H, W, D
        src = rearrange(
            src,
            "b d n_view h w -> b d h (w n_view)",
        )
        pos = rearrange(
            pos,
            "b n_view h w d -> b d h (w n_view)",
        )  # will b d h*w*n_view in the following transformer

        # deal with token patchfy
        noise_tokens = (
            rearrange(
                noise_tokens,
                "b s n d (ph p1) (pw p2) -> b s n (d p1 p2) ph pw",
                p1=self.patch_size,
                p2=self.patch_size,
            )
            if noise_tokens is not None
            else None
        )

        if is_tokens_pad is not None:  # B T -> B T' N_view*H'*W' -> B T'*N_view*H'*W'
            is_tokens_pad = (
                is_tokens_pad.unsqueeze(2)
                .repeat(1, 1, self.num_pred_token_per_timestep)
                .reshape(bs, -1)
            )
        else:
            is_tokens_pad = torch.zeros(
                bs,
                self.num_pred_token_per_timestep * self.num_temporal_token,
                dtype=torch.bool,
            ).to(qpos.device)
            is_pad = torch.zeros(bs, self.num_queries, dtype=torch.bool).to(qpos.device)

        if self.token_pe_type == "learned":
            query_embed_token = self.query_embed_token.weight
        else:
            query_embed_token = self.query_embed_token_fixed.to(qpos.device)
        # print('detr query_embed_token', query_embed_token.shape)
        hs_action, hs_token = self.transformer(
            src,  # obeserved image token
            None,
            self.query_embed.weight,  # for action token pe
            query_embed_token,  # for future token pe
            pos,  # obeserved image token pe
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight,  # latent & proprio token pe
            noisy_actions,
            noise_tokens,
            denoise_steps,
            self.denoise_step_pos_embed.weight,  # denoise step token pe
            is_actions_pad,
            is_tokens_pad,
        )

        a_hat = self.action_head(hs_action)
        is_pad_a_hat = self.is_pad_head(hs_action)
        pred_token = (
            self.token_head(hs_token) if hs_token is not None else None
        )  # B T'*N*H'*W' 6*self.patch_size*self.patch_size
        is_pad_token_hat = self.is_pad_head(hs_token) if hs_token is not None else None
        is_pad_hat = (
            torch.cat([is_pad_a_hat, is_pad_token_hat], axis=1)
            if is_pad_token_hat is not None
            else is_pad_a_hat
        )

        pred_token = (
            rearrange(
                pred_token,
                "b (t n hp wp) (c ph pw) -> b t n c (hp ph) (wp pw)",
                t=self.num_temporal_token,
                n=len(self.future_camera_names), 
                hp=self.image_h,
                wp=self.image_w,
                ph=self.patch_size,
                pw=self.patch_size,
            )
            if pred_token is not None
            else None
        )

        return a_hat, is_pad_hat, pred_token, [mu, logvar]


class DETRVAE_Denoise_Token_Prediction_Dual_Visual_Token(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        num_queries,
        camera_names,
        history_step,
        predict_frame,
        image_downsample_rate,
        temporal_downsample_rate,
        disable_vae_latent,
        disable_resnet,
        patch_size=5,
        token_pe_type="learned",
        image_height=480,
        image_width=640,
        tokenizer_model_temporal_rate=8,
        tokenizer_model_spatial_rate=16,
    ):
        """Initializes the model.
        both resnet feature and token feature are used for token prediction & action prediction
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
        self.transformer_share_decoder = transformer.share_decoder
        self.predict_only_last = transformer.predict_only_last
        self.encoder = encoder
        self.disable_vae_latent = disable_vae_latent
        self.disable_resnet = disable_resnet
        hidden_dim = transformer.d_model  # 512
        self.hidden_dim = hidden_dim
        self.action_head = nn.Linear(hidden_dim, state_dim)  # TODO replace MLP
        if self.transformer_share_decoder == False:
            self.token_head = nn.Linear(
                hidden_dim, 6 * patch_size * patch_size
            )  # HardCode TODO replace MLP
        else:  # default share decoder
            self.token_head = nn.Sequential(
                nn.Linear(
                    hidden_dim, hidden_dim
                ),  # Hardcode patch size * path size * patch dim
                nn.SiLU(),
                nn.Linear(hidden_dim, 6 * patch_size * patch_size),
            )

        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # for decoder as PE
        self.diffusion_timestep_type = transformer.diffusion_timestep_type
        if backbones is not None:  # resnet encoder to extract visual feature
            self.backbones = nn.ModuleList(backbones)
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.input_proj_token = nn.Conv2d(
                6, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False
            )  # Hardcode deal with image token patch size = 5
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)  # proprioception
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            14, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + history_step + num_queries, hidden_dim),
        )  # [CLS], qpos, a_seq

        # decoder extra parameters vae latent to decoder token space
        self.history_step = history_step
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2 + history_step, hidden_dim
        )  # learned position embedding for proprio and latent and denpoe
        self.denoise_step_pos_embed = nn.Embedding(
            1, hidden_dim
        )  # learned position embedding for denoise step
        # setting for token prediction hyper
        if self.predict_only_last:
            self.num_temporal_token = 1
        else:
            self.num_temporal_token = math.ceil(
                predict_frame
                // temporal_downsample_rate
                / tokenizer_model_temporal_rate
            )  # align with tokenizer output
        self.patch_size = patch_size
        self.image_h = (
            image_height
            // image_downsample_rate
            // tokenizer_model_spatial_rate
            // self.patch_size
        )  #
        self.image_w = (
            image_width
            // image_downsample_rate
            // tokenizer_model_spatial_rate
            // self.patch_size
        )  #
        self.num_pred_token_per_timestep = (
            self.image_h * self.image_w * len(camera_names)
        )
        # setting for token prediction position embedding
        if token_pe_type == "learned":
            self.current_embed_token = nn.Embedding(
                1 * self.num_pred_token_per_timestep, hidden_dim
            )  # for current image token encoder
            self.query_embed_token = nn.Embedding(
                self.num_temporal_token * self.num_pred_token_per_timestep, hidden_dim
            )  # for decoder as PE TODO replace with temporal spatial PE

        spatial_temporal_pe = get_spatial_temporal_positional_encoding(
            self.image_h, self.image_w, self.num_temporal_token + 1, hidden_dim
        )
        self.current_embed_token_fixed = spatial_temporal_pe[0].view(
            -1, hidden_dim
        )  # for current image token encoder
        self.query_embed_token_fixed = spatial_temporal_pe[1:].view(
            -1, hidden_dim
        )  # for decoder as PE  # (seq_len, height, width, dim) -> (seq_len*height*width, dim)

        self.token_pe_type = token_pe_type
        print(
            "predict token shape",
            (
                self.num_pred_token_per_timestep,
                self.num_temporal_token,
                self.image_h,
                self.image_w,
                self.patch_size,
            ),
        )

    def forward(
        self,
        qpos,
        current_image,
        env_state,
        actions,
        is_pad,
        noisy_actions,
        noise_tokens,
        is_tokens_pad,
        denoise_steps,
    ):
        # qpos: batch, 1+ history, qpos_dim
        # current_image: 1. batch, T', num_cam, 3, height, width - image_norm T' = 1+ history defualt = 1
        #                2. batch, T', num_cam, 6, height', width' - image_token  (current_image_norm, current_image_tokens)
        # env_state: None
        # actions: batch, seq, action_dim clean action for vae encoder
        # is_pad: batch, seq, for vae encoder
        # noisy_actions: batch, seq, action_dim, noisy action for denoise
        # noise_tokens: batch, seq, num_cam, 6, height', width', noise token for denoise
        # is_tokens_pad: batch, seq
        # denoise_steps: int, the step of denoise
        is_training = actions is not None  # train or val
        bs = qpos.shape[0]
        is_actions_pad = is_pad

        ### Obtain latent z from action sequence
        if is_training and not self.disable_vae_latent:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = (
                torch.unsqueeze(qpos_embed, axis=1)
                if len(qpos_embed.shape) == 2
                else qpos_embed
            )  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2 + self.history_step), False).to(
                qpos.device
            )  # False: not a padding

            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)  # predict mu and logvar
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        # Image observation features and position embeddings for resnet visual encoder
        all_cam_features = []
        all_cam_pos = []
        if self.backbones is not None:
            image = current_image[0]  # shape: batch, T', num_cam, 3, height, width
            for t in range(self.history_step + 1):  # only current frame
                for cam_id, cam_name in enumerate(self.camera_names):
                    features, pos = self.backbones[0](
                        image[:, t, cam_id]
                    )  # resnet feature and pos
                    features = features[0]
                    pos = pos[0]
                    all_cam_features.append(self.input_proj(features))
                    all_cam_pos.append(pos)  # resnet feature and pos
            src = torch.cat(
                all_cam_features, axis=3
            )  # B C H W*num_view*T  # fold camera dimension into width dimension
            pos = torch.cat(all_cam_pos, axis=3)  # B C H W*num_view*T

        # Image observation features and position embeddings for visual tokenizer encoder, only 1 frame
        all_token_features = []
        all_token_pos = []
        current_visual_token = current_image[
            1
        ]  # shape: batch, T', num_cam, 6,  height', width' Depth - image_token
        for cam_id, cam_name in enumerate(self.camera_names):
            token_features = current_visual_token[:, -1, cam_id]  # B C H W
            token_features = self.input_proj_token(token_features)  # B C H W
            token_pos = self.current_embed_token_fixed.to(
                qpos.device
            )  # height*width, C
            all_token_features.append(token_features)
            all_token_pos.append(token_pos)
        addition_visual_token = torch.cat(
            all_token_features, axis=3
        )  # B C H W*num_view
        addition_visual_token_pos = torch.cat(all_token_pos, axis=0)  # H*W*num_view C

        proprio_input = self.input_proj_robot_state(qpos)

        # deal with token patchfy
        noise_tokens = rearrange(
            noise_tokens,
            "b s n d (ph p1) (pw p2) -> b s n (d p1 p2) ph pw",
            p1=self.patch_size,
            p2=self.patch_size,
        )  # b seq_len num_cam 6*patch_size*atch_size height width

        if (
            is_tokens_pad is not None
        ):  # B seq_len -> B seq_len*self.num_pred_token_per_timestep useless for decoder
            is_tokens_pad = (
                is_tokens_pad.unsqueeze(2)
                .repeat(1, 1, self.num_pred_token_per_timestep)
                .reshape(bs, -1)
            )
        else:
            is_tokens_pad = torch.zeros(
                bs,
                self.num_pred_token_per_timestep * self.num_temporal_token,
                dtype=torch.bool,
            ).to(qpos.device)

        if self.token_pe_type == "learned":
            query_embed_token = self.query_embed_token.weight
        else:
            query_embed_token = self.query_embed_token_fixed.to(
                qpos.device
            )  # consider the visual token from current frame
        hs_action, hs_token = self.transformer(
            src,
            None,
            self.query_embed.weight,
            query_embed_token,
            pos,
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight,
            noisy_actions,
            noise_tokens,
            denoise_steps,
            self.denoise_step_pos_embed.weight,
            is_actions_pad,
            is_tokens_pad,
            addition_visual_token,
            addition_visual_token_pos,
        )

        a_hat = self.action_head(hs_action)
        is_pad_a_hat = self.is_pad_head(hs_action)
        pred_token = self.token_head(
            hs_token
        )  # B T'*N*H'*W' 6*self.patch_size*self.patch_size
        is_pad_token_hat = self.is_pad_head(hs_token)
        is_pad_hat = torch.cat([is_pad_a_hat, is_pad_token_hat], axis=1)

        pred_token = rearrange(
            pred_token,
            "b (t n hp wp) (c ph pw) -> b t n c (hp ph) (wp pw)",
            t=self.num_temporal_token,
            n=len(self.camera_names),
            hp=self.image_h,
            wp=self.image_w,
            ph=self.patch_size,
            pw=self.patch_size,
        )

        ## if no visal observation,
        # qpos = self.input_proj_robot_state(qpos)
        # env_state = self.input_proj_env_state(env_state)
        # transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
        # hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        return a_hat, is_pad_hat, pred_token, [mu, logvar]


class DETRVAE_Denoise_Pixel_Prediction(nn.Module):
    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        num_queries,
        camera_names,
        history_step,
        predict_frame,
        image_downsample_rate,
        temporal_downsample_rate,
        disable_vae_latent,
        disable_resnet,
        patch_size=5,
        token_pe_type="learned",
        image_height=480,
        image_width=640,
        resize_rate=8,
    ):
        """Initializes the model.
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
        self.transformer_share_decoder = transformer.share_decoder
        self.predict_only_last = transformer.predict_only_last
        self.encoder = encoder
        self.disable_vae_latent = disable_vae_latent
        self.disable_resnet = disable_resnet
        hidden_dim = transformer.d_model  # 512
        self.hidden_dim = hidden_dim
        self.action_head = nn.Linear(hidden_dim, state_dim)  # TODO replace MLP
        if self.transformer_share_decoder == False:
            self.pixel_head = nn.Linear(
                hidden_dim, 3 * patch_size * patch_size
            )  # HardCode TODO replace MLP
        else:
            self.pixel_head = nn.Sequential(
                nn.Linear(
                    hidden_dim, hidden_dim
                ),  # Hardcode patch size * path size * patch dim
                nn.SiLU(),
                nn.Linear(hidden_dim, 3 * patch_size * patch_size),
            )

        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)  # for decoder as PE
        self.diffusion_timestep_type = transformer.diffusion_timestep_type
        if backbones is not None and disable_resnet == False:
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)  # proprioception
        elif backbones is not None and disable_resnet == True:
            self.pos_encoder = backbones[0][1]  # for 2D PositionEmbedding
            self.backbones = None  # HardCDcode
            # Hardcode divide the latent feature representation into non-overlapping patches
            self.input_proj_token = nn.Conv2d(
                6, hidden_dim, kernel_size=5, stride=5, bias=False
            )  # Hardcode deal with image token pa
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)  # proprioception
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            14, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + history_step + num_queries, hidden_dim),
        )  # [CLS], qpos, a_seq

        # decoder extra parameters vae latent to decoder token space
        self.history_step = history_step
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2 + history_step, hidden_dim
        )  # learned position embedding for proprio and latent and denpoe
        self.denoise_step_pos_embed = nn.Embedding(
            1, hidden_dim
        )  # learned position embedding for denoise step
        # setting for token prediction
        # self.num_temporal_token = predict_frame // temporal_downsample_rate // tokenizer_model_temporal_rate
        if self.predict_only_last:
            self.num_frames = 1
        else:
            self.num_frames = math.ceil(predict_frame // temporal_downsample_rate)
        self.patch_size = patch_size
        self.image_h = (
            image_height // image_downsample_rate // resize_rate // self.patch_size
        )  #
        self.image_w = (
            image_width // image_downsample_rate // resize_rate // self.patch_size
        )  #
        self.num_pred_pixel_per_timestep = (
            self.image_h * self.image_w * len(camera_names)
        )

        if token_pe_type == "learned":
            self.query_embed_token = nn.Embedding(
                self.num_frames * self.num_pred_pixel_per_timestep, hidden_dim
            )  # for decoder as PE TODO replace with temporal spatial PE
        self.query_embed_token_fixed = get_spatial_temporal_positional_encoding(
            self.image_h, self.image_w, self.num_frames, hidden_dim
        ).view(
            -1, hidden_dim
        )  # for decoder as PE  # (seq_len, height, width, dim) -> (seq_len*height*width, dim)
        self.token_pe_type = token_pe_type
        print(
            "predict pixel shape",
            (
                self.num_frames,
                self.num_pred_pixel_per_timestep,
                self.image_h,
                self.image_w,
                self.patch_size,
            ),
        )

    def forward(
        self,
        qpos,
        current_image,
        env_state,
        actions,
        is_pad,
        noisy_actions,
        noise_tokens,
        is_tokens_pad,
        denoise_steps,
    ):
        # qpos: batch, 1+ history, qpos_dim
        # current_image: 1. batch, T', num_cam, 3, height, width - image_norm T' = 1+ history
        #                2. batch, T', num_cam*6, height', width' - image_token
        # env_state: None
        # actions: batch, seq, action_dim clean action for vae encoder
        # is_pad: batch, seq, for vae encoder
        # noisy_actions: batch, seq, action_dim, noisy action for denoise
        # noise_tokens: batch, seq, num_cam, 6, height', width', noise token for denoise
        # is_tokens_pad: batch, seq
        # denoise_steps: int, the step of denoise

        # print('qpos',qpos.shape)
        # print('current_image',current_image[0].shape)
        # print('actions',actions.shape)
        # print('is_pad',is_pad.shape)
        # print('noisy_actions',noisy_actions.shape)
        # print('noise_tokens',noise_tokens.shape)
        # print('is_tokens_pad',is_tokens_pad.shape)
        # print('denoise_steps',denoise_steps)

        is_training = actions is not None  # train or val
        bs = qpos.shape[0]
        is_actions_pad = is_pad

        ### Obtain latent z from action sequence
        if is_training and not self.disable_vae_latent:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = (
                torch.unsqueeze(qpos_embed, axis=1)
                if len(qpos_embed.shape) == 2
                else qpos_embed
            )  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2 + self.history_step), False).to(
                qpos.device
            )  # False: not a padding

            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)  # predict mu and logvar
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        # Image observation features and position embeddings
        all_cam_features = []
        all_cam_pos = []
        if self.backbones is not None and self.disable_resnet == False:
            image = current_image[0]  # shape: batch, T', num_cam, 3, height, width
            for t in range(self.history_step + 1):
                for cam_id, cam_name in enumerate(self.camera_names):
                    features, pos = self.backbones[0](image[:, t, cam_id])
                    features = features[0]
                    pos = pos[0]
                    all_cam_features.append(self.input_proj(features))
                    all_cam_pos.append(pos)
        else:
            image = current_image[
                1
            ]  # [:,:self.num_temporal_token] # shape: batch, T', num_cam, 6, height', width' hardcode keep consistent with token prediction
            for t in range(self.history_step + 1):
                for cam_id, cam_name in enumerate(self.camera_names):
                    features = image[:, t, cam_id]  # B C H W
                    features = self.input_proj(features)
                    pos = self.pos_encoder(features).to(features.dtype)
                    all_cam_features.append(features)
                    all_cam_pos.append(pos)
        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos)
        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, axis=3)  # B C H W*num_view*T
        pos = torch.cat(all_cam_pos, axis=3)  # B C H W*num_view*T
        # deal with token patchfy
        noise_tokens = rearrange(
            noise_tokens,
            "b s n d (ph p1) (pw p2) -> b s n (d p1 p2) ph pw",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        # print('noise_tokens after rearrange',noise_tokens.shape)
        if is_tokens_pad is not None:
            is_tokens_pad = (
                is_tokens_pad.unsqueeze(2)
                .repeat(1, 1, self.num_pred_pixel_per_timestep)
                .reshape(bs, -1)
            )
        else:
            is_tokens_pad = torch.zeros(
                bs, self.num_frames * self.num_pred_pixel_per_timestep, dtype=torch.bool
            ).to(qpos.device)
            is_pad = torch.zeros(bs, self.num_queries, dtype=torch.bool).to(qpos.device)

        if self.token_pe_type == "learned":
            query_embed_token = self.query_embed_token.weight
        else:
            query_embed_token = self.query_embed_token_fixed.to(qpos.device)

        hs_action, hs_token = self.transformer(
            src,
            None,
            self.query_embed.weight,
            query_embed_token,
            pos,
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight,
            noisy_actions,
            noise_tokens,
            denoise_steps,
            self.denoise_step_pos_embed.weight,
            is_actions_pad,
            is_tokens_pad,
        )

        a_hat = self.action_head(hs_action)
        is_pad_a_hat = self.is_pad_head(hs_action)
        pred_images = self.pixel_head(
            hs_token
        )  # B T'*N*H'*W' 6*self.patch_size*self.patch_size
        is_pad_image_hat = self.is_pad_head(hs_token)
        is_pad_hat = torch.cat([is_pad_a_hat, is_pad_image_hat], axis=1)

        pred_images = rearrange(
            pred_images,
            "b (t n hp wp) (c ph pw) -> b t n c (hp ph) (wp pw)",
            t=self.num_frames,
            n=len(self.camera_names),
            hp=self.image_h,
            wp=self.image_w,
            ph=self.patch_size,
            pw=self.patch_size,
        )

        return a_hat, is_pad_hat, pred_images, [mu, logvar]


class DETRVAEDino(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(
        self, backbones, transformer, encoder, state_dim, num_queries, camera_names
    ):
        """Initializes the model.
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
        # self.cls_token_head = nn.Linear(hidden_dim, 384)
        self.num_cls_tokens = 50
        self.query_embed = nn.Embedding(num_queries + self.num_cls_tokens, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            14, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for proprio and latent

        # settings for cls token prediction
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # self.n_patch = (self.image_size//self.patch_size)**2
        # self.k = 1 # number of next frames
        # self.n_patch = (self.img_h//self.patch_size)*(self.img_w//self.patch_size)*(self.k)
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.n_patch, hidden_dim), requires_grad=False)  # (1, n_patch, h)
        # self.patch_embed = nn.Embedding(self.n_patch, hidden_dim)
        # self.decoder_embed = nn.Linear(hidden_dim, hidden_dim, bias=True)

        decoder_depth = 2  # hardcode
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    hidden_dim,
                    16,
                    4,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_pred = nn.Linear(hidden_dim, 384, bias=True)  # decoder to patch

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.image_size//self.patch_size), cls_token=False)
        # decoder_pos_embed = get_2d_sincos_pos_embed_v2(self.decoder_pos_embed.shape[-1], (self.img_h//self.patch_size, self.img_w//self.patch_size))
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0).repeat(1,self.k,1))

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                qpos.device
            )  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
                features = features[0]  # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
            )[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self.transformer(
                transformer_input, None, self.query_embed.weight, self.pos.weight
            )[0]
        a_hat = self.action_head(hs[:, : self.num_queries])
        is_pad_hat = self.is_pad_head(hs[:, : self.num_queries])

        for blk in self.decoder_blocks:
            cls_token = blk(hs[:, self.num_queries :])
        cls_token = self.decoder_norm(cls_token)
        cls_token_hat = self.decoder_pred(cls_token)
        return a_hat, is_pad_hat, [mu, logvar], cls_token_hat


class DETRVAEjpeg(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(
        self, backbones, transformer, encoder, state_dim, num_queries, camera_names
    ):
        """Initializes the model.
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
        self.num_jpeg = (
            20 * 4
        )  # 80 tokens for the 20 next frame jpegs, 4 token will represent 1 jpeg
        self.query_embed = nn.Embedding(num_queries + self.num_jpeg, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            14, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for proprio and latent

        # settings for cls token prediction
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # self.n_patch = (self.image_size//self.patch_size)**2
        # self.k = 1 # number of next frames
        # self.n_patch = (self.img_h//self.patch_size)*(self.img_w//self.patch_size)*(self.k)
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.n_patch, hidden_dim), requires_grad=False)  # (1, n_patch, h)
        # self.patch_embed = nn.Embedding(self.n_patch, hidden_dim)
        # self.decoder_embed = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.jpeg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 400)
        )

        # self.jpeg_head = nn.Linear(hidden_dim, 400)

        # decoder_depth = 2 # hardcode
        # self.decoder_blocks = nn.ModuleList([
        #     Block(hidden_dim, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
        #     for i in range(decoder_depth)])

        # self.decoder_norm = nn.LayerNorm(hidden_dim)
        # self.decoder_pred = nn.Linear(hidden_dim, 400, bias=True) # decoder to patch

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.image_size//self.patch_size), cls_token=False)
        # decoder_pos_embed = get_2d_sincos_pos_embed_v2(self.decoder_pos_embed.shape[-1], (self.img_h//self.patch_size, self.img_w//self.patch_size))
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0).repeat(1,self.k,1))

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                qpos.device
            )  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
                features = features[0]  # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
            )[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self.transformer(
                transformer_input, None, self.query_embed.weight, self.pos.weight
            )[0]
        a_hat = self.action_head(hs[:, : self.num_queries])
        is_pad_hat = self.is_pad_head(hs[:, : self.num_queries])

        # for blk in self.decoder_blocks:
        #     jpeg_token = blk(hs[:,self.num_queries:])
        # jpeg_token = self.decoder_norm(jpeg_token)
        # jpeg_token_hat = self.decoder_pred(jpeg_token)
        jpeg_token_hat = self.jpeg_head(hs[:, self.num_queries :])

        return a_hat, is_pad_hat, [mu, logvar], jpeg_token_hat


class DETRVAEjpeg_diffusion(nn.Module):
    # add timestep
    """This is the DETR module that performs object detection"""

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        num_queries,
        camera_names,
        disable_vae_latent=False,
        predict_frame=20,
        jpeg_token_num=4,
        jpeg_dim=400,
    ):
        """Initializes the model.
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
        self.disable_vae_latent = disable_vae_latent
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.predict_frame = predict_frame
        self.jpeg_token_num = jpeg_token_num
        self.jpeg_dim = jpeg_dim
        self.num_jpeg = (
            predict_frame * jpeg_token_num
        )  # 80 tokens for the 20 next frame jpegs, 4 token will represent 1 jpeg  TODO tune
        self.query_embed = nn.Embedding(num_queries + self.num_jpeg, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            14, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for proprio and latent

        self.jpeg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, jpeg_dim),
        )

    def forward(
        self,
        qpos,
        image,
        env_state,
        actions=None,
        is_pad=None,
        jpegs=None,
        is_jpeg_pad=None,
        denoise_steps=None,
        is_training=True,
    ):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        # is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training and self.disable_vae_latent == False:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                qpos.device
            )  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
                features = features[0]  # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            # need timestep?
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
                actions,
                jpegs,
                denoise_steps,
            )[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self.transformer(
                transformer_input, None, self.query_embed.weight, self.pos.weight
            )[0]
        a_hat = self.action_head(hs[:, : self.num_queries])
        # is_pad_hat = self.is_pad_head(hs[:, :self.num_queries])
        is_pad_hat = self.is_pad_head(hs)
        # for blk in self.decoder_blocks:
        #     jpeg_token = blk(hs[:,self.num_queries:])
        # jpeg_token = self.decoder_norm(jpeg_token)
        # jpeg_token_hat = self.decoder_pred(jpeg_token)
        jpeg_token_hat = self.jpeg_head(hs[:, self.num_queries :])

        return a_hat, jpeg_token_hat, is_pad_hat, [mu, logvar]


class DETRVAEjpeg_diffusion_seperate(nn.Module):
    # add timestep
    """This is the DETR module that performs object detection"""

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        num_queries,
        camera_names,
        disable_vae_latent=False,
        predict_frame=20,
        jpeg_token_num=4,
        jpeg_dim=400,
    ):
        """Initializes the model.
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
        self.disable_vae_latent = disable_vae_latent
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.predict_frame = predict_frame
        self.jpeg_token_num = jpeg_token_num
        self.jpeg_dim = jpeg_dim
        self.num_jpeg = (
            predict_frame * jpeg_token_num
        )  # 80 tokens for the 20 next frame jpegs, 4 token will represent 1 jpeg  TODO tune
        self.query_embed_action = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_jpeg = nn.Embedding(self.num_jpeg, hidden_dim)
        # self.query_embed = nn.Embedding(num_queries+self.num_jpeg, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            14, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for proprio and latent

        self.jpeg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, jpeg_dim),  # todo
        )

    def forward(
        self,
        qpos,
        image,
        env_state,
        actions=None,
        is_pad=None,
        denoise_steps=None,
        is_training=True,
    ):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        # is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training and self.disable_vae_latent == False:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                qpos.device
            )  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED
                features = features[0]  # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            # need timestep?
            pos = torch.cat(all_cam_pos, axis=3)
            # hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, actions, jpegs, denoise_steps,self.predict_frame, self.jpeg_token_num)[0]
            hs_action, hs_jpeg = self.transformer(
                src,
                None,
                self.query_embed_action.weight,
                self.query_embed_jpeg.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
                actions,
                denoise_steps,
            )
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
        # print(hs_action.shape,hs_jpeg.shape)
        a_hat = self.action_head(hs_action)
        is_pad_action_hat = self.is_pad_head(hs_action)
        is_pad_jpeg_hat = self.is_pad_head(hs_jpeg)
        # print(a_hat.shape)
        # print(is_pad_action_hat.shape,is_pad_jpeg_hat.shape)
        is_pad_hat = torch.cat([is_pad_action_hat, is_pad_jpeg_hat], axis=1)
        jpeg_token_hat = self.jpeg_head(hs_jpeg)

        return a_hat, jpeg_token_hat, is_pad_hat, [mu, logvar]


class DETRVAE_nf_diffusion(nn.Module):
    """This is the DETR module that performs object detection"""

    def __init__(
        self,
        backbones,
        transformer,
        encoder,
        state_dim,
        num_queries,
        camera_names,
        predict_frame,
        disable_vae_latent=False,
    ):
        """Initializes the model.
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
            self.input_proj = nn.Conv2d(
                backbones[0].num_channels, hidden_dim, kernel_size=1
            )
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            14, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim)
        )  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for proprio and latent

        # settings for next frame prediction
        self.patch_size = 16
        self.img_h, self.img_w = 224, 224
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # self.n_patch = (self.image_size//self.patch_size)**2
        self.k = predict_frame  # number of next frames
        self.n_patch = (
            (self.img_h // self.patch_size) * (self.img_w // self.patch_size) * (self.k)
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patch, hidden_dim), requires_grad=False
        )  # (1, n_patch, h)
        self.patch_embed = nn.Embedding(self.n_patch, hidden_dim)
        self.decoder_embed = nn.Linear(hidden_dim, hidden_dim, bias=True)

        decoder_depth = 2  # hardcode
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    hidden_dim,
                    16,
                    4,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_pred = nn.Linear(
            hidden_dim, self.patch_size**2 * 3, bias=True
        )  # decoder to patch

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.image_size//self.patch_size), cls_token=False)
        decoder_pos_embed = get_2d_sincos_pos_embed_v2(
            self.decoder_pos_embed.shape[-1],
            (self.img_h // self.patch_size, self.img_w // self.patch_size),
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed)
            .float()
            .unsqueeze(0)
            .repeat(1, self.k, 1)
        )
        self.disable_vae_latent = disable_vae_latent
        # fwd_params = sum(p.numel() for p in self.decoder_blocks.parameters() if p.requires_grad)

    def forward(
        self,
        qpos,
        image,
        env_state,
        actions=None,
        noisy_actions=None,
        is_pad=None,
        denoise_steps=None,
    ):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        # print('is_training',is_training)
        bs, _ = qpos.shape
        # ### Obtain latent z from action sequence
        # print('detr image shape',image.shape)
        if is_training and not self.disable_vae_latent:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
                bs, 1, 1
            )  # (bs, 1, hidden_dim)
            encoder_input = torch.cat(
                [cls_embed, qpos_embed, action_embed], axis=1
            )  # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(
                qpos.device
            )  # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(
                encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
            )
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, : self.latent_dim]
            logvar = latent_info[:, self.latent_dim :]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
                qpos.device
            )
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            if is_training:
                next_frame_images = image[:, 1:]  # should resize?
                image = image[:, :1]
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id])  # HARDCODED?
                features = features[0]  # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            # query_embed = torch.cat([self.query_embed.weight, self.patch_embed.weight], axis=0)
            # should change
            # print('src',src.shape)
            hs_action, hs_patch = self.transformer(
                src,
                None,
                self.query_embed.weight,
                self.patch_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
                noisy_actions,
                denoise_steps,
            )
            # hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self.transformer(
                transformer_input, None, self.query_embed.weight, self.pos.weight
            )[0]

        a_hat = self.action_head(hs_action)
        is_pad_hat = self.is_pad_head(hs_action)

        # print('hs',hs_action.shape)
        # print('a_hat',a_hat.shape)
        # next frame prediction
        mask_token = self.mask_token
        mask_tokens = mask_token.repeat(bs, self.n_patch, 1)
        mask_tokens = mask_tokens + self.decoder_pos_embed.repeat(bs, 1, 1)

        obs_pred = self.decoder_embed(hs_patch)
        obs_pred_ = torch.cat([obs_pred, mask_tokens], dim=1)
        # print('obs_pred_',obs_pred_.shape)
        for blk in self.decoder_blocks:
            obs_pred_ = blk(obs_pred_)
        obs_pred_ = self.decoder_norm(obs_pred_)
        obs_preds = self.decoder_pred(obs_pred_)
        # print('obs_preds',obs_preds.shape)
        # print(self.n_patch)
        obs_preds = obs_preds[:, self.n_patch :]
        # print('obs_preds',obs_preds.shape)
        if is_training:
            # next_frame_images = image[:,1:]
            # print('next_frame_images',next_frame_images.shape)
            next_frame_images = nn.functional.interpolate(
                next_frame_images.reshape(bs, -1, *next_frame_images.shape[-2:]),
                size=(self.img_h, self.img_w),
            )
            # print('next_frame_images',next_frame_images.shape)
            p = self.patch_size
            h_p = self.img_h // p
            w_p = self.img_w // p
            obs_targets = next_frame_images.reshape(
                shape=(bs, self.k, 3, h_p, p, w_p, p)
            )
            obs_targets = obs_targets.permute(0, 1, 3, 5, 4, 6, 2)
            obs_targets = obs_targets.reshape(
                shape=(bs, h_p * w_p * self.k, (p**2) * 3)
            )
        else:
            obs_targets = torch.zeros_like(obs_preds)

        return a_hat, is_pad_hat, [mu, logvar], [obs_preds, obs_targets]


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """Initializes the model.
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
        self.action_head = nn.Linear(1000, state_dim)  # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5),
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(
                input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2
            )
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0]  # take the last layer feature
            pos = pos[0]  # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1)  # 768 each
        features = torch.cat([flattened_features, qpos], axis=1)  # qpos: 14
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
    d_model = args.hidden_dim  # 256
    dropout = args.dropout  # 0.1
    nhead = args.nheads  # 8
    dim_feedforward = args.dim_feedforward  # 2048
    num_encoder_layers = args.enc_layers  # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm  # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(
        d_model, nhead, dim_feedforward, dropout, activation, normalize_before
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_diffusion(args):
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)  # fixed
    transformer = build_transformer_denoise(
        args
    )  # decoder input noisy input & PE & time_embedding
    encoder = build_encoder(args)
    model = DETRVAE_Denoise(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,  # add additional denoise step
        disable_vae_latent=args.disable_vae_latent,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_diffusion_tp(args):  # token prediction
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for camera_id in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)
    transformer = build_transformer_diffusion_prediction(args)
    encoder = build_encoder(args)

    model = DETRVAE_Denoise_Token_Prediction(  # TODO design for token pred
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,  # add additional denoise step
        history_step=args.history_step,
        predict_frame=args.predict_frame,
        image_downsample_rate=args.image_downsample_rate,
        temporal_downsample_rate=args.temporal_downsample_rate,
        disable_vae_latent=args.disable_vae_latent,
        disable_resnet=args.disable_resnet,
        patch_size=args.patch_size,
        token_pe_type=args.token_pe_type,
        image_width=args.image_width,
        image_height=args.image_height,
        tokenizer_model_spatial_rate=args.tokenizer_model_spatial_rate,
        tokenizer_model_temporal_rate=args.tokenizer_model_temporal_rate,
        resize_rate=args.resize_rate,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_diffusion_tp_with_dual_visual_token(args):  # token prediction
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for i in range(len(args.camera_names)):
        backbone = build_backbone(args)
        backbones.append(backbone)  # fixed

    transformer = build_transformer_diffusion_prediction_with_dual_visual_token(
        args
    )  # TODO design for token pred decoder input noisy input & PE & time_embedding
    encoder = build_encoder(args)

    model = DETRVAE_Denoise_Token_Prediction_Dual_Visual_Token(  # TODO design for token pred
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,  # add additional denoise step
        history_step=args.history_step,
        predict_frame=args.predict_frame,
        image_downsample_rate=args.image_downsample_rate,
        temporal_downsample_rate=args.temporal_downsample_rate,
        disable_vae_latent=args.disable_vae_latent,
        disable_resnet=args.disable_resnet,
        patch_size=args.patch_size,
        token_pe_type=args.token_pe_type,
        image_width=args.image_width,
        image_height=args.image_height,
        tokenizer_model_spatial_rate=args.tokenizer_model_spatial_rate,
        tokenizer_model_temporal_rate=args.tokenizer_model_temporal_rate,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_diffusion_pp(args):  # pixel prediction
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    if args.disable_resnet:  # HARDCODE
        for param in backbone.parameters():
            param.requires_grad = False
    backbones.append(backbone)  # fixed
    # refer to Transformer_diffusion build_transformer_diffusion
    #  TODO unified prediction
    transformer = build_transformer_diffusion_pixel_prediction(
        args
    )  # TODO design for token pred decoder input noisy input & PE & time_embedding
    encoder = build_encoder(args)
    # TODO fix
    model = DETRVAE_Denoise_Pixel_Prediction(  # TODO design for token pred
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,  # add additional denoise step
        history_step=args.history_step,
        predict_frame=args.predict_frame,
        image_downsample_rate=args.image_downsample_rate,
        temporal_downsample_rate=args.temporal_downsample_rate,
        disable_vae_latent=args.disable_vae_latent,
        disable_resnet=args.disable_resnet,
        patch_size=args.patch_size,
        token_pe_type=args.token_pe_type,
        image_width=args.image_width,
        image_height=args.image_height,
        resize_rate=args.resize_rate,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


# discard
def build_seg(args):
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    # backbones = []
    # backbone = build_backbone(args)
    # backbones.append(backbone)

    # transformer = build_transformer(args)

    # encoder = build_encoder(args)

    # model = DETRVAESeg(
    #     backbones,
    #     transformer,
    #     encoder,
    #     state_dim=state_dim,
    #     num_queries=args.num_queries,
    #     camera_names=args.camera_names,
    # )

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("number of parameters: %.2fM" % (n_parameters/1e6,))

    # return model


def build_dino(args):
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAEDino(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_jpeg(args):
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAEjpeg(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_jpeg_diffusion(args):
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer_diffusion(args)

    encoder = build_encoder(args)

    model = DETRVAEjpeg_diffusion(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        disable_vae_latent=args.disable_vae_latent,
        predict_frame=args.predict_frame,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_jpeg_diffusion_seperate(args):
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    print("predict_frame", args.predict_frame)
    print("jpeg_token_num", args.jpeg_token_num)
    print("jpeg_dim", args.jpeg_dim)
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer_diffusion_seperate(args)

    encoder = build_encoder(args)

    model = DETRVAEjpeg_diffusion_seperate(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        disable_vae_latent=args.disable_vae_latent,
        predict_frame=args.predict_frame,
        jpeg_token_num=args.jpeg_token_num,
        jpeg_dim=args.jpeg_dim,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_nf_diffusion_seperate(args):
    state_dim = 14  # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    print("predict_frame", args.predict_frame)
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer_diffusion_seperate(args)

    encoder = build_encoder(args)

    model = DETRVAE_nf_diffusion(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        disable_vae_latent=args.disable_vae_latent,
        predict_frame=args.predict_frame,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model


def build_cnnmlp(args):
    state_dim = 14  # TODO hardcode

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
    print("number of parameters: %.2fM" % (n_parameters / 1e6,))

    return model
