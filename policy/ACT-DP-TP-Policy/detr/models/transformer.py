# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import IPython

e = IPython.embed


class Transformer(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        mask,
        query_embed,
        pos_embed,
        latent_input=None,
        proprio_input=None,
        additional_pos_embed=None,
    ):
        # TODO flatten only when input has H and W
        if len(src.shape) == 4:  # has H and W
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            # mask = mask.flatten(1)

            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # seq, bs, dim
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

            addition_input = torch.stack([latent_input, proprio_input], axis=0)
            src = torch.cat([addition_input, src], axis=0)
        else:
            assert len(src.shape) == 3
            # flatten NxHWxC to HWxNxC
            bs, hw, c = src.shape
            src = src.permute(1, 0, 2)
            pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        shape_len = tgt.shape[0]
        tgt_mask = torch.zeros(shape_len, shape_len).to(tgt.device)
        tgt_mask[:100, 100:] = float(
            "-inf"
        )  
        tgt_mask[100:, :100] = float("-inf")  #
        hs = self.decoder(
            tgt,
            memory,
            tgt_mask,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        hs = hs.transpose(1, 2)
        return hs


class Transformer_Denoise(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        causal_mask=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self._reset_parameters()

        self.action_embed = nn.Sequential(
            nn.Linear(14, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.d_model = d_model
        self.nhead = nhead
        self.denoise_step_pos_embed = nn.Embedding(1, d_model)
        self.causal_mask = causal_mask
        print("apply causal_mask:", causal_mask)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight,actions, denoise_steps)

    def forward(
        self,
        src,
        mask,
        query_embed,
        pos_embed,
        latent_input=None,
        proprio_input=None,
        additional_pos_embed=None,
        noisy_actions=None,
        denoise_steps=None,
    ):
        # TODO flatten only when input has H and W
        # encoder don't need change
        # src: image embedding mask: mask
        # query_embed: decoder PE
        # pos_embed: encoder PE
        # latent_input: vae latent or tacile latent
        # proprio_input: proprio
        # additional_pos_embed: proprio + proprio
        # denoise_embed: denoise timestep embedding
        # noisy_actions: noisy actions

        if len(src.shape) == 4:  # has H and W b d h (w n_view)
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)  # h*w, bs, c
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # N bs dim
            # mask = mask.flatten(1)
            # N_add Dim
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # seq, bs, dim
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

            latent_input = latent_input.unsqueeze(1) if len(latent_input.shape) ==2 else latent_input  # B 1 D
            addition_input = torch.cat([latent_input, proprio_input], axis=1).permute(
                1, 0, 2
            )  #  B T+1 D -> T+1 B D
            
            src = torch.cat([addition_input, src], axis=0)
        else:
            assert len(src.shape) == 3
            # flatten NxHWxC to HWxNxC
            bs, hw, c = src.shape
            src = src.permute(1, 0, 2)
            pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = self.action_embed(noisy_actions).permute(
            1, 0, 2
        )  # TODO Change to noise tgt B T D -> T B D
        denoise_embed = get_timestep_embedding(denoise_steps, self.d_model)  # B -> B D
        # print(denoise_embed.shape)
        denoise_embed = self.time_embed(denoise_embed).unsqueeze(0)  # B D -> 1 B D
        memory = self.encoder(
            src, src_key_padding_mask=mask, pos=pos_embed
        )  # cross attention
        denoise_step_pos_embed = self.denoise_step_pos_embed.weight.unsqueeze(1).repeat(
            1, bs, 1
        )  # 1 D -> 1 B D
        memory = torch.cat([memory, denoise_embed], axis=0)
        pos_embed = torch.cat([pos_embed, denoise_step_pos_embed], axis=0)
        seq_len = tgt.shape[0]
        if self.causal_mask:
            tgt_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf")), diagonal=1
            ).to(tgt.device)
        else:
            tgt_mask = torch.zeros(seq_len, seq_len).to(tgt.device)
        hs = self.decoder(
            tgt,
            memory,
            tgt_mask,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )  # TODO
        hs = hs.transpose(1, 2) # 1 T B D -> 1 B T D
        return hs


class Transformer_Denoise_AdLN(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        causal_mask=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerEncoderLayer_AdLN(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerEncoder_AdLN(
                decoder_layer,
            num_decoder_layers,
            decoder_norm,
        )

        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self._reset_parameters()

        self.action_embed = nn.Sequential(
            nn.Linear(14, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.d_model = d_model
        self.nhead = nhead
        self.denoise_step_pos_embed = nn.Embedding(1, d_model)

        self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
        self.norm_after_pool = nn.LayerNorm(d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src, # B D H W*num_view
        mask,  # None
        query_embed, #  H D
        pos_embed, # 1 D H W*num_view
        latent_input=None, # B 1 D
        proprio_input=None, # B 1 D
        additional_pos_embed=None, # 1+1 D
        noisy_actions=None,     # B H D
        denoise_steps=None, # B 
    ):

        if len(src.shape) == 4:  # has H and W b d h (w n_view)
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)  # h*w, bs, c
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # N bs dim
            # N_add Dim
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # seq, bs, dim
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0) # pro. + visual token PE

            latent_input = latent_input.unsqueeze(1) if len(latent_input.shape) ==2 else latent_input  # B 1 D
            addition_input = torch.cat([latent_input, proprio_input], axis=1).permute(
                1, 0, 2
            )  #  B T+1 D -> T+1 B D
            src = torch.cat([addition_input, src], axis=0) 
        else:
            assert len(src.shape) == 3
            # flatten NxHWxC to HWxNxC
            bs, hw, c = src.shape
            src = src.permute(1, 0, 2)
            pos_embed = pos_embed.unsqueeze(1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        memory = self.encoder(
            src, src_key_padding_mask=mask, pos=pos_embed
        )  # cross attention N B D 
        # N B D => B D N => B D 1 => B D
        memory = self.global_1d_pool(memory.permute(1, 2, 0)).squeeze(-1) # B D
        memory = self.norm_after_pool(memory)
        denoise_embed = get_timestep_embedding(denoise_steps, self.d_model)  # B -> B D
        denoise_embed = self.time_embed(denoise_embed)  # B -> B D  
        
        condition = memory + denoise_embed # B D as condition for modulation
        
        
        tgt = self.action_embed(noisy_actions).permute(
            1, 0, 2
        )  
        
        hs = self.decoder(
            tgt,
            condition,  
            pos=query_embed
        ) # Actually need is_pad?
        
        hs = hs.transpose(1, 2) # 1 T B D -> 1 B T D
        return hs
    
class Transformer_Denoise_Tactile(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        causal_mask=False,
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation, normalize_before
            )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder_alter(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self._reset_parameters()

        self.action_embed = nn.Sequential(
            nn.Linear(14, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.d_model = d_model
        self.nhead = nhead
        self.denoise_step_pos_embed = nn.Embedding(1, d_model)
        self.causal_mask = causal_mask
        print("apply causal_mask:", causal_mask)    
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(
        self,
        src,
        mask,
        query_embed,
        pos_embed,
        tactile_input,
        tactile_pos,
        proprio_input=None,
        additional_pos_embed=None,
        noisy_actions=None,
        denoise_steps=None,):
        
        if len(src.shape) == 4:  # has H and W b d h (w n_view)
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)  # h*w, bs, c
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # N bs dim
            
            tactile_input = tactile_input.flatten(2).permute(2, 0, 1)  # h*w*4, bs, c
            tactile_pos = tactile_pos.flatten(2).permute(2, 0, 1).repeat(1, bs, 1) # h*w, bs, c
            # N_add Dim
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # seq, bs, dim
            proprio_input = proprio_input.permute(1, 0, 2) # B 1 D -> 1 B D
            # vision-state input
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)
            src = torch.cat([proprio_input, src], axis=0)
            # tactile-state input
            # print('proprio_input', proprio_input.shape, 'tactile_input', tactile_input.shape)
            src_tactile = torch.cat([proprio_input,tactile_input], axis=0)
            src_tactile_pos = torch.cat([additional_pos_embed,tactile_pos], axis=0)
            
        tgt = self.action_embed(noisy_actions).permute(
            1, 0, 2
        )  # TODO Change to noise tgt B T D -> T B D
        denoise_embed = get_timestep_embedding(denoise_steps, self.d_model)  # B -> B D
        denoise_embed = self.time_embed(denoise_embed).unsqueeze(0)  # B D -> 1 B D
        denoise_step_pos_embed = self.denoise_step_pos_embed.weight.unsqueeze(1).repeat(
            1, bs, 1
        )  # 1 D -> 1 B D
        
        # encoder vision-state information
        memory = self.encoder(
            src, src_key_padding_mask=mask, pos=pos_embed
        )  # cross attention
        memory = torch.cat([memory, denoise_embed], axis=0)
        pos_embed = torch.cat([pos_embed, denoise_step_pos_embed], axis=0)
        seq_len = tgt.shape[0]
        
        # tactile-state information
        memory_tactile = torch.cat([src_tactile, denoise_embed], axis=0)
        tactile_pos_embed = torch.cat([src_tactile_pos, denoise_step_pos_embed], axis=0)
        tgt_mask = torch.zeros(seq_len, seq_len).to(tgt.device)
        hs = self.decoder(
            tgt,
            memory,
            memory_tactile,
            tgt_mask,
            memory_key_padding_mask=mask,
            pos = pos_embed,
            pos_alter = tactile_pos_embed,
            query_pos=query_embed,
        )
        hs = hs.transpose(1, 2)
        return hs
        
        
class Transformer_diffusion_prediction(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        num_queries=100,
        share_decoder=True,
        patch_size=5,
        diffusion_timestep_type="cat",
        attention_type="v1",
        predict_frame=16,
        predict_only_last=False,
        token_dim=6,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        if share_decoder == False:
            self.decoder_token = copy.deepcopy(self.decoder)

        self.share_decoder = share_decoder
        self.diffusion_timestep_type = diffusion_timestep_type
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self._reset_parameters()

        self.action_embed = nn.Sequential(
            nn.Linear(14, d_model),  # TODO action dim
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.token_embed = nn.Sequential(
            nn.Linear(
                token_dim * patch_size * patch_size, d_model
            ),  # Hardcode patch size * path size * patch dim
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.d_model = d_model
        self.nhead = nhead
        self.chunksize = num_queries
        self.attention_type = attention_type
        self.predict_frame = predict_frame
        self.predict_only_last = predict_only_last
        self.token_dim = token_dim

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # hs_action, hs_token = self.transformer(src, None, self.query_embed.weight, self.query_embed_toekn.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, noisy_actions, noise_tokens, denoise_steps, self.denoise_step_pos_embed.weight)
    def forward(
        self,
        src,
        mask,
        query_embed,
        query_embed_token,
        pos_embed,
        latent_input=None,
        proprio_input=None,
        additional_pos_embed=None,
        noisy_actions=None,
        noisy_tokens=None,
        denoise_steps=None,
        denoise_step_pos_embed=None,
        is_pad=None,
        is_pad_token=None,
    ):
        # src: B D H W*num_view
        # mask:None
        # query_embed: chunksize D for action query
        # query_embed_token: Num_tokens D for token query
        # pos_embed: 1 D H W*num_view for current frame token
        # latent_input: B D
        # proprio_input: B T' D
        # additional_pos_embed: B T'+1 D, include proprio and latent
        # noisy_actions: B chunksize D
        # noisy_tokens: B T' N D H' W', T' = predict_frame / temporal_compression_rate
        # denoise_steps: B
        # denoise_step_pos_embed:1 D
        # is_pad: B chunksize
        # is_pad_token: B T' N H' W'
        if len(src.shape) == 4:  # has H and W
            bs, c, h, w = src.shape  # B D H W*num_view*T
            src = src.flatten(2).permute(
                2, 0, 1
            )  # H*W*num_view*T, bs, c deal with visual features from resnet
            pos_embed = (
                pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            )  # H*W*num_view*T, bs, c

            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            query_embed_token = query_embed_token.unsqueeze(1).repeat(
                1, bs, 1
            )  # TODO temporal-spatial position embedding FOR predicted frame token
            # print('transformer query_embed_token', query_embed_token.shape,'query_embed', query_embed.shape)
            query_embed_all = torch.cat(
                [query_embed, query_embed_token], axis=0
            )  # chunksize + num_tokens, bs, c
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # seq, bs, dim
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

            latent_input = latent_input.unsqueeze(1)  # B 1 D
            addition_input = torch.cat([latent_input, proprio_input], axis=1).permute(
                1, 0, 2
            )  #  B T+1 D -> T+1 B D
            src = torch.cat([addition_input, src], axis=0)

            denoise_step_pos_embed = denoise_step_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # 1 D -> 1 B D
        else:
            assert len(src.shape) == 3

        # encoder
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # add denoise timestep embedding for decoder denoise step
        denoise_embed = self.time_embed(
            get_timestep_embedding(denoise_steps, self.d_model)
        ).unsqueeze(
            0
        )  # B D -> 1 B D
        if (
            self.diffusion_timestep_type == "cat"
        ):  # TODO add tokenizer visual token & timestep embedding
            memory = torch.cat([memory, denoise_embed], axis=0)
            pos_embed = torch.cat([pos_embed, denoise_step_pos_embed], axis=0)
        # elif self.diffusion_timestep_type == 'vis_cat':
        #     memory = torch.cat([memory, visual_token, denoise_embed], axis=0) # visual token map
        #     pos_embed = torch.cat([pos_embed, pos_embed_visual_token, denoise_step_pos_embed], axis=0) # pos_embed_visual_token
        else:
            memory = memory + denoise_embed

        tgt_action = self.action_embed(noisy_actions).permute(1, 0, 2)  # H B  D

        # if noisy_tokens is None:

        # tgt_key_padding_mask = torch.zeros_like(tgt_token).sum(-1).bool() # T'*N*H'*W' B

        if self.share_decoder and noisy_tokens is not None:
            noisy_tokens = noisy_tokens.permute(
                0, 1, 2, 4, 5, 3
            )  #  B T' N D H' W' ->  B T' N H' W' D
            tgt_token = (
                self.token_embed(noisy_tokens)
                .reshape(bs, -1, self.d_model)
                .permute(1, 0, 2)
            )  # B T' N H' W' D -> T'*N*H'*W' B D
            # print('transformer',tgt_action.shape,tgt_token.shape )
            tgt = torch.cat([tgt_action, tgt_token], axis=0)  # H1+ H2 B  D
            seq_len = tgt.shape[0]
            bs = tgt.shape[1]
            # tgt_key_padding_mask = torch.cat([is_pad, is_pad_token], axis=1) # B N+M is
            is_pad_token_zero = torch.zeros_like(is_pad_token).bool()  # HARD CODE
            if is_pad is None:
                is_pad = torch.zeros(bs, self.chunksize).bool().to(src.device)
            tgt_key_padding_mask = torch.cat(
                [is_pad, is_pad_token_zero], axis=1
            )  # HARD CODE avoid nan when all token is pad

            seq_len = tgt.shape[0]
            tgt_mask = torch.zeros(seq_len, seq_len).to(
                tgt.device
            )  # chunksize + num_pred_token_per_frame*predict_frame
            # TODO design mask
            if self.attention_type == "v1":
                tgt_mask[0 : self.chunksize, self.chunksize :] = float(
                    "-inf"
                )  # action prediction cannot attend to token prediction
                tgt_mask[self.chunksize :, self.predict_frame : self.chunksize] = float(
                    "-inf"
                )  # token prediction cannot attend to action token after token prediction
            elif self.attention_type == "v2":
                tgt_mask[0 : self.chunksize, self.chunksize :] = float(
                    "-inf"
                )  # action prediction cannot attend to token prediction
                tgt_mask[self.chunksize :, 0 : self.chunksize] = float(
                    "-inf"
                )  # token prediction cannot attend to action prediction
            elif self.attention_type == "v3":  # v1= v3 sad
                tgt_mask[0 : self.chunksize, self.chunksize :] = float(
                    "-inf"
                )  # action prediction cannot attend to token prediction
                if (
                    self.predict_frame < self.chunksize
                ):  # don't need attent action after target frame
                    tgt_mask[self.chunksize :, self.predict_frame : self.chunksize] = (
                        float("-inf")
                    )
            elif self.attention_type == "causal":
                predict_frame = noisy_tokens.shape[1]
                num_pred_token_per_frame = tgt_token.shape[0] // predict_frame
                chunk_size = query_embed.shape[0]
                temporal_compression_rate = chunk_size // predict_frame
                tgt_mask = torch.full((seq_len, seq_len), float("-inf")).to(
                    tgt.device
                )  # seq_len = chunksize + num_pred_token_per_frame*predict_frame
                # tgt_mask[:chunk_size, :chunk_size] = torch.triu(torch.full((chunk_size, seq_len), float('-inf')), diagonal=1).to(tgt.device)
                tgt_mask[:chunk_size, :chunk_size] = torch.zeros(
                    (chunk_size, chunk_size)
                ).to(tgt.device)
                for t in range(predict_frame):
                    tgt_mask[
                        chunk_size
                        + t * num_pred_token_per_frame : chunk_size
                        + (t + 1) * num_pred_token_per_frame,
                        0 : t * temporal_compression_rate,
                    ] = 0
                    tgt_mask[
                        chunk_size
                        + t * num_pred_token_per_frame : chunk_size
                        + (t + 1) * num_pred_token_per_frame,
                        chunk_size : chunk_size + (t + 1) * num_pred_token_per_frame,
                    ] = 0
            # print(tgt.shape, memory.shape, tgt_key_padding_mask.shape, pos_embed.shape, query_embed_all.shape)
            hs = self.decoder(
                tgt,
                memory,
                tgt_mask,
                memory_key_padding_mask=mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                pos=pos_embed,
                query_pos=query_embed_all,
            ).transpose(1, 2)[
                0
            ]  # TODO 1 H B D
            hs_action = hs[:, : self.chunksize]  # B H D
            hs_token = hs[:, self.chunksize :]  # B H D -> None
            # simplify version
            # tgt_mask = torch.full((seq_len, seq_len), float('-inf')).to(tgt.device)
            # # Action-to-Action mask
            # tgt_mask[:chunk_size, :chunk_size] = torch.triu(torch.full((chunk_size, chunk_size), float('-inf')), diagonal=1)

            # # Frame-to-All mask
            # frame_indices = torch.arange(chunk_size, seq_len).view(predict_frame, num_pred_token_per_frame).to(tgt.device)
            # action_indices = torch.arange(chunk_size).to(tgt.device)

            # for t in range(predict_frame):
            #     tgt_mask[frame_indices[t], action_indices[:t * temporal_compression_rate]] = 0
            #     tgt_mask[frame_indices[t], frame_indices[:t + 1].flatten()] = 0

        elif self.share_decoder and noisy_tokens is None:
            hs = self.decoder(
                tgt_action,
                memory,
                memory_key_padding_mask=mask,
                tgt_key_padding_mask=is_pad,
                pos=pos_embed,
                query_pos=query_embed,
            ).transpose(1, 2)[0]
            hs_action = hs
            hs_token = None

        else:
            hs_action = self.decoder(
                tgt_action,
                memory,
                memory_key_padding_mask=mask,
                tgt_key_padding_mask=is_pad,
                pos=pos_embed,
                query_pos=query_embed,
            ).transpose(1, 2)[0]
            hs_token = self.decoder_token(
                tgt_token,
                memory,
                memory_key_padding_mask=mask,  # tgt_key_padding_mask = is_pad_token, # is_pad_token is nonetype ! fix
                pos=pos_embed,
                query_pos=query_embed_token,
            ).transpose(1, 2)[0]
        # hs_action B chunksize D
        # hs_token  B T'*N*H'*W' D
        # print("tgt_token has NaN:", torch.isnan(tgt_token).any())
        # print("memory has NaN:", torch.isnan(memory).any())
        # print('tgt_key_padding_mask has NaN:', torch.isnan(is_pad_token).any())
        # print("pos_embed has NaN:", torch.isnan(pos_embed).any())
        # print("query_embed_token has NaN:", torch.isnan(query_embed_token).any())
        # print("hs_token has NaN:", torch.isnan(hs_token).any()) # has nan
        return hs_action, hs_token


class Transformer_diffusion_prediction_with_dual_visual_token(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        num_queries=100,
        share_decoder=True,
        patch_size=5,
        diffusion_timestep_type="cat",
        attention_type="v1",
        predict_frame=16,
        predict_only_last=False,
        token_dim=6,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        if share_decoder == False:
            self.decoder_token = copy.deepcopy(self.decoder)

        self.share_decoder = share_decoder
        self.diffusion_timestep_type = diffusion_timestep_type
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self._reset_parameters()
        # TODO change action dim
        token_dim = token_dim
        self.action_embed = nn.Sequential(
            nn.Linear(14, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.token_embed = nn.Sequential(
            nn.Linear(
                token_dim * patch_size * patch_size, d_model
            ),  # Hardcode patch size * path size * patch dim
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.d_model = d_model
        self.nhead = nhead
        self.chunksize = num_queries
        self.attention_type = attention_type
        self.predict_frame = predict_frame
        self.predict_only_last = predict_only_last

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # hs_action, hs_token = self.transformer(src, None, self.query_embed.weight, self.query_embed_toekn.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, noisy_actions, noise_tokens, denoise_steps, self.denoise_step_pos_embed.weight)
    def forward(
        self,
        src,
        mask,
        query_embed,
        query_embed_token,
        pos_embed,
        latent_input=None,
        proprio_input=None,
        additional_pos_embed=None,
        noisy_actions=None,
        noisy_tokens=None,
        denoise_steps=None,
        denoise_step_pos_embed=None,
        is_pad=None,
        is_pad_token=None,
        addition_visual_token=None,
        addition_visual_token_pos=None,
    ):
        # src: B D H W*num_view*T
        # mask:None
        # query_embed: chunksize D for action query
        # query_embed_token: Num_tokens D for token query
        # pos_embed: 1 D H W*num_view*T for current frame token
        # latent_input: B D
        # proprio_input: B T' D
        # additional_pos_embed: T'+1 D, include proprio and latent
        # noisy_actions: B chunksize D
        # noisy_tokens: B T' N D H' W', T' = predict_frame / temporal_compression_rate
        # denoise_steps: B
        # denoise_step_pos_embed:1 D
        # is_pad: B chunksize
        # is_pad_token: B T' N H' W'
        # addition_visual_token: B D H' W'*num_view
        # addition_visual_token_pos: H'*W'*num_view D for visual token position embedding

        if len(src.shape) == 4:  # has H and W
            bs, c, h, w = src.shape  # B D H W*num_view*T
            # For encoder
            src = src.flatten(2).permute(
                2, 0, 1
            )  # H*W*num_view*T, bs, c deal with visual features from resnet
            addition_visual_token = addition_visual_token.flatten(2).permute(
                2, 0, 1
            )  # H*W*num_view, bs, c visual token from tokenzier
            latent_input = latent_input.unsqueeze(1)  # B 1 D
            addition_input = torch.cat([latent_input, proprio_input], axis=1).permute(
                1, 0, 2
            )  #  B T+1 D -> T+1 B D

            pos_embed = (
                pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            )  # H*W*num_view*T, bs, c
            addition_visual_token_pos = addition_visual_token_pos.unsqueeze(1).repeat(
                1, bs, 1
            )  # H*W*num_view, bs, c
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # seq, bs, dim

            # only consider the visual token from resnet for encoder, robot state + visual token
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)
            src = torch.cat([addition_input, src], axis=0)

            # For decoder
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            query_embed_token = query_embed_token.unsqueeze(1).repeat(
                1, bs, 1
            )  # TODO temporal-spatial position embedding FOR predicted frame token
            query_embed_all = torch.cat(
                [query_embed, query_embed_token], axis=0
            )  # chunksize + num_tokens, bs, c

            denoise_step_pos_embed = denoise_step_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # 1 D -> 1 B D
        else:
            assert len(src.shape) == 3

        # encoder TODO add visual token from tokenizer here?
        memory = self.encoder(
            src, src_key_padding_mask=mask, pos=pos_embed
        )  # resnet visual feature + proprio + latent

        # add denoise timestep embedding for decoder denoise step or current visual token
        denoise_embed = self.time_embed(
            get_timestep_embedding(denoise_steps, self.d_model)
        ).unsqueeze(
            0
        )  # B D -> 1 B D
        if (
            self.diffusion_timestep_type == "cat"
        ):  # only resnet visual token + proprio + latent + timestep embedding
            memory = torch.cat([memory, denoise_embed], axis=0)
            pos_embed = torch.cat([pos_embed, denoise_step_pos_embed], axis=0)
        elif (
            self.diffusion_timestep_type == "vis_cat"
        ):  # add visual token from tokenizer
            memory = torch.cat(
                [memory, addition_visual_token, denoise_embed], axis=0
            )  # visual token map
            pos_embed = torch.cat(
                [pos_embed, addition_visual_token_pos, denoise_step_pos_embed], axis=0
            )  # pos_embed_visual_token
        elif self.diffusion_timestep_type == "add":
            memory = memory + denoise_embed

        # Noisy action and token
        tgt_action = self.action_embed(noisy_actions).permute(1, 0, 2)  # H B  D
        noisy_tokens = noisy_tokens.permute(
            0, 1, 2, 4, 5, 3
        )  #  B T' N D H' W' ->  B T' N H' W' D
        tgt_token = (
            self.token_embed(noisy_tokens)
            .reshape(bs, -1, self.d_model)
            .permute(1, 0, 2)
        )  # B T' N H' W' D -> T'*N*H'*W' B D

        if self.share_decoder:
            tgt = torch.cat([tgt_action, tgt_token], axis=0)  # H1+ H2 B  D
            seq_len = tgt.shape[0]
            bs = tgt.shape[1]
            is_pad_token_zero = torch.zeros_like(is_pad_token).bool()  # HARD CODE
            if is_pad is None:  # if action is pad
                is_pad = torch.zeros(bs, self.chunksize).bool().to(src.device)
            tgt_key_padding_mask = torch.cat(
                [is_pad, is_pad_token_zero], axis=1
            )  # HARD CODE avoid nan when all token is pad add causal mechanism
            seq_len = tgt.shape[0]
            tgt_mask = torch.zeros(seq_len, seq_len).to(tgt.device)
            # TODO design mask
            if self.attention_type == "v1":
                tgt_mask[0 : self.chunksize, self.chunksize :] = float(
                    "-inf"
                )  # action prediction cannot attend to token prediction
                tgt_mask[self.chunksize :, self.predict_frame : self.chunksize] = float(
                    "-inf"
                )  # token prediction cannot attend to action token after token prediction
            elif self.attention_type == "v2":
                tgt_mask[0 : self.chunksize, self.chunksize :] = float(
                    "-inf"
                )  # action prediction cannot attend to token prediction
                tgt_mask[self.chunksize :, 0 : self.chunksize] = float(
                    "-inf"
                )  # token prediction cannot attend to action prediction
            elif self.attention_type == "v3":
                tgt_mask[0 : self.chunksize, self.chunksize :] = float(
                    "-inf"
                )  # action prediction cannot attend to token prediction

            hs = self.decoder(
                tgt,
                memory,
                tgt_mask,
                memory_key_padding_mask=mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                pos=pos_embed,
                query_pos=query_embed_all,
            ).transpose(1, 2)[
                0
            ]  # TODO 1 H B D
            hs_action = hs[:, : self.chunksize]  # B H D
            hs_token = hs[:, self.chunksize :]  # B H D
        else:
            hs_action = self.decoder(
                tgt_action,
                memory,
                memory_key_padding_mask=mask,
                tgt_key_padding_mask=is_pad,
                pos=pos_embed,
                query_pos=query_embed,
            ).transpose(1, 2)[0]
            hs_token = self.decoder_token(
                tgt_token,
                memory,
                memory_key_padding_mask=mask,  # tgt_key_padding_mask = is_pad_token, # is_pad_token is nonetype ! fix
                pos=pos_embed,
                query_pos=query_embed_token,
            ).transpose(1, 2)[0]
        return hs_action, hs_token


class Transformer_diffusion_prediction_pixel(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        num_queries=100,
        share_decoder=True,
        patch_size=5,
        diffusion_timestep_type="cat",
        attention_type="v1",
        predict_frame=16,
        predict_only_last=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        if share_decoder == False:
            self.decoder_token = copy.deepcopy(self.decoder)

        self.share_decoder = share_decoder
        self.diffusion_timestep_type = diffusion_timestep_type
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self._reset_parameters()

        self.action_embed = nn.Sequential(
            nn.Linear(14, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.token_embed = nn.Sequential(
            nn.Linear(
                3 * patch_size * patch_size, d_model
            ),  # Hardcode patch size * path size * patch dim
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.d_model = d_model
        self.nhead = nhead
        self.chunksize = num_queries
        self.attention_type = attention_type
        self.predict_frame = predict_frame
        self.predict_only_last = predict_only_last

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # hs_action, hs_token = self.transformer(src, None, self.query_embed.weight, self.query_embed_toekn.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, noisy_actions, noise_tokens, denoise_steps, self.denoise_step_pos_embed.weight)
    def forward(
        self,
        src,
        mask,
        query_embed,
        query_embed_token,
        pos_embed,
        latent_input=None,
        proprio_input=None,
        additional_pos_embed=None,
        noisy_actions=None,
        noisy_tokens=None,
        denoise_steps=None,
        denoise_step_pos_embed=None,
        is_pad=None,
        is_pad_token=None,
    ):
        # src: B D H W*num_view*T
        # mask:None
        # query_embed: chunksize D for action query
        # query_embed_token: Num_tokens D for token query
        # pos_embed: 1 D H W*num_view*T for current frame token
        # latent_input: B D
        # proprio_input: B T' D
        # additional_pos_embed: B T'+1 D, include proprio and latent
        # noisy_actions: B chunksize D
        # noisy_tokens: B T' N D H' W', T' = predict_frame / temporal_compression_rate
        # denoise_steps: B
        # denoise_step_pos_embed:1 D
        # is_pad: B chunksize
        # is_pad_token: B T' N H' W'
        if len(src.shape) == 4:  # has H and W
            bs, c, h, w = src.shape  # B D H W*num_view*T
            src = src.flatten(2).permute(
                2, 0, 1
            )  # H*W*num_view*T, bs, c deal with visual features from resnet
            # ?TODO check the shape of pos_embed
            pos_embed = (
                pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            )  # H*W*num_view*T, bs, c

            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            query_embed_token = query_embed_token.unsqueeze(1).repeat(
                1, bs, 1
            )  # TODO temporal-spatial position embedding FOR predicted frame token
            query_embed_all = torch.cat(
                [query_embed, query_embed_token], axis=0
            )  # chunksize + num_tokens, bs, c
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # seq, bs, dim
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

            latent_input = latent_input.unsqueeze(1)  # B 1 D
            addition_input = torch.cat([latent_input, proprio_input], axis=1).permute(
                1, 0, 2
            )  #  B T+1 D -> T+1 B D
            src = torch.cat([addition_input, src], axis=0)

            denoise_step_pos_embed = denoise_step_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # 1 D -> 1 B D
        else:
            assert len(src.shape) == 3

        # encoder
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # add denoise timestep embedding for decoder denoise step
        denoise_embed = self.time_embed(
            get_timestep_embedding(denoise_steps, self.d_model)
        ).unsqueeze(
            0
        )  # B D -> 1 B D
        if (
            self.diffusion_timestep_type == "cat"
        ):  # TODO add tokenizer visual token & timestep embedding
            memory = torch.cat([memory, denoise_embed], axis=0)
            pos_embed = torch.cat([pos_embed, denoise_step_pos_embed], axis=0)
        # elif self.diffusion_timestep_type == 'vis_cat':
        #     memory = torch.cat([memory, visual_token, denoise_embed], axis=0) # visual token map
        #     pos_embed = torch.cat([pos_embed, pos_embed_visual_token, denoise_step_pos_embed], axis=0) # pos_embed_visual_token
        else:
            memory = memory + denoise_embed

        tgt_action = self.action_embed(noisy_actions).permute(1, 0, 2)  # H B  D
        noisy_tokens = noisy_tokens.permute(
            0, 1, 2, 4, 5, 3
        )  #  B T' N D H' W' ->  B T' N H' W' D
        tgt_token = (
            self.token_embed(noisy_tokens)
            .reshape(bs, -1, self.d_model)
            .permute(1, 0, 2)
        )  # B T' N H' W' D -> T'*N*H'*W' B D
        # tgt_key_padding_mask = torch.zeros_like(tgt_token).sum(-1).bool() # T'*N*H'*W' B

        if self.share_decoder:
            tgt = torch.cat([tgt_action, tgt_token], axis=0)  # H1+ H2 B  D
            seq_len = tgt.shape[0]
            bs = tgt.shape[1]
            # tgt_key_padding_mask = torch.cat([is_pad, is_pad_token], axis=1) # B N+M is
            is_pad_token_zero = torch.zeros_like(is_pad_token).bool()  # HARD CODE
            if is_pad is None:
                is_pad = torch.zeros(bs, self.chunksize).bool().to(src.device)
            tgt_key_padding_mask = torch.cat(
                [is_pad, is_pad_token_zero], axis=1
            )  # HARD CODE avoid nan when all token is pad

            seq_len = tgt.shape[0]
            tgt_mask = torch.zeros(seq_len, seq_len).to(tgt.device)
            # TODO design mask
            if self.attention_type == "v1":
                tgt_mask[0 : self.chunksize, self.chunksize :] = float(
                    "-inf"
                )  # action prediction cannot attend to token prediction
                tgt_mask[self.chunksize :, self.predict_frame : self.chunksize] = float(
                    "-inf"
                )  # token prediction cannot attend to action token after token prediction
            elif self.attention_type == "v2":
                tgt_mask[0 : self.chunksize, self.chunksize :] = float(
                    "-inf"
                )  # action prediction cannot attend to token prediction
                tgt_mask[self.chunksize :, 0 : self.chunksize] = float(
                    "-inf"
                )  # token prediction cannot attend to action prediction
            elif self.attention_type == "v3":
                tgt_mask[0 : self.chunksize, self.chunksize :] = float(
                    "-inf"
                )  # action prediction cannot attend to token prediction

            # print(tgt.shape, memory.shape, tgt_key_padding_mask.shape, pos_embed.shape, query_embed_all.shape)
            hs = self.decoder(
                tgt,
                memory,
                tgt_mask,
                memory_key_padding_mask=mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                pos=pos_embed,
                query_pos=query_embed_all,
            ).transpose(1, 2)[
                0
            ]  # TODO 1 H B D
            hs_action = hs[:, : self.chunksize]  # B H D
            hs_token = hs[:, self.chunksize :]  # B H D
        else:
            hs_action = self.decoder(
                tgt_action,
                memory,
                memory_key_padding_mask=mask,
                tgt_key_padding_mask=is_pad,
                pos=pos_embed,
                query_pos=query_embed,
            ).transpose(1, 2)[0]
            hs_token = self.decoder_token(
                tgt_token,
                memory,
                memory_key_padding_mask=mask,  # tgt_key_padding_mask = is_pad_token, # is_pad_token is nonetype ! fix
                pos=pos_embed,
                query_pos=query_embed_token,
            ).transpose(1, 2)[0]
        # hs_action B chunksize D
        # hs_token  B T'*N*H'*W' D
        # print("tgt_token has NaN:", torch.isnan(tgt_token).any())
        # print("memory has NaN:", torch.isnan(memory).any())
        # print('tgt_key_padding_mask has NaN:', torch.isnan(is_pad_token).any())
        # print("pos_embed has NaN:", torch.isnan(pos_embed).any())
        # print("query_embed_token has NaN:", torch.isnan(query_embed_token).any())
        # print("hs_token has NaN:", torch.isnan(hs_token).any()) # has nan
        return hs_action, hs_token


class Transformer_diffusion(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        jpeg_dim=400,
        num_jpeg=80,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(14, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.jpeg_embed = nn.Sequential(
            nn.Linear(jpeg_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.num_jpeg = num_jpeg

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        mask,
        query_embed,
        pos_embed,
        latent_input=None,
        proprio_input=None,
        additional_pos_embed=None,
        noisy_actions=None,
        noisy_jpegs=None,
        denoise_steps=None,
    ):
        # TODO flatten only when input has H and W
        if len(src.shape) == 4:  # has H and W
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            # mask = mask.flatten(1)

            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # seq, bs, dim
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

            addition_input = torch.stack([latent_input, proprio_input], axis=0)
            src = torch.cat([addition_input, src], axis=0)
        else:
            assert len(src.shape) == 3
            # flatten NxHWxC to HWxN
        tgt_action = self.action_embed(noisy_actions).permute(1, 0, 2)  # B H D
        tgt_jpeg = self.jpeg_embed(noisy_jpegs).permute(1, 0, 2)  # B H. D
        tgt = torch.cat([tgt_action, tgt_jpeg], axis=0)  # H1+ H2 B  D

        denoise_embed = get_timestep_embedding(denoise_steps, self.d_model)  # B -> B D
        denoise_embed = self.time_embed(denoise_embed)
        memory = self.encoder(
            src, src_key_padding_mask=mask, pos=pos_embed
        )  # cross attentionc# TODO mak mask to focus on self-time-step
        memory = memory + denoise_embed.unsqueeze(0)
        seq_len = tgt.shape[0]
        tgt_mask = torch.zeros(seq_len, seq_len).to(tgt.device)
        tgt_mask[: -self.num_jpeg, -self.num_jpeg :] = float(
            "-inf"
        )  
        tgt_mask[-self.num_jpeg :, : -self.num_jpeg] = float("-inf")  #
        hs = self.decoder(
            tgt,
            memory,
            tgt_mask,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )  # TODO 1 H B D
        hs = hs.transpose(1, 2)  # 1 B H D
        return hs


class Transformer_diffusion_seperate(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        share_decoder=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder_action = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )
        # self.decoder_jpeg = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                 return_intermediate=return_intermediate_dec)
        if share_decoder:
            self.decoder_jpeg = self.decoder_action
        else:
            self.decoder_jpeg = copy.deepcopy(self.decoder_action)

        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.action_embed = nn.Sequential(
            nn.Linear(14, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src,
        mask,
        query_embed_action=None,
        query_embed_jpeg=None,
        pos_embed=None,
        latent_input=None,
        proprio_input=None,
        additional_pos_embed=None,
        noisy_actions=None,
        denoise_steps=None,
    ):
        # TODO flatten only when input has H and W
        if len(src.shape) == 4:  # has H and W
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed_action = query_embed_action.unsqueeze(1).repeat(1, bs, 1)
            query_embed_jpeg = query_embed_jpeg.unsqueeze(1).repeat(1, bs, 1)
            # mask = mask.flatten(1)

            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # seq, bs, dim
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

            addition_input = torch.stack([latent_input, proprio_input], axis=0)
            src = torch.cat([addition_input, src], axis=0)
        else:
            assert len(src.shape) == 3
            # flatten NxHWxC to HWxN

        memory = self.encoder(
            src, src_key_padding_mask=mask, pos=pos_embed
        )  # cross attentionc# TODO mak mask to focus on self-time-step
        denoise_embed = get_timestep_embedding(denoise_steps, self.d_model)  # B -> B D
        denoise_embed = self.time_embed(denoise_embed)
        memory_action = memory + denoise_embed.unsqueeze(0)

        tgt_action = self.action_embed(noisy_actions).permute(1, 0, 2)  # B H D
        # print('tgt_action', tgt_action.shape)
        hs_action = self.decoder_action(
            tgt_action,
            memory_action,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed_action,
        )
        # print('hs_action before', hs_action.shape)
        hs_action = hs_action.transpose(1, 2)  # 1 B H D
        # print('hs_action after', hs_action.shape)
        tgt_jpeg = torch.zeros_like(query_embed_jpeg).cuda()
        # print('tgt_jpeg', tgt_jpeg.shape)
        hs_jpeg = self.decoder_jpeg(
            tgt_jpeg,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed_jpeg,
        )
        hs_jpeg = hs_jpeg.transpose(1, 2)  # 1 B H D

        return hs_action[0], hs_jpeg[0]

    def inference_only_actin(
        self,
        src,
        mask,
        query_embed_action=None,
        pos_embed=None,
        latent_input=None,
        proprio_input=None,
        additional_pos_embed=None,
        noisy_actions=None,
        denoise_steps=None,
    ):
        if len(src.shape) == 4:  # has H and W
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed_action = query_embed_action.unsqueeze(1).repeat(1, bs, 1)
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(
                1, bs, 1
            )  # seq, bs, dim
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)

            addition_input = torch.stack([latent_input, proprio_input], axis=0)
            src = torch.cat([addition_input, src], axis=0)
        else:
            assert len(src.shape) == 3

        memory = self.encoder(
            src, src_key_padding_mask=mask, pos=pos_embed
        )  # cross attentionc# TODO mak mask to focus on self-time-step
        denoise_embed = get_timestep_embedding(denoise_steps, self.d_model)  # B -> B D
        denoise_embed = self.time_embed(denoise_embed)
        memory_action = memory + denoise_embed.unsqueeze(0)

        tgt_action = self.action_embed(noisy_actions).permute(1, 0, 2)  # B H D

        hs_action = self.decoder_action(
            tgt_action,
            memory_action,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed_action,
        )
        return hs_action, None


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoder_AdLN(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        condition,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                condition,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output.unsqueeze(0) #1 T B D
    

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerDecoder_alter(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        memory_alter,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        pos_alter: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt
        memory_list = [memory, memory_alter]
        pos_list = [pos, pos_alter]
        intermediate = []
        num_layer = 0
        for layer in self.layers:
            index = num_layer % 2
            output = layer(
                output,
                memory_list[index],
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos_list[index],
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)

class TransformerEncoderLayer_AdLN(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        ) # neccesary for adaLN

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_pre(
        self,
        src, # B T D
        condition, # B D
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(6, dim=1) # B D
        
        src2 = self.norm1(src)
        src2 = modulate(src2, shift_msa, scale_msa)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(gate_msa.unsqueeze(0) *src2)
        
        src2 = self.norm2(src)
        src2 = modulate(src2, shift_mlp, scale_mlp)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(gate_mlp.unsqueeze(0) *src2)
        return src

    def forward(
        self,
        src,
        condition,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):

        return self.forward_pre(src, condition,src_mask, src_key_padding_mask, pos)



class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_denoise(args):
    print(f"Using {args.condition_type} for condition")
    if args.condition_type == "adaLN":
        return Transformer_Denoise_AdLN(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers,
            num_decoder_layers=args.dec_layers,
            normalize_before=args.pre_norm,
            return_intermediate_dec=True,
        )
    return Transformer_Denoise(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        causal_mask=args.causal_mask,
    )

def build_transformer_denoise_tactile(args):
    return Transformer_Denoise_Tactile(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        causal_mask=args.causal_mask,
    )


def build_transformer_diffusion_prediction(args):
    return Transformer_diffusion_prediction(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        num_queries=args.num_queries,
        share_decoder=args.share_decoder,
        patch_size=args.patch_size,
        diffusion_timestep_type=args.diffusion_timestep_type,
        attention_type=args.attention_type,
        predict_frame=args.predict_frame,
        predict_only_last=args.predict_only_last,
        token_dim=args.token_dim,
    )


def build_transformer_diffusion_prediction_with_dual_visual_token(args):
    return Transformer_diffusion_prediction_with_dual_visual_token(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        num_queries=args.num_queries,
        share_decoder=args.share_decoder,
        patch_size=args.patch_size,
        diffusion_timestep_type=args.diffusion_timestep_type,
        attention_type=args.attention_type,
        predict_frame=args.predict_frame,
        predict_only_last=args.predict_only_last,
    )


def build_transformer_diffusion(args):
    return Transformer_diffusion(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        jpeg_dim=args.jpeg_dim,  # Hard codes
        num_jpeg=args.jpeg_token_num * args.predict_frame,  # Hard codes
        return_intermediate_dec=True,
    )


def build_transformer_diffusion_pixel_prediction(args):
    return Transformer_diffusion_prediction_pixel(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        num_queries=args.num_queries,
        share_decoder=args.share_decoder,
        patch_size=args.patch_size,
        diffusion_timestep_type=args.diffusion_timestep_type,
        attention_type=args.attention_type,
        predict_frame=args.predict_frame,
        predict_only_last=args.predict_only_last,
    )


def build_transformer_diffusion_seperate(args):
    return Transformer_diffusion_seperate(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        share_decoder=args.share_decoder,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos=False,
    downscale_freq_shift=1,
    scale=1,
    max_period=10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    return emb

class Tactile_ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tactile_ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=3, stride=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size=5, stride=5)
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))
        self.activation = nn.SiLU()
        self.bn1 = nn.BatchNorm2d(out_channels//2)  # LN for conv1 output
        self.bn2 = nn.BatchNorm2d(out_channels)    # LN for conv2 output
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        return x
    
class Tactile_Encoder(nn.Module):
    def __init__(self,tactile_dim,dropout):
        super().__init__()
        self.conv_ll = Tactile_ConvBlock(3, tactile_dim)
        self.conv_lr = Tactile_ConvBlock(3, tactile_dim)
        self.conv_rl = Tactile_ConvBlock(3, tactile_dim)
        self.conv_rr = Tactile_ConvBlock(3, tactile_dim)
        self.feature_extractor = nn.ModuleList([
            self.conv_ll,
            self.conv_lr,
            self.conv_rl,
            self.conv_rr
        ])
        self.input_proj_tacile = nn.Conv2d(
            tactile_dim,
            tactile_dim,
            kernel_size=1,
        )
        
        # self.query_tokens = nn.Parameter(torch.randn(16, tactile_dim)) 
        # self.attn = nn.MultiheadAttention(embed_dim=tactile_dim, num_heads=8, dropout=dropout,batch_first=True)
        self.tactile_dim = tactile_dim
        
    def forward(self,tactile_data):
        #B 4 C H W
        tactile_data = tactile_data[:,0]
        B = tactile_data.shape[0] 
        tactile_feature_list = []
        for i in range(tactile_data.shape[1]):
            tactile_feature = self.feature_extractor[i](tactile_data[:,i])
            tactile_feature = self.input_proj_tacile(tactile_feature)
            tactile_feature_list.append(tactile_feature)
        tactile_features_raw = torch.stack(tactile_feature_list, dim=2) # B C 4 H W
        # print('before attention tactile feature raw shape', tactile_features_raw.shape)
        # tactile_features = tactile_features_raw.view(B, tactile_features_raw.size(1), -1) # B C 4*H*W
        # print('before attention tactile feature shape', tactile_features.shape)
        # tactile_features = tactile_features.permute(0, 2, 1) # B H*W*4 C   
        return tactile_features_raw # b d 4 h w 
        # print('before attention tactile feature shape', tactile_features.shape)
        # query = self.query_tokens.unsqueeze(0).repeat(B, 1, 1)
        # # print('query shape', query.shape)
        # attn_output, _ = self.attn(query, tactile_features, tactile_features) # B 16 D
        # return attn_output
    
if __name__ == "__main__":
    tactile_data = torch.randn(2, 1, 4, 3, 960, 960).cuda()
    tactile_encoder = Tactile_Encoder(512, 0.1).cuda()
    tactile_feature = tactile_encoder(tactile_data)
    print(tactile_feature.shape)  # B 16 D
    total_params = sum(p.numel() for p in tactile_encoder.parameters() if p.requires_grad)
    total_params_in_million = total_params / 1e6
    print(f"Total trainable parameters: {total_params_in_million:.2f} M")