import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math
from torch.nn.modules.batchnorm import _BatchNorm
from collections import OrderedDict
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from torch.nn import functional as F

_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)


# def normalize_data(action_data, stats, norm_type, data_type="action"):

#     if norm_type == "minmax":
#         action_max = torch.from_numpy(stats[data_type + "_max"]).float().cuda()
#         action_min = torch.from_numpy(stats[data_type + "_min"]).float().cuda()
#         action_data = (action_data - action_min) / (action_max - action_min) * 2 - 1
#     elif norm_type == "gaussian":
#         action_mean = torch.from_numpy(stats[data_type + "_mean"]).float().cuda()
#         action_std = torch.from_numpy(stats[data_type + "_std"]).float().cuda()
#         action_data = (action_data - action_mean) / action_std
#     return action_data


def tensor2numpy(input_tensor: torch.Tensor, range_min: int = -1) -> np.ndarray:
    """Converts tensor in [-1,1] to image(dtype=np.uint8) in range [0..255].

    Args:
        input_tensor: Input image tensor of Bx3xHxW layout, range [-1..1].
    Returns:
        A numpy image of layout BxHxWx3, range [0..255], uint8 dtype.
    """
    if range_min == -1:
        input_tensor = (input_tensor.float() + 1.0) / 2.0
    ndim = input_tensor.ndim
    output_image = input_tensor.clamp(0, 1).cpu().numpy()
    output_image = output_image.transpose((0,) + tuple(range(2, ndim)) + (1,))
    return (output_image * _UINT8_MAX_F + 0.5).astype(np.uint8)


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class RandomShiftsAug(nn.Module):
    def __init__(self, pad_h, pad_w):
        super().__init__()
        self.pad_h = pad_h
        self.pad_w = pad_w
        print(f"RandomShiftsAug: pad_h {pad_h}, pad_w {pad_w}")

    def forward(self, x):
        orignal_shape = x.shape
        n, h, w = x.shape[0], x.shape[-2], x.shape[-1]  # n,T,M,C,H,W
        x = x.view(n, -1, h, w)  # n,T*M*C,H,W
        padding = (
            self.pad_w,
            self.pad_w,
            self.pad_h,
            self.pad_h,
        )  # left, right, top, bottom padding
        x = F.pad(x, padding, mode="replicate")

        h_pad, w_pad = h + 2 * self.pad_h, w + 2 * self.pad_w
        eps_h = 1.0 / h_pad
        eps_w = 1.0 / w_pad

        arange_h = torch.linspace(
            -1.0 + eps_h, 1.0 - eps_h, h_pad, device=x.device, dtype=x.dtype
        )[:h]
        arange_w = torch.linspace(
            -1.0 + eps_w, 1.0 - eps_w, w_pad, device=x.device, dtype=x.dtype
        )[:w]

        arange_h = arange_h.unsqueeze(1).repeat(1, w).unsqueeze(2)  # h w 1
        arange_w = arange_w.unsqueeze(1).repeat(1, h).unsqueeze(2)  # w h 1

        # print(arange_h.shape, arange_w.shape)
        base_grid = torch.cat([arange_w.transpose(1, 0), arange_h], dim=2)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).repeat(
            n, 1, 1, 1
        )  # Repeat for batch [B, H, W, 2]

        shift_h = torch.randint(
            0, 2 * self.pad_h + 1, size=(n, 1, 1, 1), device=x.device, dtype=x.dtype
        ).float()
        shift_w = torch.randint(
            0, 2 * self.pad_w + 1, size=(n, 1, 1, 1), device=x.device, dtype=x.dtype
        ).float()
        shift_h *= 2.0 / h_pad
        shift_w *= 2.0 / w_pad

        grid = base_grid + torch.cat([shift_w, shift_h], dim=3)
        x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
        return x.view(orignal_shape)


class EMAModel:
    """
    Exponential Moving Average of models weights, support multi-gpus
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=0.75,
        min_value=0.0,
        max_value=0.9999,
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        self.averaged_model = model
        self.averaged_model.eval()
        # self.averaged_model.requires_grad_(False)
        for param in self.averaged_model.parameters():
            param.detach_()

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)

        # old_all_dataptrs = set()
        # for param in new_model.parameters():
        #     data_ptr = param.data_ptr()
        #     if data_ptr != 0:
        #         old_all_dataptrs.add(data_ptr)

        all_dataptrs = set()
        for module, ema_module in zip(
            new_model.modules(), self.averaged_model.modules()
        ):
            for param, ema_param in zip(
                module.parameters(recurse=False), ema_module.parameters(recurse=False)
            ):
                # iterative over immediate parameters only.
                if isinstance(param, dict):
                    raise RuntimeError("Dict parameter not supported")

                # data_ptr = param.data_ptr()
                # if data_ptr != 0:
                #     all_dataptrs.add(data_ptr)

                if isinstance(module, _BatchNorm):
                    # skip batchnorms
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(
                        param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay
                    )

        # verify that iterating over module and then parameters is identical to parameters recursively.
        # assert old_all_dataptrs == all_dataptrs
        self.optimization_step += 1


class EpisodicDataset_Unified(torch.utils.data.Dataset):
    """
    only for head_camera
    Args:
        norm_stats: normalization stats for qpos and action
        chunksize: chunk size
        history_steps: number of history steps
        predict_frame: number of future frames to predict
    Output:
        A unified dataset for all the datasets
        image_data [0~1]: history_steps+1 Num_view C H W
        qpos_data [normalized]: history_steps+1 D
        action_data [raw]: chunk_size D
        is_pad: chunk_size
        future_imgs_data [0~1]: predict_frame Num_view C H W
        is_pad_img : predict_frame
    """

    def __init__(
        self,
        head_camera,
        state,
        action,
        episode_ends,
        stats,
        chunk_size=50,
        history_steps=0,
        predict_frame=0,
        temporal_downsample_rate=1,
        predict_only_last=False,
    ):
        super(EpisodicDataset_Unified).__init__()
        self.head_camera = head_camera # T H W C 
        self.state = state
        self.action = action
        self.episode_ends = episode_ends
        self.norm_stats = stats
        self.chunk_size = chunk_size
        self.history_steps = history_steps
        self.predict_frame = predict_frame
        self.temporal_downsample_rate = temporal_downsample_rate
        self.predict_only_last = predict_only_last

    def __len__(self):
        return len(self.head_camera)

    def _get_episode_idx(self, idx):
        for i, end_idx in enumerate(self.episode_ends):
            if idx < end_idx:
                start_idx = self.episode_ends[i - 1] if i > 0 else 0
                end_idx = end_idx
                return start_idx, end_idx

    def __getitem__(self, index):
        start_idx, end_idx = self._get_episode_idx(index)
        index = min(index, end_idx - 2)  # Avoid the last index of episode
        # get observation qpos and image data
        past_start_ts = max(start_idx, index - self.history_steps)
        past_padding_needed = self.history_steps - (index - past_start_ts)
        obs_qpos = self.state[past_start_ts : index + 1]
        obs_img = self.head_camera[past_start_ts : index + 1]
        if past_padding_needed > 0:
            padding_qpos = np.tile(self.state[start_idx], (past_padding_needed, 1))
            padding_image = np.tile(
                self.head_camera[start_idx], (past_padding_needed, 1, 1, 1)
            )
            obs_qpos = np.concatenate([padding_qpos, obs_qpos], axis=0)
            obs_img = np.concatenate(
                [padding_image, obs_img], axis=0
            )  # (history_steps+1, 3, H, W)
        # get action chunk data
        original_action_shape = (self.chunk_size, *self.action.shape[1:])
        gt_action = np.zeros(original_action_shape)
        action_len = min(self.chunk_size, end_idx - index - 1)
        # print(action_len)
        # print(self.action[index+1:index+1+action_len].shape)
        gt_action[:action_len] = self.action[
            index + 1 : index + 1 + action_len
        ]  # move left one step,due to action = qpos
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1
        # get future image data
        if self.predict_frame > 0:
            future_frames_len = min(self.predict_frame, end_idx - index - 1)
            future_images = self.head_camera[
                index : index + future_frames_len + 1
            ]  # (future_frames_len+1, 3, H, W)
            is_pad_img = np.zeros(self.predict_frame + 1)
            future_padding_needed = self.predict_frame - future_frames_len

            if future_padding_needed > 0:
                is_pad_img[-future_padding_needed:] = 1
                last_frame = self.head_camera[end_idx - 1]  # (3, H, W)
                padding_future_frames = np.broadcast_to(
                    last_frame, (future_padding_needed, *last_frame.shape)
                )
                future_images = np.concatenate(
                    [future_images, padding_future_frames], axis=0
                )

            future_images = future_images[:: self.temporal_downsample_rate][
                1:
            ]  # (predict_frames, 3, H, W)
            is_pad_img = is_pad_img[:: self.temporal_downsample_rate][
                1:
            ]  # (predict_frames)

            if self.predict_only_last:
                future_images = future_images[-1:]
                is_pad_img = is_pad_img[-1:]

        # construct observations
        image_data = (
            torch.from_numpy(obs_img).unsqueeze(1).float()
        )  # (history_steps+1, 1, 3, H, W) add num_view
        qpos_data = torch.from_numpy(obs_qpos).float()
        action_data = torch.from_numpy(gt_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        if self.predict_frame > 0:
            future_imgs_data = torch.from_numpy(future_images).unsqueeze(1).float()
            is_pad_img = torch.from_numpy(is_pad_img).bool()
        else:
            future_imgs_data = 0
            is_pad_img = 0

        # normalize image and change dtype to float
        image_data = image_data / 255.0  # history_steps+1 N C H W
        future_imgs_data = (
            future_imgs_data / 255.0 if future_imgs_data is not None else None
        )
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]

        return image_data, qpos_data, action_data, is_pad, future_imgs_data, is_pad_img


class EpisodicDataset_Unified_Multiview(torch.utils.data.Dataset):
    """
    only for head_camera
    Args:
        norm_stats: normalization stats for qpos and action
        chunksize: chunk size
        history_steps: number of history steps
        predict_frame: number of future frames to predict
    Output:
        A unified dataset for all the datasets
        image_data [0~1]: history_steps+1 Num_view C H W
        qpos_data [normalized]: history_steps+1 D
        action_data [raw]: chunk_size D
        is_pad: chunk_size
        future_imgs_data [0~1]: predict_frame Num_view C H W
        is_pad_img : predict_frame
    """

    def __init__(
        self,
        head_camera,
        state,
        action,
        episode_ends,
        stats,
        chunk_size=50,
        history_steps=0,
        predict_frame=0,
        temporal_downsample_rate=1,
        predict_only_last=False,
    ):
        super(EpisodicDataset_Unified_Multiview).__init__()
        self.head_camera = head_camera # T H W C 
        self.state = state
        self.action = action
        self.episode_ends = episode_ends
        self.norm_stats = stats
        self.chunk_size = chunk_size
        self.history_steps = history_steps
        self.predict_frame = predict_frame
        self.temporal_downsample_rate = temporal_downsample_rate
        self.predict_only_last = predict_only_last

    def __len__(self):
        return len(self.head_camera)

    def _get_episode_idx(self, idx):
        for i, end_idx in enumerate(self.episode_ends):
            if idx < end_idx:
                start_idx = self.episode_ends[i - 1] if i > 0 else 0
                end_idx = end_idx
                return start_idx, end_idx

    def __getitem__(self, index):
        start_idx, end_idx = self._get_episode_idx(index)
        index = min(index, end_idx - 2)  # Avoid the last index of episode
        # get observation qpos and image data
        past_start_ts = max(start_idx, index - self.history_steps)
        past_padding_needed = self.history_steps - (index - past_start_ts)
        obs_qpos = self.state[past_start_ts : index + 1]
        obs_img = self.head_camera[past_start_ts : index + 1] # his+1 N_view 3 H W 
        if past_padding_needed > 0:
            padding_qpos = np.tile(self.state[start_idx], (past_padding_needed, 1))
            padding_image = np.tile(
                self.head_camera[start_idx], (past_padding_needed, 1, 1, 1, 1)
            )
            obs_qpos = np.concatenate([padding_qpos, obs_qpos], axis=0)
            obs_img = np.concatenate(
                [padding_image, obs_img], axis=0
            )  # (history_steps+1, N_view, 3, H, W)
        # get action chunk data
        original_action_shape = (self.chunk_size, *self.action.shape[1:])
        gt_action = np.zeros(original_action_shape)
        action_len = min(self.chunk_size, end_idx - index - 1)

        gt_action[:action_len] = self.action[
            index + 1 : index + 1 + action_len
        ]  # move left one step,due to action = qpos
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1
        # get future image data
        if self.predict_frame > 0:
            future_frames_len = min(self.predict_frame, end_idx - index - 1)
            future_images = self.head_camera[
                index : index + future_frames_len + 1
            ]  # (future_frames_len+1,N_view, 3, H, W)
            is_pad_img = np.zeros(self.predict_frame + 1)
            future_padding_needed = self.predict_frame - future_frames_len

            if future_padding_needed > 0:
                is_pad_img[-future_padding_needed:] = 1
                last_frame = self.head_camera[end_idx - 1]  # (N_view, 3, H, W)
                padding_future_frames = np.broadcast_to(
                    last_frame, (future_padding_needed, *last_frame.shape)
                )
                future_images = np.concatenate(
                    [future_images, padding_future_frames], axis=0
                )

            future_images = future_images[:: self.temporal_downsample_rate][
                1:
            ]  # (predict_frames, 3, H, W)
            is_pad_img = is_pad_img[:: self.temporal_downsample_rate][
                1:
            ]  # (predict_frames)

            if self.predict_only_last:
                future_images = future_images[-1:]
                is_pad_img = is_pad_img[-1:]

        # construct observations
        image_data = (
            torch.from_numpy(obs_img).float()
        )  # (history_steps+1, N_view,, 3, H, W) add num_view
        qpos_data = torch.from_numpy(obs_qpos).float()
        action_data = torch.from_numpy(gt_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        if self.predict_frame > 0:
            future_imgs_data = torch.from_numpy(future_images).float() # TODO fixed single view?
            is_pad_img = torch.from_numpy(is_pad_img).bool()
        else:
            future_imgs_data = 0
            is_pad_img = 0

        # normalize image and change dtype to float
        image_data = image_data / 255.0  # history_steps+1 N C H W
        future_imgs_data = (
            future_imgs_data / 255.0 if future_imgs_data is not None else None
        )
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]

        return image_data, qpos_data, action_data, is_pad, future_imgs_data, is_pad_img


class EpisodicDataset_Unified_vision_tacile(torch.utils.data.Dataset):
    """
    only for head_camera
    Args:
        norm_stats: normalization stats for qpos and action
        chunksize: chunk size
    Output:
        A unified dataset for all the datasets
        image_data [0~1]: 1 Num_view C H W
        qpos_data [normalized]: 1 D
        action_data [raw]: chunk_size D
        is_pad: chunk_size
        ll_tactile_data [normalized]: C H W 
        lr_tactile_data [normalized]: C H W 
        rr_tactile_data [normalized]: C H W 
        rl_tactile_data [normalized]: C H W 
    """

    def __init__(
        self,
        head_camera,
        state,
        action,
        episode_ends,
        stats,
        chunk_size=50,
    ):
        super(EpisodicDataset_Unified_vision_tacile).__init__()
        self.head_camera = head_camera # T H W C 
        self.state = state
        self.action = action
        self.episode_ends = episode_ends
        self.norm_stats = stats
        self.chunk_size = chunk_size # tacile 0~189 
    def __init__(
        self,
        head_camera, # N N_view C H W
        state,
        action,
        tactile, # N 4 C H W
        episode_ends,
        stats,
        chunk_size=50,
        
    ):
        super(EpisodicDataset_Unified_vision_tacile).__init__()
        self.head_camera = head_camera # N N_view C H W
        self.state = state
        self.action = action
        self.tactile = tactile # N 4 C H W
        self.episode_ends = episode_ends
        self.norm_stats = stats
        self.chunk_size = chunk_size 
    
    def __len__(self):
        return len(self.head_camera)        
    
    def _get_episode_idx(self, idx):
        for i, end_idx in enumerate(self.episode_ends):
            if idx < end_idx:
                start_idx = self.episode_ends[i - 1] if i > 0 else 0
                end_idx = end_idx
                return start_idx, end_idx   
    def __getitem__(self, index):
        start_idx, end_idx = self._get_episode_idx(index)    
        index = min(index, end_idx - 2)
        obs_qpos = self.state[index : index + 1] # 1 D
        obs_img = self.head_camera[index : index + 1] # 1 N_view 3 H W 
        obs_tacile = self.tactile[index : index + 1] # 1 4 C H W
        original_action_shape = (self.chunk_size, *self.action.shape[1:])
        gt_action = np.zeros(original_action_shape)
        action_len = min(self.chunk_size, end_idx - index - 1)
        gt_action[:action_len] = self.action[
            index + 1 : index + 1 + action_len
        ]
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1 
        
        image_data = (
            torch.from_numpy(obs_img).float()
        )  # (1, N_view,, 3, H, W) add num_view
        qpos_data = torch.from_numpy(obs_qpos).float()
        action_data = torch.from_numpy(gt_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        tacile_data = torch.from_numpy(obs_tacile).float() # N 4 C H W

        image_data = image_data / 255.0 # 1 N_view C H W
        tacile_data = tacile_data / 200.0 # 1 4 C H W
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ] # 1 D
        
        return image_data, qpos_data, action_data, is_pad, tacile_data


class EpisodicDataset_Unified_Multiview_Simple(torch.utils.data.Dataset):
    
    def __init__(self,
        data_dir,
        train_index,
        episode_ends,
        stats,
        chunk_size,
        history_steps,
    ):
        super(EpisodicDataset_Unified_Multiview_Simple).__init__()
        self.data_dir = data_dir
        self.train_index = train_index
        self.episode_ends = episode_ends
        self.norm_stats = stats
        self.chunk_size = chunk_size
        self.history_steps = history_steps
    
    def __len__(self):
        return len(self.train_index)
    
    def _get_episode_idx(self, idx): # return episode index and episode length
        for i, end_idx in enumerate(self.episode_ends):
            if idx < end_idx:
                start_idx = self.episode_ends[i - 1] if i > 0 else 0
                end_idx = end_idx
                episode_len = end_idx - start_idx
                frame_idx = idx - start_idx
                return i,frame_idx, episode_len
    def __getitem__(self, index):
        episode_idx, frame_idx, episode_len = self._get_episode_idx(index)
        data_path = os.path.join(self.data_dir,f'episode{episode_idx}.pt') 
        episode_data = torch.load(data_path,weights_only=False)
        # get observation qpos and image data
        past_frame_index = max(0, frame_idx - self.history_steps)
        head_camera = episode_data['head_camera_arrays'][past_frame_index:frame_idx+1] # T C H W
        state = episode_data['state_arrays'][past_frame_index:frame_idx+1] # T D
        past_padding_needed = self.history_steps+1 - head_camera.shape[0]
        if past_padding_needed > 0:
           first_head_camera = episode_data['head_camera_arrays'][0] # C H W
           first_head_camera = first_head_camera.unsqueeze(0).repeat(past_padding_needed, 1, 1, 1)
           first_state = episode_data['state_arrays'][0] # D 
           first_state = first_state.unsqueeze(0).repeat(past_padding_needed, 1)
           head_camera = torch.cat([first_head_camera, head_camera], dim=0)
           state = torch.cat([first_state, state], dim=0)
        # get action chunk data
        action_len = min(self.chunk_size, episode_len - frame_idx - 1)
        action = episode_data['joint_action_arrays'][frame_idx+1:frame_idx+1+action_len] # T D
        original_action_shape = (self.chunk_size, *action.shape[1:])
        gt_action = torch.zeros(original_action_shape)
        gt_action[:action_len] = action
        is_pad = torch.zeros(self.chunk_size).bool()
        is_pad[action_len:] = 1

        image_data = head_camera.unsqueeze(1).float() / 255.0  # (T, N, C, H, W)
        qpos_data = (state.float() - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ] # T D
        action_data = gt_action.float()
        
        return image_data, qpos_data, action_data, is_pad, 0, 0
    
    
def get_norm_stats(state, action):
    all_qpos_data = torch.from_numpy(np.array(state))
    all_action_data = torch.from_numpy(np.array(action))
    # normalize action data
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping
    action_max = torch.amax(all_action_data, dim=[0], keepdim=True)
    action_min = torch.amin(all_action_data, dim=[0], keepdim=True)

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "action_max": action_max.numpy().squeeze(),
        "action_min": action_min.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
    }

    return stats


def split_episode_indices(head_camera, state, action, episode_ends, train_ratio=0.98):
    num_episodes = len(episode_ends)
    shuffledm_episodes_indices = np.random.permutation(num_episodes)
    train_episode_index = shuffledm_episodes_indices[: int(train_ratio * num_episodes)]
    val_episode_index = shuffledm_episodes_indices[int(train_ratio * num_episodes) :]

    train_head_camera = []
    train_state = []
    train_action = []
    train_episode_ends = []
    train_samples = 0

    val_head_camera = []
    val_state = []
    val_action = []
    val_episode_ends = []
    val_samples = 0

    prev_end = 0
    for index, end in enumerate(episode_ends):
        sampled_head_camera = head_camera[prev_end:end]
        sampled_state = state[prev_end:end]
        sampled_action = action[prev_end:end]
        if index in train_episode_index:
            train_head_camera.append(sampled_head_camera)
            train_state.append(sampled_state)
            train_action.append(sampled_action)
            train_samples += len(sampled_action)
            train_episode_ends.append(train_samples)
        else:
            val_head_camera.append(sampled_head_camera)
            val_state.append(sampled_state)
            val_action.append(sampled_action)
            val_samples += len(sampled_action)
            val_episode_ends.append(val_samples)
        prev_end = end
    train_head_camera = np.concatenate(train_head_camera, axis=0)
    train_state = np.concatenate(train_state, axis=0)
    train_action = np.concatenate(train_action, axis=0)
    train_episode_ends = np.stack(train_episode_ends)

    val_head_camera = np.concatenate(val_head_camera, axis=0)
    val_state = np.concatenate(val_state, axis=0)
    val_action = np.concatenate(val_action, axis=0)
    val_episode_ends = np.stack(val_episode_ends)

    return (
        train_head_camera,
        train_state,
        train_action,
        train_episode_ends,
        val_head_camera,
        val_state,
        val_action,
        val_episode_ends,
    )


def load_data_unified(
    data_dir,
    task_name,
    head_camera_type,
    num_episodes=100,
    train_ratio=0.9,
    batch_size_train=32,
    batch_size_val=32,
    chunk_size=100,
    history_step=0,
    predict_frame=0,
    temporal_downsample_rate=1,
    predict_only_last=False,
    distributed=False,
):
    zarr_path = os.path.join(
        data_dir, f"{task_name}_{head_camera_type}_{num_episodes}.zarr"
    )
    print(f"Loading data from {zarr_path}")
    zarr_root = zarr.open(zarr_path, mode="r")
    head_camera = zarr_root["data/head_camera"]
    state = zarr_root["data/state"]
    action = zarr_root["data/action"]
    episode_ends = zarr_root["meta/episode_ends"]
    stats = get_norm_stats(state, action)
    # split the dataset
    (
        train_head_camera,
        train_state,
        train_action,
        train_episode_ends,
        val_head_camera,
        val_state,
        val_action,
        val_episode_ends,
    ) = split_episode_indices(head_camera, state, action, episode_ends, train_ratio)
    print(
        f"Train episodes: {len(train_episode_ends)}, Validation episodes: {len(val_episode_ends)}"
    )
    print(
        f"Train samples: {train_episode_ends[-1]}, Validation samples: {val_episode_ends[-1]}"
    )
    # create dataset
    train_dataset = EpisodicDataset_Unified(
        train_head_camera,
        train_state,
        train_action,
        train_episode_ends,
        stats,
        chunk_size,
        history_step,
        predict_frame,
        temporal_downsample_rate,
        predict_only_last,
    )
    val_dataset = EpisodicDataset_Unified(
        val_head_camera,
        val_state,
        val_action,
        val_episode_ends,
        stats,
        chunk_size,
        history_step,
        predict_frame,
        temporal_downsample_rate,
        predict_only_last,
    )

    if distributed:
        print("Using distributed sampler-----------------------------")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=(train_sampler is None),
        pin_memory=True,
        num_workers=4,
        sampler=train_sampler,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        sampler=val_sampler,
    )

    return train_dataloader, val_dataloader, train_sampler, stats


def load_data_unified_multiview(
    data_dir,
    task_name,
    head_camera_type,
    num_episodes=100,
    camera_name=['head_camra'],
    train_ratio=0.9,
    batch_size_train=32,
    batch_size_val=32,
    chunk_size=100,
    history_step=0,
    predict_frame=0,
    temporal_downsample_rate=1,
    predict_only_last=False,
    distributed=False,
):
    zarr_path = os.path.join(
        data_dir, f"{task_name}_{head_camera_type}_{num_episodes}.zarr"
    )
    print(f"Loading data from {zarr_path}")
    zarr_root = zarr.open(zarr_path, mode="r")
    # head_camera = zarr_root["data/head_camera"]
    camera_image = []
    for camera_name in camera_name:
        camera_image.append(zarr_root[f"data/{camera_name}"])
    head_camera = np.stack(camera_image, axis=1) # T N H W C
    state = zarr_root["data/state"]
    action = zarr_root["data/action"]
    episode_ends = zarr_root["meta/episode_ends"]
    stats = get_norm_stats(state, action)
    # split the dataset
    (
        train_head_camera, # N_train N_view H W C
        train_state,
        train_action,
        train_episode_ends,
        val_head_camera,
        val_state,
        val_action,
        val_episode_ends,
    ) = split_episode_indices(head_camera, state, action, episode_ends, train_ratio)
    print(
        f"Train episodes: {len(train_episode_ends)}, Validation episodes: {len(val_episode_ends)}"
    )
    print(
        f"Train samples: {train_episode_ends[-1]}, Validation samples: {val_episode_ends[-1]}"
    )
    # create dataset
    train_dataset = EpisodicDataset_Unified_Multiview(
        train_head_camera,
        train_state,
        train_action,
        train_episode_ends,
        stats,
        chunk_size,
        history_step,
        predict_frame,
        temporal_downsample_rate,
        predict_only_last,
    )
    val_dataset = EpisodicDataset_Unified_Multiview(
        val_head_camera,
        val_state,
        val_action,
        val_episode_ends,
        stats,
        chunk_size,
        history_step,
        predict_frame,
        temporal_downsample_rate,
        predict_only_last,
    )

    if distributed:
        print("Using distributed sampler-----------------------------")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=(train_sampler is None),
        pin_memory=True,
        num_workers=4,
        sampler=train_sampler,

    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        sampler=val_sampler,
    )

    return train_dataloader, val_dataloader, train_sampler, stats


def load_data_unified_vision_tacile(
    data_dir,
    task_name='classify_tactile', 
    head_camera_type='D435',
    num_episodes=100,
    camera_name=['head_camra'],
    batch_size_train=32,
    chunk_size=100,
):
    zarr_path = os.path.join(
        data_dir, f"{task_name}_{head_camera_type}_{num_episodes}.zarr"
    )
    print(f"Loading data from {zarr_path}")
    zarr_root = zarr.open(zarr_path, mode="r")
    camera_image = []
    for camera_name in camera_name:
        camera_image.append(zarr_root[f"data/{camera_name}"])
    head_camera = np.stack(camera_image, axis=1) # N N_view C H W 
    state = zarr_root["data/state"]
    action = zarr_root["data/action"]
    episode_ends = zarr_root["meta/episode_ends"]
    stats = get_norm_stats(state, action)
    ll_tactile = zarr_root["data/ll_tactile"]
    lr_tactile = zarr_root["data/lr_tactile"]
    rl_tactile = zarr_root["data/rl_tactile"]
    rr_tactile = zarr_root["data/rr_tactile"] # Ns C H W 
    tactile = np.stack(
        [ll_tactile, lr_tactile, rl_tactile, rr_tactile], axis=1
    )  # N 4 C H W
    print(
        f"Train episodes: {len(episode_ends)}"
    )
    print(
        f"Train samples: {episode_ends[-1]}"
        
    )
    train_dataset =  EpisodicDataset_Unified_vision_tacile(
        head_camera,
        state,
        action,
        tactile,
        episode_ends,
        stats,
        chunk_size,
    )
    train_sampler = None
        
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=(train_sampler is None),
        pin_memory=True,
        num_workers=4,
        sampler=train_sampler,
        prefetch_factor=2,
    )
    return train_dataloader, train_sampler, stats


def load_data_unified_multiview_from_pt(
    data_dir='data/data_pt',
    task_name='bowls_stack',
    head_camera_type='D435',
    num_episodes=100,
    camera_name=['head_camra'],
    train_ratio=0.9,
    batch_size_train=32,
    batch_size_val=32,
    chunk_size=100,
    history_step=0,
    predict_frame=0,
    temporal_downsample_rate=1,
    predict_only_last=False,
    distributed=False):
    
    data_path = os.path.join(
        data_dir, f"{task_name}_{head_camera_type}"
    )
    train_val_split_file = data_path +f'/train_val_split_{num_episodes}.pt'
    episode_ends_file = data_path +f'/episode_ends.pt'
    train_index = torch.load(train_val_split_file,weights_only=False)['train_indices']
    val_index = torch.load(train_val_split_file,weights_only=False)['val_indices']
    episode_ends = list(torch.load(episode_ends_file,weights_only=False))[0]
    # episode_ends = torch.load(episode_ends_file,weights_only=False)['episode_ends'] 
    print(f"Loading data from {data_path}")
    print(f"Train episodes: {len(train_index)}, Validation episodes: {len(val_index)}")
    stats_path = data_path +f'/dataset_stats.pkl'
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f) # for normalization
    print(f'Loading stats from {stats_path}')
    # print(stats["qpos_mean"])
    # print(stats["qpos_std"]) 
    train_dataset = EpisodicDataset_Unified_Multiview_Simple(
        data_path,
        train_index,
        episode_ends,
        stats,
        chunk_size,
        history_step,
    )
    val_dataset = EpisodicDataset_Unified_Multiview_Simple(
        data_path,
        val_index,
        episode_ends,
        stats,
        chunk_size,
        history_step,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        sampler=None,
        prefetch_factor=2
    )
    
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        sampler=None,
        prefetch_factor=2
    )
    
    return train_dataloader, val_dataloader, None, stats

### helper functions
def convert_weigt(obj):
    newmodel = OrderedDict()
    for k, v in obj.items():
        if k.startswith("module."):
            newmodel[k[7:]] = v
        else:
            newmodel[k] = v
    return newmodel


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([~torch.optim.Optimizer]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (int):
            The number of steps for the warmup phase.
        num_training_steps (int):
            The total number of training steps.
        num_cycles (float, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (int, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule(optimizer, last_epoch: int = -1) -> LambdaLR:
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def normalize_data(action_data, stats, norm_type, data_type="action"):

    if norm_type == "minmax":
        action_max = torch.from_numpy(stats[data_type + "_max"]).float().to(action_data.device)
        action_min = torch.from_numpy(stats[data_type + "_min"]).float().to(action_data.device)
        action_data = (action_data - action_min) / (action_max - action_min) * 2 - 1
    elif norm_type == "gaussian":
        action_mean = torch.from_numpy(stats[data_type + "_mean"]).float().to(action_data.device)
        action_std = torch.from_numpy(stats[data_type + "_std"]).float().to(action_data.device)
        action_data = (action_data - action_mean) / action_std
    return action_data


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
import random


def set_seed(seed):
    random.seed(seed)  #
    np.random.seed(seed)  #
    torch.manual_seed(seed)  #
    torch.cuda.manual_seed(seed)  #
    torch.cuda.manual_seed_all(seed)  #


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


def create_multiview_video(image_list, save_path, fps=30):
    """
    Convert a list of multi-view images into a video and save it to the specified path.

    Args:
        image_list (list): A list containing T elements, where each element is a NumPy array of shape (k, c, h, w),
                           representing k views of RGB images.
        save_path (str): The path to save the video file, the filename should end with '.mp4' or other video formats.
        fps (int): Frames per second of the video, default is 30.
    """
    T = len(image_list)  # Number of frames
    k, c, h, w = image_list[
        0
    ].shape  # Get the shape of each element, k is the number of views, c is the channels, h and w are height and width

    # Ensure the input data format
    assert c == 3, "Each image must be RGB (3 channels)"

    # Output resolution, width is the sum of the widths of k views, height remains unchanged
    output_width = k * w
    output_height = h

    # Create video writer using OpenCV, specify the save path, codec (e.g., MP4V), fps and resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use mp4 codec
    video_writer = cv2.VideoWriter(
        save_path, fourcc, fps, (output_width, output_height)
    )

    # Iterate over each frame
    for t in range(T):
        # Get all views for the t-th frame
        frame = image_list[t]  # shape: (k, c, h, w)

        # Horizontally concatenate k views
        views = []
        for i in range(k):
            # Extract the i-th view, ensure it's in (h, w, c) format
            img = (
                frame[i].transpose(1, 2, 0) * 255
            )  # Convert from (c, h, w) to (h, w, c)
            views.append(img)

        # Horizontally concatenate the images of k views, resulting in (h, k*w, 3)
        combined_image = np.concatenate(views, axis=1)

        # Ensure image data type is uint8 and pixel values are between 0 and 255
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

        # Convert RGB image to BGR (OpenCV uses BGR format)
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

        # Write the concatenated image to the video
        video_writer.write(combined_image)

    # Release the video writer
    video_writer.release()


import pickle


def extract_and_save_subset(data_dir, task_name, head_camera_type, num_episodes=20):
    zarr_path = os.path.join(data_dir, f"{task_name}_{head_camera_type}_100.zarr")
    print(f"Loading data from {zarr_path}")
    zarr_root = zarr.open(zarr_path, mode="r")
    head_camera = zarr_root["data/head_camera"]
    state = zarr_root["data/state"]
    action = zarr_root["data/action"]
    tcp_action = zarr_root["data/tcp_action"]
    episode_ends = zarr_root["meta/episode_ends"]

    sub_episode_ends = episode_ends[:num_episodes]
    sub_head_camera = head_camera[: sub_episode_ends[-1]]
    sub_state = state[: sub_episode_ends[-1]]
    sub_action = action[: sub_episode_ends[-1]]
    sub_tcp_action = tcp_action[: sub_episode_ends[-1]]

    save_dir = os.path.join(
        data_dir, f"{task_name}_{head_camera_type}_{num_episodes}.zarr"
    )
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    action_chunk_size = (100, sub_action.shape[1])
    state_chunk_size = (100, sub_state.shape[1])
    joint_chunk_size = (100, sub_tcp_action.shape[1])
    head_camera_chunk_size = (100, *sub_head_camera.shape[1:])
    zarr_data.create_dataset(
        "head_camera",
        data=sub_head_camera,
        chunks=head_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "tcp_action",
        data=sub_tcp_action,
        chunks=action_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=sub_state,
        chunks=state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=sub_action,
        chunks=joint_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=sub_episode_ends,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )
    print("action shape:", sub_action.shape)
    print(f"Saved data to {save_dir}")


if __name__ == "__main__":
    task_name = "put_bottles_dustbin"
    head_camera_type = "D435"
    num_episodes = 1000
    train_ratio = 0.99
    batch_size_train = 128  # 2 min for dataloader
    batch_size_val = 4
    chunk_size = 20
    history_step = 0
    predict_frame = 20
    temporal_downsample_rate = 5
    predict_only_last = False
    distributed = False
    camera_name=['head_camera']
    data_dir = '/attached/remote-home2/xhl/8_kaust_pj/RoboTwin/data/data_pt'
    train_dataloader, train_sampler, _, stats = load_data_unified_multiview_from_pt(
        data_dir,
        task_name,
        head_camera_type,
        num_episodes,
        camera_name,
        train_ratio,
        batch_size_train,
        batch_size_val,
        chunk_size,
        history_step)
    from tqdm import tqdm
    for i, (
        image_data,
        qpos_data,
        action_data,
        is_pad,
        future_imgs_data,
        is_pad_img,
    ) in enumerate(tqdm(train_dataloader)):
        image = image_data.cuda()
        # print(
        #     image_data.shape, # B his+1 N_view C H W
        #     qpos_data.shape, # B his+1 14
        #     action_data.shape,
        #     is_pad.shape,
        #     future_imgs_data.shape,
        #     is_pad_img.shape,
        # )
        # break   
    
    # train_dataloader, train_sampler, stats = load_data_unified_vision_tacile(
    #     data_dir,
    #     task_name,
    #     head_camera_type,
    #     num_episodes,
    #     camera_name, 
    #     batch_size_train,
    #     chunk_size,
    # )
    # print("Train dataloader:", len(train_dataloader))
    # for i, (
    #     image_data,
    #     qpos_data,
    #     action_data,
    #     is_pad,
    #     tacile_data,
    # ) in enumerate(train_dataloader):
    #     print(
    #         f"Batch {i}:",
    #         f"Image data shape: {image_data.shape}",  # (B, 1, N_view, C, H, W)
    #         f"Qpos data shape: {qpos_data.shape}",    # (B, 1, D)
    #         f"Action data shape: {action_data.shape}",  # (B, chunk_size, D)
    #         f"Is pad shape: {is_pad.shape}",          # (B, chunk_size)
    #         f"Tactile data shape: {tacile_data.shape}",  # (B, 1, 4, C, H, W)
    #     )
    #     break
    # DATA_DIR = '/attached/remote-home2/xhl/8_kaust_pj/RoboTwin/data/data_zarr'  # TODO: change this to the path of the zarr files
    # camera_names = ["head_camera", "front_camera", "left_camera", "right_camera"]
    # # camera_names = ["head_camera"]
    # train_dataloader, val_dataloader, train_sampler, stats = load_data_unified_multiview(
    #     DATA_DIR,
    #     task_name,
    #     head_camera_type,
    #     num_episodes,
    #     camera_names,
    #     train_ratio,
    #     batch_size_train,
    #     batch_size_val,
    #     chunk_size,
    #     history_step,
    #     predict_frame,
    #     temporal_downsample_rate,
    #     predict_only_last,
    #     distributed,
    # )
    # print("Train dataloader:", len(train_dataloader))
    # print("Val dataloader:", len(val_dataloader))
    
    # for i, (
    #     image_data,
    #     qpos_data,
    #     action_data,
    #     is_pad,
    #     future_imgs_data,
    #     is_pad_img,
    # ) in enumerate(train_dataloader):
    #     print(
    #         image_data.shape, # B his+1 N_view C H W
    #         qpos_data.shape, # B his+1 14
    #         action_data.shape,
    #         is_pad.shape,
    #         future_imgs_data.shape,
    #         is_pad_img.shape,
    #     )
    #     break
    
    # train_dataloader, val_dataloader, train_sampler, stats = load_data_unified(
    #     DATA_DIR,
    #     task_name,
    #     head_camera_type,
    #     num_episodes,
    #     train_ratio,
    #     batch_size_train,
    #     batch_size_val,
    #     chunk_size,
    #     history_step,
    #     predict_frame,
    #     temporal_downsample_rate,
    #     predict_only_last,
    #     distributed,
    # )
    # print("Train dataloader:", len(train_dataloader))
    # print("Val dataloader:", len(val_dataloader))
    # import time
    # from tqdm import tqdm
    # start_time = time.time()
    # for i, (image_data, qpos_data, action_data, is_pad, future_imgs_data, is_pad_img) in enumerate(tqdm(train_dataloader)):
    #     print(image_data.max(), image_data.min()) # 0~1
    #     # print(image_data.shape, qpos_data.shape, action_data.shape, is_pad.shape)
    #     break
    # for i, (
    #     image_data,
    #     qpos_data,
    #     action_data,
    #     is_pad,
    #     future_imgs_data,
    #     is_pad_img,
    # ) in enumerate(train_dataloader):
    #     print(
    #         image_data.shape,
    #         qpos_data.shape,
    #         action_data.shape,
    #         is_pad.shape,
    #         future_imgs_data.shape,
    #         is_pad_img.shape,
    #     )
    #     break

    # data_dir = "data_zarr"
    # camera_type = "D435"
    # task_name = "dual_bottles_pick_easy"
    # num_episodes = 20
    # extract_and_save_subset(data_dir, task_name, camera_type, num_episodes)
