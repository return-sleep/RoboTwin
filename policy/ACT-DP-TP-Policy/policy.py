import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from diffusers import DDIMScheduler, DDPMScheduler
from detr.main import *
import numpy as np
import mediapy as media
from collections import deque
from utils_robotwin import normalize_data, tensor2numpy, kl_divergence, RandomShiftsAug
import os

# from cosmos_tokenizer.video_lib import CausalVideoTokenizer
# from cosmos_tokenizer.image_lib import ImageTokenizer


def get_tokenizer(model_name):
    print(f"Loading tokenizer {model_name}")
    current_dir = os.path.dirname(__file__)
    checkpoint_enc = (
        f"{current_dir}/Cosmos-Tokenizer/pretrained_ckpts/{model_name}/encoder.jit"
    )
    checkpoint_dec = (
        f"{current_dir}/Cosmos-Tokenizer/pretrained_ckpts/{model_name}/decoder.jit"
    )
    model_type = model_name[18]  # I or V
    if model_type == "I":
        encoder = ImageTokenizer(checkpoint_enc=checkpoint_enc)
        decoder = ImageTokenizer(checkpoint_dec=checkpoint_dec)
    elif model_type == "V":
        encoder = CausalVideoTokenizer(checkpoint_enc=checkpoint_enc)
        decoder = CausalVideoTokenizer(checkpoint_dec=checkpoint_dec)

    for param in encoder.parameters():  # frozen
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False
    return encoder, decoder


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        print(f"KL Weight {self.kl_weight}")
        self.args = args_override

    def __call__(
        self, qpos, image, actions=None, is_pad=None, is_training=False, instances=None
    ):
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if actions is not None:  # training time
            # image = self.aug(image) if is_training else image # disbale aug
            image = normalize(image)
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            if self.args["segmentation"]:
                # ACT with segmentation
                a_hat, is_pad_hat, (mu, logvar), (mask_classes, outputs_seg_masks) = (
                    self.model(qpos, image, env_state, actions, is_pad)
                )
            else:
                # Vanilla ACT
                a_hat, is_pad_hat, (mu, logvar) = self.model(
                    qpos, image, env_state, actions, is_pad
                )

            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")

            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]

            if self.args["segmentation"]:
                targets = self.prepare_targets(instances, image)
                losses_seg = self.criterion(
                    {"pred_logits": mask_classes, "pred_masks": outputs_seg_masks},
                    targets,
                )

                for k in list(losses_seg.keys()):
                    if k in self.criterion.weight_dict:
                        losses_seg[k] *= (
                            self.criterion.weight_dict[k] * self.segloss_weight
                        )
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses_seg.pop(k)
                loss_dict.update(losses_seg)
                loss_dict["loss"] = (
                    loss_dict["l1"]
                    + loss_dict["kl"] * self.kl_weight
                    + loss_dict["loss_ce"]
                    + loss_dict["loss_mask"]
                    + loss_dict["loss_dice"]
                )
            else:
                loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            image = normalize(image)
            if self.args["segmentation"]:
                a_hat, is_pad_hat, (mu, logvar), (mask_classes, outputs_seg_masks) = (
                    self.model(qpos, image, env_state, actions, is_pad)
                )
                for mask_cls_result, mask_pred_result in zip(
                    mask_classes, outputs_seg_masks
                ):
                    sem_seg = torch.zeros(
                        (mask_pred_result.shape[1], mask_pred_result.shape[2], 3),
                        device=mask_pred_result.device,
                    )
                    keep = mask_cls_result.softmax(-1)[:, 0] > 0.5
                    for ii, mask in enumerate(mask_pred_result):
                        if keep[ii]:
                            sem_seg[mask.sigmoid() > 0.5, :] = torch.tensor(
                                self.colors[ii],
                                device=mask_pred_result.device,
                                dtype=sem_seg.dtype,
                            )
                    self.seg = sem_seg.cpu().numpy()
                # sem_seg = self.semantic_inference(mask_cls_result, mask_pred_result)
                #    import matplotlib.pyplot as plt
                #    plt.subplot(122)
                #    plt.imshow(sem_seg.cpu().numpy()/255)
                #    plt.savefig('seg.png')
                #    import pdb; pdb.set_trace()
            else:
                a_hat, _, (_, _) = self.model(
                    qpos, image, env_state
                )  # no action, sample from prior

            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    def prepare_targets(self, targets, image):
        # h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            targets_per_image = targets_per_image.to(image.device)
            gt_masks = targets_per_image.gt_masks
            # padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device)
            # padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": gt_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    # for robotwin
    def reset_obs(self, stats, norm_type):
        self.stats = stats
        self.norm_type = norm_type

    def update_obs(self, obs):
        self.obs_image = (
            torch.from_numpy(obs["head_cam"]).unsqueeze(0).unsqueeze(0).float().cuda()
        )  # 1 1 C H W 0~1
        obs_qpos = torch.from_numpy(obs["agent_pos"]).unsqueeze(0).float().cuda()
        self.obs_qpos = normalize_data(
            obs_qpos, self.stats, "gaussian", data_type="qpos"
        )  # qpos mean std

    def get_action(self):
        a_hat = self(self.obs_qpos, self.obs_image).detach().cpu().numpy()  # B T K
        # unnormalize
        if self.norm_type == "minmax":
            a_hat = (a_hat + 1) / 2 * (
                self.stats["action_max"] - self.stats["action_min"]
            ) + self.stats["action_min"]
        elif self.norm_type == "gaussian":
            a_hat = a_hat * self.stats["action_std"] + self.stats["action_mean"]
        return a_hat[0]  # chunksize 14


class ACTDiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACTDiffusion_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        print(args_override.keys())
        print(f"KL Weight {self.kl_weight}")
        if "sim" in args_override["task_name"]:  # for aloha env
            self.aug = RandomShiftsAug(15, 20)  # TODO acording to the task
        else:
            self.aug = RandomShiftsAug(8, 10)  # for robotwin env
        self.history_steps = args_override["history_step"]
        self.obs_image = deque(maxlen=self.history_steps + 1)
        self.obs_qpos = deque(maxlen=self.history_steps + 1)
        # diffusion setup
        self.num_inference_steps = args_override["num_inference_steps"]
        self.num_queries = args_override["num_queries"]
        num_train_timesteps = args_override["num_train_timesteps"]
        prediction_type = args_override["prediction_type"]
        beta_schedule = args_override["beta_schedule"]
        noise_scheduler = (
            DDIMScheduler if args_override["schedule_type"] == "DDIM" else DDPMScheduler
        )
        noise_scheduler = noise_scheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

        self.noise_scheduler = noise_scheduler
        self.loss_type = args_override["loss_type"]
        print("num_train_timesteps", {args_override["num_train_timesteps"]})
        print("schedule_type", {args_override["schedule_type"]})
        print("beta_schedule", {args_override["beta_schedule"]})
        print("prediction_type", {args_override["prediction_type"]})
        print(f"Loss Type {self.loss_type}")

    def train_model(self, qpos, image, actions, is_pad=None):
        """
        qpos: B his+1 14
        image: B his+1 N_view 3 H W 
        """
        env_state = None
        noise = torch.randn_like(actions).to(actions.device)
        bsz = actions.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=actions.device,
        )
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        pred, is_pad_hat, [mu, logvar] = self.model(
            qpos, image, env_state, noisy_actions, is_pad, denoise_steps=timesteps
        )

        pred_type = self.noise_scheduler.config.prediction_type

        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = actions
        elif pred_type == "v_prediction":
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = (
                self.noise_scheduler.alpha_t[timesteps],
                self.noise_scheduler.sigma_t[timesteps],
            )
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * actions
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss_dict = {}
        if self.loss_type == "l2":
            loss = F.mse_loss(pred, target, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction="none")
        diffusion_loss = (loss * ~is_pad.unsqueeze(-1)).mean()
        diffusion_loss_name = pred_type + "_diffusion_loss_" + self.loss_type
        loss_dict[diffusion_loss_name] = diffusion_loss

        if mu is not None and logvar is not None:  # for CVAE module
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = (
                loss_dict[diffusion_loss_name] + loss_dict["kl"] * self.kl_weight
            )
        else:
            loss_dict["loss"] = loss_dict[diffusion_loss_name]
        return loss_dict

    # ===================inferece ===============
    def conditional_sample(self, qpos, image, is_pad):
        """
        diffusion process to generate actions
        """
        if len(image.shape) == 5:  # B N C H W
            qpos = qpos.unsqueeze(1)
            image = image.unsqueeze(1)
        env_state = None
        model = self.model
        scheduler = self.noise_scheduler
        batch = image.shape[0]
        action_shape = (batch, self.num_queries, 14)
        actions = torch.randn(action_shape, device=qpos.device, dtype=qpos.dtype)
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            timesteps = torch.full((batch,), t, device=qpos.device, dtype=torch.long)
            model_output, is_pad_hat, [mu, logvar] = model(
                qpos,
                image,
                env_state,
                actions,
                is_pad,
                denoise_steps=timesteps,
                is_training=False,
            )
            actions = scheduler.step(model_output, t, actions).prev_sample
        return actions

    def __call__(self, qpos, image, actions=None, is_pad=None, is_training=True):
        # qpos: B D
        # image: B Num_view C H W
        # actions: B T K
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if actions is not None:  # training time
            image = self.aug(image) if is_training else image
            image = normalize(image)
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]
            loss_dict = self.train_model(qpos, image, actions, is_pad)
            return loss_dict
        else:  # inference time
            image = normalize(image)
            a_hat = self.conditional_sample(qpos, image, is_pad)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


## use visual tokenization
"""
Input:
        A unified dataset for all the datasets
        image_data [0~1]: history_steps+1 Num_view C H W
        qpos_data [normalized]: history_steps+1 D
        action_data [raw]: chunk_size D
        is_pad: chunk_size
        future_imgs_data [0~1]: predict_frame Num_view C H W
        is_pad_img : predict_frame
"""


class ACTPolicyDiffusion_with_Token_Prediction(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model = build_diffusion_tp_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.camera_num = len(args_override["camera_names"])
        self.kl_weight = args_override["kl_weight"]
        print(f"KL Weight {self.kl_weight}")
        # memory buffer
        self.history_steps = args_override["history_step"]
        self.obs_image = deque(maxlen=self.history_steps + 1)
        self.obs_qpos = deque(maxlen=self.history_steps + 1)
        # visual tokenization
        if "sim" in args_override["task_name"]:  # for aloha env
            self.aug = RandomShiftsAug(15, 20)
        else:
            self.aug = RandomShiftsAug(8, 10)  # for robotwin env
        # tokenizer model and shape
        self.tokenizer_model_type = args_override["tokenizer_model_name"][17:19]  # VI
        self.tokenizer_enc, self.tokenizer_dec = get_tokenizer(
            args_override["tokenizer_model_name"]
        )
        self.token_dim = args_override["token_dim"]
        self.num_temporal_token = self.model.num_temporal_token
        self.token_h = (
            args_override["image_height"]
            // args_override["image_downsample_rate"]
            // args_override["tokenizer_model_spatial_rate"]
            // args_override["resize_rate"]
        )
        self.token_w = (
            args_override["image_width"]
            // args_override["image_downsample_rate"]
            // args_override["tokenizer_model_spatial_rate"]
            // args_override["resize_rate"]
        )
        print(
            "token shape",
            "token_h",
            self.token_h,
            "token_w",
            self.token_w,
            "token_dim",
            self.token_dim,
        )
        # video prediction hyperparameters
        self.temporal_compression = args_override[
            "tokenizer_model_temporal_rate"
        ]  # temporal compression
        self.predict_only_last = args_override["predict_only_last"]
        self.prediction_weight = args_override["prediction_weight"]
        self.imitate_weight = args_override["imitate_weight"]
        self.predict_frame = args_override["predict_frame"]
        self.temporal_downsample_rate = args_override[
            "temporal_downsample_rate"
        ]  # uniformly sample
        self.resize_rate = args_override["resize_rate"]
        print("tokenizer_model_type", self.tokenizer_model_type)
        print("predict_frame", self.predict_frame)
        print("prediction_weight", self.prediction_weight)
        print("imitate_weight", self.imitate_weight)

        # diffusion hyperparameters
        self.num_inference_steps = args_override["num_inference_steps"]
        self.num_queries = args_override["num_queries"]
        num_train_timesteps = args_override["num_train_timesteps"]
        prediction_type = args_override["prediction_type"]
        beta_schedule = args_override["beta_schedule"]
        noise_scheduler = (
            DDIMScheduler if args_override["schedule_type"] == "DDIM" else DDPMScheduler
        )
        noise_scheduler = noise_scheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )
        self.noise_scheduler = noise_scheduler
        pred_type = self.noise_scheduler.config.prediction_type
        self.loss_type = args_override["loss_type"]
        self.diffusion_loss_name = pred_type + "_diffusion_loss_" + self.loss_type

        print("num_train_timesteps", {args_override["num_train_timesteps"]})
        print("schedule_type", {args_override["schedule_type"]})
        print("beta_schedule", {args_override["beta_schedule"]})
        print("prediction_type", {args_override["prediction_type"]})
        print(f"Loss Type {self.loss_type}")

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.vis_idx = 0

    def train_model(self, qpos, image, env_state, actions, is_pad, is_image_pad=None):
        # qpos: B T' D;  T' = history_steps+1
        # image: B T+1 N C H W ,T = history_steps+1+predict_frame 0~1
        # actions: B H D, H = chunk_size
        # is_pad: B H, is_valid
        # is_image_pad: B predict_frame
        env_state = None
        bsz = actions.shape[0]
        is_tokens_pad = torch.ones(
            bsz, self.num_temporal_token, device=actions.device, dtype=torch.bool
        )  # B T/t  length after temporal compression
        if self.predict_only_last:
            is_tokens_pad = is_image_pad
        else:
            valid_is_tokens_pad = is_image_pad[
                :, :: self.temporal_compression
            ]  # avoid meaningless token
            is_tokens_pad[:, : valid_is_tokens_pad.shape[-1]] = valid_is_tokens_pad

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=actions.device,
        )

        # image tokenization ; another image normalization for resnet
        current_image_norm = self.normalize(
            image[:, 0 : self.history_steps + 1]  # image[:, 0:1]
        )  # B 1 N C H W TODO image[:, self.history_steps,self.history_steps+1]
        # TOOD only compress current and future frames image[:, self.history_steps:,:]
        image_tokens = self.get_visual_token(
            image[
                :, self.history_steps :, 0:1, :, :: self.resize_rate, :: self.resize_rate
            ]  # resize future frame
        )  # B T+1 N C H W
        # image_tokens = self.get_visual_token(
        #     image[..., :: self.resize_rate, :: self.resize_rate]  # resize future frame
        # )  # B T/t + 1 N D H' W' including history_steps+1+predict_frame
        current_image_tokens = image_tokens[:, 0:1]  # B 1 N D H' W' useless
        future_image_tokens = image_tokens[:, 1:]  # B T N D H' W'
        self.image_tokens_shape = future_image_tokens.shape  # B T N D H' W'
        # TODO can use differnent noise_scheduler for image token & actions
        actions_noise = torch.randn_like(actions).to(actions.device)
        token_noise = torch.randn_like(future_image_tokens).to(
            future_image_tokens.device
        )
        noisy_actions = self.noise_scheduler.add_noise(
            actions, actions_noise, timesteps
        )
        noise_tokens = self.noise_scheduler.add_noise(
            future_image_tokens, token_noise, timesteps
        )  # future image token
        # use detr-diffusion model to predict actions & image tokens
        a_hat, is_pad_hat, pred_token, (mu, logvar) = self.model(
            qpos,  # B his+1 D
            (current_image_norm, current_image_tokens),  # B 1 N C H W
            env_state,
            actions,  # B T D
            is_pad,
            noisy_actions,  # B T D
            noise_tokens,  # B T' N_view D H' W'
            is_tokens_pad,
            denoise_steps=timesteps,
        )

        # prediction type
        pred_type = self.noise_scheduler.config.prediction_type

        if pred_type == "epsilon":
            target_action = actions_noise
            target_token = token_noise
        elif pred_type == "sample":
            target_action = actions
            target_token = future_image_tokens
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # calculate diffusion loss
        loss_dict = {}
        if self.loss_type == "l2":
            loss = F.mse_loss(a_hat, target_action, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(a_hat, target_action, reduction="none")
        diffusion_loss = (loss * ~is_pad.unsqueeze(-1)).mean()
        loss_dict[self.diffusion_loss_name] = diffusion_loss
        # just vis diffusion l2 loss
        diffusion_l2 = F.mse_loss(a_hat, target_action, reduction="none")
        diffusion_l2 = (diffusion_l2 * ~is_pad.unsqueeze(-1)).mean().detach()
        loss_dict["diffusion_l2"] = diffusion_l2

        tokens_loss = F.mse_loss(
            pred_token, target_token, reduction="none"
        )  # B T N D H' W'
        is_tokens_pad = (
            is_tokens_pad.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )  # B T N D H' W'
        tokens_loss = (tokens_loss * ~is_tokens_pad).mean()
        loss_dict["loss_prediction"] = tokens_loss

        if mu is not None and logvar is not None:
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = (
                loss_dict[self.diffusion_loss_name] * self.imitate_weight
                + loss_dict["kl"] * self.kl_weight
                + loss_dict["loss_prediction"] * self.prediction_weight
            )
        else:
            loss_dict["loss"] = (
                loss_dict[self.diffusion_loss_name] * self.imitate_weight
                + loss_dict["loss_prediction"] * self.prediction_weight
            )

        return loss_dict, (current_image_tokens, target_token, pred_token)

    def conditional_sample(self, qpos, image):
        # qpos: B 1 D or B D
        # image: B 1 N C H W or B N C H W
        if len(image.shape) == 5:  # B N C H W single frame
            qpos = qpos.unsqueeze(1)
            image = image.unsqueeze(1)
        env_state = None
        model = self.model
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)
        # process image observation
        current_image_norm = self.normalize(image[:, 0 : self.history_steps + 1])  # B 1 N C H W
        # initial noise action & token
        batch = image.shape[0]
        action_shape = (batch, self.num_queries, 14)
        actions = torch.randn(action_shape, device=qpos.device, dtype=qpos.dtype)
        tokens = None  # TODO discard token prediction while evaluation
        for t in scheduler.timesteps:
            timesteps = torch.full((batch,), t, device=qpos.device, dtype=torch.long)
            model_action_output, is_pad_hat, model_token_output, (mu, logvar) = model(
                qpos,
                (current_image_norm, None),
                env_state,
                None,
                None,
                actions,
                tokens,
                None,
                denoise_steps=timesteps,
            )
            actions = scheduler.step(model_action_output, t, actions).prev_sample
        return actions, tokens, mu, logvar

    def __call__(
        self,
        qpos,
        image,
        actions=None,
        is_pad=None,
        future_imgs=None,
        is_pad_img=None,
        is_training=True,
        save_rec=False,
    ):
        env_state = None
        if actions is not None:  # training time
            # print(image.shape, future_imgs.shape) B 1 N C H W & B H N C H W
            all_image = torch.cat(
                [image, future_imgs], dim=1
            )  # B H+1+T' N C H W same resize maybe just use image_sample_size
            all_image = self.aug(all_image) if is_training else all_image
            loss_dict, (current_image_tokens, target_token, pred_token) = (
                self.train_model(
                    qpos, all_image, env_state, actions, is_pad, is_pad_img
                )
            )  # B H

            if save_rec == True:  # show the visulization result
                # a_hat, pred_token, _, _ = self.conditional_sample(qpos, image) # generate bug?
                # tokens_loss = F.mse_loss(pred_token, target_token)
                # print('conditional tokens_loss', tokens_loss) # not predict from noise
                print("is_image_pad rate ", is_pad_img[0].sum() / 20)
                raw_videos = (
                    all_image[:, :, 0].permute(0, 2, 1, 3, 4) * 2 - 1
                )  # B T C H W -> B C T H W -1,1
                rec_videos = self.generate_video_by_codes(
                    current_image_tokens, pred_token
                )
                rec_gt_videos = self.generate_video_by_codes(
                    current_image_tokens, target_token
                )
                error_videos = (rec_gt_videos - rec_videos).clip(-1, 1)

                raw_videos = tensor2numpy(raw_videos)[0]  # C T H W
                rec_videos = tensor2numpy(rec_videos)[0]  # T H W C
                rec_gt_videos = tensor2numpy(rec_gt_videos)[0]  # T H W C
                error_videos = tensor2numpy(error_videos)[0]  # T H W C
                vis_video = np.concatenate(
                    [raw_videos, rec_gt_videos, rec_videos, error_videos], axis=1
                )  # T H*N W C
                return loss_dict, vis_video
            else:
                return loss_dict

        else:  # inference time
            qpos = qpos  # B 1 D or B D
            image = image  # B 1 N C H W or B N C H W
            # print(image.shape, image.max(), image.min())
            a_hat, pred_token, _, _ = self.conditional_sample(qpos, image)
            # print('prediction action', a_hat.shape)
            return a_hat  # B H D

    # visual tokenization generate tokens & reconstruct video
    def get_visual_token(self, input_tensor):
        # input_tensor: B T N C H W range [0,1] -> [-1,1]
        input_tensor = input_tensor * 2 - 1  # B T N C H W
        input_tensor = input_tensor.permute(0, 2, 3, 1, 4, 5)  # B N C T H W
        horizon = input_tensor.shape[3]
        C, H, W = input_tensor.shape[2], input_tensor.shape[4], input_tensor.shape[5]
        self.num_view = input_tensor.shape[1]
        codes_list = []
        # refer to Cosmos-Tokenizer/cosmos_tokenizer/video_lib.py Line 103
        for view_idx in range(
            self.num_view
        ):  # deal with each view for video tokenization B C T H W
            if self.tokenizer_model_type == "DV":  # encoder for video tokenization
                (indices, codes) = self.tokenizer_enc._enc_model(
                    input_tensor[:, view_idx]  # B C T H W
                )[
                    :-1
                ]  # B D T' H' W'
            elif self.tokenizer_model_type == "DI":  # encoder for image tokenization
                input_tensor_image = (
                    input_tensor[:, view_idx].permute(0, 2, 1, 3, 4).view(-1, C, H, W)
                )  # B C T H W -> B*T C H W
                (indices, codes) = self.tokenizer_enc._enc_model(input_tensor_image)[
                    :-1
                ]  # B*T D H' W'
                codes = codes.view(
                    -1, horizon, codes.shape[1], codes.shape[2], codes.shape[3]
                ).permute(
                    0, 2, 1, 3, 4
                )  # B T+1 D H' W' -> B D T+1 H' W'
            elif (
                self.tokenizer_model_type == "CV"
            ):  # encoder for video tokenization TODO check
                (codes,) = self.tokenizer_enc._enc_model(input_tensor[:, view_idx])[
                    :-1
                ]  # B 16 T' H' W'
                codes = codes / 15.0  # nomalize to [-1,1] HardCode
                # TODO codes should normlize to [-1,1]
            elif self.tokenizer_model_type == "CI":
                input_tensor_image = (
                    input_tensor[:, view_idx].permute(0, 2, 1, 3, 4).view(-1, C, H, W)
                )  # B C T H W -> B*T C H W
                (codes,) = self.tokenizer_enc._enc_model(input_tensor_image)[
                    :-1
                ]  # B*T D H' W'
                codes = codes.view(
                    -1, horizon, codes.shape[1], codes.shape[2], codes.shape[3]
                ).permute(
                    0, 2, 1, 3, 4
                )  # B T+1 D H' W' -> B D T+1 H' W'
                codes = codes / 15.0  # nomalize to [-1,1] HardCode
            codes_list.append(codes.detach())  # Important
        codes = torch.stack(codes_list, dim=1)  # B N D  T'  H' W' ->  # B T+1 N D H' W'
        codes = codes.permute(0, 3, 1, 2, 4, 5).float()  # B T+1 N D H' W'
        return codes

    def generate_video_by_codes(self, current_codes, pred_codes):
        # Input: B T'+1 D H' W' simgle view video tokens
        # Output: B C T+1  H W range [-1,1]
        codes = torch.cat([current_codes, pred_codes], dim=1)[
            :, :, 0
        ]  # B T+1 N D H' W' -> B T'+1 D H' W'# HARD CODE
        codes = codes.permute(0, 2, 1, 3, 4).to(dtype=torch.bfloat16)  # B D T'+1 H' W'
        # suitable for DV only
        if self.tokenizer_model_type == "DV":
            h = self.tokenizer_dec._dec_model.post_quant_conv(
                codes
            )  # problem shoudl fixed TODO
        elif self.tokenizer_model_type == "CV":
            h = codes * 15  # unnormalize to original
        reconstructed_videos = self.tokenizer_dec._dec_model.decoder(
            h
        ).detach()  # B C T+1 H W -1,1

        return reconstructed_videos

    # For ROBOTWIN
    def reset_obs(self, stats=None, norm_type="minmax"):
        self.obs_image.clear()
        self.obs_qpos.clear()
        self.stats = stats
        self.norm_type = norm_type

    def update_obs(self, obs):
        image_data = (
            torch.from_numpy(obs["head_cam"]).unsqueeze(0).unsqueeze(0).float().cuda()
        )  # B 1 C H W 0~1
        obs_qpos = torch.from_numpy(obs["agent_pos"]).unsqueeze(0).float().cuda()  # B D
        qpos_data = normalize_data(
            obs_qpos, self.stats, "gaussian", data_type="qpos"
        )  # qpos mean std

        if len(self.obs_image) == 0:
            for _ in range(self.history_steps + 1):
                self.obs_image.append(image_data)  # B T N C H W
                self.obs_qpos.append(qpos_data)
        else:
            self.obs_image.append(image_data)
            self.obs_qpos.append(qpos_data)

    def get_action(self):
        stacked_obs_image = torch.stack(
            list(self.obs_image), dim=1
        )  # 1 n+1 1 3 H W raw
        stacked_obs_qpos = torch.stack(list(self.obs_qpos), dim=1)  # 1 n+1 14
        a_hat = (
            self(stacked_obs_qpos, stacked_obs_image).detach().cpu().numpy()
        )  # 1 chunksize 14
        if self.norm_type == "minmax":
            a_hat = (a_hat + 1) / 2 * (
                self.stats["action_max"] - self.stats["action_min"]
            ) + self.stats["action_min"]
        elif self.norm_type == "gaussian":
            a_hat = a_hat * self.stats["action_std"] + self.stats["action_mean"]
        return a_hat[0]  # chunksize 14


class ACTPolicyDiffusion_with_Pixel_Prediction(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model = build_diffusion_pp_model_and_optimizer(args_override)  # TODO
        self.model = model  # CVAE decoder
        self.kl_weight = args_override["kl_weight"]
        print(f"KL Weight {self.kl_weight}")

        # memory buffer
        self.history_steps = args_override["history_step"]
        self.obs_image = deque(maxlen=self.history_steps + 1)
        self.obs_qpos = deque(maxlen=self.history_steps + 1)
        # self.obs_depth = deque(maxlen=self.history_steps+1)
        # visual tokenization
        if "sim" in args_override["task_name"]:
            self.aug = RandomShiftsAug(15, 20)  #
            self.aug = RandomShiftsAug(8, 10)
        self.predict_only_last = args_override["predict_only_last"]
        self.prediction_weight = args_override["prediction_weight"]
        self.imitate_weight = args_override["imitate_weight"]
        self.predict_frame = args_override["predict_frame"]
        self.temporal_downsample_rate = args_override["temporal_downsample_rate"]
        self.resize_rate = args_override["resize_rate"]
        self.image_height = args_override["image_height"]
        self.image_width = args_override["image_width"]
        # T N C H W
        self.future_images_shape = (
            args_override["predict_frame"] // args_override["temporal_downsample_rate"],
            len(args_override["camera_names"]),
            3,
            self.image_height // self.resize_rate,
            self.image_width // self.resize_rate,
        )
        print("predict_frame", self.predict_frame)
        print("prediction_weight", self.prediction_weight)
        print("imitate_weight", self.imitate_weight)
        # diffusion step
        self.num_inference_steps = args_override["num_inference_steps"]
        self.num_queries = args_override["num_queries"]
        num_train_timesteps = args_override["num_train_timesteps"]
        prediction_type = args_override["prediction_type"]
        beta_schedule = args_override["beta_schedule"]
        noise_scheduler = (
            DDIMScheduler if args_override["schedule_type"] == "DDIM" else DDPMScheduler
        )
        noise_scheduler = noise_scheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )
        self.noise_scheduler = noise_scheduler
        pred_type = self.noise_scheduler.config.prediction_type
        self.loss_type = args_override["loss_type"]
        self.diffusion_loss_name = pred_type + "_diffusion_loss_" + self.loss_type

        print("num_train_timesteps", {args_override["num_train_timesteps"]})
        print("schedule_type", {args_override["schedule_type"]})
        print("beta_schedule", {args_override["beta_schedule"]})
        print("prediction_type", {args_override["prediction_type"]})
        print(f"Loss Type {self.loss_type}")
        # discard resnet
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.vis_idx = 0

    def train_model(self, qpos, image, env_state, actions, is_pad, is_image_pad=None):
        # qpos: B T' D;  T' = history_steps+1
        # image: B T+1 N C H W ,T = history_steps+1+predict_frame 0~1s
        # actions: B H D H = chunk_size
        # is_pad: B H
        # is_image_pad: B predict_frame
        env_state = None
        bsz = actions.shape[0]

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=actions.device,
        )
        current_image_norm = self.normalize(image[:, 0:1])
        future_images = (
            image[:, 1:, :, :, :: self.resize_rate, :: self.resize_rate] * 2 - 1
        )  # B T N C H W scale to [-1,1]
        # diffusion process
        actions_noise = torch.randn_like(actions).to(actions.device)
        pixel_noise = torch.randn_like(future_images).to(future_images.device)
        noisy_actions = self.noise_scheduler.add_noise(
            actions, actions_noise, timesteps
        )
        noise_pixel = self.noise_scheduler.add_noise(
            future_images, pixel_noise, timesteps
        )  # future image token
        # predict clean data
        a_hat, is_pad_hat, pred_images, (mu, logvar) = self.model(
            qpos,
            (current_image_norm, None),
            env_state,
            actions,
            is_pad,
            noisy_actions,
            noise_pixel,
            is_image_pad,
            denoise_steps=timesteps,
        )

        # prediction type
        pred_type = self.noise_scheduler.config.prediction_type

        if pred_type == "epsilon":
            target_action = actions_noise
            target_images = pixel_noise
        elif pred_type == "sample":
            target_action = actions
            target_images = future_images
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # calculate diffusion loss
        loss_dict = {}
        if self.loss_type == "l2":
            loss = F.mse_loss(a_hat, target_action, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(a_hat, target_action, reduction="none")
        diffusion_loss = (loss * ~is_pad.unsqueeze(-1)).mean()
        loss_dict[self.diffusion_loss_name] = diffusion_loss
        diffusion_l2 = F.mse_loss(a_hat, target_action, reduction="none")
        diffusion_l2 = (diffusion_l2 * ~is_pad.unsqueeze(-1)).mean().detach()
        loss_dict["diffusion_l2"] = diffusion_l2

        pixel_loss = F.mse_loss(
            pred_images, target_images, reduction="none"
        )  # B T N C H W
        is_image_pad = (
            is_image_pad.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )  # B T N C H W
        pixel_loss = (pixel_loss * ~is_image_pad).mean()
        loss_dict["loss_prediction_pixel"] = pixel_loss

        if mu is not None and logvar is not None:
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = (
                loss_dict[self.diffusion_loss_name] * self.imitate_weight
                + loss_dict["kl"] * self.kl_weight
                + loss_dict["loss_prediction_pixel"] * self.prediction_weight
            )
        else:
            loss_dict["loss"] = (
                loss_dict[self.diffusion_loss_name] * self.imitate_weight
                + loss_dict["loss_prediction_pixel"] * self.prediction_weight
            )

        return loss_dict, (image[:, 0:1], target_images, pred_images)

    def conditional_sample(self, qpos, image):
        # qpos: B 1 D
        # image: B 1 N C H W
        if len(image.shape) == 5:
            qpos = qpos.unsqueeze(1)
            image = image.unsqueeze(1)
        env_state = None
        model = self.model
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)
        # process image observation
        current_image_norm = self.normalize(image[:, 0:1])  # B 1 N C H W
        batch = image.shape[0]
        action_shape = (batch, self.num_queries, 14)
        future_images_shape = (batch, *self.future_images_shape)  # B T N 3 H' W'
        actions = torch.randn(action_shape, device=qpos.device, dtype=qpos.dtype)
        pixels = torch.randn(future_images_shape, device=qpos.device, dtype=qpos.dtype)
        # denoise
        for t in scheduler.timesteps:
            timesteps = torch.full((batch,), t, device=qpos.device, dtype=torch.long)
            model_action_output, is_pad_hat, model_pixel_output, (mu, logvar) = model(
                qpos,
                (current_image_norm, None),
                env_state,
                None,
                None,
                actions,
                pixels,
                None,
                denoise_steps=timesteps,
            )
            actions = scheduler.step(model_action_output, t, actions).prev_sample
            pixels = scheduler.step(model_pixel_output, t, pixels).prev_sample
        return actions, pixels, mu, logvar

    def __call__(
        self,
        qpos,
        image,
        actions=None,
        is_pad=None,
        future_imgs=None,
        is_pad_img=None,
        is_training=True,
    ):
        env_state = None
        if actions is not None:  # training time
            all_image = torch.cat(
                [image, future_imgs], dim=1
            )  # B T N C H W same resize maybe just use image_sample_size
            all_image = self.aug(all_image) if is_training else all_image
            loss_dict, (current_image, target_images, pred_images) = self.train_model(
                qpos, all_image, env_state, actions, is_pad, is_pad_img
            )  # B H
            return loss_dict
        else:
            qpos = qpos  # B 1 D
            image = image  # B 1 N C H W 0~1
            a_hat, pred_images, _, _ = self.conditional_sample(qpos, image)
            return a_hat  # B H D


#  discard
class ACTPolicy_NextFrame(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_NF_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        self.nextframe_weight = 1
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if actions is not None:  # training time
            curr_image = image[:, 0:1]
            next_image = image[:, 1:]
            curr_image = normalize(curr_image)
            image = torch.cat(
                [curr_image, next_image], dim=1
            )  # B T C H W normalize currernt image not funture

            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar), (obs_preds, obs_targets) = self.model(
                qpos, image, env_state, actions, is_pad
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            obs_loss = ((obs_preds.sigmoid() - obs_targets) ** 2).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["next_frame"] = obs_loss
            loss_dict["loss"] = (
                loss_dict["l1"]
                + loss_dict["kl"] * self.kl_weight
                + loss_dict["next_frame"] * self.nextframe_weight
            )
            return loss_dict
        else:  # inference time
            image = normalize(image)
            a_hat, _, (_, _), (obs_preds, obs_targets) = self.model(
                qpos, image, env_state
            )  # no action, sample from prior

            # next frame prediction
            bs = a_hat.shape[0]
            patch_size = 16
            # image_size = 224
            img_h, img_w = self.model.img_h, self.model.img_w
            ph, pw = img_h // patch_size, img_w // patch_size
            nf_pred = obs_preds.sigmoid().reshape(
                shape=(bs, ph, pw, patch_size, patch_size, 3)
            )
            nf_pred = nf_pred.permute(0, 5, 1, 3, 2, 4)
            nf_pred = nf_pred.reshape(shape=(bs, 3, img_h, img_w))

            # import matplotlib.pyplot as plt
            # plt.imshow(nf_pred[0].cpu().numpy().transpose(1,2,0))
            # plt.savefig('tmp.png')
            # plt.clf()
            # plt.close()
            # import pdb; pdb.set_trace()
            # self.next_frames.append({'next_frame':nf_pred[0].cpu().numpy().transpose(1,2,0)})
            self.next_frame = nf_pred[0].cpu().numpy().transpose(1, 2, 0)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


if __name__ == "__main__":

    from cosmos_tokenizer.networks import TokenizerConfigs
    from cosmos_tokenizer.utils import (
        get_filepaths,
        get_output_filepath,
        read_video,
        resize_video,
        write_video,
    )

    # # from cosmos_tokenizer.video_lib import CausalVideoTokenizer
    # from cosmos_tokenizer.image_lib import ImageTokenizer

    #     tokenizer_type = 'DV'
    #     spatial_compression = 16
    #     temporal_compression = 8
    #     mode = 'torch'
    #     temporal_window = 17
    #     dtype = 'bfloat16'
    #     device = 'cuda'
    #     model_name = "Cosmos-Tokenizer-DV8x16x16"
    #     checkpoint_enc = f'Cosmos-Tokenizer/pretrained_ckpts/{model_name}/encoder.jit'
    #     checkpoint_dec = f'Cosmos-Tokenizer/pretrained_ckpts/{model_name}/decoder.jit'
    #     # load model
    #     tokenizer_config = TokenizerConfigs[tokenizer_type].value
    #     tokenizer_config.update(dict(spatial_compression=spatial_compression))
    #     tokenizer_config.update(dict(temporal_compression=temporal_compression))
    #     autoencoder = CausalVideoTokenizer(
    #     checkpoint= None,
    #     checkpoint_enc=checkpoint_enc,
    #     checkpoint_dec=checkpoint_dec,
    #     tokenizer_config=tokenizer_config,
    #     device=device,
    #     dtype=dtype,
    # )
    # # T C HW
    # input_tensor = read_video('Cosmos-Tokenizer/test_robot_data/sim_transfer_cube_scripted/episode_0.mp4')
    # print(input_tensor.dtype) # uint8 255

    # batch_video  = np.array(input_tensor[49:98:3,::2,::2])[np.newaxis, ...] # B T H W C
    #     output_video = autoencoder(batch_video, temporal_window=temporal_window)[0] # T H W C np.unit8 255

    #     autoencoder.get_latent_codes(batch_video)# B C T' H' W'  # B 6 T'  H' W'

    #     encoder = CausalVideoTokenizer(checkpoint_enc=checkpoint_enc)
    #     decoder = CausalVideoTokenizer(checkpoint_dec=checkpoint_dec)
    #     batch_video_tensor = torch.from_numpy(batch_video).cuda().permute(0,4,1,2,3) / 127.5 - 1 # B C T H W -1,1
    #     print(batch_video_tensor.shape)
    #     output_index, output_code = encoder._enc_model(batch_video_tensor)[:-1] # Input B C T H W  -1,1
    #     print(output_index.shape, output_code.shape)
    #     h = decoder._dec_model.post_quant_conv(output_code) # input shape B 6 T'  H' W', output shape B 16 T'  H' W'
    #     reconstructed_tensor = decoder._dec_model.decoder(h).detach()  # B C T H W -1,1
    #     rec_video = tensor2numpy(reconstructed_tensor)[0] # T H W C

    #     print('diff', (rec_video - output_video).mean())
    #     print('rec diff', (rec_video - batch_video).mean())
    #     print('out diff', (output_video - batch_video).mean())

    #     vis_path = 'vis_0.mp4'
    #     vis_video = np.concatenate([batch_video[0], output_video, rec_video], axis=2) # T H W*2 C
    #     media.write_video(vis_path, vis_video, fps=3)

    # model_name = "Cosmos-Tokenizer-DI16x16"
    # encoder = ImageTokenizer(checkpoint_enc=f'Cosmos-Tokenizer/pretrained_ckpts/{model_name}/encoder.jit')
    # decoder = ImageTokenizer(checkpoint_dec=f'Cosmos-Tokenizer/pretrained_ckpts/{model_name}/decoder.jit')
    # input_tensor = torch.randn(32, 3, 480, 640).to('cuda').to(torch.bfloat16)  # [B, C, T, H, W]
    # (indices, codes) = encoder.encode(input_tensor)
    # print(codes.shape)
    # print(codes.max(), codes.min())
    # reconstructed_tensor = decoder.decode(indices) # input index
    # print(reconstructed_tensor.shape)
    model_name = "Cosmos-Tokenizer-CV4x8x8"
    encoder = CausalVideoTokenizer(
        checkpoint_enc=f"Cosmos-Tokenizer/pretrained_ckpts/{model_name}/encoder.jit"
    )
    decoder = CausalVideoTokenizer(
        checkpoint_dec=f"Cosmos-Tokenizer/pretrained_ckpts/{model_name}/decoder.jit"
    )
    # B N C T H W
    input_tensor = (
        -torch.ones(16, 5, 3, 21, 480, 640).to("cuda").to(torch.bfloat16)
    )  # [B, C, T, H, W]
    for view_idx in range(5):
        (codes,) = encoder._enc_model(input_tensor[:, view_idx])[:-1]  # B 16 T' H' W'
        codes = codes.detach()
        print(codes.shape)
        print(codes.max(), codes.min())
