import os
import json
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from diffusers import DDIMScheduler, DDPMScheduler
import numpy as np
import mediapy as media
from collections import deque
from detr.main import *
from utils_robotwin import normalize_data, tensor2numpy, kl_divergence, RandomShiftsAug
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        args_override["num_queries"] = args_override["chunk_size"]
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model
        self.camera_num = len(args_override["camera_names"])
        self.obs_image = None
        self.obs_qpos = None

    def __call__(
        self, qpos, image, actions=None, is_pad=None, is_training=False, instances=None
    ):
        env_state = None
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        image = normalize(image)

        a_hat, _, (_, _) = self.model(
            qpos, image, env_state
        )  # no action, sample from prior

        return a_hat

    # for robotwin
    def reset_obs(self, stats, norm_type):
        self.stats = stats
        self.norm_type = norm_type

    def update_obs(self, obs):
        # self.obs_image = (
        #     torch.from_numpy(obs["head_cam"]).unsqueeze(0).unsqueeze(0).float().cuda()
        # )  # 1 1 C H W 0~1
        image_data_list = []
        for camera_name in self.model.camera_names:
            assert camera_name in obs, f"camera {camera_name} not in obs"
            camera_image = torch.from_numpy(obs[camera_name]).float().cuda()
            image_data_list.append(camera_image)
        self.obs_image = torch.stack(image_data_list, dim=0).unsqueeze(0)  # B N C H W
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
        args_override["num_queries"] = args_override["chunk_size"]
        model, optimizer = build_ACTDiffusion_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.camera_num = len(args_override["camera_names"])
        self.obs_image = None
        self.obs_qpos = None
        # diffusion setup
        self.num_inference_steps = args_override["num_inference_steps"]
        self.num_queries = args_override["num_queries"]
        num_train_timesteps = args_override["num_train_steps"]
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
        print("num_train_timesteps", {args_override["num_train_steps"]})
        print("schedule_type", {args_override["schedule_type"]})
        print("beta_schedule", {args_override["beta_schedule"]})
        print("prediction_type", {args_override["prediction_type"]})
        print(f"Loss Type {self.loss_type}")

    # ===================inferece ===============
    def conditional_sample(self, qpos, image, is_pad):
        """
        diffusion process to generate actions
        """
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

        image = normalize(image)
        a_hat = self.conditional_sample(qpos, image, is_pad)
        return a_hat

    def reset_obs(self, stats, norm_type):
        self.stats = stats
        self.norm_type = norm_type

    def update_obs(self, obs):
        # self.obs_image = (
        #     torch.from_numpy(obs["head_cam"]).unsqueeze(0).unsqueeze(0).float().cuda()
        # )  # 1 1 C H W 0~1
        image_data_list = []
        for camera_name in self.model.camera_names:
            assert camera_name in obs, f"camera {camera_name} not in obs"
            camera_image = torch.from_numpy(obs[camera_name]).float().cuda()
            image_data_list.append(camera_image)
        self.obs_image = torch.stack(image_data_list, dim=0).unsqueeze(0)  # B N C H W
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


class ACTPolicyDiffusion_with_Token_Prediction(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        args_override["num_queries"] = args_override["chunk_size"]
        model = build_diffusion_tp_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.camera_num = len(args_override["camera_names"])
        # memory buffer
        self.history_steps = args_override["history_step"]
        self.obs_image = deque(maxlen=self.history_steps + 1)
        self.obs_qpos = deque(maxlen=self.history_steps + 1)
        # tokenizer model and shape
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
        print("predict_frame", self.predict_frame)
        print("prediction_weight", self.prediction_weight)
        print("imitate_weight", self.imitate_weight)

        # diffusion hyperparameters
        self.num_inference_steps = args_override["num_inference_steps"]
        self.num_queries = args_override["num_queries"]
        num_train_timesteps = args_override["num_train_steps"]
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

        print("num_train_timesteps", {args_override["num_train_steps"]})
        print("schedule_type", {args_override["schedule_type"]})
        print("beta_schedule", {args_override["beta_schedule"]})
        print("prediction_type", {args_override["prediction_type"]})
        print(f"Loss Type {self.loss_type}")

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def conditional_sample(self, qpos, image):
        # qpos: B 1 D
        # image: B 1 N C H W or B N C H W
        if len(image.shape) == 5:  # B N C H W
            qpos = qpos.unsqueeze(1)
            image = image.unsqueeze(1)
        env_state = None
        model = self.model
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps)
        # process image observation
        current_image_norm = self.normalize(image[:, 0 : self.history_steps + 1])  # B his+1 N C H W
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
        # inference time
        qpos = qpos  # B 1 D
        image = image  # B 1 N C H W
        a_hat, pred_token, _, _ = self.conditional_sample(qpos, image)
        return a_hat  # B H D

    # For ROBOTWIN
    def reset_obs(self, stats=None, norm_type="minmax"):
        self.obs_image.clear()
        self.obs_qpos.clear()
        self.stats = stats
        self.norm_type = norm_type

    def update_obs(self, obs):  # TODO adjust the obs format
        image_data_list = []
        for camera_name in self.model.camera_names:
            assert camera_name in obs, f"camera {camera_name} not in obs"
            camera_image = torch.from_numpy(obs[camera_name]).float().cuda()
            image_data_list.append(camera_image)
        image_data = torch.stack(image_data_list, dim=0).unsqueeze(0)  # B=1 N C H W
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
        )  # 1 n+1 1 3 H W raw time dimension
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


class ACT:
    def __init__(self, ckpt_folder: str):
        ckpt_file = os.path.join(ckpt_folder, "policy_lastest_seed_0.ckpt")
        config_file = os.path.join(
            ckpt_folder,
            "train_args_config.json",
        )
        self.policy = self.get_policy(ckpt_file, config_file, None, "cuda:0")

    def update_obs(self, observation):
        self.policy.update_obs(observation)  # add obs to deque

    def get_action(self, observation=None):
        action = self.policy.get_action()
        return action  # chunksize 14 cpu numpy

    def get_last_obs(self):
        return self.policy.obs_qpos, self.policy.obs_image

    def get_policy(self, ckpt_file, config_file, output_dir, device):
        with open(config_file, "r", encoding="utf-8") as file:
            policy_config = json.load(file)
        policy = ACTPolicy(policy_config)
        policy.load_state_dict(
            torch.load(ckpt_file, map_location=device)["state_dict"], strict=False
        )
        stats = torch.load(ckpt_file)["stats"]
        norm_type = policy_config["norm_type"]
        policy.norm_type = norm_type
        policy.stats = stats
        device = torch.device(device)
        policy.eval()
        return policy


class ACT_DP:
    def __init__(self, ckpt_folder: str):
        ckpt_file = os.path.join(ckpt_folder, "policy_lastest_seed_0.ckpt")
        config_file = os.path.join(
            ckpt_folder,
            "train_args_config.json",
        )
        self.policy = self.get_policy(ckpt_file, config_file, None, "cuda:0")

    def update_obs(self, observation):
        self.policy.update_obs(observation)  # add obs to deque

    def get_action(self, observation=None):
        action = self.policy.get_action()
        return action  # chunksize 14 cpu numpy

    def get_last_obs(self):
        return self.policy.obs_qpos, self.policy.obs_image

    def get_policy(self, ckpt_file, config_file, output_dir, device):
        with open(config_file, "r", encoding="utf-8") as file:
            policy_config = json.load(file)
        policy = ACTDiffusionPolicy(policy_config)
        policy.load_state_dict(
            torch.load(ckpt_file, map_location=device)["state_dict"], strict=False
        )
        stats = torch.load(ckpt_file)["stats"]
        norm_type = policy_config["norm_type"]
        policy.norm_type = norm_type
        policy.stats = stats
        device = torch.device(device)
        policy.eval()
        return policy


class ACT_DP_TP:
    def __init__(self, ckpt_folder: str):
        ckpt_file = os.path.join(ckpt_folder, "policy_lastest_seed_0.ckpt")
        config_file = os.path.join(
            ckpt_folder,
            "train_args_config.json",
        )
        self.policy = self.get_policy(ckpt_file, config_file, None, "cuda:0")

    def update_obs(self, observation):
        self.policy.update_obs(observation)  # add obs to deque

    def get_action(self, observation=None):
        action = self.policy.get_action()
        return action  # chunksize 14 cpu numpy

    def get_last_obs(self):
        return self.policy.obs_qpos[-1], self.policy.obs_image[-1]

    def get_policy(self, ckpt_file, config_file, output_dir, device):
        with open(config_file, "r", encoding="utf-8") as file:
            policy_config = json.load(file)
        policy = ACTPolicyDiffusion_with_Token_Prediction(policy_config)
        policy.load_state_dict(
            torch.load(ckpt_file, map_location=device)["state_dict"], strict=False
        )
        stats = torch.load(ckpt_file)["stats"]
        norm_type = policy_config["norm_type"]
        policy.norm_type = norm_type
        policy.stats = stats
        device = torch.device(device)
        policy.eval()
        return policy


def encode_obs(observation):  # For ACT-DP DP
    head_cam = (
        np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255
    )
    front_cam = (
        np.moveaxis(observation["observation"]["front_camera"]["rgb"], -1, 0) / 255
    )
    left_cam = (
        np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255
    )
    right_cam = (
        np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255
    )
    obs = dict(
        head_camera=head_cam,
        front_camera=front_cam,
        left_camera=left_cam,
        right_camera=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]
    return obs


def get_model(ckpt_folder):
    # f"./policy/{policy_name}/checkpoints/", usr_args.ckpt_folder
    print("ckpt_folder: ", ckpt_folder)
    policy_subname = ckpt_folder.split("/")[-4]
    print("policy_subname: ", policy_subname)  # TODO add act and act_dp policy
    if policy_subname == "act":
        return ACT(ckpt_folder)
    elif policy_subname == "act_dp":
        return ACT_DP(ckpt_folder)
    elif policy_subname == "act_dp_tp":
        return ACT_DP_TP(ckpt_folder)
    else:
        raise ValueError(f"Unknown policy name: {policy_subname}")


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)  # obs = observation
    # ======== Get Action ========
    if (
        len(model.policy.obs_image) == 0 or model.policy.obs_image == None
    ):  # TODO check if this is necessary designed for first timestep
        model.update_obs(obs)  # overlap with last command
    actions = model.get_action()  # perform inference per chunk

    for action in actions:
        TASK_ENV.take_action(action)  # print(action.shape)
        observation = (
            TASK_ENV.get_obs()
        )  # TODO process observation which type observation is
        obs = encode_obs(observation)  # TODO process observation
        model.update_obs(obs)


def reset_model(model):
    if isinstance(model.policy.obs_image, deque):
        model.policy.obs_image.clear()
        model.policy.obs_qpos.clear()
    else:
        model.policy.obs_image = None
        model.policy.obs_qpos = None


if __name__ == "__main__":
    ckpt_folder = "checkpoints/bottle_adjust/act_dp_tp/20_20_5_4_cosine_warmup/seed_0/num_epochs_300"
    model = get_model(ckpt_folder)

    obs_agent = np.random.rand(14).astype(np.float32)
    head_image = np.random.rand(3, 240, 320).astype(np.float32)
    obs = {
        "agent_pos": obs_agent,
        "head_camera": head_image,
    }
    reset_model(model)
    model.update_obs(obs)
    action = model.get_action()
    print(action.shape)
    # eval(None, model, obs)
