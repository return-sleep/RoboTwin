## Conda Environment
```
conda activate RoboTwin_Challenge 
cd policy/ACT-DP-TP-Policy
pip install -r requirements.txt
```
## Data Preparation
```python

cd ../..
# raw data path: data/{task_name}_{head_camera_type}

python script/pkl2zarr_mypolicy.py {task_name} {expert_data_num} 0 
# python script/pkl2zarr_mypolicy.py blocks_stack_hard 600 0
# zarr data path data/data_zarr/{task_name}_{head_camera_type}_{expert_data_num}.zarr

# modify the abs data path DATA_DIR = ''
# policy/ACT-DP-TP-Policy/train_policy_robotwin.py Line 34 

```

## Model Training
```python
cd policy/ACT-DP-TP-Policy

CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/act_dp/train_dp_single_parallel.sh {task_name} {num_episodes} {chunk_size} {history_step} {batch_size}

# CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/act_dp/train_dp_single_parallel.sh blocks_stack_hard 600 20 0 256
```
## Hyperparameters for 24G 4090

| task_name | chunk_size | history_step | batch_size |
|-----------|------------|--------------|------------|
| empty_cup_place    | 20     | 0        | 256   |
| dual_shoes_place    | 20     | 0        | 256   |
| bowls_stack    | 20      | 2        | 128     |
| blocks_stack_hard    | 20     | 0        | 256   |
| put_bottles_dustbin    | 20     | 2        | 128   |

## Evaluation Score (update 2025.04.27, 600 episodes)

| task_name | score | 
|-----------|------------|
| empty_cup_place    | 98     | 
| dual_shoes_place    | 93.4     | 
| bowls_stack    | 93.0      |  
| blocks_stack_hard    | 95.50     | 
| put_bottles_dustbin | 89.50     | 
|classify_tactile| 14|

## Trouble shooting:
- If you encounter the error `sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cuda:1) `
Modify the code in the file `/envs/RoboTwin_Challenge/lib/python3.10/site-packages/diffusers/schedulers/scheduling_ddim.py` as follows:
```python
#Line 485
timesteps = timesteps.to(original_samples.device).cpu()
#Line 497
noisy_samples = sqrt_alpha_prod.to(original_samples.device) * original_samples + sqrt_one_minus_alpha_prod.to(original_samples.device) * noise
```