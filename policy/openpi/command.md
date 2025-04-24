``` python 
source .venv/bin/activate
cd policy/openpi/
bash process_data_pi.sh $task_name $head_camera_type $expert_data_num # 10min
bash generate.sh ${hdf5_path} ${repo_id} # 3min
# change src/openpi/training/config.py Line 393
repo_id="your_repo_id"
# compute norm_stat for dataset train_config_name = pi0_base_aloha_robotwin_lora 
uv run scripts/compute_norm_stats.py --config-name ${train_config_name} # 30min
# train_config_name: The name corresponding to the config in _CONFIGS, such as pi0_base_aloha_full
# model_name: You can choose any name for your model
# gpu_use: if not using multi gpu,set to gpu_id like 0;else set like 0,1,2,3
bash finetune.sh ${train_config_name} ${model_name} ${gpu_use}
```