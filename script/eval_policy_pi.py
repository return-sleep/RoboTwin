
import sys
sys.path.append('./') 
sys.path.insert(0, './policy/openpi')
from pi_model import *

import torch  
import os
import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib

from argparse import ArgumentParser

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

def class_decorator(task_name):
    envs_module = importlib.import_module(f'envs.{task_name}')
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, '../task_config/_camera_config.yml')

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f'camera {camera_type} is not defined'
    return args[camera_type]

def main(usr_args):
    task_name = usr_args.task_name
    head_camera_type = usr_args.head_camera_type
    train_config_name = usr_args.train_config_name
    model_name = usr_args.model_name
    checkpoint_num = usr_args.checkpoint_num
    seed = usr_args.seed

    with open(f'./task_config/{task_name}.yml', 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    args['head_camera_type'] = head_camera_type
    head_camera_config = get_camera_config(args['head_camera_type'])
    args['head_camera_fovy'] = head_camera_config['fovy']
    args['head_camera_w'] = head_camera_config['w']
    args['head_camera_h'] = head_camera_config['h']
    head_camera_config = 'fovy' + str(args['head_camera_fovy']) + '_w' + str(args['head_camera_w']) + '_h' + str(args['head_camera_h'])
    
    wrist_camera_config = get_camera_config(args['wrist_camera_type'])
    args['wrist_camera_fovy'] = wrist_camera_config['fovy']
    args['wrist_camera_w'] = wrist_camera_config['w']
    args['wrist_camera_h'] = wrist_camera_config['h']
    wrist_camera_config = 'fovy' + str(args['wrist_camera_fovy']) + '_w' + str(args['wrist_camera_w']) + '_h' + str(args['wrist_camera_h'])

    front_camera_config = get_camera_config(args['front_camera_type'])
    args['front_camera_fovy'] = front_camera_config['fovy']
    args['front_camera_w'] = front_camera_config['w']
    args['front_camera_h'] = front_camera_config['h']
    front_camera_config = 'fovy' + str(args['front_camera_fovy']) + '_w' + str(args['front_camera_w']) + '_h' + str(args['front_camera_h'])

    # output camera config
    print('============= Camera Config =============\n')
    print('Head Camera Config:\n    type: '+ str(args['head_camera_type']) + '\n    fovy: ' + str(args['head_camera_fovy']) + '\n    camera_w: ' + str(args['head_camera_w']) + '\n    camera_h: ' + str(args['head_camera_h']))
    print('Wrist Camera Config:\n    type: '+ str(args['wrist_camera_type']) + '\n    fovy: ' + str(args['wrist_camera_fovy']) + '\n    camera_w: ' + str(args['wrist_camera_w']) + '\n    camera_h: ' + str(args['wrist_camera_h']))
    print('Front Camera Config:\n    type: '+ str(args['front_camera_type']) + '\n    fovy: ' + str(args['front_camera_fovy']) + '\n    camera_w: ' + str(args['front_camera_w']) + '\n    camera_h: ' + str(args['front_camera_h']))
    print('\n=======================================')

    task = class_decorator(args['task_name'])
    args['model_name'] = model_name
    args['checkpoint_id'] = checkpoint_num

    st_seed = 100000 * (1+seed)
    suc_nums = []
    test_num = 100
    topk = 1
    
    model = PI0(task_name,train_config_name,model_name,checkpoint_num)
   
    model.random_set_language()
    st_seed, suc_num = test_policy(task, args, model, st_seed, test_num=test_num)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]
    save_dir = Path(f'result_pi0/{train_config_name}_{task_name}_{usr_args.head_camera_type}') 
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f'ckpt_{checkpoint_num}_seed_{seed}.txt'
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(file_path, 'w') as file:
        file.write(f'Timestamp: {current_time}\n\n')

        file.write(f'Checkpoint Num: {checkpoint_num}\n')
        
        file.write('Successful Rate of Diffenent checkpoints:\n')
        file.write('\n'.join(map(str, np.array(suc_nums) / test_num)))
        file.write('\n\n')
        file.write(f'TopK {topk} Success Rate (every):\n')
        file.write('\n'.join(map(str, np.array(topk_success_rate) / test_num)))
        file.write('\n\n')
        file.write(f'TopK {topk} Success Rate:\n')
        file.write(f'\n'.join(map(str, np.array(topk_success_rate) / (topk * test_num))))
        file.write('\n\n')

    print(f'Data has been saved to {file_path}')
    

def test_policy(Demo_class, args, policy, st_seed, test_num=20):
    expert_check = True
    task_name = args["task_name"]
    print("Task name: ", args["task_name"])


    Demo_class.suc = 0
    Demo_class.test_num =0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    

    now_seed = st_seed
    while succ_seed < test_num:
        render_freq = args['render_freq']
        args['render_freq'] = 0
        
        if expert_check:
            try:
                Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
                Demo_class.play_once()
                Demo_class.close()
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(' -------------')
                print('Error: ', stack_trace)
                print(' -------------')
                Demo_class.close()
                now_seed += 1
                args['render_freq'] = render_freq
                print('error occurs !')
                continue

        if (not expert_check) or ( Demo_class.plan_success and Demo_class.check_success() ):
            succ_seed +=1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args['render_freq'] = render_freq
            continue

        args['render_freq'] = render_freq
        args['expert_seed'] = now_seed

        Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
        Demo_class.apply_pi(policy, args)

        now_id += 1
        Demo_class.close()
        if Demo_class.render_freq:
            Demo_class.viewer.close()
        # policy.env_runner.reset_obs()
        policy.reset_obsrvationwindows()
        policy.random_set_language()
        print(f"{task_name} success rate: {Demo_class.suc}/{Demo_class.test_num}, current seed: {now_seed}\n")
        Demo_class._take_picture()
        now_seed += 1

    return now_seed, Demo_class.suc

if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()
    
    parser = ArgumentParser()

    parser.add_argument('task_name', type=str, default='block_hammer_beat')
    parser.add_argument('head_camera_type', type=str)
    parser.add_argument('train_config_name', type=str)
    parser.add_argument('model_name', type=str)
    parser.add_argument('checkpoint_num', type=int)
    parser.add_argument('seed', type=int, default=0)
    usr_args = parser.parse_args()
    
    main(usr_args)
