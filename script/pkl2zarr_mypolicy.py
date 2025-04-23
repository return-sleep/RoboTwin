import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import einops
import cv2


def main():
    parser = argparse.ArgumentParser(description='Process some episodes.')
    parser.add_argument('task_name', type=str, default='block_hammer_beat',
                        help='The name of the task (e.g., block_hammer_beat)')
    parser.add_argument('expert_data_num', type=int, default=50,
                        help='Number of episodes to process (e.g., 50)')
    parser.add_argument('current_ep', type=int, default=0,
                        help='Number of episodes to start (e.g., 50)')
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    current_ep = args.current_ep
    setting = "D435"
    load_dir = f'data_200/{task_name}_{setting}'
    
    total_count = 0

    save_dir = f'data/data_zarr/{task_name}_{setting}_{num}.zarr'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    head_camera_arrays, front_camera_arrays, left_camera_arrays, right_camera_arrays = [], [], [], []
    episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = [], [], [], []
    
    if task_name == 'classify_tactile':
       ll_tactile_arrays, lr_tactile_arrays, rl_tactile_arrays, rr_tactile_arrays = [], [], [], [] 
    
    while os.path.isdir(load_dir+f'/episode{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        file_num = 0
        
        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
            with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
                data = pickle.load(file)
            
            head_img = data['observation']['head_camera']['rgb']
            # print(f'head_img shape: {head_img.shape}')
            # front_img = data['observation']['front_camera']['rgb']
            # left_img = data['observation']['left_camera']['rgb']
            # right_img = data['observation']['right_camera']['rgb']
            # action = data['endpose']
            joint_action = data['joint_action']

            head_camera_arrays.append(head_img)
            # front_camera_arrays.append(front_img)
            # left_camera_arrays.append(left_img)
            # right_camera_arrays.append(right_img)
            # action_arrays.append(action)
            state_arrays.append(joint_action)
            joint_action_arrays.append(joint_action)
            
            if task_name == 'classify_tactile':
                ll_tactile_arrays.append(data['vision_tactile']['ll_tactile']['rgb'])
                lr_tactile_arrays.append(data['vision_tactile']['lr_tactile']['rgb'])   
                rl_tactile_arrays.append(data['vision_tactile']['rl_tactile']['rgb'])
                rr_tactile_arrays.append(data['vision_tactile']['rr_tactile']['rgb'])

            file_num += 1
            total_count += 1
            
            del data 

        current_ep += 1

        episode_ends_arrays.append(total_count)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    # action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)
    head_camera_arrays = np.array(head_camera_arrays)
    # front_camera_arrays = np.array(front_camera_arrays)
    # left_camera_arrays = np.array(left_camera_arrays)
    # right_camera_arrays = np.array(right_camera_arrays)
    joint_action_arrays = np.array(joint_action_arrays)

    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW
    # front_camera_arrays = np.moveaxis(front_camera_arrays, -1, 1)  # NHWC -> NCHW
    # left_camera_arrays = np.moveaxis(left_camera_arrays, -1, 1)  # NHWC -> NCHW
    # right_camera_arrays = np.moveaxis(right_camera_arrays, -1, 1)  # NHWC -> NCHW
    
    
    
    if task_name == 'classify_tactile':
        ll_tactile_arrays = np.array(ll_tactile_arrays) 
        lr_tactile_arrays = np.array(lr_tactile_arrays)
        rl_tactile_arrays = np.array(rl_tactile_arrays)
        rr_tactile_arrays = np.array(rr_tactile_arrays)

        ll_tactile_arrays = np.moveaxis(ll_tactile_arrays, -1, 1) #  NHWC -> NCHW
        lr_tactile_arrays = np.moveaxis(lr_tactile_arrays, -1, 1)
        rl_tactile_arrays = np.moveaxis(rl_tactile_arrays, -1, 1)
        rr_tactile_arrays = np.moveaxis(rr_tactile_arrays, -1, 1)
    
    pre_save_dir = f'data/data_zarr/{task_name}_{setting}_{args.current_ep}.zarr'
    
    if os.path.exists(pre_save_dir):
        print('loading data from existing file')
        zarr_root = zarr.open(pre_save_dir, mode="r")
        pre_state_arrays = zarr_root["data/state"] # B C H W
        pre_joint_action_arrays = zarr_root["data/action"]
        pre_episode_ends_arrays = zarr_root["meta/episode_ends"]
        pre_head_camera_arrays = zarr_root["data/head_camera"]
        
        state_arrays = np.concatenate((pre_state_arrays, state_arrays), axis=0)
        joint_action_arrays = np.concatenate((pre_joint_action_arrays, joint_action_arrays), axis=0)
        episode_ends_arrays = np.concatenate((pre_episode_ends_arrays, episode_ends_arrays), axis=0)
        head_camera_arrays = np.concatenate((pre_head_camera_arrays, head_camera_arrays), axis=0)
        if task_name == 'classify_tactile':
            pre_ll_tactile_arrays = zarr_root["data/ll_tactile"]
            pre_lr_tactile_arrays = zarr_root["data/lr_tactile"]
            pre_rl_tactile_arrays = zarr_root["data/rl_tactile"]
            pre_rr_tactile_arrays = zarr_root["data/rr_tactile"]
            ll_tactile_arrays = np.concatenate((pre_ll_tactile_arrays, ll_tactile_arrays), axis=0)
            lr_tactile_arrays = np.concatenate((pre_lr_tactile_arrays, lr_tactile_arrays), axis=0)
            rl_tactile_arrays = np.concatenate((pre_rl_tactile_arrays, rl_tactile_arrays), axis=0)
            rr_tactile_arrays = np.concatenate((pre_rr_tactile_arrays, rr_tactile_arrays), axis=0)
    
    print('begin to save zarr')        
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    # action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    joint_chunk_size = (100, joint_action_arrays.shape[1])
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
    # front_camera_chunk_size = (100, *front_camera_arrays.shape[1:])
    # left_camera_chunk_size = (100, *left_camera_arrays.shape[1:])
    # right_camera_chunk_size = (100, *right_camera_arrays.shape[1:])
    zarr_data.create_dataset('head_camera', data=head_camera_arrays, chunks=head_camera_chunk_size, overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('front_camera', data=front_camera_arrays, chunks=front_camera_chunk_size, overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('left_camera', data=left_camera_arrays, chunks=left_camera_chunk_size, overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('right_camera', data=right_camera_arrays, chunks=right_camera_chunk_size, overwrite=True, compressor=compressor)
    # zarr_data.create_dataset('tcp_action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=joint_action_arrays, chunks=joint_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    if task_name == 'classify_tactile':
        tactile_chunk_size = (100, *ll_tactile_arrays.shape[1:]) 
        zarr_data.create_dataset('ll_tactile', data=ll_tactile_arrays, chunks=tactile_chunk_size, overwrite=True, compressor=compressor)
        zarr_data.create_dataset('lr_tactile', data=lr_tactile_arrays, chunks=tactile_chunk_size, overwrite=True, compressor=compressor)
        zarr_data.create_dataset('rl_tactile', data=rl_tactile_arrays, chunks=tactile_chunk_size, overwrite=True, compressor=compressor)
        zarr_data.create_dataset('rr_tactile', data=rr_tactile_arrays, chunks=tactile_chunk_size, overwrite=True, compressor=compressor)


def main_alter():
    import torch
    parser = argparse.ArgumentParser(description='Process some episodes.')
    parser.add_argument('task_name', type=str, default='put_bottles_dustbin',
                        help='The name of the task (e.g., block_hammer_beat)')
    parser.add_argument('expert_data_num', type=int, default=1,
                        help='Number of episodes to process (e.g., 50)')
    parser.add_argument('current_ep', type=int, default=0,
                        help='Number of episodes to start (e.g., 50)')
    args = parser.parse_args()
    
    setting = "D435"
    task_name = args.task_name
    num = args.expert_data_num
    current_ep = args.current_ep
    
    load_dir = f'data_200/{task_name}_{setting}'
    save_dir = os.path.join('data/data_pt',f'{task_name}_{setting}')
    os.makedirs(save_dir,exist_ok=True)
    total_count = 0
    
    episode_ends_arrays = []
    while os.path.isdir(load_dir+f'/episode{current_ep}') and current_ep < num:
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        file_num = 0 
        head_camera_arrays, state_arrays, joint_action_arrays = [], [], []
        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
            with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
                data = pickle.load(file)  
            head_img = data['observation']['head_camera']['rgb']    
            joint_action = data['joint_action']

            head_camera_arrays.append(head_img)
            state_arrays.append(joint_action)
            joint_action_arrays.append(joint_action)
            file_num += 1
            total_count += 1
            
            del data
        
        head_camera_arrays = np.array(head_camera_arrays)
        state_arrays = np.array(state_arrays)
        joint_action_arrays = np.array(joint_action_arrays)
        head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW
        save_path = os.path.join(save_dir,f'episode{current_ep}.pt')
        
        state_tensor = torch.tensor(state_arrays)
        joint_action_tensor = torch.tensor(joint_action_arrays)
        head_camera_tensor = torch.tensor(head_camera_arrays)
        torch.save({
            'state_arrays': state_tensor,
            'joint_action_arrays': joint_action_tensor,
            'head_camera_arrays': head_camera_tensor
        }, save_path)         
        
        current_ep += 1
        episode_ends_arrays.append(total_count)
    
    episode_ends_arrays = np.array(episode_ends_arrays)
    episode_ends_tensor = torch.tensor(episode_ends_arrays)
    episode_ends_save_path = os.path.join(save_dir,f'episode_ends.pt')
    torch.save({'episode_ends':episode_ends_tensor},episode_ends_save_path )
    

    all_indexes = torch.arange(episode_ends_tensor[-1])
    train_size = int(len(all_indexes) * 0.99)
    indices = torch.randperm(len(all_indexes))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    train_indices_list = train_indices.tolist()
    test_indices_list = test_indices.tolist()
    torch.save({'train_indices': train_indices_list, 'val_indices': test_indices_list}, save_dir +f'/train_val_split_{num}.pt')


if __name__ == '__main__':
    # main()
    main_alter()
    # load_dir = 'data/classify_tactile_D435'
    # current_ep = 0
    # file_num = 0 
    # with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
    #     data = pickle.load(file)
    #     print(data.keys())
    #     print(data['observation'].keys())
    #     print(data['vision_tactile'].keys())
    #     print(data['vision_tactile']['ll_tactile']['rgb'].shape)
    # load_dir = 'data/data_zarr/classify_tactile_D435_20.zarr'
    # zarr_root = zarr.open(load_dir, mode="r")
    # head_camera = zarr_root["data/head_camera"]
    # state = zarr_root["data/state"]
    # action = zarr_root["data/action"]
    # ll_tactile = zarr_root["data/ll_tactile"]
    # lr_tactile = zarr_root["data/lr_tactile"]
    # rl_tactile = zarr_root["data/rl_tactile"]
    # rr_tactile = zarr_root["data/rr_tactile"]
    # episode_ends = zarr_root["meta/episode_ends"]
    # print(head_camera.shape)
    # print(state.shape)
    # print(action.shape)
    # print(ll_tactile.shape)
    # print(lr_tactile.shape)
    # print(rl_tactile.shape)
    # print(rr_tactile.shape)
    # print(np.max(rr_tactile))
    # print(np.min(rr_tactile))
    # print(episode_ends[0]) # 0~189