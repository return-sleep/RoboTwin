import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
load_dir = 'data/classify_tactile_D435'
current_ep = 30 

head_frames = []
tactile_frames = []

file_num =0
while os.path.exists(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl'):
    with open(load_dir+f'/episode{current_ep}'+f'/{file_num}.pkl', 'rb') as file:
        data = pickle.load(file)
    head_img = data['observation']['head_camera']['rgb']
    ll_tactile= data['vision_tactile']['ll_tactile']['rgb']
    lr_tactile= data['vision_tactile']['lr_tactile']['rgb']
    rl_tactile= data['vision_tactile']['rl_tactile']['rgb']
    rr_tactile= data['vision_tactile']['rr_tactile']['rgb'] # H W C
    tactile_concat = np.hstack((ll_tactile, lr_tactile, rl_tactile, rr_tactile)) 
    head_frames.append(head_img)
    tactile_frames.append(tactile_concat)
    file_num += 1

fps = 15
head_h, head_w, _ = head_frames[0].shape
tactile_h, tactile_w, _ = tactile_frames[0].shape
head_out = cv2.VideoWriter(f'episode{current_ep}_head.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (head_w, head_h))
for frame in head_frames:
    head_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
head_out.release()

tactile_out = cv2.VideoWriter(f'episode{current_ep}_tactile.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (tactile_w, tactile_h))
for frame in tactile_frames:
    tactile_out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
tactile_out.release()
