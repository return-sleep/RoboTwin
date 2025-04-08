#!/bin/bash
#SBATCH --job-name=act_dp_tp       # name
#SBATCH --mem=100G # memory pool for all cores`
#SBATCH --nodes=1                    # nodes
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:a100:1            # number of gpus
#SBATCH --time 23:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=./outputs/act_dp_tp_%x-%j.out           # output file name
#SBATCH --error=./outputs/act_dp_tp_%x-%j.err            # error file nameW
#SBATCH --mail-user=xuh0e@kaust.edu.sa #Your Email address assigned for your job
#SBATCH --mail-type=ALL #Receive an email for ALL Job S

task_name=$1
chunk_size=$2
history_step=$3
cuda=$4
batch_size=$5
num_epochs=$6
num_episodes=$7
backbone=resnet50



seed=0
lr_schedule_type=cosine_warmup

echo "Processing $task_name"
CUDA_VISIBLE_DEVICES=$cuda python3 train_policy_robotwin.py \
    --task_name  $task_name \
    --ckpt_dir checkpoints/$task_name/single_${chunk_size}_${history_step}_${num_epochs}_${num_episodes}_scale/act_dp \
    --policy_class ACT_diffusion --hidden_dim 512  --batch_size $batch_size --dim_feedforward 3200 \
    --chunk_size $chunk_size  --norm_type minmax --disable_vae_latent \
    --num_epochs  $num_epochs \
    --lr 1e-4  --lr_schedule_type $lr_schedule_type  \
    --seed $seed --num_episodes $num_episodes  \
    --kl_weight 10 \
    --dist-url 'tcp://localhost:10001' \
    --world-size 1 \
    --rank 0 \
    --gpu 0  \
    --history_step $history_step \
    --disable_multi_view --backbone $backbone  \