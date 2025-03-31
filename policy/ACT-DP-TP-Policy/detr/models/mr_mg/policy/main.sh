#!/usr/bin/env bash
cd /opt/tiger/GR_MG
GPUS_PER_NODE=8 # number of gpus per machine
MASTER_ADDR={master_address}":"{port} # modify it with your own address and port
NNODES=1 # number of machines
JOB_ID=107
torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank 0 \
    --rdzv_endpoint $MASTER_ADDR \
    --rdzv_id $JOB_ID \
    --rdzv_backend c10d \
    policy/main.py \
    --config ${@:1} \
    --devices $GPUS_PER_NODE \
    --num_nodes $NNODES