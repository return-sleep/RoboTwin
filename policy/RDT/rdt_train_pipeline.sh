task_name=$1
bash generate.sh rdt_${task_name} # ${model_name}
# model_config/${model_name}.yml, set the GPU to be used (modify cuda_visible_device)
bash process_data_rdt.sh ${task_name} D435 100 0
mv processed_data/${task_name}_D435_100 training_data/rdt_${task_name}
# bash finetune.sh rdt_${model_name}