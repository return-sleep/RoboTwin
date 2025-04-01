for task_name in  bowls_stack blocks_stack_hard; do
    echo $task_name
    bash run_task.sh $task_name 0  
    scp -P 20088 -r data/${task_name}_D435 xhl@10.176.54.118:/attached/remote-home2/xhl/8_kaust_pj/RoboTwin/data/
done