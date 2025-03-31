for task_name in put_bottles_dustbin dual_shoes_place bowls_stack blocks_stack_hard; do
    bash run_task.sh $task_name 0  
    python script/pkl2zarr_mypolicy.py $task_name 100
    cd data/data_zarr
    tar -czf ${task_name}_D435_100.zarr.tar.gz ${task_name}_D435_100.zarr  
    scp -P 20176 ${task_name}_D435_100.zarr.tar.gz xhl@10.176.54.116:/attached/remote-home2/xhl/8_kaust_pj/RoboTwin/data
    rm -rf ${task_name}_D435_100.zarr.tar.gz
    cd ..
    rm -rf ${task_name}_D435_100.zarr  
    cd ..
    rm -rf ${task_name}_D435_100
done