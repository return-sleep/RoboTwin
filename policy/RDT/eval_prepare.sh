CONFIG_NAME="$1"
# bash eval.sh pick_apple_messy D435 rdt_pick_apple_messy 10000 0 0 # mask offline embedding instruction    
rsync -avz checkpoints/$CONFIG_NAME/checkpoint-10000 xuhuilin@10.154.55.147:/mnt/mydisk/Project_folder/0_RoboTwin/RoboTwin/policy/RDT/checkpoints/$CONFIG_NAME/
scp training_data/instruction_pt_files/${CONFIG_NAME#rdt_}_*.pt xuhuilin@10.154.55.147:/mnt/mydisk/Project_folder/0_RoboTwin/RoboTwin/policy/RDT/training_data/instruction_pt_files/
