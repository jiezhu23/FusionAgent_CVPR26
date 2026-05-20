# Set wandb to only initialize on main process
export WANDB_INIT_ON_PRIMARY_PROCESS_ONLY=true
# Optimize CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


ACCELERATE_PATH=~/anaconda3/envs/fusionagent/bin/accelerate
GPU_IDS=0,1,2,3
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Run training with accelerate

# $ACCELERATE_PATH launch \
#   --gpu_ids ${GPU_IDS} \
#   --num_processes=${NUM_GPUS} \
#   --main_process_port 29511 \
#   src/fusionagent/fusionagent_grpo.py \
#   --configs src/fusionagent/configs/train_config_test_ltcc_fewshot.yaml \
#   --use_accelerate \

$ACCELERATE_PATH launch \
  --gpu_ids ${GPU_IDS} \
  --num_processes=${NUM_GPUS} \
  --main_process_port 29513 \
  src/fusionagent/fusionagent_grpo.py \
  --configs src/fusionagent/configs/train_config_test_ccvid.yaml \
  --use_accelerate \

# $ACCELERATE_PATH launch \
#   --gpu_ids ${GPU_IDS} \
#   --num_processes=${NUM_GPUS} \
#   --main_process_port 29524 \
#   src/fusionagent/fusionagent_grpo.py \
#   --configs src/fusionagent/configs/train_config_test_ltcc.yaml \
#   --use_accelerate \
  
# $ACCELERATE_PATH launch \
#   --gpu_ids ${GPU_IDS} \
#   --num_processes=${NUM_GPUS} \
#   --main_process_port 29539 \
#   src/fusionagent/fusionagent_grpo.py \
#   --configs src/fusionagent/configs/train_config_test_mevid.yaml \
#   --use_accelerate \