PYTHON_PATH=~/anaconda3/envs/fusionagent/bin/python
GPU_IDS=0
export CUDA_VISIBLE_DEVICES=${GPU_IDS}


# /--------------For single model evaluation--------------/
# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode adaface \
#   --dataset ccvid \
#   --dataset_type test \
#   --train_batch 128 \
#   --num_sample 4 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode adaface \
#   --dataset mevid \
#   --dataset_type test \
#   --train_batch 128 \
#   --num_sample 4 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode adaface \
#   --dataset ltcc \
#   --dataset_type test \
#   --train_batch 128 \
#   --num_sample 4 \
#   --eval_mode feat \

# /--------------For merging score matrices--------------/

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode aim \
#   --dataset ltcc \
#   --dataset_type test \
#   --eval_mode score \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode kprpe,adaface,cal-ccvid,biggait \
#   --dataset ccvid \
#   --dataset_type test \
#   --eval_mode gather \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode kprpe,adaface,cal-mevid,agrl \
#   --dataset mevid \
#   --dataset_type test \
#   --eval_mode gather \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode kprpe,adaface,cal-ccvid,biggait \
#   --dataset ccvid \
#   --dataset_type train \
#   --eval_mode gather \

# /--------------For training score matrices curation--------------/

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode adaface \
#   --dataset ccvid \
#   --dataset_type train \
#   --train_batch 128 \
#   --num_sample 4 \
#   --max_batch 2000 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode adaface \
#   --dataset mevid \
#   --dataset_type train \
#   --train_batch 128 \
#   --num_sample 4 \
#   --max_batch 2000 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode adaface \
#   --dataset ltcc \
#   --dataset_type train \
#   --train_batch 128 \
#   --num_sample 4 \
#   --max_batch 2000 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode biggait \
#   --dataset ccvid \
#   --dataset_type train \
#   --train_batch 128 \
#   --num_sample 4 \
#   --max_batch 2000 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode cal-ccvid \
#   --dataset ccvid \
#   --dataset_type train \
#   --train_batch 128 \
#   --num_sample 4 \
#   --max_batch 2000 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode adaface \
#   --dataset ltcc \
#   --dataset_type train \
#   --train_batch 128 \
#   --num_sample 4 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode cal-ltcc \
#   --dataset ltcc \
#   --dataset_type train \
#   --train_batch 128 \
#   --num_sample 4 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode aim \
#   --dataset ltcc \
#   --dataset_type train \
#   --train_batch 128 \
#   --num_sample 4 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode adaface \
#   --dataset mevid \
#   --dataset_type train \
#   --train_batch 128 \
#   --num_sample 4 \
#   --max_batch 2000 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode cal-mevid \
#   --dataset mevid \
#   --dataset_type train \
#   --train_batch 128 \
#   --num_sample 4 \
#   --max_batch 2000 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode agrl \
#   --dataset mevid \
#   --dataset_type train \
#   --train_batch 128 \
#   --num_sample 4 \
#   --max_batch 2000 \
#   --eval_mode feat \


# /--------------FOR few shot setting--------------/
# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode adaface \
#   --dataset ltcc \
#   --dataset_type train \
#   --train_batch 1 \
#   --num_sample 1 \
#   --few_shot 10 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode cal-ltcc \
#   --dataset ltcc \
#   --dataset_type train \
#   --train_batch 1 \
#   --num_sample 1 \
#   --few_shot 10 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --mode aim \
#   --dataset ltcc \
#   --dataset_type train \
#   --train_batch 1 \
#   --num_sample 1 \
#   --few_shot 10 \
#   --eval_mode feat \

# $PYTHON_PATH src/fusionagent/extract_features.py \
#   --dataset ltcc \
#   --dataset_type train \
#   --few_shot 10 \
#   --eval_mode gather \
