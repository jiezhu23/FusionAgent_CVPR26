PYTHON_PATH=~/anaconda3/envs/fusionagent/bin/python
GPU_IDS=0
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# $PYTHON_PATH src/fusionagent/precompute_center.py \
#   --mode adaface \
#   --dataset ccvid \
#   --root /localscratch/zhujie4/data/ \
#   --backbone_cfg ./src/fusionagent/WBModules/model_cfg_ccvid.yaml \
#   --save_path ./src/fusionagent/mod_center_feat/


# $PYTHON_PATH src/fusionagent/precompute_center.py \
#   --mode adaface \
#   --dataset mevid \
#   --root /localscratch/zhujie4/data/ \
#   --backbone_cfg ./src/fusionagent/WBModules/model_cfg_mevid.yaml \
#   --save_path ./src/fusionagent/mod_center_feat/
  
  
$PYTHON_PATH src/fusionagent/precompute_center.py \
  --mode adaface \
  --dataset ltcc \
  --root /localscratch/zhujie4/data/ \
  --backbone_cfg ./src/fusionagent/WBModules/model_cfg_ltcc.yaml \
  --save_path ./src/fusionagent/mod_center_feat/