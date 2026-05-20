export CUDA_VISIBLE_DEVICES=6

python src/fusionagent/app.py \
    --configs src/fusionagent/configs/train_config_test_ccvid.yaml \
    --ckpt_path src/fusionagent/checkpoints/ccvid-grpo-200step-Rmetricv2_train-Rformatv2-actfusion_topk10-adacot-frac06_1029_1259 \
    --share \