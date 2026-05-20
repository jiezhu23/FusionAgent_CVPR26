cd src/virft
pip install -e ".[dev]"

# Addtional modules
pip install wandb
pip install matplotlib
pip install peft
pip install omegaconf
pip install easydict
# pip install lmdeploy
pip install tensorboardx
# pip install beautifulsoup4
pip install timm
pip install qwen_vl_utils torchvision
pip install kornia
pip install imageio
pip install yacs
pip install timm
pip install fvcore
pip install imgaug
pip install h5py
pip install numpy==1.26.4 # one package to update to 2.x
pip install checkpoints
# upgrade transformers and trl (Do not upgrade trl, it will cause error）

# flash-attn
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# vLLM support 
pip install vllm==0.7.2

# fix transformers version (DO NOT upgrade transformers, it will cause OOM error)
pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
pip install trl==0.14.0
