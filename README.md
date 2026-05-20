<div align="center">

# FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition

[![arXiv](https://img.shields.io/badge/arXiv-2603.26908-b31b1b)](https://arxiv.org/abs/2603.26908)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue)](https://cvpr.thecvf.com/Conferences/2026)
[![Project Page](https://img.shields.io/badge/Project_Page-FusionAgent-green)](https://fusionagent.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-red)](LICENSE)

<p align="center">
  <a href="https://github.com/jiezhu23"><strong>Jie Zhu</strong></a><sup>1</sup>
  &nbsp;·&nbsp;
  <a href="#"><strong>Xiao Guo</strong></a><sup>1</sup>
  &nbsp;·&nbsp;
  <a href="https://cse.msu.edu/~suyiyan1/"><strong>Yiyang Su</strong></a><sup>1</sup>
  &nbsp;·&nbsp;
  <strong>Anil Jain</strong><sup>1</sup>
  &nbsp;·&nbsp;
  <a href="https://www.cse.msu.edu/~liuxm/"><strong>Xiaoming Liu</strong></a><sup>1,2</sup>
</p>

<sup>1</sup>Michigan State University&nbsp;&nbsp;&nbsp;
<sup>2</sup>University of North Carolina at Chapel Hill

</div>

FusionAgent is a novel agentic framework for whole-body human recognition. It uses a Multimodal Large Language Model (MLLM) as a reasoning agent to **dynamically select** which expert recognition models (face, gait, body) to invoke on a per-sample basis, replacing static score-fusion strategies. The agent is trained via Reinforcement Fine-Tuning (RFT/GRPO) with a metric-based reward, and introduces **Anchor-based Confidence Top-k (ACT)** score-fusion to handle score misalignment across heterogeneous expert models.

## News

- Training code, checkpoints, and scripts have been released.
- Preprocessed datasets and pre-computed score matrices have been released.
- The paper is available on [arXiv](https://arxiv.org/abs/2603.26908).
- Our paper has been accepted to **CVPR 2026**! 🎉

## Highlights

- **Dynamic Model Selection:** The MLLM agent uses ReAct-style multi-turn reasoning to decide which subset of expert models to invoke per sample, instead of always running all models.
- **ACT Score-Fusion:** Anchor-based Confidence Top-k fusion anchors on the most confident model and integrates complementary predictions in a confidence-aware manner.
- **Efficient & Explainable:** FusionAgent outperforms SoTA methods while invoking fewer models, and supports both Chain-of-Thought (interpretable) and Direct Answering (fast) inference modes.

## Project Structure

```
FusionAgent_CVPR26/
├── src/
│   ├── fusionagent/
│   │   ├── checkpoints/            # Pre-trained backbone model weights
│   │   │   ├── adaface/            # AdaFace face recognition model
│   │   │   ├── ...
│   │   ├── configs/                # Training config YAML files
│   │   ├── data/                   # Dataset loaders and transforms
│   │   │   └── datasets/           # Per-dataset loaders (CCVID, LTCC, MEVID)
│   │   ├── WBModules/              # Backbone model wrappers
│   │   │   ├── Biggait/
│   │   │   ├── ...
│   │   ├── trainer/                # Custom GRPO trainer
│   │   ├── utils/                  # Evaluation metrics and utilities
│   │   ├── mod_center_feat/        # Precomputed center features for training
│   │   ├── train_feats/            # Precomputed training score matrices
│   │   ├── test_feats/             # Precomputed test features and score matrices
│   │   ├── fusionagent_grpo.py     # Main training entry point
│   │   ├── extract_features.py     # Feature and score matrix extraction
│   │   ├── precompute_center.py    # Center feature precomputation
│   │   └── app.py                  # Gradio interactive demo
│   └── scripts/
│       ├── fusionagent_grpo.sh             # Training script
│       ├── fusionagent_app.sh              # Gradio demo launch script
│       ├── fusionagent_precompute_center.sh
│       └── fusionagent_extract_scores.sh
```

## Getting Started

### 1. Environment Setup

```bash
git clone https://github.com/jiezhu23/FusionAgent_CVPR26.git
cd FusionAgent_CVPR26

conda create -n fusionagent python=3.10
conda activate fusionagent

pip install -e .
```

If `flash-attn` installation fails due to a CUDA/PyTorch version mismatch, install the wheel matching your environment or remove `--attn_implementation flash_attention_2` from the relevant config file.

> **Before running any scripts**, update the following path fields to match your environment:
> - **Training configs** (`src/fusionagent/configs/train_config_test_*.yaml`): set `root` to your dataset directory.
> - **Backbone configs** (`src/fusionagent/WBModules/model_cfg_*.yaml`): set all `*_cache_path` / `*_backbone_path` fields to the absolute paths of your downloaded model weights, and set `HF_TOKEN` to your HuggingFace token.
> - **Launch scripts** (`src/scripts/*.sh`): set `PYTHON_PATH` / `ACCELERATE_PATH` to the Python/accelerate binary in your conda environment.

### 2. Download Preprocessed Datasets

For convenience, we provide preprocessed datasets as `.h5` files. Download them from:

- **[[Google Drive]](https://drive.google.com/drive/folders/1TBt4HrJlm-Y-IO5SA7IAamZlWvj1vHQU?usp=sharing)** (`./Dataset/`)

Place the downloaded dataset files under your data root directory (e.g., `/path/to/data/`). For example, the CCVID dataset includes `ccvid.h5` (body images) and `ccvid_face.h5` (face images):

- `ccvid.h5`: key `'session1/001_01/00001'` → body image array.
- `ccvid_face.h5`: key `'session1/001_01/00001/face_0'` → face image array. A single body image may contain zero or multiple detected faces.

Update the `root` field in the training config (e.g., `src/fusionagent/configs/train_config_test_ccvid.yaml`) to point to your data root.

### 3. Download Pre-trained Models

Download pre-trained backbone model weights from:

- **[[Google Drive]](https://drive.google.com/drive/folders/1xoEIDuTezea-oLBW9VXG0rTSJF2llJNE?usp=sharing)**

Place the downloaded weights in `src/fusionagent/checkpoints/`, e.g.:
- `src/fusionagent/checkpoints/CAL/ltcc-checkpoint.pth.tar`
- `src/fusionagent/checkpoints/BigGait/BigGait__Dinov2_Gaitbase_Frame30-40000.pt`

**AdaFace (HuggingFace):** These models are downloaded automatically from HuggingFace. Set `HF_TOKEN` and the cache paths (`adaface_cache_path`) in the backbone config file for your dataset (e.g., `src/fusionagent/WBModules/model_cfg_ccvid.yaml`).

**BigGait DINOv2 backbone:** Update `pretrained_dinov2` in `src/fusionagent/WBModules/Biggait/configs/BigGait.yaml` and `BigGait_L.yaml` to the absolute path of your downloaded DINOv2 checkpoint (e.g., `/xxx/xxx/dinov2_vits14_pretrain.pth`).

### 4. Precompute Center Features for Training Set

Center features are used to construct the per-identity representation during training. Run the precompute script for each dataset and backbone, e.g.:

```bash
# CCVID dataset, AdaFace backbone
python src/fusionagent/precompute_center.py \
  --mode adaface \
  --dataset ccvid \
  --root /path/to/data/ \
  --backbone_cfg ./src/fusionagent/WBModules/model_cfg_ccvid.yaml \
  --save_path ./src/fusionagent/mod_center_feat/
```

Or use the provided script (edit `PYTHON_PATH` and paths first):

```bash
bash src/scripts/fusionagent_precompute_center.sh
```

By default, output is saved to `src/fusionagent/mod_center_feat/`. For `adaface` on `ccvid`, this produces:
1. `adaface_center_n100_ccvid_CC.h5` — 100-frame features per video.
2. `adaface_center_ccvid_CC.h5` — center features aggregated from the above.

### 5. Precompute Score Matrices

We provide pre-computed score matrices for all datasets in the download link above (`src/fusionagent/test_feats/scoremats_{dataset}.h5`). To extract them yourself:

```bash
# Step 1: Extract per-model features for the test set
python src/fusionagent/extract_features.py \
  --mode adaface \
  --dataset ccvid \
  --dataset_type test \
  --train_batch 128 \
  --num_sample 4 \
  --eval_mode feat

# Step 2: Gather multi-model score matrices
python src/fusionagent/extract_features.py \
  --mode adaface,cal-ccvid,biggait \
  --dataset ccvid \
  --dataset_type test \
  --eval_mode gather
```

For the **training set** (required for RFT training), extract training score matrices similarly with `--dataset_type train`. Refer to `src/scripts/fusionagent_extract_scores.sh` for the full set of commands across all datasets and backbones.

### 6. Training FusionAgent

Update `ACCELERATE_PATH`, `GPU_IDS`, and the YAML config paths in the script for your environment. The default config trains on CCVID with 4 GPUs:

```bash
bash src/scripts/fusionagent_grpo.sh
```

The entry point is `src/fusionagent/fusionagent_grpo.py`. Key training configs (in `src/fusionagent/configs/train_config_test_{dataset}.yaml`):

| Parameter | Default | Description |
|---|---|---|
| `model` | `Qwen/Qwen2.5-VL-3B-Instruct` | Base MLLM backbone |
| `train_method` | `grpo` | Training method |
| `max_steps` | `200` | Training steps |
| `num_generations` | `6` | GRPO rollout samples |
| `use_lora` | `true` | LoRA fine-tuning |
| `lora_r` | `64` | LoRA rank |
| `reward_funcs` | `format_v2, metric, tool_success_rate, answer_accuracy` | Reward functions |

Configs for LTCC and MEVID are available at `src/fusionagent/configs/train_config_test_ltcc.yaml` and `src/fusionagent/configs/train_config_test_mevid.yaml`.

### 7. Gradio Demo

To launch the interactive demo with a trained checkpoint:

```bash
bash src/scripts/fusionagent_app.sh
```

Or directly:

```bash
python src/fusionagent/app.py \
  --configs src/fusionagent/configs/train_config_test_ccvid.yaml \
  --ckpt_path src/fusionagent/checkpoints/<your-checkpoint> \
  --share
```

## Troubleshooting

**BigGait: DINOv2 path not found**

Open `src/fusionagent/WBModules/Biggait/configs/BigGait.yaml` and update `pretrained_dinov2` to the correct absolute path of your DINOv2 pretrained checkpoint:

```yaml
model_cfg:
  pretrained_dinov2: /absolute/path/to/dinov2_vits14_pretrain.pth
```

**AdaFace: `ModuleNotFoundError` when loading**

If you encounter an error such as `ModuleNotFoundError: No module named 'checkpoints.adaface'` or `No such file or directory: .../models.py`, open `src/fusionagent/checkpoints/adaface/wrapper.py` and change line 4 to use an absolute import:

```python
# Change this:
from checkpoints.adaface.models import get_model
# Or this:
from .models import get_model

# To this:
from models import get_model
```

Then clear the transformers dynamic module cache:

```bash
rm -rf ~/.cache/modules/transformers_modules/adaface/
```

## Citation

If you find this project useful for your research, please consider citing our paper:

```bibtex
@inproceedings{zhu2026fusionagent,
  title     = {FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition},
  author    = {Jie Zhu and Xiao Guo and Yiyang Su and Anil Jain and Xiaoming Liu},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Acknowledgements

This codebase builds on [Visual-RFT](https://github.com/Liuziyu77/Visual-RFT) for GRPO training, [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) as the MLLM backbone, and open-source implementations of [QME](https://github.com/jiezhu23/QME_ICCV25), [BigGait](https://github.com/ShiqiYu/OpenGait), [CAL](https://github.com/guyuchao/CAL), [AGRL](https://github.com/weleen/AGRL.pytorch), and [AdaFace](https://github.com/mk-minchul/AdaFace). We thank the authors for their open-source contributions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
