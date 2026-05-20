"""
Credited from https://www.kaggle.com/code/alphadua/finetune-qwen2-5vl-grpo

# modified by Jie Zhu

Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import textwrap
import json
import copy
import gc
import random
import time
from argparse import Namespace
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Dict, List
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
    BitsAndBytesConfig
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available, get_json_schema
from transformers.trainer import EvalLoopOutput
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from accelerate.utils import gather_object
import asyncio
import inspect
import json
import h5py
import yaml

from fusionagent.utils import XMLLikeTagParser, ConversationParser, SYSTEM_PROMPT_AGENT, SYSTEM_PROMPT_AGENT_FAST, TOOL_PROMPT
from fusionagent.WBModules import DEFAULT_BACKBONE_CFG, MODEL_MAPPING_DICT
from fusionagent.tool_func import RuntimeCtx
from fusionagent.utils.eval_metrics import test_score, normalize_score, act_score_fusion
from fusionagent.utils.file_utils import load_scoremats_from_h5

if is_peft_available():
    from peft import PeftConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        model: Union[str, PreTrainedModel] = None,
        model_cache_dir: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        reward_weights: Optional[list[float]] = None,
        tools: Optional[list[Union[dict, Callable]]] = None,
        agent_config: Optional[Any] = None,
        agent_raw_dataset: Optional[Any] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs,
                                                                        torch_dtype=torch.bfloat16,
                                                                        cache_dir=model_cache_dir)
            elif "Qwen2.5-VL" in model_id:
                # resume from lora checkpoint
                if resume_from_checkpoint is not None:
                    self.resume_from_checkpoint = resume_from_checkpoint
                    if os.path.exists(os.path.join(resume_from_checkpoint, "adapter_config.json")):
                        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs,
                                                                           torch_dtype=torch.bfloat16,
                                                                           cache_dir=model_cache_dir)
                        lora_model = PeftModel.from_pretrained(model, resume_from_checkpoint)
                        model = lora_model.merge_and_unload()
                        print(f"Loaded and merged LoRA checkpoint from {resume_from_checkpoint}")
                        if hasattr(model, "peft_config"):
                            delattr(model, "peft_config")
                    else:
                        raise NotImplementedError("Resume from checkpoint is not supported for non-lora model")
                else:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs,
                                                                           torch_dtype=torch.bfloat16,
                                                                           cache_dir=model_cache_dir)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            print(peft_config)
            # The following block is crucial for using PEFT with gradient checkpointing.
            # The correct order is to first wrap the model with PEFT and then enable GC.
            model = get_peft_model(model, peft_config)
            if args.gradient_checkpointing:
                # Step 1: Enable gradient checkpointing on the PeftModel.
                # This uses PEFT's own implementation that is aware of the adapters.
                model.gradient_checkpointing_enable()
                # Step 2: Disable the Trainer's native gradient checkpointing.
                args.gradient_checkpointing = False
                print('Using PEFT with Gradient checkpointing, manually disable it for Trainer config')

            model.enable_input_require_grads()
            print(model.print_trainable_parameters())

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs
        
        # Reward weights
        if reward_weights is not None:
            if len(reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)
            
        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments

        # Build name->callable mapping for tool execution at runtime
        self.tools = tools
        self.tool_funcs: dict[str, Callable] = {}
        if isinstance(tools, list):
            for func in tools:
                try:
                    self.tool_funcs[func.__name__] = func
                except Exception:
                    pass
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        # self.generation_config = GenerationConfig(
        #     max_new_tokens=self.max_completion_length,
        #     do_sample=True,  
        #     temperature=1, # HACK
        #     num_return_sequences=self.num_generations,
        #     pad_token_id=pad_token_id,
        # )
        self.turn_gen_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=1, # HACK
                num_return_sequences=1, # only one completion for each turn
                pad_token_id=pad_token_id,
            )
        
        self.generation_config_val = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=False,  
            num_return_sequences=1, # only one completion for validation
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False
        print(f"Running with {self.accelerator.num_processes} GPUs.")
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        # ------------------------ Tool runtime context ------------------------
        # Defer heavy non-LLM model initializations here to keep separation of concerns
        self.agent_config = agent_config
        self.agent_raw_dataset = agent_raw_dataset
        # self.tool_model_list: dict[str, Any] = {}
        # self.center_feats: Optional[dict[str, Any]] = None
        try:
            if (self.agent_config is not None) and (self.agent_raw_dataset is not None):
                self._initialize_tool()
        except Exception as init_err:
            # Do not crash training if tool init fails; tools are optional for some reward functions
            print(f"[WARN] Tool runtime initialization failed: {init_err}")
        # Build system prompt
        tool_json_schema = [str(get_json_schema(tool)) for tool in self.tools]
        tool_prompt = TOOL_PROMPT.format(TOOL_SCHEMA="\n".join(tool_json_schema))
        model_type_dict = {k: v for k, v in MODEL_MAPPING_DICT.items() if k in RuntimeCtx.model_list}
        self.system_prompt = SYSTEM_PROMPT_AGENT.format(TOOL_PROMPT=tool_prompt, MODEL_TYPE_DICT=model_type_dict)
        self.system_prompt_fast = SYSTEM_PROMPT_AGENT_FAST.format(TOOL_PROMPT=tool_prompt, MODEL_TYPE_DICT=model_type_dict)
        print(f"System prompt: {self.system_prompt}")

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values_videos, video_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def _prepare_conversation(self, inputs: list[dict[str, Union[torch.Tensor, Any]]], prompt_type: str = "cot") -> list[dict[str, Union[torch.Tensor, Any]]]:
        outputs = []
        assert prompt_type in ["fast", "cot"], "Invalid prompt type"
        if prompt_type == "fast": #[fast, cot]
            system_prompt = self.system_prompt_fast
        else:
            system_prompt = self.system_prompt
        for i in inputs:
            user_content = [{"type": "video", "video": [img for img in i['body_image_keys']]}]
            user_content.append({"type": "text", "text": "Identify the person in the video from the dataset."})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            outputs.append({"prompt": messages})
        return outputs
    
    @staticmethod
    def load_and_resize_image(img_data, factor=28, resize_h=256, resize_w=128):
        """
        Load and resize the image if the height or width is less than the factor for Qwen-VL model.
        """
        img = Image.fromarray(np.array(img_data))
        width, height = img.size
        # if height < factor or width < factor:
        img = img.resize((resize_w, resize_h))
        return img

    def apply_chat_template_with_tools(self, messages: list[dict[str, str]]) -> dict[str, str]:
        tool_json_schema = [str(get_json_schema(tool)) for tool in self.tools]
        tool_prompt = TOOL_PROMPT.format(TOOL_SCHEMA="\n".join(tool_json_schema))

        prompts_text = [maybe_apply_chat_template(example, self.processing_class, tools=tool_json_schema)["prompt"] for example in messages]
        # add tool json schema to the prompt
        return prompts_text, tool_prompt

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # randomly sample the prompt type
        prompt_type = "fast" if np.random.rand() < 0.4 else "cot"
        msg = self._prepare_conversation(inputs, prompt_type)
        prompts_text = [maybe_apply_chat_template(example, self.processing_class, tools=self.tools)["prompt"] for example in msg]
        labels = [input_item['subject_id'] for input_item in inputs]
        video_inputs = [[self.load_and_resize_image(self.agent_raw_dataset.body_data[f]) for f in input_item['body_image_keys']] for input_item in inputs]
        
        # prompt_inputs = self.processing_class(
        #     text=prompts_text,
        #     videos=video_inputs,
        #     return_tensors="pt",
        #     padding=True,
        #     padding_side="left",
        #     add_special_tokens=False,
        # )
        # prompt_inputs = super()._prepare_inputs(prompt_inputs)

        # ---------------- Agent-style multi-turn generation with tool execution ----------------
        device = self.accelerator.device

        # Keep original conversations for logging and reward formatting (B)
        base_conversations = msg  # list[list[dict]] per example
        # For compatibility with downstream reward code that expands `prompts`, keep `prompts` as B here
        prompts = base_conversations

        # Expanded prompts used internally for per-generation rollouts (B*G)
        gen_prompts = [conv for conv in base_conversations for _ in range(self.num_generations)]

        # Prepare video inputs expanded per generation
        video_inputs_expanded = []
        for v in video_inputs:
            for _ in range(self.num_generations):
                video_inputs_expanded.append(v)

        # Prepare tool kwargs (pil image list, subject_id, has_face, etc.) for tool execution
        def build_tools_kwargs(sample_item: dict) -> list[dict]:
            # Fallback to at least one placeholder to avoid empty sequence
            if len(sample_item['body_image_keys']) == 0:
                sample = {
                    "body_pil_image": [],
                    "face_pil_image": [],
                    "body_image_keys": sample_item['body_image_keys'],
                    "face_image_keys": sample_item['face_image_keys'],
                    "subject_id": sample_item['subject_id'],
                    "has_face": sample_item['has_face'],
                }
            else:
                sample = {
                    "body_pil_image": [Image.fromarray(np.array(self.agent_raw_dataset.body_data[f])) for f in sample_item['body_image_keys']],
                    "face_pil_image": [Image.fromarray(np.array(self.agent_raw_dataset.face_data[f])) for f in sample_item['face_image_keys']],
                    "body_image_keys": sample_item['body_image_keys'],
                    "face_image_keys": sample_item['face_image_keys'],
                    "subject_id": sample_item['subject_id'],
                    "has_face": sample_item['has_face'],
                }
            return sample

        # Prepare per-candidate containers
        num_candidates = len(gen_prompts)
        final_completion_texts: list[str] = []  # final assistant answers for each candidate
        
        all_input_ids_list: list[torch.Tensor] = []
        all_attention_mask_list: list[torch.Tensor] = []
        all_completion_mask_list: list[torch.Tensor] = []
        # all_pixel_values_videos_list: list[torch.Tensor] = []
        # all_video_grid_thw_list: list[torch.Tensor] = []

        # Multi-turn generation loop to produce full conversations
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            for cand_idx in range(num_candidates):
                sample_idx = cand_idx // self.num_generations
                tools_kwargs = build_tools_kwargs(inputs[sample_idx])
                current_messages = copy.deepcopy(gen_prompts[cand_idx]['prompt'])  # start from base system+user

                # Accumulators for the current candidate
                cand_input_ids_parts: list[torch.Tensor] = []
                cand_attention_mask_parts: list[torch.Tensor] = []
                cand_completion_mask_parts: list[torch.Tensor] = []
                
                # Initial prompt
                initial_prompt_example = {"prompt": current_messages}
                initial_prompt_text = maybe_apply_chat_template(initial_prompt_example, self.processing_class, tools=self.tools)["prompt"]
                initial_inputs = self.processing_class(
                    text=[initial_prompt_text],
                    videos=[video_inputs_expanded[cand_idx]],
                    return_tensors="pt",
                    padding=False, # No padding for incremental building
                    add_special_tokens=False,
                )
                initial_inputs = super()._prepare_inputs(initial_inputs)

                cand_input_ids_parts.append(initial_inputs["input_ids"].squeeze(0))
                cand_attention_mask_parts.append(initial_inputs["attention_mask"].squeeze(0))
                cand_completion_mask_parts.append(torch.zeros_like(initial_inputs["input_ids"].squeeze(0)))


                for _turn in range(self.agent_config.max_turns):

                    # Prepare inputs for generation using accumulated parts
                    turn_input_ids = torch.cat(cand_input_ids_parts).unsqueeze(0)
                    if self.max_prompt_length is not None:
                        turn_input_ids = turn_input_ids[:, -self.max_prompt_length:]
                    
                    # We need to provide attention_mask for generate
                    turn_attention_mask = torch.cat(cand_attention_mask_parts).unsqueeze(0)
                    if self.max_prompt_length is not None:
                         turn_attention_mask = turn_attention_mask[:, -self.max_prompt_length:]

                    # The video input is consistent for the whole conversation
                    turn_out_ids = unwrapped_model.generate(
                        input_ids=turn_input_ids,
                        attention_mask=turn_attention_mask,
                        pixel_values_videos = initial_inputs["pixel_values_videos"],
                        video_grid_thw = initial_inputs["video_grid_thw"],
                        generation_config=self.turn_gen_config
                    )
                    
                    turn_completion_ids = turn_out_ids[0, turn_input_ids.shape[1]:]
                    turn_text = self.processing_class.decode(turn_completion_ids, skip_special_tokens=True)
                    
                    # Append assistant response parts
                    cand_input_ids_parts.append(turn_completion_ids)
                    cand_attention_mask_parts.append(torch.ones_like(turn_completion_ids))
                    cand_completion_mask_parts.append(torch.ones_like(turn_completion_ids))
                    current_messages.append({"role": "assistant", "content": turn_text})

                    xml_parser = XMLLikeTagParser(turn_text)
                    # first detect the tool call tag
                    if 'tool_call' in xml_parser.get_ordered_tags():
                        tool_input = xml_parser.parse_tool_calls()
                        tool_result = self.execute_tool_from_response(tool_input, tools_kwargs)
                        tool_result_str = str(tool_result)
                        tool_message = {"role": "system", "content": f"<tool_result>{tool_result_str}</tool_result>"}
                        current_messages.append(tool_message)
                        
                        # We need to tokenize the tool result to append to our history
                        tool_result_text = maybe_apply_chat_template({"prompt": [tool_message]}, self.processing_class, tools=self.tools)["prompt"]
                        tool_result_ids = self.processing_class(text=tool_result_text, add_special_tokens=False).input_ids
                        tool_result_ids = torch.tensor(tool_result_ids, device=device).squeeze(0)

                        cand_input_ids_parts.append(tool_result_ids)
                        cand_attention_mask_parts.append(torch.ones_like(tool_result_ids))
                        cand_completion_mask_parts.append(torch.zeros_like(tool_result_ids))
                        continue
                    elif 'answer' in xml_parser.get_ordered_tags():
                        break
                    else:
                        error_message = {"role": "system", "content": "error: no tool call or answer found in the response, continue to next turn"}
                        current_messages.append(error_message)
                        
                        error_text = maybe_apply_chat_template({"prompt": [error_message]}, self.processing_class, tools=self.tools)["prompt"]
                        error_ids = self.processing_class(text=error_text, add_special_tokens=False).input_ids
                        error_ids = torch.tensor(error_ids, device=device).squeeze(0)
                        
                        cand_input_ids_parts.append(error_ids)
                        cand_attention_mask_parts.append(torch.ones_like(error_ids))
                        cand_completion_mask_parts.append(torch.zeros_like(error_ids))
                        continue

                final_completion_texts.append(current_messages)
                all_input_ids_list.append(torch.cat(cand_input_ids_parts))
                all_attention_mask_list.append(torch.cat(cand_attention_mask_parts))
                all_completion_mask_list.append(torch.cat(cand_completion_mask_parts))
                # all_pixel_values_videos_list.append(initial_inputs["pixel_values_videos"])
                # all_video_grid_thw_list.append(initial_inputs["video_grid_thw"])
                                
                # clear the RuntimeCtx
                RuntimeCtx.clear_storage()
                
        # Pad all sequences to the same length
        def pad_and_stack(tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
            max_len = max(t.size(0) for t in tensors)
            padded_tensors = []
            for t in tensors:
                padding_needed = max_len - t.size(0)
                if padding_needed > 0:
                    padded_tensors.append(torch.nn.functional.pad(t, (padding_needed, 0), value=pad_value))
                else:
                    padded_tensors.append(t)
            return torch.stack(padded_tensors)

        all_input_ids = pad_and_stack(all_input_ids_list, self.processing_class.pad_token_id) # (B*G, L)
        all_attention_mask = pad_and_stack(all_attention_mask_list, 0) # (B*G, L)
        completion_masks = pad_and_stack(all_completion_mask_list, 0) # (B*G, L)
        del all_input_ids_list, all_attention_mask_list, all_completion_mask_list
        gc.collect()
        # all_pixel_values_videos = torch.cat(all_pixel_values_videos_list, 0)
        # all_video_grid_thw = torch.cat(all_video_grid_thw_list, 0)
        
        # Get logps for the full sequences by iterating over each candidate to save memory
        # NOTE: assume per device batch size is 1
        full_logps_list = []
        full_ref_logps_list = []

        for i in range(num_candidates):
            input_ids_cand = all_input_ids[i].unsqueeze(0)
            attention_mask_cand = all_attention_mask[i].unsqueeze(0)
            pixel_values_videos_cand = initial_inputs["pixel_values_videos"]
            video_grid_thw_cand = initial_inputs["video_grid_thw"]

            logps_cand = self._get_per_token_logps(
                model, input_ids_cand, attention_mask_cand, pixel_values_videos_cand, video_grid_thw_cand
            )
            full_logps_list.append(logps_cand.squeeze(0))
            
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_logps_cand = self._get_per_token_logps(
                        self.ref_model, input_ids_cand, attention_mask_cand, pixel_values_videos_cand, video_grid_thw_cand
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_logps_cand = self._get_per_token_logps(
                            model, input_ids_cand, attention_mask_cand, pixel_values_videos_cand, video_grid_thw_cand
                        )
                full_ref_logps_list.append(ref_logps_cand.squeeze(0))

        full_logps = torch.stack(full_logps_list)
        full_ref_logps = torch.stack(full_ref_logps_list)

        per_token_logps = full_logps
        per_token_kl = torch.exp(full_ref_logps - full_logps) - (full_ref_logps - full_logps) - 1
        del full_logps_list, full_ref_logps_list, all_input_ids, all_attention_mask, full_ref_logps
        gc.collect()
        
        # We need to shift the mask to the left to align with the logits, which are for predicting the *next* token.
        # input_ids[i] -> logit[i-1] -> predicts input_ids[i]
        completion_mask = completion_masks[:, 1:].int()

        # The loss is computed only on the tokens that were generated by the assistant.
        # Other tokens (prompts, tool results) have a mask of 0.

        # The tensors are already padded, but we apply the mask.
        per_token_logps = per_token_logps * completion_mask
        per_token_kl = per_token_kl * completion_mask


        # Get the whole conversation history for reward calculation
        conversations = [ConversationParser(conv) for conv in final_completion_texts]
        # if is_conversational(inputs[0]):
        #     conversations = [[{"role": "assistant", "content": completion}] for completion in conversations]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        labels = [label for label in labels for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, conversations)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, conversations)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                # prepare for some reward calculations
                reward_kwargs['prompts'] = prompts
                reward_kwargs['labels'] = labels
                reward_kwargs['completion_length'] = completion_mask.sum(1).float().cpu().tolist()
                reward_kwargs['num_generations'] = self.num_generations
                reward_kwargs['accelerator'] = self.accelerator
                reward_kwargs['test_feats'] = RuntimeCtx.test_feats
                reward_kwargs['train_feats'] = RuntimeCtx.train_feats
                reward_kwargs['top_k'] = self.agent_config.top_k
                reward_kwargs['main_frac'] = self.agent_config.main_frac
                reward_kwargs['prompt_type'] = prompt_type
                _result = reward_func(conversations=conversations, **reward_kwargs)
                # detect if the reward function is async function (api call)
                if inspect.isawaitable(_result):
                    output_reward_func = asyncio.run(_result)
                else:
                    output_reward_func = _result
                # for combined reward function, return 3 values
                if isinstance(output_reward_func, tuple):
                    output_reward_func, _func_names, _reward_per_func = output_reward_func
                    _reward_per_func = _reward_per_func.to(device)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        #---------------Log a random response for visualization----------------
        # print the last sample group for visualization
        if self.accelerator.is_main_process:

            # Select a random sample from the last batch
            selected_idx = torch.randint(0, len(prompts), (1,)).item()
            
            # Log to wandb at specific steps
            # if wandb.run is not None:
            if wandb.run is not None and self.state.global_step % 30 == 0:
            
                train_sample_table = wandb.Table(columns=["Conversation", "Image"])                
                # Handle image if available
                sampled_video_index = selected_idx // self.num_generations
                wandb_image = wandb.Image(video_inputs[sampled_video_index][0]) # select the first frame of the video
                
                # Add data to table
                train_sample_table.add_data(
                    str(conversations[selected_idx]),
                    wandb_image
                )
                
                # Log the table
                wandb.log({
                    "visual/train_sample": train_sample_table,
                }, commit=False)
                
            # Print to console
            print('/---------------Random Selected Response-----------------/')
            print(f"[Response]:\n{conversations[selected_idx]}")
            print('/--------------------------------------------------------/\n')
        #----------------------------------------------------------------------
    
        
        # Sum the rewards from all reward functions
        # rewards = rewards_per_func.sum(dim=1) # (B*G,)
        # Or apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        #-----------------------------------------------------------
        
        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        torch.cuda.empty_cache()
        
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Evaluation loop for GRPO trainer.
        """
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        
        self.model.eval()
        if self.ref_model is not None:
            self.ref_model.eval()

        prompt_types = ["cot", "fast"]
        overall_metrics = {}

        for prompt_type in prompt_types:
            # Step 0: Initialization
            model_names = list(RuntimeCtx.model_list.keys())
            dataset_name = self.agent_config.DATA.DATASET
            model_name_to_idx = {name: i for i, name in enumerate(model_names)}
            model_index_to_name = {i: name for name, i in model_name_to_idx.items()}
            N = len(model_names)
            num_eval_samples = len(dataloader.dataset)
            local_model_mask = torch.zeros((num_eval_samples, N), dtype=torch.bool, device=self.accelerator.device)
            local_anchor_model_index = torch.zeros((num_eval_samples,), dtype=torch.int, device=self.accelerator.device)
            local_acc_answer = torch.zeros((num_eval_samples,), dtype=torch.float, device=self.accelerator.device)
            eval_results = {}
            local_conversations = []
            
            # total_flops = 0.0
            start_time = time.time()

            with torch.inference_mode():
                for batch in tqdm(dataloader, desc=f"Evaluating ({prompt_type})"):
                    for i, example in enumerate(batch):
                        original_index = example['index']
                        example['body_image_keys'] = example['body_image_keys'][:self.agent_config.num_sample]
                        example['face_image_keys'] = example['face_image_keys'][:self.agent_config.num_sample]
                        example['has_face'] = example['has_face'][:self.agent_config.num_sample]
                        
                        msg = self._prepare_conversation([example], prompt_type=prompt_type)
                        video_input = [self.load_and_resize_image(self.agent_raw_dataset.body_data[f]) for f in example['body_image_keys']]
                        
                        cand_input_ids_parts: list[torch.Tensor] = []
                        cand_attention_mask_parts: list[torch.Tensor] = []
                        cand_completion_mask_parts: list[torch.Tensor] = []
                        
                        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                            num_params = sum(p.numel() for p in unwrapped_model.parameters())
                            tools_kwargs = {
                                "body_pil_image": [Image.fromarray(np.array(self.agent_raw_dataset.body_data[f])) for f in example['body_image_keys']],
                                "face_pil_image": [Image.fromarray(np.array(self.agent_raw_dataset.face_data[f])) for f in example['face_image_keys']],
                                "body_image_keys": example['body_image_keys'],
                                "face_image_keys": example['face_image_keys'],
                                "subject_id": example['subject_id'],
                                "has_face": example['has_face'],
                            }
                            current_messages = copy.deepcopy(msg[0]['prompt'])
                            
                            initial_prompt_example = {"prompt": current_messages}
                            initial_prompt_text = maybe_apply_chat_template(initial_prompt_example, self.processing_class, tools=self.tools)["prompt"]                       
                            initial_inputs = self.processing_class(
                                text=[initial_prompt_text],
                                videos=[video_input],
                                return_tensors="pt",
                                padding=False,
                                add_special_tokens=False,
                            )
                            initial_inputs = super()._prepare_inputs(initial_inputs)
                            
                            cand_input_ids_parts.append(initial_inputs["input_ids"].squeeze(0))
                            cand_attention_mask_parts.append(initial_inputs["attention_mask"].squeeze(0))
                            cand_completion_mask_parts.append(torch.zeros_like(initial_inputs["input_ids"].squeeze(0)))
                            
                            for _turn in range(self.agent_config.max_turns):
                                turn_input_ids = torch.cat(cand_input_ids_parts).unsqueeze(0)
                                if self.max_prompt_length is not None:
                                    turn_input_ids = turn_input_ids[:, -self.max_prompt_length:]
                                
                                turn_attention_mask = torch.cat(cand_attention_mask_parts).unsqueeze(0)
                                if self.max_prompt_length is not None:
                                    turn_attention_mask = turn_attention_mask[:, -self.max_prompt_length:]
                                
                                prompt_len = turn_input_ids.shape[1]
                                turn_out_ids = unwrapped_model.generate(
                                    input_ids=turn_input_ids,
                                    attention_mask=turn_attention_mask,
                                    pixel_values_videos=initial_inputs["pixel_values_videos"],
                                    video_grid_thw=initial_inputs["video_grid_thw"],
                                    generation_config=self.generation_config_val
                                )
                                completion_len = turn_out_ids.shape[1] - prompt_len
                                
                                turn_completion_ids = turn_out_ids[0, turn_input_ids.shape[1]:]
                                turn_text = self.processing_class.decode(turn_completion_ids, skip_special_tokens=True)
                                
                                cand_input_ids_parts.append(turn_completion_ids)
                                cand_attention_mask_parts.append(torch.ones_like(turn_completion_ids))
                                cand_completion_mask_parts.append(torch.ones_like(turn_completion_ids))
                                current_messages.append({"role": "assistant", "content": turn_text})
                                                           
                                xml_parser = XMLLikeTagParser(turn_text)
                                if 'tool_call' in xml_parser.get_ordered_tags():
                                    tool_input = xml_parser.parse_tool_calls()
                                    tool_result = self.execute_tool_from_response(tool_input, tools_kwargs)
                                    if tool_result is not None and isinstance(tool_result, dict):
                                        if 'model_name' in tool_result:
                                            model_name = tool_result['model_name']
                                            score_mat = RuntimeCtx.test_feats['score_mats'][model_name][original_index]
                                            pred_identity_id = str(RuntimeCtx.test_feats['g_pids'][score_mat.argmax()].item())
                                            tool_result['pred_identity_id'] = f'{dataset_name}_{pred_identity_id}'
    
                                    tool_result_str = str(tool_result)
                                    tool_message = {"role": "system", "content": f"<tool_result>{tool_result_str}</tool_result>"}
                                    current_messages.append(tool_message)
                                    
                                    tool_result_text = maybe_apply_chat_template({"prompt": [tool_message]}, self.processing_class, tools=self.tools)["prompt"]
                                    tool_result_ids = self.processing_class(text=tool_result_text, add_special_tokens=False).input_ids
                                    tool_result_ids = torch.tensor(tool_result_ids, device=self.accelerator.device).squeeze(0)
                                    
                                    cand_input_ids_parts.append(tool_result_ids)
                                    cand_attention_mask_parts.append(torch.ones_like(tool_result_ids))
                                    cand_completion_mask_parts.append(torch.zeros_like(tool_result_ids))
                                    continue
                                elif 'answer' in xml_parser.get_ordered_tags():
                                    if example['subject_id'] in xml_parser.get_contents('answer'):
                                        local_acc_answer[original_index] = 1
                                    break
                                else: # Error case
                                    error_message = {"role": "system", "content": "error: no tool call or answer found in the response, continue to next turn"}
                                    current_messages.append(error_message)
                                    
                                    error_text = maybe_apply_chat_template({"prompt": [error_message]}, self.processing_class, tools=self.tools)["prompt"]
                                    error_ids = self.processing_class(text=error_text, add_special_tokens=False).input_ids
                                    error_ids = torch.tensor(error_ids, device=self.accelerator.device).squeeze(0)
                                    
                                    cand_input_ids_parts.append(error_ids)
                                    cand_attention_mask_parts.append(torch.ones_like(error_ids))
                                    cand_completion_mask_parts.append(torch.zeros_like(error_ids))
                                    continue
                            
                            local_conversations.append({
                                "index": original_index,
                                "conversation": current_messages,
                                "label": example['subject_id'],
                            })
                            RuntimeCtx.clear_storage()
    
                        parser = ConversationParser(current_messages)
                        model_list = []
                        for turn in parser.parsed_turns:
                            if 'tool_call' in turn.get_ordered_tags():
                                try:
                                    tool_input = turn.parse_tool_calls()
                                    for call in tool_input:
                                        if 'tool_kwargs' in call and 'model_name' in call['tool_kwargs']:
                                            model_list.append(call['tool_kwargs']['model_name'])
                                except Exception:
                                    continue
                        for model_name in model_list:
                            if model_name in model_name_to_idx:
                                local_anchor_model_index[original_index] = model_name_to_idx[model_name]
                                break                    
                        for model_name in set(model_list):
                            if model_name in model_name_to_idx:
                                local_model_mask[original_index, model_name_to_idx[model_name]] = True
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_sample = round(total_time / num_eval_samples, 2) if num_eval_samples > 0 else 0.0
            
            gathered_masks_list = self.accelerator.gather_for_metrics(local_model_mask)
            gathered_anchor_model_index = self.accelerator.gather_for_metrics(local_anchor_model_index)
            gathered_acc_answer = self.accelerator.gather_for_metrics(local_acc_answer)
            gathered_conversations = gather_object(local_conversations)
            
            if isinstance(gathered_masks_list, list):
                model_mask = torch.stack(gathered_masks_list, dim=0).any(dim=0).cpu()
                anchor_model_index = torch.stack(gathered_anchor_model_index, dim=0).max(dim=0)[0].cpu()
                acc_answer = torch.stack(gathered_acc_answer, dim=0).any(dim=0).cpu()
                acc_answer_rate = (acc_answer.sum() / num_eval_samples).item()
            else:
                model_mask = gathered_masks_list.reshape(self.accelerator.num_processes, num_eval_samples, N).any(dim=0).cpu()
                anchor_model_index = gathered_anchor_model_index.reshape(self.accelerator.num_processes, num_eval_samples).max(dim=0)[0].cpu()
                acc_answer = gathered_acc_answer.reshape(self.accelerator.num_processes, num_eval_samples).any(dim=0).cpu()
                acc_answer_rate = (acc_answer.sum() / num_eval_samples).item()
            
            if self.accelerator.is_main_process:
                if hasattr(self, "_globalstep_last_logged"):
                    torch.save({'model_mask': model_mask, 'model_index': model_name_to_idx, 'anchor_model_index': anchor_model_index}, os.path.join(self.args.output_dir, f"model_mask_step{self._globalstep_last_logged}_{prompt_type}.pt"))
                    output_path = os.path.join(self.args.output_dir, f"conversations_step{self._globalstep_last_logged}_{prompt_type}.jsonl")
                else:
                    torch.save({'model_mask': model_mask, 'model_index': model_name_to_idx, 'anchor_model_index': anchor_model_index}, os.path.join(self.args.output_dir, f"model_mask_zeroshot_{prompt_type}.pt"))
                    output_path = os.path.join(self.args.output_dir, f"conversations_zeroshot_{prompt_type}.jsonl")
                all_conversations = sorted(gathered_conversations, key=lambda x: x['index'])
                with open(output_path, 'w') as f:
                    for item in all_conversations:
                        f.write(json.dumps(item) + '\n')
                        
            print(f"Model usage statistics ({prompt_type}): {model_mask.sum(dim=0).tolist()}")
            
            test_feats = RuntimeCtx.test_feats
            score_mats = {m: test_feats['score_mats'][m].cpu().numpy() for m in model_names}
            merge_score_mats = {m: test_feats['merge_score_mats'][m].cpu().numpy() for m in model_names}
      
            model_index_to_name = {i: name for name, i in model_name_to_idx.items()}
            anchor_model_list = [model_index_to_name[i.item()] for i in anchor_model_index]
            fused_score_mat, fused_merge_score_mat = act_score_fusion(score_mats, merge_score_mats, model_mask, anchor_model_list, top_k=10, norm_method='zscore')
    
            fused_result = test_score(
                fused_score_mat,
                fused_merge_score_mat,
                test_feats['q_pids'], test_feats['q_camids'], test_feats['q_clothes_ids'], 
                test_feats['g_pids'], test_feats['g_camids'], test_feats['g_clothes_ids'], test_feats['unique_g_pids'], 
                dataset=test_feats['dataset_name'], log=True, seed=45)
            
            for k, v in fused_result.items():
                if k in ['GR_top1', 'GR_mAP', r'TAR@1.00%FAR', r'FNIR@1.00%FPIR']:
                    eval_results[f"{metric_key_prefix}_{prompt_type}_{k}"] = v
            
            eval_results[f"{metric_key_prefix}_{prompt_type}_performance"] = fused_result['GR_top1'] + fused_result['GR_mAP'] + fused_result[r'TAR@1.00%FAR'] - fused_result[r'FNIR@1.00%FPIR']
            eval_results[f"{metric_key_prefix}_{prompt_type}_answer_accuracy"] = acc_answer_rate
            eval_results[f"{metric_key_prefix}_{prompt_type}_avg_time_s"] = avg_time_per_sample
            # eval_results[f"{metric_key_prefix}_{prompt_type}_flops"] = total_flops_all_processes
            overall_metrics.update(eval_results)

        self.accelerator.wait_for_everyone()
        
        # metris for tracking the best model
        overall_metrics['eval_performance'] = overall_metrics[f"{metric_key_prefix}_cot_performance"]
        
        return EvalLoopOutput(
            predictions=[],
            label_ids=[],
            metrics=overall_metrics,
            num_samples=num_eval_samples
        )

    # ---------------------------- Tool Functions ----------------------------
    def _initialize_tool(self) -> None:
        """
        Initialize center features and tool models (separate from the MLLM).
        This mirrors the debug initialization flow while keeping it encapsulated.
        """
        from fusionagent.WBModules import Model_Wrapper  # lazy import to avoid circular deps
        from fusionagent.data import VID_DATASET
        from fusionagent.utils.load_center_utils import load_center_feat

        BACKBONE_CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '../WBModules', f'model_cfg_{self.agent_config.DATA.DATASET}.yaml'), 'r'))
        config = self.agent_config
        dataset_type = "vid" if getattr(config.DATA, "DATASET", None) in VID_DATASET else "img"

        # Build models listed in BACKBONE_CFG
        model_names: list[str] = BACKBONE_CFG["model_list"]
        tool_model_list: dict[str, Any] = {}
        for mode in model_names:
            model = Model_Wrapper(
                mode=mode,
                backbone_cfg=BACKBONE_CFG,
                config=config,
                is_training=False,
                dataset_type=dataset_type,
            )
            model.eval().to(self.accelerator.device)
            tool_model_list[mode] = model

        # Center features for retrieval-based tools
        # init the score matrices for train set
        if hasattr(self.agent_config, 'few_shot') and self.agent_config.few_shot is not None:
            center_feats = load_center_feat(model_names, self.agent_raw_dataset, 'cpu', few_shot=self.agent_config.few_shot)
            train_feats = load_scoremats_from_h5(f'./src/fusionagent/train_feats/scoremats_{self.agent_config.DATA.DATASET}_train_fewshot{self.agent_config.few_shot}.h5', model_names)
        else:
            center_feats = load_center_feat(model_names, self.agent_raw_dataset, 'cpu')
            train_feats = load_scoremats_from_h5(f'./src/fusionagent/train_feats/scoremats_{self.agent_config.DATA.DATASET}_train.h5', model_names)

        # print(f"Rank {self.accelerator.process_index}: Center feats: {center_feats['center_pids'].device}")
        # self.tool_model_list = tool_model_list
        
        train_feats['dataset_name'] = self.agent_config.DATA.DATASET
            
        # init the score matrices for test set
        test_feats = load_scoremats_from_h5(f'./src/fusionagent/test_feats/scoremats_{self.agent_config.DATA.DATASET}.h5', model_names)
        test_feats['dataset_name'] = self.agent_config.DATA.DATASET
        
        # Prime the global runtime context for tools (hidden params not exposed to agent)
        RuntimeCtx.set(model_list=tool_model_list, center_feats=center_feats, 
                       dataset_name=self.agent_config.DATA.DATASET, 
                       train_feats=train_feats, test_feats=test_feats)
        
    def execute_tool_from_response(self, response: List[Dict[str, Any]], 
                                   sequence: Optional[Dict[str, Any]] = None) -> Any:
        """
        Parse a <tool_call>...</tool_call> block and execute the mapped tool with
        pre-initialized models and center features. Example response:
        <tool_call>{"name": "get_result_tool", "parameters": {"model_name": "adaface"}}</tool_call>
        """
        if not isinstance(response, list):
            raise ValueError("'response' must be a list of tool call dicts")
        
        # Runtime context: sequence, model_list, center_feats
        sequence = sequence or {}
        # Update hidden runtime context so tool wrappers can access without schema exposure
        RuntimeCtx.set(sequence=sequence)
        # print(f"Rank {self.accelerator.process_index}: RuntimeCtx image keys: {RuntimeCtx.sequence['body_image_keys']}")
        last_result: Any = None

        for call in response:
            if not isinstance(call, dict):
                continue
            tool_name = call.get("name")
            if not tool_name:
                last_result = {"error": "invalid tool call format"}
                continue

            # Normalize params to tool_kwargs structure
            tool_kwargs = call.get("tool_kwargs") or {}
            if not isinstance(tool_kwargs, dict):
                last_result = {"error": "tool_kwargs must be a dict"}
                continue
            
            # Merge everything into tool_kwargs
            tool_kwargs.update({"sequence": sequence})

            tool_func = self.tool_funcs.get(tool_name)
            if tool_func is None:
                # Unknown tool name; skip gracefully
                last_result = {"error": f"unknown tool name: {tool_name}, available tools: {list(self.tool_funcs.keys())}"}
                continue
            
            try:
                input_args = {}
                sig = inspect.signature(tool_func)
                for args in sig.parameters:
                    # hard code for model_list and center_feats
                    if args == "model_list":
                        input_args[args] = self.tool_model_list
                    else:
                        input_args[args] = tool_kwargs.get(args)
                last_result = tool_func(**input_args)
            except Exception as exec_err:
                # print(f"Error in calling tool {tool_name}: {exec_err}, {traceback.format_exc()}")
                last_result = {"error": str(exec_err), "tool": tool_name}

        return last_result
