import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import string
import argparse
import wandb
import logging
from time import time
from omegaconf import OmegaConf
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List
from accelerate import Accelerator
from datasets import Dataset as HFDataset
from transformers import set_seed
from math_verify import parse, verify
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from fusionagent.trainer import Qwen2VLGRPOTrainer
from fusionagent.data import VID_DATASET, build_interleave_dataset
from fusionagent.configs.default_img import get_img_config
from fusionagent.configs.default_vid import get_vid_config
from fusionagent.tool_func import face_detector_tool, get_result_tool
from fusionagent.reward_func import conversation_format_reward, tool_success_rate_reward, metric_reward, answer_accuracy_reward, multi_turn_reward, conversation_format_reward_v2
# from codebleu import calc_codebleu

reward_funcs_registry = {
    "format": conversation_format_reward,
    "format_v2": conversation_format_reward_v2,
    "multi_turn": multi_turn_reward,
    "metric": metric_reward,
    "tool_success_rate": tool_success_rate_reward,
    "answer_accuracy": answer_accuracy_reward,
}
tool_funcs_registry = {
    # "face_detector_tool": face_detector_tool,
    "get_result_tool": get_result_tool,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='/research/cvl-zhujie4/FusionAgent_CVPR26/src/fusionagent/configs/debug.yaml')
    parser.add_argument('--image_root', type=str, default='')
    parser.add_argument('--load_quantized', action='store_true')
    parser.add_argument('--use_accelerate', action='store_true', help='Whether to use Accelerate for distributed training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--test_only', action='store_true', help='Only run evaluation on the test set')

    args = parser.parse_args()
    configs = OmegaConf.load(args.configs)
    configs = OmegaConf.to_container(configs, resolve=True)

    # add configs to args
    for k, v in configs.items():
        setattr(args, k, v)

    return args

def main(args, accelerator):
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # # Fix random seeds for reproducibility
    set_seed(args.seed)
    print(f"✅ Random seed set to {args.seed}.")

    run_name = args.wandb_run_name
    os.makedirs(os.path.join(args.output_dir, run_name), exist_ok=True)    
    
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in args.reward_funcs]
    # Get tool functions
    args.tool_funcs = ['get_result_tool']
    tool_funcs = [tool_funcs_registry[func] for func in args.tool_funcs]

    # Build agent dataset (InterleaveDataset) and wrap to HF Dataset for GRPO
    if args.dataset in VID_DATASET:
        config = get_vid_config(args)
    else:
        config = get_img_config(args)

    start_time = time()
    train_interleave, query_interleave, gallery_interleave, raw_dataset = build_interleave_dataset(config, max_samples=args.max_samples)
    print(f"Time taken to build interleave dataset: {time() - start_time:.2f} seconds")

    dataset = {"train": HFDataset.from_list(train_interleave), "test": HFDataset.from_list(query_interleave)}
    print(f"Time taken to build HF dataset: {time() - start_time:.2f} seconds")
    # dataset = {"train": HFDataset.from_generator(lambda: train_interleave), "test": HFDataset.from_generator(lambda: query_interleave)}
      
    # Configure LoRA with optimized parameters
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",  # To save memory
        fan_in_fan_out=False,  # Set to True for specific modules that require it
    ) if args.use_lora and not args.test_only else None
        
    trainer_agent = Qwen2VLGRPOTrainer if args.train_method == "grpo" else None
    print("using: ", trainer_agent)
    if args.use_lora and 'zero3' in args.deepspeed:
        print("Detected using lora, deepspeed zero 3 will be disabled right now")
        args.deepspeed = None
    
    training_args = GRPOConfig(
        output_dir=os.path.join(args.output_dir, run_name),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size =args.per_device_eval_batch_size ,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.mixed_precision,
        bf16=args.bf16,
        deepspeed=args.deepspeed if args.deepspeed is not None and not args.use_lora else None,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.eval_steps if hasattr(args, 'eval_steps') else args.save_steps,
        save_total_limit=1,
        report_to="wandb" if args.use_wandb and accelerator.is_main_process else 'none',
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=True, 
        dataloader_prefetch_factor=args.prefetch_factor,
        max_prompt_length=args.max_prompt_length,  
        max_completion_length=args.max_completion_length,  
        optim="adamw_torch", # More efficient implementation
        max_grad_norm=1.0,  # Helps with training stability
        ddp_find_unused_parameters=False,  # Important for DDP
        gradient_checkpointing=args.gradient_checkpointing,
        save_on_each_node=False,
        eval_strategy="steps", # do not evaluate during training for fully finetuning
        eval_steps=args.eval_steps if hasattr(args, 'eval_steps') else 10,
        eval_delay=args.eval_delay,
        load_best_model_at_end=False,
        metric_for_best_model="eval_performance",
        greater_is_better=True,
        seed=args.seed,
    )

    # Initialize the GRPO trainer
    trainer = trainer_agent(
        model=args.model,
        model_cache_dir=args.model_cache_dir,
        resume_from_checkpoint=args.resume_from_checkpoint if 'resume_from_checkpoint' in args else None,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"] if training_args.eval_strategy != "no" else None, # use test_dataset for evaluation
        peft_config=peft_config,
        # attn_implementation=model_args.attn_implementation,
        max_pixels=args.max_pixels,
        # min_pixels=args.min_pixels,
        reward_weights=args.reward_weights,
        tools=tool_funcs,
        agent_config=config,
        agent_raw_dataset=raw_dataset,
    )

    if args.test_only:
        logger.info("Running evaluation only...")
        eval_metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {eval_metrics}")
    else:
        if args.few_shot is not None:
            logger.info(f"Zero-shot testing before training for few-shot setting")
            eval_metrics = trainer.evaluate()
        # Train and push the model to the Hub
        trainer.train()
    
        # Save and push to hub
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=args.dataset_name)

    # clean trainer and dataset
    # del trainer
    # del dataset
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()
    
    # Close wandb on main process only
    if args.use_wandb and  accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":

    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if hasattr(args, 'api_url'):
        API_URL = args.api_url    
    # Initialize accelerator early to check main process
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if args.mixed_precision else ("bf16" if args.bf16 else "no"),
    )
    args.wandb_run_name = args.wandb_run_name + "_" + datetime.now().strftime("%m%d_%H%M")
    # Initialize wandb only on the main process
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            save_code=True,
            config=args,
            settings=wandb.Settings(code_dir=os.path.dirname(os.path.abspath(__file__)))
        )
    main(args, accelerator)
