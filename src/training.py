"""
RLVR Training Script using TRL's GRPOTrainer.

This script is the entry point for running a full training run. It ties together every other
component in the project: it reads a YAML configuration file that specifies the model, LoRA
setup, GRPO hyperparameters, and reward weights; it uses Tau2BenchRLVREnvironment to build
the training dataset from τ²-bench tasks; it loads the language model with LoRA adapters
applied via the PEFT library; and it hands everything to TRL's GRPOTrainer to run the actual
reinforcement learning loop.

The algorithm being used is Group Relative Policy Optimization (GRPO), which was introduced
by the DeepSeek team as part of DeepSeek-R1. GRPO was chosen specifically because it does
not require a separate critic (value) network, which is what makes PPO expensive. Instead,
GRPO generates K completions for the same prompt, scores each one with the reward function,
and computes each completion's advantage as its reward minus the mean reward of the group.
This group-relative baseline reduces gradient variance without requiring any learned baseline
model. In this project K is set to 2 in the CPU configuration, meaning the model generates
two candidate responses for each training prompt and learns to prefer whichever one the
reward function scores higher.

LoRA (Low-Rank Adaptation) is used to make training feasible without a GPU. Rather than
updating all 3 billion parameters of Qwen2.5-3B-Instruct, LoRA inserts small trainable
adapter matrices into the query, key, value, and output projection layers of the attention
blocks. The number of trainable parameters is reduced from billions to a few million,
which means gradients and optimizer states fit comfortably in CPU memory. Two LoRA
configurations were used across the training runs: r=16 (alpha=32) for Run 1 and r=32
(alpha=64) for Run 2. A higher rank gives the adapter more expressive capacity at the
cost of slightly more memory and compute.

The script also supports a --dry-run flag that builds the dataset and prints sample prompts
without loading the model or starting training. This is useful for verifying that the
environment, tool schemas, and prompts are correctly constructed before committing to a
multi-hour training run.

Usage:
    python -m src.training --config configs/training_config_3b_cpu.yaml
    python -m src.training --dry-run
"""

import argparse
import os
import sys

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from src.environment import Tau2BenchRLVREnvironment
from src.rewards import make_reward_fn


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_dataset(config: dict) -> Dataset:
    """Build training dataset from τ²-bench environment."""
    env = Tau2BenchRLVREnvironment(
        domain=config["environment"]["domain"],
        task_split=config["environment"]["task_split"],
        max_turns=config["environment"]["max_turns"],
    )

    # Try to load real τ²-bench tasks; fall back to representative episodes
    env.try_load_tau2_tasks()

    dataset = env.build_training_dataset()
    print(f"Built training dataset with {len(dataset)} prompts")
    print(f"Domain: {config['environment']['domain']}")
    print(f"Sample prompt (first 200 chars): {dataset[0]['prompt'][:200]}...")
    return dataset


def setup_model_and_tokenizer(config: dict):
    """Load the base model with QLoRA quantization."""
    model_name = config["model"]["name"]

    # Quantization config for memory-efficient training
    bnb_config = None
    if config["model"].get("load_in_4bit", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Select dtype and device_map based on hardware availability
    if torch.cuda.is_available():
        model_dtype = torch.bfloat16
        device_map = "auto"
    else:
        model_dtype = torch.float32
        device_map = None
        bnb_config = None  # bitsandbytes requires CUDA

    dtype_override = config["model"].get("dtype", None)
    if dtype_override == "float32":
        model_dtype = torch.float32

    print(f"Loading model: {model_name} (dtype={model_dtype}, device_map={device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=model_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for generation

    return model, tokenizer


def setup_lora(config: dict) -> LoraConfig:
    """Configure LoRA adapters for parameter-efficient training."""
    lora_cfg = config["lora"]
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )


def setup_grpo_config(config: dict, output_dir: str) -> GRPOConfig:
    """Configure the GRPO trainer."""
    grpo = config["grpo"]

    use_cpu = not torch.cuda.is_available()

    return GRPOConfig(
        output_dir=output_dir,
        num_generations=grpo["num_generations"],
        max_completion_length=grpo["max_completion_length"],
        temperature=grpo["temperature"],
        beta=grpo["beta"],
        num_train_epochs=grpo["num_train_epochs"],
        per_device_train_batch_size=grpo["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo["gradient_accumulation_steps"],
        learning_rate=grpo["learning_rate"],
        lr_scheduler_type=grpo["lr_scheduler_type"],
        warmup_ratio=grpo["warmup_ratio"],
        max_grad_norm=grpo["max_grad_norm"],
        optim=grpo["optimizer"],
        bf16=grpo["bf16"] if torch.cuda.is_available() else False,
        logging_steps=grpo["logging_steps"],
        save_steps=grpo["save_steps"],
        seed=grpo["seed"],
        use_cpu=use_cpu,
        report_to="wandb" if config.get("wandb", {}).get("project") else "none",
        remove_unused_columns=False,
    )


def train(config: dict):
    """Main training loop."""
    output_dir = os.path.join("outputs", f"rlvr_{config['environment']['domain']}")

    # Build dataset
    dataset = build_dataset(config)

    # Setup model
    model, tokenizer = setup_model_and_tokenizer(config)
    lora_config = setup_lora(config)

    # Setup reward function
    reward_cfg = config["reward"]
    reward_fn = make_reward_fn(
        domain=config["environment"]["domain"],
        task_completion_weight=reward_cfg["task_completion_weight"],
        policy_compliance_weight=reward_cfg["policy_compliance_weight"],
        efficiency_weight=reward_cfg["efficiency_weight"],
        format_weight=reward_cfg["format_weight"],
    )

    # Setup GRPO config
    grpo_config = setup_grpo_config(config, output_dir)

    # Initialize wandb
    if config.get("wandb", {}).get("project"):
        os.environ["WANDB_PROJECT"] = config["wandb"]["project"]
        if config["wandb"].get("entity"):
            os.environ["WANDB_ENTITY"] = config["wandb"]["entity"]

    # Create trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    # Train
    print("Starting GRPO training...")
    print(f"  Model: {config['model']['name']}")
    print(f"  Domain: {config['environment']['domain']}")
    print(f"  Num generations per prompt (K): {config['grpo']['num_generations']}")
    print(f"  Learning rate: {config['grpo']['learning_rate']}")
    print(f"  Epochs: {config['grpo']['num_train_epochs']}")
    print(f"  Output: {output_dir}")

    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"Training complete. Model saved to {output_dir}/final")


def main():
    parser = argparse.ArgumentParser(description="RLVR Training for τ²-bench")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml")
    parser.add_argument("--domain", type=str, default=None, help="Override domain from config")
    parser.add_argument("--model", type=str, default=None, help="Override model from config")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--dry-run", action="store_true", help="Build dataset and exit")
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply overrides
    if args.domain:
        config["environment"]["domain"] = args.domain
    if args.model:
        config["model"]["name"] = args.model
    if args.lr:
        config["grpo"]["learning_rate"] = args.lr
    if args.epochs:
        config["grpo"]["num_train_epochs"] = args.epochs

    if args.dry_run:
        print("=== DRY RUN: Building dataset only ===")
        dataset = build_dataset(config)
        print(f"\nDataset size: {len(dataset)}")
        print(f"Columns: {dataset.column_names}")
        print(f"\n--- Sample Prompt ---")
        print(dataset[0]["prompt"])
        print(f"\n--- Ground Truth ---")
        print(dataset[0]["ground_truth_actions"])
        return

    train(config)


if __name__ == "__main__":
    main()
