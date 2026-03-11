"""
Evaluation script for the RLVR-trained model on τ²-bench.

Runs the trained model through τ²-bench evaluation episodes and reports
metrics on task completion, policy compliance, efficiency, and format quality.

Usage:
    python -m src.evaluate --model outputs/rlvr_retail/final --domain retail
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.environment import Tau2BenchRLVREnvironment
from src.rewards import (
    task_completion_reward,
    policy_compliance_reward,
    efficiency_reward,
    format_compliance_reward,
    compute_composite_reward,
)


def load_model(model_path: str, base_model: str = None):
    """Load the trained model (LoRA adapter + base)."""
    # Check if this is a LoRA adapter or full model
    adapter_config_path = os.path.join(model_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        # LoRA adapter — need base model
        if base_model is None:
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            base_model = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-3B-Instruct")

        print(f"Loading base model: {base_model}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        print(f"Loading full model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a single response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,  # Low temp for deterministic eval
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def evaluate(model_path: str, domain: str, base_model: str = None, num_samples: int = None):
    """Run evaluation and report metrics."""
    print(f"=== Evaluation: domain={domain} ===\n")

    # Load model
    model, tokenizer = load_model(model_path, base_model)

    # Build eval dataset
    env = Tau2BenchRLVREnvironment(domain=domain, task_split="train")
    dataset = env.build_training_dataset()

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} prompts\n")

    # Run evaluation
    results = []
    for i, sample in enumerate(dataset):
        prompt = sample["prompt"]
        gt_actions = sample["ground_truth_actions"]

        response = generate_response(model, tokenizer, prompt)

        # Compute individual rewards
        gt_parsed = json.loads(gt_actions)
        r_task = task_completion_reward(response, gt_parsed)
        r_policy = policy_compliance_reward(response, prompt, domain)
        r_efficiency = efficiency_reward(response, len(gt_parsed))
        r_format = format_compliance_reward(response)
        r_composite = compute_composite_reward(response, prompt, gt_actions, domain)

        results.append({
            "task_id": sample["task_id"],
            "task_completion": r_task,
            "policy_compliance": r_policy,
            "efficiency": r_efficiency,
            "format_compliance": r_format,
            "composite": r_composite,
            "response_length": len(response.split()),
        })

        print(f"[{i+1}/{len(dataset)}] {sample['task_id']}: "
              f"task={r_task:.2f} policy={r_policy:.2f} format={r_format:.2f} "
              f"composite={r_composite:.2f}")

    # Aggregate metrics
    print("\n=== Aggregate Results ===")
    metrics = {}
    for key in ["task_completion", "policy_compliance", "efficiency", "format_compliance", "composite"]:
        values = [r[key] for r in results]
        avg = sum(values) / len(values)
        metrics[key] = avg
        print(f"  {key:25s}: {avg:.4f}")

    avg_length = sum(r["response_length"] for r in results) / len(results)
    print(f"  {'avg_response_length':25s}: {avg_length:.1f} words")

    # Save results
    output_path = os.path.join(os.path.dirname(model_path), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "per_task": results}, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLVR model on τ²-bench")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--base-model", type=str, default=None, help="Base model for LoRA")
    parser.add_argument("--domain", type=str, default="retail")
    parser.add_argument("--num-samples", type=int, default=None)
    args = parser.parse_args()

    evaluate(args.model, args.domain, args.base_model, args.num_samples)


if __name__ == "__main__":
    main()
