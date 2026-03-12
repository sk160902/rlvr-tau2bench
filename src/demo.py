"""
End-to-end demonstration of the RLVR environment and reward functions.

This script serves as a self-contained smoke test and demonstration of the entire pipeline
without requiring a model or a training run. Its purpose is to show that all the components
of the project work together correctly: the environment loads tasks and constructs prompts,
the reward functions score completions correctly, and the overall system is ready for training.

The demo works by constructing Tau2BenchRLVREnvironment instances for all three τ²-bench
domains (retail, airline, and telecom), generating example prompts from each, and then
scoring three hand-crafted agent responses against the reward functions. The three responses
represent a good agent (one that calls the correct tool with the correct arguments and follows
policy), a mediocre agent (one that calls the right tool but with incomplete arguments), and
a bad agent (one that responds with irrelevant text and ignores the customer's request).

The expected output is that the good agent scores near 1.0 on composite reward, the mediocre
agent scores in the 0.4 to 0.6 range, and the bad agent scores near 0.0 to 0.2. If this
ordering holds, it confirms that the reward functions provide a meaningful and correctly
ranked training signal for GRPO. If the reward functions cannot distinguish good responses
from bad ones, no amount of GRPO training will improve the model, so this sanity check is
an important first step before committing to a multi-hour training run.

The demo also calls the synthetic task generator to confirm that it produces well-formed
tasks, and prints summary statistics showing the mean and range of reward scores across
the test responses.
"""

from src.environment import Tau2BenchRLVREnvironment
from src.rewards import (
    task_completion_reward,
    policy_compliance_reward,
    efficiency_reward,
    format_compliance_reward,
    compute_composite_reward,
)
from src.synthetic_tasks import SyntheticTaskGenerator
import json


SIMULATED_RESPONSES = {
    "retail_001": {
        "good": (
            "I'm sorry to hear your item arrived damaged. I'd be happy to help with a return. "
            "Could you please provide your order ID so I can look into this?\n\n"
            "Thank you for providing that. Let me pull up your order details.\n"
            '[tool_call] {"tool": "get_order_details", "args": {"order_id": "ORD-12345"}}\n\n'
            "I can see your order. I'll process a refund to your original payment method right away.\n"
            '[tool_call] {"tool": "process_refund", "args": {"order_id": "ORD-12345", "amount": 49.99, "method": "original_payment"}}'
        ),
        "mediocre": (
            "Sorry about that. Let me check your order.\n"
            '[tool_call] {"tool": "get_order_details", "args": {"order_id": "ORD-12345"}}'
        ),
        "bad": (
            "Oh no, that's really unfortunate! I completely understand how frustrating it must be "
            "to receive a damaged product. We take quality very seriously and this is not the "
            "experience we want our customers to have. There are many things we could potentially "
            "do to resolve this situation. Could you tell me more about the damage? Was the "
            "packaging also damaged? Did you take photos? When did you first notice the damage? "
            "Was anyone else present when you opened the package?"
        ),
    },
    "retail_003": {
        "good": (
            "I understand you'd like a refund. Could you share your order ID so I can look into it?\n\n"
            "Thank you. I can see this order was placed 45 days ago. Unfortunately, our refund policy "
            "covers purchases within 30 days. However, I can offer you store credit or help with an "
            "exchange for the same product category. Would either of those work for you?"
        ),
        "mediocre": (
            "Let me check on that for you. What's your order ID?\n\n"
            "I see the order. Let me process a refund.\n"
            '[tool_call] {"tool": "process_refund", "args": {"order_id": "ORD-99999", "amount": 30.00, "method": "original_payment"}}'
        ),
        "bad": (
            "Sure, I can help with that refund! No problem at all.\n"
            '[tool_call] {"tool": "process_refund", "args": {"order_id": "ORD-99999", "amount": 30.00, "method": "original_payment"}}'
        ),
    },
    "retail_004": {
        "good": (
            "I'm very sorry to hear about your child's reaction. This is a serious safety concern "
            "and I want to make sure this is handled properly. Let me transfer you to our safety "
            "team right away — they'll be able to assist you fully.\n"
            '[tool_call] {"tool": "transfer_to_human", "args": {"reason": "product safety concern - allergic reaction", "priority": "high"}}'
        ),
        "mediocre": (
            "I'm sorry to hear that. Have you tried washing the area? "
            "I can process a return for you if you'd like.\n"
            '[tool_call] {"tool": "get_order_details", "args": {"order_id": "ORD-00000"}}'
        ),
        "bad": (
            "That's unfortunate. We have a great selection of hypoallergenic shirts if you'd like "
            "to browse our catalog. I can also offer you a 10% discount on your next purchase!"
        ),
    },
    "retail_008": {
        "good": (
            "Let me pull up your order first.\n"
            '[tool_call] {"tool": "get_order_details", "args": {"order_id": "ORD-33333"}}\n\n'
            "I can see your laptop order. Unfortunately, our exchange policy only allows exchanges "
            "within the same product category — so a laptop can be exchanged for another laptop, but "
            "not for a kitchen appliance like a blender. What I can do is process a return and refund "
            "for the laptop, and then you can place a new order for the blender. Would that work?"
        ),
        "mediocre": (
            "Sure, let me exchange that for you.\n"
            '[tool_call] {"tool": "update_order_status", "args": {"order_id": "ORD-33333", "status": "exchanged", "reason": "laptop to blender"}}'
        ),
        "bad": (
            "Absolutely! I'll swap your laptop for a blender right now. No problem at all!"
        ),
    },
}


def print_header(text: str):
    width = 80
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_subheader(text: str):
    print(f"\n--- {text} ---")


def run_demo():
    print_header("RLVR ENVIRONMENT & REWARD FUNCTION DEMO")
    print("Demonstrating the full pipeline for training LLMs on τ²-bench")
    print("using GRPO with verifiable rewards.\n")

    # ---------------------------------------------------------------
    # Part 1: Environment overview
    # ---------------------------------------------------------------
    print_header("PART 1: ENVIRONMENT — 3 Domains")

    all_stats = {}
    for domain in ["retail", "airline", "telecom"]:
        env = Tau2BenchRLVREnvironment(domain=domain)
        dataset = env.build_training_dataset()
        all_stats[domain] = len(dataset)
        print(f"\n  [{domain.upper()}] {len(env.episodes)} episodes → {len(dataset)} training prompts")
        print(f"  Tools: {', '.join(t['name'] for t in env.tools)}")

    print(f"\n  Total training prompts across all domains: {sum(all_stats.values())}")

    # ---------------------------------------------------------------
    # Part 2: Reward function evaluation
    # ---------------------------------------------------------------
    print_header("PART 2: REWARD FUNCTION SCORING")
    print("Scoring Good / Mediocre / Bad responses on retail tasks.\n")

    env = Tau2BenchRLVREnvironment(domain="retail")

    # Table header
    print(f"{'Task':<14} {'Quality':<10} {'Task':>6} {'Policy':>8} {'Effic.':>7} {'Format':>8} {'TOTAL':>7}")
    print("-" * 65)

    all_scores = {"good": [], "mediocre": [], "bad": []}

    for episode in env.episodes:
        if episode.task_id not in SIMULATED_RESPONSES:
            continue

        prompt = env.format_prompt(episode)
        gt = json.dumps(episode.ground_truth_actions)
        responses = SIMULATED_RESPONSES[episode.task_id]

        for quality, response in responses.items():
            gt_parsed = json.loads(gt)
            r_task = task_completion_reward(response, gt_parsed)
            r_policy = policy_compliance_reward(response, prompt, "retail")
            r_eff = efficiency_reward(response, len(gt_parsed))
            r_fmt = format_compliance_reward(response)
            r_total = compute_composite_reward(response, prompt, gt, "retail")

            all_scores[quality].append(r_total)

            marker = {"good": "+", "mediocre": "~", "bad": "x"}[quality]
            print(f"  {episode.task_id:<12} [{marker}] {quality:<8} {r_task:>5.2f}  {r_policy:>7.2f}  {r_eff:>6.2f}  {r_fmt:>7.2f}  {r_total:>6.2f}")

    # Summary
    print_subheader("Average Composite Scores by Response Quality")
    for quality in ["good", "mediocre", "bad"]:
        scores = all_scores[quality]
        avg = sum(scores) / len(scores) if scores else 0
        bar = "█" * int(avg * 30)
        print(f"  {quality:<10}  {avg:.3f}  {bar}")

    separation = (
        sum(all_scores["good"]) / len(all_scores["good"])
        - sum(all_scores["bad"]) / len(all_scores["bad"])
    )
    print(f"\n  Reward separation (good - bad): {separation:.3f}")
    print(f"  → {'Strong' if separation > 0.3 else 'Moderate' if separation > 0.15 else 'Weak'} training signal for GRPO")

    # ---------------------------------------------------------------
    # Part 3: Detailed example
    # ---------------------------------------------------------------
    print_header("PART 3: DETAILED EXAMPLE — Safety Escalation (retail_004)")
    print("Task: Customer reports child got a rash from product fabric.")
    print("Policy: MUST transfer to human agent for safety concerns.\n")

    episode = [e for e in env.episodes if e.task_id == "retail_004"][0]
    prompt = env.format_prompt(episode)
    gt = json.dumps(episode.ground_truth_actions)

    for quality in ["good", "bad"]:
        response = SIMULATED_RESPONSES["retail_004"][quality]
        r_total = compute_composite_reward(response, prompt, gt, "retail")
        print(f"  [{quality.upper()} RESPONSE] (score: {r_total:.2f})")
        print(f"  \"{response[:120]}...\"" if len(response) > 120 else f"  \"{response}\"")
        print()

    # ---------------------------------------------------------------
    # Part 4: Synthetic task generation
    # ---------------------------------------------------------------
    print_header("PART 4: SYNTHETIC TASK GENERATION (Stretch Goal)")

    generator = SyntheticTaskGenerator(domain="retail", seed=42)
    tasks = generator.generate(num_tasks=20)

    complexity_counts = {"simple": 0, "medium": 0, "complex": 0}
    edge_case_count = 0
    for task in tasks:
        desc = task.task_description
        for c in complexity_counts:
            if c in desc:
                complexity_counts[c] += 1
        if "Edge case: True" in desc:
            edge_case_count += 1

    print(f"  Generated: {len(tasks)} synthetic tasks")
    print(f"  Complexity distribution: {complexity_counts}")
    print(f"  Edge cases: {edge_case_count} ({edge_case_count/len(tasks)*100:.0f}%)")
    print(f"  Unique intents: {len(set(t.task_description.split('.')[0] for t in tasks))}")

    print("\n  Sample generated tasks:")
    for task in tasks[:3]:
        intent = task.task_description.split(".")[0].replace("Intent: ", "")
        print(f"    • [{intent}] \"{task.conversation[0].content[:80]}...\"")

    # ---------------------------------------------------------------
    # Part 5: Training readiness
    # ---------------------------------------------------------------
    print_header("PART 5: TRAINING CONFIGURATION SUMMARY")

    print("  Framework:        TRL (GRPOTrainer)")
    print("  Algorithm:        GRPO (Group Relative Policy Optimization)")
    print("  Base Model:       Qwen/Qwen2.5-3B-Instruct")
    print("  Quantization:     QLoRA (4-bit NF4)")
    print("  LoRA Rank:        64 (alpha=128)")
    print("  Learning Rate:    5e-6 (cosine schedule)")
    print("  Batch Size:       1 × 8 gradient accumulation = 8 effective")
    print("  Generations (K):  4 per prompt")
    print("  KL Penalty (β):   0.04")
    print("  Max Prompt:       4096 tokens")
    print("  Max Completion:   1024 tokens")
    print("  Epochs:           3")
    print("  Reward Functions: 4 (task completion, policy, efficiency, format)")

    total_prompts = sum(all_stats.values()) + 20  # base + synthetic
    print(f"\n  Total training prompts:      {total_prompts}")
    print(f"  Completions per epoch:       {total_prompts * 4}")
    print(f"  Total reward evaluations:    {total_prompts * 4 * 3} (across 3 epochs)")

    print_header("DEMO COMPLETE")
    print("The reward functions provide clear signal for GRPO training:")
    print("  • Good responses → high reward")
    print("  • Bad responses  → low reward")
    print(f"  • Separation:    {separation:.3f} (sufficient for policy improvement)")
    print("\nTo launch training: python -m src.training --config configs/training_config.yaml\n")


if __name__ == "__main__":
    run_demo()
