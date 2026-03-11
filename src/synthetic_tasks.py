"""
Stretch Goal: Synthetic Task Generation for τ²-bench.

Generates novel tasks that conform to the τ²-bench format and constraints.
Tasks are created by composing atomic customer service scenarios with
varying complexity, policy edge cases, and multi-step resolutions.

Each synthetic task specifies:
    - A user persona with a specific intent
    - A conversation opening
    - Ground-truth actions the agent should take
    - Policy rules that apply
    - Expected outcome
"""

import json
import random
from dataclasses import dataclass, field
from typing import Optional

from src.environment import ConversationTurn, Episode


@dataclass
class TaskTemplate:
    """Template for generating synthetic tasks."""
    intent: str
    complexity: str  # "simple", "medium", "complex"
    user_openings: list[str]
    required_tools: list[str]
    policy_rules_tested: list[str]
    edge_case: bool = False


# Task template library for the retail domain
RETAIL_TEMPLATES = [
    TaskTemplate(
        intent="return_damaged_item",
        complexity="simple",
        user_openings=[
            "I received a broken {product}. I want my money back.",
            "My {product} arrived damaged. Can I get a refund?",
            "The {product} I ordered is defective. I need to return it.",
        ],
        required_tools=["get_order_details", "process_refund"],
        policy_rules_tested=["verify_identity", "refund_window"],
    ),
    TaskTemplate(
        intent="exchange_same_category",
        complexity="simple",
        user_openings=[
            "I want to exchange my {product} for a different {variant}.",
            "Can I swap my {product} for a different size?",
        ],
        required_tools=["get_order_details", "get_product_info", "update_order_status"],
        policy_rules_tested=["verify_identity", "same_category_exchange"],
    ),
    TaskTemplate(
        intent="exchange_cross_category",
        complexity="medium",
        user_openings=[
            "I bought a {product} but actually need a {other_product}. Can I exchange?",
            "Instead of the {product}, I'd rather have a {other_product}.",
        ],
        required_tools=["get_order_details"],
        policy_rules_tested=["same_category_exchange"],
        edge_case=True,
    ),
    TaskTemplate(
        intent="late_refund_request",
        complexity="medium",
        user_openings=[
            "I bought this {product} about {days} days ago and want a refund.",
            "I've had the {product} for {days} days, can I still return it?",
        ],
        required_tools=["get_order_details"],
        policy_rules_tested=["refund_window"],
        edge_case=True,
    ),
    TaskTemplate(
        intent="safety_complaint",
        complexity="complex",
        user_openings=[
            "The {product} caused an allergic reaction on my skin.",
            "My child got hurt using the {product}. This is a safety hazard!",
            "The {product} started smoking when I plugged it in. This is dangerous!",
        ],
        required_tools=["transfer_to_human"],
        policy_rules_tested=["safety_escalation"],
    ),
    TaskTemplate(
        intent="multi_issue",
        complexity="complex",
        user_openings=[
            "I have two issues: the {product} from order {order1} is damaged, and I also need to return the {product2} from order {order2}.",
            "I need help with multiple orders. {order1} has a wrong item and {order2} needs cancellation.",
        ],
        required_tools=["get_order_details", "process_refund", "update_order_status"],
        policy_rules_tested=["verify_identity", "refund_window", "confirm_before_action"],
    ),
    TaskTemplate(
        intent="price_match_request",
        complexity="medium",
        user_openings=[
            "I found my {product} cheaper at another store. Can you match the price?",
            "Your competitor sells the {product} for ${lower_price}. I want the difference refunded.",
        ],
        required_tools=[],
        policy_rules_tested=[],
        edge_case=True,  # Not covered by standard policy — agent should handle gracefully
    ),
]

# Product catalog for template filling
PRODUCTS = {
    "electronics": ["wireless headphones", "bluetooth speaker", "laptop charger", "USB hub", "smart watch"],
    "clothing": ["winter jacket", "running shoes", "cotton t-shirt", "denim jeans", "wool sweater"],
    "home": ["coffee maker", "desk lamp", "throw pillow", "wall clock", "kitchen knife set"],
}

PRODUCT_VARIANTS = {
    "wireless headphones": ["black version", "white version", "noise-cancelling model"],
    "running shoes": ["size 10", "size 11", "wide fit"],
    "cotton t-shirt": ["medium", "large", "blue one"],
    "winter jacket": ["size S", "size M", "waterproof version"],
}

ORDER_IDS = [f"ORD-{random.randint(10000, 99999)}" for _ in range(50)]


class SyntheticTaskGenerator:
    """
    Generates novel τ²-bench-compatible tasks by composing templates
    with randomized parameters.
    """

    def __init__(self, domain: str = "retail", seed: int = 42):
        self.domain = domain
        self.rng = random.Random(seed)
        self.templates = RETAIL_TEMPLATES  # Extend for other domains

    def generate(self, num_tasks: int = 20) -> list[Episode]:
        """Generate a batch of synthetic tasks."""
        tasks = []
        for i in range(num_tasks):
            template = self.rng.choice(self.templates)
            task = self._instantiate_template(template, task_index=i)
            tasks.append(task)
        return tasks

    def _instantiate_template(self, template: TaskTemplate, task_index: int) -> Episode:
        """Fill a template with random concrete values."""
        # Pick products
        category = self.rng.choice(list(PRODUCTS.keys()))
        product = self.rng.choice(PRODUCTS[category])
        other_category = self.rng.choice([c for c in PRODUCTS if c != category])
        other_product = self.rng.choice(PRODUCTS[other_category])
        variant = self.rng.choice(PRODUCT_VARIANTS.get(product, ["different color"]))
        order_id = self.rng.choice(ORDER_IDS)
        order_id2 = self.rng.choice([o for o in ORDER_IDS if o != order_id])
        days = self.rng.randint(31, 60) if template.edge_case else self.rng.randint(1, 28)
        lower_price = round(self.rng.uniform(10, 80), 2)

        # Fill template
        opening = self.rng.choice(template.user_openings).format(
            product=product,
            variant=variant,
            other_product=other_product,
            days=days,
            order1=order_id,
            order2=order_id2,
            product2=self.rng.choice(PRODUCTS[self.rng.choice(list(PRODUCTS.keys()))]),
            lower_price=lower_price,
        )

        # Build ground truth actions
        gt_actions = self._build_ground_truth(template, order_id, product)

        # Build conversation
        conversation = [ConversationTurn(role="user", content=opening)]

        # Build task description
        description = (
            f"Intent: {template.intent}. "
            f"Complexity: {template.complexity}. "
            f"Policy rules tested: {', '.join(template.policy_rules_tested)}. "
            f"Edge case: {template.edge_case}."
        )

        return Episode(
            task_id=f"synthetic_{self.domain}_{task_index:04d}",
            domain=self.domain,
            system_prompt="",  # Will be filled by environment
            conversation=conversation,
            available_tools=[],  # Will be filled by environment
            ground_truth_actions=gt_actions,
            task_description=description,
        )

    def _build_ground_truth(self, template: TaskTemplate, order_id: str, product: str) -> list[dict]:
        """Build expected actions based on template and policy rules."""
        actions = []

        # Identity verification is always first if needed
        if "verify_identity" in template.policy_rules_tested:
            actions.append({"type": "message", "content": "ask for order ID"})

        # Tool calls
        for tool in template.required_tools:
            if tool == "get_order_details":
                actions.append({"type": "tool_call", "tool": tool, "args": {"order_id": order_id}})
            elif tool == "process_refund":
                amount = round(random.uniform(10, 200), 2)
                actions.append({
                    "type": "tool_call", "tool": tool,
                    "args": {"order_id": order_id, "amount": amount, "method": "original_payment"},
                })
            elif tool == "update_order_status":
                actions.append({
                    "type": "tool_call", "tool": tool,
                    "args": {"order_id": order_id, "status": "exchanged", "reason": "customer request"},
                })
            elif tool == "get_product_info":
                actions.append({"type": "tool_call", "tool": tool, "args": {"product_id": product.replace(" ", "-")}})
            elif tool == "transfer_to_human":
                actions.append({
                    "type": "tool_call", "tool": tool,
                    "args": {"reason": template.intent, "priority": "high"},
                })

        # Edge case responses
        if template.edge_case:
            if "refund_window" in template.policy_rules_tested:
                actions.append({"type": "message", "content": "explain 30-day refund policy, offer store credit or exchange"})
            if "same_category_exchange" in template.policy_rules_tested:
                actions.append({"type": "message", "content": "explain same-category exchange policy, offer refund + new purchase"})
            if not template.policy_rules_tested:
                actions.append({"type": "message", "content": "explain this is not covered by current policy, offer alternatives"})

        return actions

    def export_to_json(self, tasks: list[Episode], output_path: str):
        """Export synthetic tasks to JSON for use with τ²-bench."""
        export = []
        for task in tasks:
            export.append({
                "task_id": task.task_id,
                "domain": task.domain,
                "task_description": task.task_description,
                "conversation": [
                    {"role": t.role, "content": t.content} for t in task.conversation
                ],
                "ground_truth_actions": task.ground_truth_actions,
            })

        with open(output_path, "w") as f:
            json.dump(export, f, indent=2)
        print(f"Exported {len(tasks)} synthetic tasks to {output_path}")


def main():
    """Generate synthetic tasks and preview them."""
    generator = SyntheticTaskGenerator(domain="retail", seed=42)
    tasks = generator.generate(num_tasks=20)

    print(f"Generated {len(tasks)} synthetic tasks\n")

    for task in tasks[:5]:
        print(f"--- {task.task_id} ---")
        print(f"Description: {task.task_description}")
        print(f"User: {task.conversation[0].content}")
        print(f"Expected actions: {len(task.ground_truth_actions)}")
        for a in task.ground_truth_actions:
            print(f"  - {a}")
        print()

    # Export
    generator.export_to_json(tasks, "synthetic_tasks_retail.json")


if __name__ == "__main__":
    main()
