"""
Reward functions for RLVR training on τ²-bench.

This module implements multiple verifiable reward functions that together
incentivize the model to become a better customer service agent. The key
principle of RLVR is that rewards must be *verifiable* — computable
programmatically without a learned reward model.

Reward Functions:
    1. Task Completion Reward  — Did the agent take the correct actions?
    2. Policy Compliance Reward — Did the agent follow domain rules?
    3. Efficiency Reward        — Did the agent solve it in few turns?
    4. Format Compliance Reward — Are tool calls properly structured?
"""

import json
import re
from typing import Optional


def compute_composite_reward(
    completion: str,
    prompt: str,
    ground_truth_actions: str,
    domain: str = "retail",
    task_completion_weight: float = 1.0,
    policy_compliance_weight: float = 0.5,
    efficiency_weight: float = 0.2,
    format_weight: float = 0.3,
) -> float:
    """
    Compute the composite reward for a single (prompt, completion) pair.

    This is the main entry point called by the GRPO trainer's reward function.
    It combines all four reward signals into a single scalar.

    Args:
        completion: The model's generated response
        prompt: The conversation context
        ground_truth_actions: JSON string of expected actions
        domain: The τ²-bench domain
        task_completion_weight: Weight for task completion reward
        policy_compliance_weight: Weight for policy compliance reward
        efficiency_weight: Weight for efficiency reward
        format_weight: Weight for format compliance reward

    Returns:
        A scalar reward in [0, 1] (weighted sum, normalized)
    """
    gt_actions = json.loads(ground_truth_actions) if isinstance(ground_truth_actions, str) else ground_truth_actions

    r_task = task_completion_reward(completion, gt_actions)
    r_policy = policy_compliance_reward(completion, prompt, domain)
    r_efficiency = efficiency_reward(completion, len(gt_actions))
    r_format = format_compliance_reward(completion)

    total_weight = task_completion_weight + policy_compliance_weight + efficiency_weight + format_weight
    composite = (
        task_completion_weight * r_task
        + policy_compliance_weight * r_policy
        + efficiency_weight * r_efficiency
        + format_weight * r_format
    ) / total_weight

    return composite


# ---------------------------------------------------------------------------
# Reward Function 1: Task Completion (Primary)
# ---------------------------------------------------------------------------

def task_completion_reward(completion: str, ground_truth_actions: list[dict]) -> float:
    """
    Measures whether the agent performed the correct actions to solve the task.

    This is the primary verifiable reward. We check:
    - Did the agent call the correct tools?
    - Did the agent use the correct arguments?
    - Did the agent provide appropriate messages?

    Scoring:
        - For each expected action, check if it appears in the completion
        - Partial credit for correct tool name with wrong arguments
        - Full credit for correct tool name + correct arguments
        - Message actions get credit if semantically relevant keywords appear

    Returns:
        Float in [0, 1] — fraction of expected actions matched
    """
    if not ground_truth_actions:
        return 1.0  # No expected actions = trivially complete

    completion_lower = completion.lower()
    matched = 0
    total = len(ground_truth_actions)

    # Parse tool calls from the completion
    tool_calls = _extract_tool_calls(completion)
    tool_names_called = {tc.get("tool", "") for tc in tool_calls}

    for action in ground_truth_actions:
        action_type = action.get("type", "")

        if action_type == "tool_call":
            expected_tool = action.get("tool", "")
            expected_args = action.get("args", {})

            # Check if the tool was called
            if expected_tool in tool_names_called:
                # Find the matching call
                for tc in tool_calls:
                    if tc.get("tool") == expected_tool:
                        # Check argument overlap
                        actual_args = tc.get("args", {})
                        arg_score = _compute_arg_overlap(expected_args, actual_args)
                        matched += 0.5 + 0.5 * arg_score  # 50% for tool name, 50% for args
                        break
            # Partial credit if the tool name appears in text even without proper format
            elif expected_tool.lower().replace("_", " ") in completion_lower:
                matched += 0.25

        elif action_type == "message":
            expected_content = action.get("content", "")
            keywords = _extract_keywords(expected_content)
            if keywords:
                keyword_matches = sum(1 for kw in keywords if kw.lower() in completion_lower)
                matched += keyword_matches / len(keywords)
            else:
                # If no specific keywords, just check the agent produced a message
                if len(completion.strip()) > 10:
                    matched += 0.5

    return min(matched / total, 1.0)


# ---------------------------------------------------------------------------
# Reward Function 2: Policy Compliance
# ---------------------------------------------------------------------------

POLICY_RULES = {
    "retail": {
        "verify_identity": {
            "triggers": ["return", "refund", "cancel", "exchange", "order"],
            "required": ["order id", "order number", "ord-", "order_id", "order"],
            "description": "Must verify identity by asking for order ID before actions",
        },
        "refund_window": {
            "triggers": ["45 day", "60 day", "2 month", "last month"],
            "violation_patterns": ["process_refund"],
            "required_response": ["30 day", "policy", "unfortunately", "unable", "alternative"],
            "description": "Cannot process refunds outside 30-day window",
        },
        "safety_escalation": {
            "triggers": ["safety", "rash", "allergic", "injury", "dangerous", "harmful"],
            "required_tool": "transfer_to_human",
            "description": "Must transfer to human for safety concerns",
        },
        "same_category_exchange": {
            "triggers": ["exchange"],
            "description": "Exchanges only within same product category",
        },
        "confirm_before_action": {
            "description": "Must confirm with customer before executing actions",
        },
    },
    "airline": {
        "verify_booking": {
            "triggers": ["change", "cancel", "modify", "upgrade"],
            "required": ["booking", "reference", "bk-"],
            "description": "Must verify booking reference before changes",
        },
        "medical_emergency": {
            "triggers": ["medical", "emergency", "hospital", "doctor"],
            "required_tool": "transfer_to_human",
            "description": "Must waive fees and escalate for medical emergencies",
        },
    },
    "telecom": {
        "verify_account": {
            "triggers": ["plan", "bill", "service", "internet", "wifi"],
            "required": ["account", "acc-"],
            "description": "Must verify account ID before accessing information",
        },
        "billing_dispute_escalation": {
            "triggers": ["dispute", "billing dispute"],
            "required_tool": "transfer_to_human",
            "description": "Must transfer billing disputes over $100 to specialist",
        },
        "troubleshoot_first": {
            "triggers": ["wifi", "internet", "slow", "dropping", "connection"],
            "required_response": ["restart", "reboot", "troubleshoot", "try", "check"],
            "description": "Must troubleshoot technical issues before submitting ticket",
        },
    },
}


def policy_compliance_reward(completion: str, prompt: str, domain: str) -> float:
    """
    Checks whether the agent's response complies with domain-specific policies.

    This reward verifies rule-following behavior without needing a learned model.
    Each domain has explicit rules (e.g., "verify identity before taking action",
    "escalate safety concerns to human agent").

    Scoring:
        - Start with 1.0 (fully compliant)
        - Deduct for each policy violation detected
        - Violations are checked based on trigger keywords in the prompt

    Returns:
        Float in [0, 1] — 1.0 means fully policy-compliant
    """
    rules = POLICY_RULES.get(domain, {})
    if not rules:
        return 1.0

    prompt_lower = prompt.lower()
    completion_lower = completion.lower()
    tool_calls = _extract_tool_calls(completion)
    tool_names = {tc.get("tool", "") for tc in tool_calls}

    violations = 0
    applicable_rules = 0

    for rule_name, rule in rules.items():
        triggers = rule.get("triggers", [])
        is_triggered = any(t in prompt_lower for t in triggers)

        if not is_triggered:
            continue

        applicable_rules += 1

        # Check if required tool was called
        if "required_tool" in rule:
            if rule["required_tool"] not in tool_names:
                violations += 1
                continue

        # Check if required keywords appear in response
        if "required" in rule:
            has_required = any(r in completion_lower for r in rule["required"])
            if not has_required:
                # Check if it's in the prompt already (identity already verified)
                already_in_prompt = any(r in prompt_lower for r in rule["required"])
                if not already_in_prompt:
                    violations += 0.5

        # Check for violation patterns (doing something forbidden)
        if "violation_patterns" in rule:
            for pattern in rule["violation_patterns"]:
                if pattern in tool_names:
                    violations += 1

        # Check required response keywords
        if "required_response" in rule:
            has_response = any(r in completion_lower for r in rule["required_response"])
            if not has_response:
                violations += 0.5

    if applicable_rules == 0:
        return 1.0

    return max(0.0, 1.0 - violations / applicable_rules)


# ---------------------------------------------------------------------------
# Reward Function 3: Efficiency
# ---------------------------------------------------------------------------

def efficiency_reward(completion: str, expected_num_actions: int) -> float:
    """
    Rewards concise, efficient responses that don't waste turns.

    An ideal agent should resolve the customer's issue in minimal exchanges.
    We measure this by comparing the response length and number of actions
    to what's expected.

    Scoring:
        - Penalize excessively verbose responses (>3x expected length)
        - Penalize responses with too many tool calls (unnecessary actions)
        - Reward responses that are appropriately concise

    Returns:
        Float in [0, 1]
    """
    # Count actions in completion
    tool_calls = _extract_tool_calls(completion)
    num_actions = len(tool_calls)

    # Length-based efficiency
    word_count = len(completion.split())
    ideal_length = max(20, expected_num_actions * 30)  # ~30 words per action

    length_ratio = word_count / ideal_length if ideal_length > 0 else 1.0
    if length_ratio <= 1.5:
        length_score = 1.0
    elif length_ratio <= 3.0:
        length_score = 1.0 - (length_ratio - 1.5) / 3.0
    else:
        length_score = 0.2  # Floor: don't penalize too harshly

    # Action count efficiency
    if expected_num_actions == 0:
        action_score = 1.0 if num_actions == 0 else 0.5
    else:
        action_ratio = num_actions / expected_num_actions
        if 0.5 <= action_ratio <= 1.5:
            action_score = 1.0
        elif action_ratio < 0.5:
            action_score = action_ratio * 2  # Scale up
        else:
            action_score = max(0.3, 1.0 - (action_ratio - 1.5) / 2.0)

    return 0.6 * length_score + 0.4 * action_score


# ---------------------------------------------------------------------------
# Reward Function 4: Format Compliance
# ---------------------------------------------------------------------------

def format_compliance_reward(completion: str) -> float:
    """
    Verifies that the agent's response uses correct formatting for tool calls.

    Proper tool call format: [tool_call] {"tool": "<name>", "args": {<args>}}

    Scoring:
        - 1.0 if all tool invocations use the correct format
        - Penalize malformed JSON
        - Penalize tool calls without the [tool_call] prefix
        - 1.0 if no tool calls needed and none attempted

    Returns:
        Float in [0, 1]
    """
    # Check for any tool-call-like patterns
    has_tool_call_prefix = "[tool_call]" in completion
    has_json_braces = "{" in completion and "}" in completion
    has_tool_keyword = any(
        kw in completion.lower()
        for kw in ["tool_call", "function_call", "get_order", "process_refund",
                    "update_order", "get_booking", "search_flights", "get_account"]
    )

    if not has_tool_keyword and not has_json_braces:
        return 1.0  # No tool calls attempted, which is fine

    if not has_tool_call_prefix and has_tool_keyword:
        # Tried to call a tool but didn't use proper format
        return 0.3

    # Use the shared tool call parser
    parsed_calls = _extract_tool_calls(completion)

    if not parsed_calls and has_tool_call_prefix:
        return 0.4  # Had prefix but malformed JSON

    if not parsed_calls and not has_tool_call_prefix:
        # No tool calls attempted at all — check if there were informal attempts
        if has_tool_keyword:
            return 0.3  # Tried to use a tool informally
        return 1.0  # No tool calls needed or attempted

    valid_calls = 0
    total_calls = len(parsed_calls)

    for tc in parsed_calls:
        if "tool" in tc and "args" in tc:
            valid_calls += 1
        elif "tool" in tc:
            valid_calls += 0.5

    if total_calls == 0:
        return 0.5

    return valid_calls / total_calls


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _extract_tool_calls(text: str) -> list[dict]:
    """Extract structured tool calls from the agent's response."""
    tool_calls = []

    # Find all [tool_call] markers and extract the JSON that follows
    parts = text.split("[tool_call]")
    for part in parts[1:]:  # Skip everything before first marker
        part = part.strip()
        # Find the outermost JSON object by matching braces
        if not part.startswith("{"):
            continue
        depth = 0
        end_idx = -1
        for i, ch in enumerate(part):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        if end_idx > 0:
            json_str = part[: end_idx + 1]
            try:
                tc = json.loads(json_str)
                tool_calls.append(tc)
            except json.JSONDecodeError:
                pass

    return tool_calls


def _compute_arg_overlap(expected: dict, actual: dict) -> float:
    """Compute the fraction of expected arguments that match the actual ones."""
    if not expected:
        return 1.0

    matched = 0
    for key, value in expected.items():
        if key in actual:
            if str(actual[key]).lower() == str(value).lower():
                matched += 1
            else:
                matched += 0.5  # Right key, wrong value = partial credit

    return matched / len(expected)


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from a description string."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "for", "and", "or", "but",
        "in", "on", "at", "to", "of", "with", "by", "from", "as", "into",
        "that", "this", "it", "its", "not", "no", "if", "then", "than",
    }
    words = re.findall(r'[a-z]+', text.lower())
    return [w for w in words if w not in stop_words and len(w) > 2]


# ---------------------------------------------------------------------------
# Reward function factory for TRL GRPOTrainer
# ---------------------------------------------------------------------------

def make_reward_fn(
    domain: str = "retail",
    task_completion_weight: float = 1.0,
    policy_compliance_weight: float = 0.5,
    efficiency_weight: float = 0.2,
    format_weight: float = 0.3,
):
    """
    Factory function that returns a reward callable compatible with TRL's
    GRPOTrainer interface.

    The GRPOTrainer calls: reward_fn(completions, prompts, **kwargs) -> list[float]

    We wrap our composite reward to handle batched inputs and extract
    metadata from the dataset.
    """

    def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        """Batched reward function for GRPOTrainer."""
        rewards = []

        # Extract ground truth from kwargs if available
        ground_truths = kwargs.get("ground_truth_actions", [None] * len(completions))

        for completion, prompt, gt in zip(completions, prompts, ground_truths):
            if gt is None:
                # Fall back to format + policy compliance only
                r = (
                    format_compliance_reward(completion) * format_weight
                    + policy_compliance_reward(completion, prompt, domain) * policy_compliance_weight
                ) / (format_weight + policy_compliance_weight)
            else:
                r = compute_composite_reward(
                    completion=completion,
                    prompt=prompt,
                    ground_truth_actions=gt,
                    domain=domain,
                    task_completion_weight=task_completion_weight,
                    policy_compliance_weight=policy_compliance_weight,
                    efficiency_weight=efficiency_weight,
                    format_weight=format_weight,
                )
            rewards.append(r)

        return rewards

    return reward_fn
