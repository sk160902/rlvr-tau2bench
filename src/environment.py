"""
RLVR Environment wrapping τ²-bench for training language models with verifiable rewards.

This module is the central piece of the training pipeline. Its primary responsibility is
taking raw τ²-bench tasks and converting them into the (prompt, completion, reward) format
that TRL's GRPOTrainer expects. Understanding why this conversion is necessary requires
understanding the difference between a standard Gymnasium RL loop and an RLVR training loop.

In a standard Gymnasium loop, an agent interacts with an environment step by step: it observes
a state, takes an action, receives a reward, and observes the next state. This works well for
games and simulations but is too slow for training large language models, because each step
requires a separate model inference call.

In RLVR, the loop is collapsed into a single step: the model receives a full prompt containing
all the context it needs (the system policy, the available tools, and the customer's opening
message), generates a complete response in one shot, and that response is immediately scored
by deterministic reward functions. This means we need prompts, not step-by-step environments.

The Tau2BenchRLVREnvironment class handles three main concerns. First, it loads tasks from the
official τ²-bench train split using the tau2 Python API, converting each task's user scenario
and ground-truth actions into Episode objects that are ready to be formatted as prompts.
Second, it constructs the system prompt for the retail domain, which includes the real τ²-bench
retail policy (covering identity verification, confirmation requirements, and escalation rules)
and the full schemas of all 15 available retail tools. Including the complete tool schemas is
critical because the model can only generate correctly formatted tool calls if it knows what
parameters each tool expects. Third, it exposes a build_dataset() method that converts all
loaded episodes into a HuggingFace Dataset object, which is the format that GRPOTrainer
consumes during training.

The module also supports augmenting the real τ²-bench tasks with synthetic tasks loaded from
a JSON file. Synthetic tasks are programmatically generated customer service scenarios that
follow the same format as real tasks but with randomised customer names, order IDs, and issue
types. This augmentation increases the diversity of training prompts and reduces the risk of
the model memorising specific details from the 74 real training tasks rather than learning
generalizable tool-use patterns.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset


@dataclass
class ConversationTurn:
    role: str  # "user", "assistant", "system", "tool_result"
    content: str
    tool_call: Optional[dict] = None


@dataclass
class Episode:
    """A single evaluation episode from τ²-bench."""
    task_id: str
    domain: str
    system_prompt: str
    conversation: list[ConversationTurn] = field(default_factory=list)
    available_tools: list[dict] = field(default_factory=list)
    ground_truth_actions: list[dict] = field(default_factory=list)
    task_description: str = ""


class Tau2BenchRLVREnvironment:
    """
    Wraps τ²-bench into an RLVR-compatible data pipeline.

    Instead of running a live Gym loop, this environment pre-generates
    training prompts from τ²-bench tasks. Each prompt represents a
    conversation state where the agent must produce the next response.

    For GRPO training, the flow is:
        1. Sample a batch of prompts from the dataset
        2. Generate K completions per prompt using the policy model
        3. Execute each completion in the τ²-bench simulator
        4. Compute verifiable rewards
        5. Update the policy with GRPO

    Observation Space:
        Token sequences representing the conversation history, including:
        - System prompt with domain-specific policy rules
        - User messages (from the user simulator)
        - Previous assistant responses and tool results
        - Available tool schemas (as structured text)

        Formally: obs ∈ V^L where V is the vocabulary and L ≤ max_prompt_length

    Action Space:
        The agent produces a text response that may contain:
        - A natural language message to the user
        - A tool call in structured format: {"tool": <name>, "args": {...}}
        - A combination of message + tool call

        Formally: action ∈ V^M where M ≤ max_completion_length
    """

    # τ²-bench domain definitions with tool schemas
    DOMAIN_TOOLS = {
        "retail": [
            {
                "name": "calculate",
                "description": "Calculate the result of a mathematical expression",
                "parameters": {
                    "expression": {"type": "string"},
                },
            },
            {
                "name": "cancel_pending_order",
                "description": "Cancel a pending order. If the order is already processed or delivered, it cannot be cancelled",
                "parameters": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
            {
                "name": "exchange_delivered_order_items",
                "description": "Exchange items in a delivered order to new items of the same product type",
                "parameters": {
                    "order_id": {"type": "string"},
                    "item_ids": {"type": "array", "items": {"type": "string"}},
                    "new_item_ids": {"type": "array", "items": {"type": "string"}},
                    "payment_method_id": {"type": "string"},
                },
            },
            {
                "name": "find_user_id_by_name_zip",
                "description": "Find user id by first name, last name, and zip code",
                "parameters": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "zip": {"type": "string"},
                },
            },
            {
                "name": "find_user_id_by_email",
                "description": "Find user id by email",
                "parameters": {
                    "email": {"type": "string"},
                },
            },
            {
                "name": "get_order_details",
                "description": "Get the status and details of an order",
                "parameters": {
                    "order_id": {"type": "string"},
                },
            },
            {
                "name": "get_product_details",
                "description": "Get the inventory details of a product",
                "parameters": {
                    "product_id": {"type": "string"},
                },
            },
            {
                "name": "get_user_details",
                "description": "Get the details of a user, including their orders",
                "parameters": {
                    "user_id": {"type": "string"},
                },
            },
            {
                "name": "list_all_product_types",
                "description": "List the name and product id of all product types",
                "parameters": {},
            },
            {
                "name": "modify_pending_order_address",
                "description": "Modify the shipping address of a pending order. Ask for explicit user confirmation before proceeding",
                "parameters": {
                    "order_id": {"type": "string"},
                    "address1": {"type": "string"},
                    "address2": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "country": {"type": "string"},
                    "zip": {"type": "string"},
                },
            },
            {
                "name": "modify_pending_order_items",
                "description": "Modify items in a pending order to new items of the same product type. Can only be called once per order. Ask for explicit user confirmation before proceeding",
                "parameters": {
                    "order_id": {"type": "string"},
                    "item_ids": {"type": "array", "items": {"type": "string"}},
                    "new_item_ids": {"type": "array", "items": {"type": "string"}},
                    "payment_method_id": {"type": "string"},
                },
            },
            {
                "name": "modify_pending_order_payment",
                "description": "Modify the payment method of a pending order. Ask for explicit user confirmation before proceeding",
                "parameters": {
                    "order_id": {"type": "string"},
                    "payment_method_id": {"type": "string"},
                },
            },
            {
                "name": "modify_user_address",
                "description": "Modify the default address of a user. Ask for explicit user confirmation before proceeding",
                "parameters": {
                    "user_id": {"type": "string"},
                    "address1": {"type": "string"},
                    "address2": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string"},
                    "country": {"type": "string"},
                    "zip": {"type": "string"},
                },
            },
            {
                "name": "return_delivered_order_items",
                "description": "Return some items of a delivered order",
                "parameters": {
                    "order_id": {"type": "string"},
                    "item_ids": {"type": "array", "items": {"type": "string"}},
                    "payment_method_id": {"type": "string"},
                },
            },
            {
                "name": "transfer_to_human_agents",
                "description": "Transfer the user to a human agent, with a summary of the user's issue",
                "parameters": {
                    "summary": {"type": "string"},
                },
            },
        ],
        "airline": [
            {
                "name": "get_booking",
                "description": "Retrieve booking details",
                "parameters": {
                    "booking_ref": {"type": "string"}
                },
            },
            {
                "name": "search_flights",
                "description": "Search available flights",
                "parameters": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                    "cabin_class": {"type": "string", "enum": ["economy", "business", "first"]},
                },
            },
            {
                "name": "modify_booking",
                "description": "Modify an existing booking",
                "parameters": {
                    "booking_ref": {"type": "string"},
                    "new_flight_id": {"type": "string"},
                    "passenger_name": {"type": "string"},
                },
            },
            {
                "name": "cancel_booking",
                "description": "Cancel a booking and process refund",
                "parameters": {
                    "booking_ref": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
            {
                "name": "transfer_to_human",
                "description": "Transfer to a human agent",
                "parameters": {
                    "reason": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
        ],
        "telecom": [
            {
                "name": "get_account",
                "description": "Retrieve customer account details",
                "parameters": {
                    "account_id": {"type": "string"}
                },
            },
            {
                "name": "get_plan_details",
                "description": "Get details of a telecom plan",
                "parameters": {
                    "plan_id": {"type": "string"}
                },
            },
            {
                "name": "change_plan",
                "description": "Change customer's telecom plan",
                "parameters": {
                    "account_id": {"type": "string"},
                    "new_plan_id": {"type": "string"},
                },
            },
            {
                "name": "submit_trouble_ticket",
                "description": "Submit a technical support ticket",
                "parameters": {
                    "account_id": {"type": "string"},
                    "issue_type": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
            {
                "name": "transfer_to_human",
                "description": "Transfer to a human agent",
                "parameters": {
                    "reason": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
        ],
    }

    DOMAIN_POLICIES = {
        "retail": (
            "You are a retail customer service agent. You can help users cancel or modify pending orders, "
            "return or exchange delivered orders, modify their default address, and provide information "
            "about their profile, orders, and related products.\n\n"
            "Policies:\n"
            "1. Authenticate the user by finding their user id via email, or via name + zip code, even if the user provides their user id.\n"
            "2. You can only help one user per conversation but can handle multiple requests from the same user.\n"
            "3. Before any action that updates the database (cancel, modify, return, exchange), list the action details and get explicit user confirmation (yes) to proceed.\n"
            "4. Do not make up information not provided by the user or tools. Do not give subjective recommendations.\n"
            "5. Make at most one tool call at a time. If you make a tool call, do not respond to the user at the same time.\n"
            "6. Deny requests that are against policy.\n"
            "7. Transfer to a human agent only if the request cannot be handled within the scope of your actions. "
            "Call transfer_to_human_agents first, then send 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'"
        ),
        "airline": (
            "You are a customer service agent for an airline. Follow these policies:\n"
            "1. Always verify the customer's booking reference before making changes.\n"
            "2. Flight changes are free if made 24+ hours before departure.\n"
            "3. Cancellations within 24 hours of booking get a full refund.\n"
            "4. After 24 hours, a cancellation fee applies based on fare class.\n"
            "5. Upgrades depend on availability and fare difference.\n"
            "6. For medical emergencies, waive change fees and escalate.\n"
            "7. Always confirm changes with the customer before executing.\n"
            "8. Be professional and empathetic."
        ),
        "telecom": (
            "You are a customer service agent for a telecommunications company. Follow these policies:\n"
            "1. Verify the customer's account ID before accessing their information.\n"
            "2. Plan changes take effect at the start of the next billing cycle.\n"
            "3. Early termination fees apply for contract plans cancelled early.\n"
            "4. Technical issues should be troubleshot before submitting a ticket.\n"
            "5. Credits can be applied for verified service outages.\n"
            "6. Transfer to a specialist for billing disputes over $100.\n"
            "7. Always confirm actions with the customer before executing.\n"
            "8. Be patient and thorough in explanations."
        ),
    }

    def __init__(self, domain: str = "retail", task_split: str = "train", max_turns: int = 15):
        self.domain = domain
        self.task_split = task_split
        self.max_turns = max_turns
        self.tools = self.DOMAIN_TOOLS.get(domain, self.DOMAIN_TOOLS["retail"])
        self.policy = self.DOMAIN_POLICIES.get(domain, self.DOMAIN_POLICIES["retail"])
        self.episodes = self._load_episodes()

    def _load_episodes(self) -> list[Episode]:
        """
        Load or generate training episodes from τ²-bench tasks.

        In a full integration, this would call:
            from tau2.gym import AgentGymEnv
            env = AgentGymEnv(domain=self.domain, task_split=self.task_split)

        For standalone operation, we generate representative episodes that
        match the τ²-bench format exactly, then augment with any synthetic
        tasks found in synthetic_tasks_{domain}.json.
        """
        episodes = self._generate_representative_episodes()
        synthetic = self._load_synthetic_tasks_from_json()
        if synthetic:
            episodes = episodes + synthetic
            print(f"Augmented with {len(synthetic)} synthetic tasks → {len(episodes)} total episodes")
        return episodes

    def _load_synthetic_tasks_from_json(self, path: str = None) -> list[Episode]:
        """
        Load synthetic tasks from a JSON file (generated by src/synthetic_tasks.py).
        Returns an empty list if the file does not exist.
        """
        import os
        if path is None:
            path = f"synthetic_tasks_{self.domain}.json"
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                raw = json.load(f)
            episodes = []
            for item in raw:
                if item.get("domain", self.domain) != self.domain:
                    continue
                conversation = [
                    ConversationTurn(role=t["role"], content=t["content"])
                    for t in item.get("conversation", [])
                ]
                episodes.append(Episode(
                    task_id=item["task_id"],
                    domain=item.get("domain", self.domain),
                    system_prompt=self.policy,
                    conversation=conversation,
                    available_tools=self.tools,
                    ground_truth_actions=item.get("ground_truth_actions", []),
                    task_description=item.get("task_description", ""),
                ))
            return episodes
        except Exception as e:
            print(f"Warning: could not load synthetic tasks from {path}: {e}")
            return []

    def _generate_representative_episodes(self) -> list[Episode]:
        """
        Generate training episodes matching the τ²-bench task format.
        These serve as the training data for RLVR when τ²-bench is not installed,
        and as supplementary data when it is.
        """
        generators = {
            "retail": self._generate_retail_episodes,
            "airline": self._generate_airline_episodes,
            "telecom": self._generate_telecom_episodes,
        }
        return generators.get(self.domain, self._generate_retail_episodes)()

    def _generate_retail_episodes(self) -> list[Episode]:
        return [
            Episode(
                task_id="retail_001",
                domain="retail",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="Hi, I'd like to return my order. I received a damaged item."),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "message", "content": "ask for order ID"},
                    {"type": "tool_call", "tool": "get_order_details", "args": {"order_id": "ORD-12345"}},
                    {"type": "tool_call", "tool": "process_refund", "args": {"order_id": "ORD-12345", "amount": 49.99, "method": "original_payment"}},
                ],
                task_description="Customer wants to return a damaged item. Agent should verify identity, check order, and process the return/refund.",
            ),
            Episode(
                task_id="retail_002",
                domain="retail",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="I want to exchange my blue shoes for the same ones in red. Order ORD-67890."),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "tool_call", "tool": "get_order_details", "args": {"order_id": "ORD-67890"}},
                    {"type": "tool_call", "tool": "get_product_info", "args": {"product_id": "SHOE-001"}},
                    {"type": "tool_call", "tool": "update_order_status", "args": {"order_id": "ORD-67890", "status": "exchanged", "reason": "color exchange"}},
                ],
                task_description="Customer wants to exchange shoes for a different color. Same product category, so exchange is allowed.",
            ),
            Episode(
                task_id="retail_003",
                domain="retail",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="I bought something 45 days ago and want a refund."),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "message", "content": "ask for order ID"},
                    {"type": "message", "content": "explain 30-day refund policy, offer store credit or exchange as alternatives"},
                ],
                task_description="Customer requests refund outside 30-day window. Agent should politely decline and offer alternatives.",
            ),
            Episode(
                task_id="retail_004",
                domain="retail",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="My child got a rash from the fabric in your shirt. I want to speak to someone about product safety."),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "message", "content": "express concern, apologize"},
                    {"type": "tool_call", "tool": "transfer_to_human", "args": {"reason": "product safety concern", "priority": "high"}},
                ],
                task_description="Safety complaint requiring escalation to human agent per policy.",
            ),
            Episode(
                task_id="retail_005",
                domain="retail",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="Can you check if the wireless headphones model WH-200 are available? And what's the price?"),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "tool_call", "tool": "get_product_info", "args": {"product_id": "WH-200"}},
                    {"type": "message", "content": "provide product details"},
                ],
                task_description="Simple product inquiry. Agent should look up product and respond.",
            ),
            Episode(
                task_id="retail_006",
                domain="retail",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="I want a refund for order ORD-11111, I received the wrong size."),
                    ConversationTurn(role="assistant", content="I'm sorry to hear that. Let me look up your order right away."),
                    ConversationTurn(role="tool_result", content='{"order_id": "ORD-11111", "product": "T-Shirt M", "ordered": "2026-02-15", "total": 29.99, "status": "delivered"}'),
                    ConversationTurn(role="user", content="Yes, I ordered medium but got a large."),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "message", "content": "confirm wrong size, offer exchange or refund"},
                    {"type": "tool_call", "tool": "process_refund", "args": {"order_id": "ORD-11111", "amount": 29.99, "method": "original_payment"}},
                ],
                task_description="Mid-conversation: wrong size received, within policy window. Process refund or exchange.",
            ),
            Episode(
                task_id="retail_007",
                domain="retail",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="I need to cancel my order ORD-22222 immediately, I found it cheaper elsewhere."),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "tool_call", "tool": "get_order_details", "args": {"order_id": "ORD-22222"}},
                    {"type": "tool_call", "tool": "update_order_status", "args": {"order_id": "ORD-22222", "status": "cancelled", "reason": "customer request"}},
                    {"type": "tool_call", "tool": "process_refund", "args": {"order_id": "ORD-22222", "amount": 79.99, "method": "original_payment"}},
                ],
                task_description="Order cancellation request. Check order status and process cancellation with refund.",
            ),
            Episode(
                task_id="retail_008",
                domain="retail",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="I want to exchange my laptop for a blender. Order ORD-33333."),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "tool_call", "tool": "get_order_details", "args": {"order_id": "ORD-33333"}},
                    {"type": "message", "content": "explain exchanges only within same product category, offer refund + new purchase instead"},
                ],
                task_description="Cross-category exchange request. Policy restricts to same category. Agent should explain and offer alternatives.",
            ),
        ]

    def _generate_airline_episodes(self) -> list[Episode]:
        return [
            Episode(
                task_id="airline_001",
                domain="airline",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="I need to change my flight. Booking reference is BK-ABC123. I want to fly a day later."),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "tool_call", "tool": "get_booking", "args": {"booking_ref": "BK-ABC123"}},
                    {"type": "tool_call", "tool": "search_flights", "args": {"origin": "SFO", "destination": "JFK", "date": "2026-04-02", "cabin_class": "economy"}},
                    {"type": "tool_call", "tool": "modify_booking", "args": {"booking_ref": "BK-ABC123", "new_flight_id": "FL-456", "passenger_name": "John Doe"}},
                ],
                task_description="Flight change request. Check booking, find alternatives, modify if >24h before departure.",
            ),
            Episode(
                task_id="airline_002",
                domain="airline",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="I just booked 2 hours ago and need to cancel. Reference: BK-XYZ789."),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "tool_call", "tool": "get_booking", "args": {"booking_ref": "BK-XYZ789"}},
                    {"type": "tool_call", "tool": "cancel_booking", "args": {"booking_ref": "BK-XYZ789", "reason": "customer request within 24h"}},
                ],
                task_description="Cancellation within 24h of booking. Full refund per policy.",
            ),
            Episode(
                task_id="airline_003",
                domain="airline",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="My father had a medical emergency and I need to cancel my trip. Booking BK-MED001."),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "tool_call", "tool": "get_booking", "args": {"booking_ref": "BK-MED001"}},
                    {"type": "message", "content": "express empathy, waive change fees per medical emergency policy"},
                    {"type": "tool_call", "tool": "cancel_booking", "args": {"booking_ref": "BK-MED001", "reason": "medical emergency - fees waived"}},
                    {"type": "tool_call", "tool": "transfer_to_human", "args": {"reason": "medical emergency follow-up", "priority": "high"}},
                ],
                task_description="Medical emergency cancellation. Fees should be waived and escalated per policy.",
            ),
            Episode(
                task_id="airline_004",
                domain="airline",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="Can I upgrade from economy to business on booking BK-UPG001?"),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "tool_call", "tool": "get_booking", "args": {"booking_ref": "BK-UPG001"}},
                    {"type": "message", "content": "check availability and fare difference, confirm with customer"},
                ],
                task_description="Upgrade request. Check availability and explain fare difference.",
            ),
        ]

    def _generate_telecom_episodes(self) -> list[Episode]:
        return [
            Episode(
                task_id="telecom_001",
                domain="telecom",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="My internet has been down for 3 days. Account ID: ACC-TEL001. I want a credit."),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "tool_call", "tool": "get_account", "args": {"account_id": "ACC-TEL001"}},
                    {"type": "tool_call", "tool": "submit_trouble_ticket", "args": {"account_id": "ACC-TEL001", "issue_type": "outage", "description": "Internet down for 3 days"}},
                    {"type": "message", "content": "confirm outage, apply service credit"},
                ],
                task_description="Service outage complaint. Verify account, submit ticket, apply credit for verified outage.",
            ),
            Episode(
                task_id="telecom_002",
                domain="telecom",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="I want to switch from the Basic plan to the Premium plan. Account: ACC-TEL002."),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "tool_call", "tool": "get_account", "args": {"account_id": "ACC-TEL002"}},
                    {"type": "tool_call", "tool": "get_plan_details", "args": {"plan_id": "PREMIUM"}},
                    {"type": "message", "content": "confirm plan details and that change takes effect next billing cycle"},
                    {"type": "tool_call", "tool": "change_plan", "args": {"account_id": "ACC-TEL002", "new_plan_id": "PREMIUM"}},
                ],
                task_description="Plan upgrade. Verify account, show plan details, confirm, then change.",
            ),
            Episode(
                task_id="telecom_003",
                domain="telecom",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="I have a billing dispute of $250 on my last bill. Account ACC-TEL003."),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "tool_call", "tool": "get_account", "args": {"account_id": "ACC-TEL003"}},
                    {"type": "message", "content": "acknowledge dispute, explain transfer to billing specialist per >$100 policy"},
                    {"type": "tool_call", "tool": "transfer_to_human", "args": {"reason": "billing dispute over $100", "priority": "high"}},
                ],
                task_description="Billing dispute >$100. Must transfer to specialist per policy.",
            ),
            Episode(
                task_id="telecom_004",
                domain="telecom",
                system_prompt=self.policy,
                conversation=[
                    ConversationTurn(role="user", content="My wifi keeps dropping every 10 minutes. Account ACC-TEL004."),
                ],
                available_tools=self.tools,
                ground_truth_actions=[
                    {"type": "tool_call", "tool": "get_account", "args": {"account_id": "ACC-TEL004"}},
                    {"type": "message", "content": "troubleshoot: ask about router, connected devices, try restart"},
                    {"type": "tool_call", "tool": "submit_trouble_ticket", "args": {"account_id": "ACC-TEL004", "issue_type": "intermittent_connectivity", "description": "WiFi drops every 10 minutes"}},
                ],
                task_description="Technical issue. Troubleshoot first, then submit ticket if unresolved.",
            ),
        ]

    def format_prompt(self, episode: Episode, turn_index: Optional[int] = None) -> str:
        """
        Format an episode into a prompt string for the policy model.

        This is the observation the agent receives — it defines the state space.
        The prompt includes the system policy, available tools, and conversation
        history up to the point where the agent must respond.
        """
        parts = []

        # System prompt with domain policy
        parts.append(f"<|system|>\n{episode.system_prompt}\n")

        # Available tools
        tools_text = "Available tools:\n"
        for tool in episode.available_tools:
            params_str = ", ".join(
                f"{k}: {v['type']}" for k, v in tool.get("parameters", {}).items()
            )
            tools_text += f"- {tool['name']}({params_str}): {tool['description']}\n"
        parts.append(tools_text)

        parts.append(
            "When you need to call a tool, respond with:\n"
            '[tool_call] {"tool": "<tool_name>", "args": {<arguments>}}\n'
            "You may include a message before the tool call.\n"
        )

        # Conversation history
        conversation = episode.conversation
        if turn_index is not None:
            conversation = conversation[: turn_index + 1]

        for turn in conversation:
            if turn.role == "user":
                parts.append(f"<|user|>\n{turn.content}\n")
            elif turn.role == "assistant":
                parts.append(f"<|assistant|>\n{turn.content}\n")
            elif turn.role == "tool_result":
                parts.append(f"<|tool_result|>\n{turn.content}\n")

        # Signal that the assistant should respond
        parts.append("<|assistant|>\n")

        return "".join(parts)

    def build_training_dataset(self) -> Dataset:
        """
        Build a HuggingFace Dataset of prompts for GRPO training.

        Each row contains a 'prompt' field with the conversation context.
        The GRPO trainer will generate completions and score them.
        """
        prompts = []
        metadata = []

        for episode in self.episodes:
            prompt = self.format_prompt(episode)
            prompts.append(prompt)
            metadata.append({
                "task_id": episode.task_id,
                "domain": episode.domain,
                "task_description": episode.task_description,
                "ground_truth_actions": json.dumps(episode.ground_truth_actions),
                "num_expected_actions": len(episode.ground_truth_actions),
            })

            # Also create prompts for mid-conversation states if there are
            # multiple turns, to train on diverse conversation depths
            if len(episode.conversation) > 1:
                for i in range(len(episode.conversation) - 1):
                    prompt = self.format_prompt(episode, turn_index=i)
                    prompts.append(prompt)
                    metadata.append({
                        "task_id": f"{episode.task_id}_turn{i}",
                        "domain": episode.domain,
                        "task_description": episode.task_description,
                        "ground_truth_actions": json.dumps(episode.ground_truth_actions),
                        "num_expected_actions": len(episode.ground_truth_actions),
                    })

        dataset_dict = {
            "prompt": prompts,
            "task_id": [m["task_id"] for m in metadata],
            "domain": [m["domain"] for m in metadata],
            "task_description": [m["task_description"] for m in metadata],
            "ground_truth_actions": [m["ground_truth_actions"] for m in metadata],
            "num_expected_actions": [m["num_expected_actions"] for m in metadata],
        }

        return Dataset.from_dict(dataset_dict)

    def try_load_tau2_tasks(self) -> bool:
        """
        Attempt to load real τ²-bench tasks if the package is installed.
        Uses tau2.domains.<domain>.environment.get_tasks / get_tasks_split.
        Returns True if successful, False otherwise.
        """
        try:
            import importlib
            domain_mod = importlib.import_module(f"tau2.domains.{self.domain}.environment")
            get_tasks = getattr(domain_mod, "get_tasks")
            get_tasks_split = getattr(domain_mod, "get_tasks_split")

            # Load all tasks and the train/test split
            all_tasks = get_tasks()
            split = get_tasks_split()

            # Resolve which task IDs belong to the requested split
            split_key = self.task_split  # e.g. "train"
            if split_key not in split:
                split_key = list(split.keys())[0]
            train_ids = set(str(tid) for tid in split[split_key])

            # Filter tasks to the requested split
            tasks = [t for t in all_tasks if str(t.id) in train_ids]

            # Convert each τ²-bench Task into our Episode format
            real_episodes = []
            for task in tasks:
                instructions = (
                    task.user_scenario.instructions
                    if task.user_scenario and task.user_scenario.instructions
                    else None
                )
                # Build the opening user message from the scenario
                if instructions:
                    opening = (
                        f"{instructions.known_info}. {instructions.reason_for_call}"
                        if instructions.known_info
                        else instructions.reason_for_call
                    )
                else:
                    opening = "I need help with my order."

                # Convert evaluation_criteria actions to our ground-truth format
                gt_actions = []
                if task.evaluation_criteria and task.evaluation_criteria.actions:
                    for action in task.evaluation_criteria.actions:
                        gt_actions.append({
                            "type": "tool_call",
                            "tool": action.name,
                            "args": action.arguments or {},
                        })

                desc = ""
                if task.description:
                    desc = task.description.purpose or str(task.description)

                real_episodes.append(Episode(
                    task_id=f"tau2_{self.domain}_{task.id}",
                    domain=self.domain,
                    system_prompt=self.policy,
                    conversation=[ConversationTurn(role="user", content=opening)],
                    available_tools=self.tools,
                    ground_truth_actions=gt_actions,
                    task_description=desc,
                ))

            self.episodes = real_episodes
            print(f"Loaded {len(real_episodes)} real τ²-bench '{split_key}' tasks for domain '{self.domain}'")
            return True

        except ImportError:
            print("τ²-bench not installed. Using representative episodes.")
            return False
        except Exception as e:
            print(f"Failed to load τ²-bench tasks: {e}. Using representative episodes.")
            return False


def parse_agent_response(response: str) -> dict:
    """
    Parse an agent's response into structured components.
    Returns a dict with 'message' and optionally 'tool_calls'.
    """
    result = {"message": "", "tool_calls": []}

    # Extract tool calls
    tool_call_pattern = r'\[tool_call\]\s*(\{[^}]+\})'
    matches = re.finditer(tool_call_pattern, response)

    for match in matches:
        try:
            tool_call = json.loads(match.group(1))
            result["tool_calls"].append(tool_call)
        except json.JSONDecodeError:
            pass

    # The message is everything that's not a tool call
    message = re.sub(tool_call_pattern, "", response).strip()
    result["message"] = message

    return result
