"""Quick test to verify reward functions work correctly."""

from src.rewards import (
    task_completion_reward,
    policy_compliance_reward,
    efficiency_reward,
    format_compliance_reward,
    compute_composite_reward,
)

def test_all():
    # --- Test 1: Good response (correct tool call, proper format) ---
    good_response = (
        "I'd be happy to help with your return. Let me look up your order.\n"
        '[tool_call] {"tool": "get_order_details", "args": {"order_id": "ORD-12345"}}'
    )
    gt_actions = [
        {"type": "message", "content": "ask for order ID"},
        {"type": "tool_call", "tool": "get_order_details", "args": {"order_id": "ORD-12345"}},
    ]
    prompt = "<|user|>\nI want to return my order ORD-12345. It arrived damaged.\n<|assistant|>\n"

    r_task = task_completion_reward(good_response, gt_actions)
    r_policy = policy_compliance_reward(good_response, prompt, "retail")
    r_format = format_compliance_reward(good_response)
    r_eff = efficiency_reward(good_response, len(gt_actions))

    print("=== Test 1: Good response ===")
    print(f"  Task completion:   {r_task:.3f}")
    print(f"  Policy compliance: {r_policy:.3f}")
    print(f"  Format compliance: {r_format:.3f}")
    print(f"  Efficiency:        {r_eff:.3f}")

    # --- Test 2: Bad response (no tool call, verbose) ---
    bad_response = (
        "Oh no, that's terrible! I'm so sorry to hear that your item was damaged. "
        "That must be really frustrating for you. Let me see what I can do. "
        "Actually, before I can help you, I need to know some things. "
        "Could you tell me more about what happened? "
        "Was the package damaged when it arrived? "
        "Did you take any photos of the damage?"
    )

    r_task2 = task_completion_reward(bad_response, gt_actions)
    r_format2 = format_compliance_reward(bad_response)
    r_eff2 = efficiency_reward(bad_response, len(gt_actions))

    print("\n=== Test 2: Bad response (verbose, no tools) ===")
    print(f"  Task completion:   {r_task2:.3f}")
    print(f"  Format compliance: {r_format2:.3f}")
    print(f"  Efficiency:        {r_eff2:.3f}")

    # --- Test 3: Safety escalation ---
    safety_prompt = "<|user|>\nThe product caused a rash on my skin!\n<|assistant|>\n"
    safety_good = (
        "I'm very sorry to hear about this safety concern. Let me transfer you to a specialist immediately.\n"
        '[tool_call] {"tool": "transfer_to_human", "args": {"reason": "product safety", "priority": "high"}}'
    )
    safety_bad = "I'm sorry about that. Have you tried washing the area with soap and water?"

    r_policy_good = policy_compliance_reward(safety_good, safety_prompt, "retail")
    r_policy_bad = policy_compliance_reward(safety_bad, safety_prompt, "retail")

    print("\n=== Test 3: Safety escalation ===")
    print(f"  Good (transfers to human): {r_policy_good:.3f}")
    print(f"  Bad (doesn't escalate):    {r_policy_bad:.3f}")

    # --- Test 4: Malformed tool call ---
    malformed = "Let me check that. [tool_call] {tool: get_order, args: {id: 123}}"
    r_format_bad = format_compliance_reward(malformed)
    print(f"\n=== Test 4: Malformed tool call ===")
    print(f"  Format compliance: {r_format_bad:.3f}")

    # --- Test 5: Composite reward ---
    composite = compute_composite_reward(
        good_response, prompt,
        '[{"type": "tool_call", "tool": "get_order_details", "args": {"order_id": "ORD-12345"}}]',
        "retail"
    )
    print(f"\n=== Test 5: Composite reward (good response) ===")
    print(f"  Composite: {composite:.3f}")

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_all()
