# RLVR Environment for τ²-bench

An RLVR (Reinforcement Learning with Verifiable Rewards) environment that trains LLMs to improve performance on the [τ²-bench](https://github.com/sierra-research/tau2-bench) customer service benchmark using [TRL](https://github.com/huggingface/trl)'s GRPO trainer.

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Dry run — verify everything works (no GPU needed)
python -m src.training --dry-run

# Generate synthetic tasks (stretch goal)
python -m src.synthetic_tasks

# Test reward functions
python -m src.test_rewards

# Full training (requires GPU with 16GB+ VRAM)
python -m src.training --config configs/training_config.yaml

# Evaluate trained model
python -m src.evaluate --model outputs/rlvr_retail/final --domain retail
```

## Architecture

```
src/
├── environment.py       # RLVR environment wrapping τ²-bench
│                        #   - Observation space: token sequences (conversation context)
│                        #   - Action space: text responses with tool calls
├── rewards.py           # 4 verifiable reward functions
│                        #   1. Task completion (correct tools + args)
│                        #   2. Policy compliance (domain rules followed)
│                        #   3. Efficiency (concise responses)
│                        #   4. Format compliance (valid JSON tool calls)
├── training.py          # GRPO training with TRL
├── evaluate.py          # Evaluation script
├── synthetic_tasks.py   # Synthetic task generator (stretch goal)
└── test_rewards.py      # Reward function verification
```

## Design

See [WRITEUP.md](WRITEUP.md) for the full design document covering:
- Framework selection rationale (TRL/GRPO)
- Benchmark selection rationale (τ²-bench)
- State/observation space definition
- Action space definition
- Reward function design (4 functions)
- Model selection and training plan
- Scaling considerations
- Stretch goals (synthetic tasks, dual environment, scaling)
