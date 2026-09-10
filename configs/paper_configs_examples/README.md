# Paper prompt configurations

These examples run the paper prompts on the BEA-2019 development set using
batch size 1. A.1--A.5 use GPT-4.1-mini. A.6 includes all four optimized,
model-specific variants:

- A.6.1: Qwen3-8B
- A.6.2: GPT-4.1-mini
- A.6.3: Claude Sonnet 4.6
- A.6.4: Gemini 3 Flash

```bash
uv run python en_main.py --config configs/paper_configs_examples/a1_vanilla_zero_shot.yaml
```

Change only the config path to run another prompt variant. A.3, A.5, A.6.1,
and A.6.3 use eight BEA-2019 train examples sampled with seed 123. A.6.2 uses
eight examples sampled with seed 712. A.6.4 has five examples embedded in the
prompt and does not sample additional examples.
