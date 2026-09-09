# Larger Context Window, Fewer Overcorrections

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-Findings%20of%20EMNLP%202026-b31b1b?style=for-the-badge&logo=googlescholar&logoColor=white)](paper.pdf)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/Environment-uv-DE5FE9?style=for-the-badge&logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![Claude Skill](https://img.shields.io/badge/Claude-GEC%20Skill-D97757?style=for-the-badge&logo=anthropic&logoColor=white)](.claude/skills/gec-prompt-optimise/SKILL.md)

</div>

Code and artifacts for **“Larger Context Window, Fewer Overcorrections: Optimizing Prompts and Batching for Minimal-Edit Grammatical Error Correction”**, accepted to **Findings of EMNLP 2026**.

We study taxonomy-grounded prompts, model-specific prompt optimization, and multi-sentence batching for English GEC. Our best configuration reaches **78.32 F0.5 on BEA-2019** and **67.08 F0.5 on CoNLL-2014**.

## Contents

- `src/`, `en_main.py`, `en_evaluation/`: API-model inference and evaluation
- `configs/`: main experiment configurations
- `src/agents/prompts/`: API-model prompt definitions
- `.claude/skills/gec-prompt-optimise/`: Claude prompt-optimization skill
- `data/`: BEA-2019 and CoNLL-2014 data
- `results/`: predictions and evaluation reports
- `paper.pdf`: camera-ready paper

## Quick start

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m spacy download en_core_web_sm
cp .env.example .env  # then add the required API key
```

Run the main configurations:

```bash
uv run python en_main.py --config configs/bea_dev.yaml
uv run python en_main.py --config configs/bea_test.yaml
uv run python en_main.py --config configs/conll14_test.yaml
```

Outputs are written to `outputs/<run_name>/`.

## More information

- Dataset layout and licensing: [`data/README.md`](data/README.md)
- Claude optimization skill: [`.claude/skills/gec-prompt-optimise/SKILL.md`](.claude/skills/gec-prompt-optimise/SKILL.md)

Credentials, model weights, raw API responses, and checkpoints are not included. Third-party datasets and models remain governed by their original terms.
