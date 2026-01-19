# SimpleStories Training

Training tiny language models (5M-35M parameters) on the [SimpleStories](https://arxiv.org/abs/2504.09184) dataset, a parameterized synthetic dataset designed for training small, interpretable language models.

## Background

This project explores the "TinyStories" line of research showing that very small transformer models (<10M parameters) can generate coherent English when trained on appropriately simple data. SimpleStories improves on the original TinyStories dataset with better diversity and preserved metadata labels.

See `../background.md` for a full literature review of tiny language models.

## Setup

```bash
# Install dependencies
make install

# Set up WandB logging (optional)
cp .env.example .env
# Edit .env with your WANDB_API_KEY
```

## Training

Train a model using one of the provided configs:

```bash
# Available model sizes: 1.25M, 5M, 11M, 30M, 35M
python simple_stories_train/train_llama.py simple_stories_train/5M_config.yaml
python simple_stories_train/train_llama.py simple_stories_train/11M_config.yaml
```

If using Python 3.12 with PyTorch < 2.4, disable torch.compile:
```bash
python simple_stories_train/train_llama.py simple_stories_train/5M_config.yaml --compile=False
```

Training logs to WandB if `wandb_project` is set in the config.

## Generation

Interactive story generation with a trained model:

```bash
python generate.py
```

## Model Architecture

All models use a Llama-style transformer architecture:
- RoPE (Rotary Position Embeddings)
- Grouped Query Attention (GQA)
- SwiGLU MLP activation
- RMSNorm

| Config | Layers | Heads | Embed | Parameters |
|--------|--------|-------|-------|------------|
| 1.25M  | 4      | 4     | 128   | ~1.25M     |
| 5M     | 6      | 4     | 256   | ~5M        |
| 11M    | 6      | 6     | 384   | ~11M       |
| 30M    | 10     | 8     | 512   | ~30M       |
| 35M    | 12     | 8     | 512   | ~35M       |

## Trained Models

| Model | Final Loss | Training Steps | Notes |
|-------|------------|----------------|-------|
| 5M    | 1.92       | 18,560 (~2 epochs) | First experiment |
| 11M   | TBD        | TBD            | |

## Example Output (5M model)

**Prompt:** "In the forest, two friends discovered"

> in the forest, two friends discovered a hidden cave filled with sparkling stones. they stepped inside, curious about what they would find. suddenly, they heard a soft voice. "help me, please!" it was a tiny fairy stuck under a rock. the fairy waved her wand and sent bright light into the cave. the cave filled with light, and the fairy said, "thank you! you are the kindest explorer!" as a gift, the fairy gave the girl a tiny star. "whenever you dream, just close your eyes and believe," she said.

## References

- [SimpleStories Paper](https://arxiv.org/abs/2504.09184) - Finke et al., 2025
- [TinyStories Paper](https://arxiv.org/abs/2305.07759) - Eldan & Li, 2023
- Training code based on [llm.c](https://github.com/karpathy/llm.c) and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
