# Claude Instructions for NanoTool Project

## Project Overview

This project investigates whether neural networks can learn to use embedded tool modules (e.g., arithmetic units) when trained via genetic algorithms or evolution strategies rather than backpropagation. The core hypothesis is that non-differentiable computation modules can be integrated directly into network architectures and that evolutionary optimization can discover effective interfaces to these tools.

## Key Documents

- **IDEA.md** - The original research questions and motivation
- **BACKGROUND.md** - Literature review covering neuroevolution, neural arithmetic, tool use, memory-augmented networks, and related work
- **references.bib** - BibTeX file for paper writing

Whenever you fetch and read a research paper, add a summary to BACKGROUND.md and an entry in references.bib.

## Research Log

Maintain a research log in **RESEARCH_LOG.md**. After completing any significant work (experiments, analysis, implementation, writing), append a dated entry summarizing what was accomplished.

**Rules for the research log:**
1. Always append new entries; never edit or delete previous content
2. Begin each entry with the date in ISO format (YYYY-MM-DD)
3. Keep entries concise but informative
4. Note key decisions, results, and next steps

## Conduct

- Maintain a professional, scientific tone in all documents and conversations
- Be precise in technical claims; distinguish speculation from established results
- When uncertain, investigate before asserting
- Document rationale for design decisions

## Project Configuration

This project uses `.claude/settings.json` to configure:
- **Default model:** Opus
- **Pre-approved operations:** Python execution, conda, git (except push), file I/O, web search
- **Requires approval:** git push, pip/conda install, file deletion
- **Denied:** destructive operations, sudo, reading .env files

## Development Environment

- **OS:** Ubuntu 22.04.5 LTS
- **CPU:** AMD Ryzen 7 7700X 8-Core Processor
- **GPU:** NVIDIA GeForce RTX 4090 (24 GB VRAM)
- **Python environment:** `py3_torch` (conda)
- **PyTorch:** 2.2.1 with CUDA 11.8

To activate the environment:
```bash
conda activate py3_torch
```

## Project Structure

```
nanotool/
├── src/                    # Core library
│   ├── evolution/          # ES/GA implementations
│   ├── networks/           # Neural network architectures
│   ├── tools/              # Non-differentiable tool modules
│   ├── encodings/          # Number encoding schemes
│   └── utils/              # Utilities
├── experiments/            # Individual experiments
│   ├── exp001_*/           # Each experiment has config.yaml, run.py, results/
│   └── ...
├── tests/                  # Unit tests
├── scripts/                # Utility scripts
└── paper/                  # Paper drafts and figures
```

## Current Status

Refer to the most recent entry in RESEARCH_LOG.md for project status.
