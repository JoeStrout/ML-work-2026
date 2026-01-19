# Design Document: Initial Experiments

## 1. Toy Problem Selection

### Criteria for a Good Toy Problem

1. **Clear benefit from tool use** — The task should be something networks struggle with, where an exact tool provides measurable advantage
2. **Simple success metric** — Binary correct/incorrect or low-dimensional error
3. **Scalable difficulty** — Can increase complexity to test generalization
4. **Fast iteration** — Training should complete in minutes or hours, not days
5. **Well-studied baseline** — Prior work exists for comparison

### Options Considered

| Problem | Pros | Cons | Verdict |
|---------|------|------|---------|
| **Modular arithmetic** | Well-studied (grokking); clear ground truth | Networks eventually learn it; benefit of tool unclear | Possible |
| **Multi-digit addition** | Networks struggle; extrapolation fails; clear tool benefit | Encoding design matters | **Recommended** |
| **Multi-digit multiplication** | Even harder for networks; strong tool benefit | May be too hard initially | Later |
| **Expression evaluation** | Tests parsing + computation | Too complex for initial tests | Later |
| **Lookup/memory** | Simple interface | Less scientifically interesting | No |

### Recommended: Multi-Digit Integer Addition

**Task:** Given two N-digit integers, produce their sum.

**Why this task:**
- Transformers and MLPs struggle with addition, especially as digit count increases (Nogueira et al., 2021)
- Networks cannot extrapolate: training on 5-digit fails on 6-digit
- An arithmetic tool guarantees correctness regardless of operand size
- Clear experimental design: compare tool-augmented vs. tool-free networks
- Natural difficulty scaling: increase N

**Experimental questions:**
1. Can evolution discover how to use an embedded addition tool?
2. Does tool-augmented network outperform tool-free network?
3. Does tool-augmented network generalize to longer operands (extrapolation)?
4. Under what conditions does the network learn to use vs. ignore the tool?

---

## 2. Network Architecture

### Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Input Encoding                        │
│         (Two N-digit integers → fixed-size vector)       │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                    Encoder Network                       │
│              (MLP: input → hidden representation)        │
└─────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
┌──────────────────────┐    ┌──────────────────────────────┐
│   Direct Pathway     │    │      Tool Pathway            │
│   (learned MLP)      │    │  ┌────────────────────────┐  │
│                      │    │  │ Decoder: hidden → ints │  │
│                      │    │  └──────────┬─────────────┘  │
│                      │    │             ▼                │
│                      │    │  ┌────────────────────────┐  │
│                      │    │  │ TOOL: exact addition   │  │
│                      │    │  │ (non-differentiable)   │  │
│                      │    │  └──────────┬─────────────┘  │
│                      │    │             ▼                │
│                      │    │  ┌────────────────────────┐  │
│                      │    │  │ Encoder: int → hidden  │  │
│                      │    │  └────────────────────────┘  │
└──────────┬───────────┘    └──────────────┬───────────────┘
           │                               │
           │         ┌─────────┐           │
           └────────►│  Gate   │◄──────────┘
                     │ (blend) │
                     └────┬────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Output Decoder                        │
│           (hidden → predicted sum digits)                │
└─────────────────────────────────────────────────────────┘
```

### Components

**1. Input Encoding**
- Each digit encoded as one-hot (10 dimensions) or binary (4 bits)
- Two operands concatenated
- Fixed maximum digit length (pad shorter numbers)

**2. Encoder Network**
- Simple MLP: Input → Hidden layers → Latent representation
- All weights evolved via ES

**3. Tool Module**
- **Decoder subnet:** Latent → discrete integers (argmax over digit logits)
- **Exact addition:** Python integer addition (non-differentiable black box)
- **Encoder subnet:** Result integer → latent representation

**4. Gate Mechanism**
- Learned scalar (evolved) passed through sigmoid
- Blends tool pathway output with direct pathway output
- `output = gate * tool_output + (1 - gate) * direct_output`

**5. Output Decoder**
- Latent → digit logits → predicted sum
- Cross-entropy or MSE loss equivalent for fitness

### Why This Architecture

- **Simplicity:** MLP is fast to evolve, easy to analyze
- **Modularity:** Tool pathway is cleanly separated
- **Interpretable gate:** Can measure how much the network relies on the tool
- **Upgradeable:** Can swap MLP for RNN/Transformer later

---

## 3. Evolution Strategy

### Algorithm: OpenAI-ES (Salimans et al., 2017)

Chosen for:
- Scalability (embarrassingly parallel)
- Simplicity (no populations to manage, just perturbations)
- Proven effectiveness on neural network optimization

### Fitness Function

```
fitness(θ) = -mean_loss over batch of addition problems

where loss = cross_entropy(predicted_digits, true_digits)
         or = (predicted_integer - true_integer)²
```

### Hyperparameters (Initial)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Population size | 50-100 | Balance exploration/compute |
| σ (noise std) | 0.02-0.05 | Standard for ES on NNs |
| Learning rate | 0.01-0.1 | Tunable |
| Batch size | 64-256 | Enough for stable fitness estimate |
| Max generations | 1000+ | Until convergence |

---

## 4. Experimental Plan

### Experiment 1: Baseline (No Tool)

- Train MLP to perform N-digit addition using ES
- Establish baseline accuracy vs. digit length
- Confirm networks struggle with extrapolation

### Experiment 2: Tool-Augmented Network

- Same task, but network has access to tool module
- Measure: accuracy, gate activation (tool usage), extrapolation

### Experiment 3: Ablations

- **Gate fixed at 0:** Force network to ignore tool
- **Gate fixed at 1:** Force network to always use tool
- **Random tool outputs:** Does network learn to avoid broken tools?

### Experiment 4: Scaling

- Vary digit length: 2, 4, 6, 8, 10, 12
- Train on N digits, test on N+2 (extrapolation test)

---

## 5. Open Design Questions

1. **Encoding scheme:** One-hot vs. binary vs. positional? Need to experiment.
2. **Tool decoder:** How to convert soft outputs to discrete integers? Argmax? Threshold?
3. **Gate granularity:** Single global gate vs. per-example gate vs. per-digit gate?
4. **Network size:** How small can we go while still seeing interesting behavior?
5. **Fitness shaping:** Raw accuracy vs. shaped rewards for partial credit?

---

## 6. Success Criteria

The core hypothesis is confirmed if:

1. Tool-augmented networks achieve higher accuracy than baseline on same-length test
2. Tool-augmented networks generalize to longer operands (extrapolation)
3. Gate activation correlates with task difficulty or tool usefulness
4. Evolution discovers tool usage without explicit supervision

If all four hold, we have evidence that neuroevolution can discover effective use of embedded non-differentiable tools.
