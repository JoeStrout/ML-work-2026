# Research Log

## 2026-01-08

**Literature review completed.**

Conducted a comprehensive survey of related work spanning neuroevolution, neural arithmetic learning, tool use in neural networks, memory-augmented architectures, and neuro-symbolic integration. Key findings:

- Evolution strategies and genetic algorithms are viable for training networks with millions of parameters (Salimans et al., Such et al.)
- Neural networks develop complex internal circuits for arithmetic but struggle with extrapolation (grokking literature, Nogueira et al.)
- Existing tool-use approaches (Toolformer, PAL) operate at the token level rather than embedding tools directly in network architecture
- The closest prior work is EDNC (evolving DNC controllers), but their external modules remain differentiable

Identified research gap: No prior work systematically explores embedding non-differentiable, exact computation modules into neural architectures and training via pure neuroevolution.

**Deliverables:** BACKGROUND.md, references.bib (40+ entries)

**Next steps:** Design initial experiments; determine network architecture, tool module interface, and encoding scheme.

---

## 2026-01-08 (continued)

**Project infrastructure and initial design completed.**

Selected multi-digit integer addition as the initial toy problem. Rationale:
- Well-documented that neural networks struggle with arithmetic extrapolation
- Clear benefit from exact tool: guaranteed correctness regardless of operand size
- Natural difficulty scaling via digit count
- Directly tests core hypothesis

Designed two-pathway architecture:
1. Direct pathway: learned MLP approximation
2. Tool pathway: decode → exact addition → encode
3. Learned gate blends the two pathways

Implemented core components:
- `src/evolution/es.py`: OpenAI-ES optimizer with antithetic sampling
- `src/tools/arithmetic.py`: Non-differentiable addition/multiplication tools
- `src/encodings/numbers.py`: Digit (one-hot) and binary encodings
- `src/networks/mlp.py`: SimpleMLP baseline and ToolAugmentedMLP

All components verified via unit tests.

Created experiment structure:
- `exp001_baseline_addition`: MLP without tool (control)
- `exp002_tool_augmented`: MLP with embedded addition tool

**Deliverables:** DESIGN.md, src/ library, experiment configs, passing tests

**Next steps:** Implement training loop; run baseline experiment; analyze results.

---

## 2026-01-08 (training loop)

**Training loop implemented and debugged.**

Implemented:
- `src/data/addition.py`: Data generation for addition tasks
- `src/training/trainer.py`: ES training loop with evaluation

Encountered and resolved stability issues:
1. Initial ES implementation diverged (fitness exploded negatively)
2. Root cause: fitness normalization combined with different batches per population member introduced high variance
3. Fix: All population members now evaluated on same batch per generation
4. Added rank-based fitness shaping (more robust than z-score normalization)
5. Added weight decay regularization

Tuned hyperparameters for stability:
- sigma: 0.02 → 0.1 (more exploration)
- learning_rate: 0.01 → 0.001 (prevent overshooting)
- weight_decay: 0.01 (regularization)

Verified learning occurs:
- 100 generations on 2-digit addition
- Digit accuracy improved from ~20% (random) to ~36%
- Fitness improved consistently (less negative loss)
- Training stable with no divergence

**Deliverables:** Training loop, data generation, tuned experiment configs

**Next steps:** Run full experiments (exp001, exp002); compare baseline vs. tool-augmented performance.

---

## 2026-01-08 (initial experiments)

**Ran exp001 (baseline) and exp002 (tool-augmented) on 4-digit addition.**

Results after 1000 generations, population 100:

| Metric | Baseline MLP | Tool-Augmented MLP |
|--------|--------------|-------------------|
| Exact match accuracy | 0.0% | 0.0% |
| Digit accuracy | 24.9% | 25.0% |
| Parameters | 12,594 | 24,309 |
| Training time | 61s | 137s |
| Final gate | N/A | 0.78 |

**Key findings:**

1. Both models performed poorly - barely above random on digit accuracy (25% vs 10% random)
2. Neither achieved any exact matches on 4-digit addition
3. The tool-augmented model learned to prefer the tool pathway (gate: 0.5 → 0.78)
4. Despite preferring the tool, accuracy did not improve

**Analysis:**

The network is attempting to use the tool (evidenced by gate increase) but cannot properly interface with it. The decoder that converts hidden representations to discrete integers for the tool is not producing correct operands. When the tool receives incorrect inputs, it produces incorrect outputs, and the fitness signal doesn't distinguish between "wrong because tool input was wrong" vs "wrong because direct pathway is wrong."

**Hypotheses for why tool pathway isn't helping:**

1. **Decoder bottleneck:** The decoder network may be too weak or the hidden representation may not contain sufficient information about the original operands
2. **Credit assignment:** ES has difficulty determining whether errors come from the decoder, the tool result encoder, or the final output layers
3. **Task too hard:** 4-digit addition (10,000 × 10,000 input space) may be too difficult; should try 2-digit first
4. **Architecture issue:** Current design may not provide useful gradients through the tool pathway

**Next steps:**

1. Reduce to 2-digit addition to see if tool learning is possible on easier task
2. Analyze what the decoder is actually producing (are the decoded numbers even close?)
3. Consider alternative architectures (e.g., shared encoding, direct number input)
4. Potentially add auxiliary loss/supervision on the decoder

---

## 2026-01-08 (ES debugging - autoencoder diagnostic)

**Conducted systematic debugging of ES on autoencoder task.**

User insight: The tool pathway essentially requires learning an autoencoder (X → hidden → X). If ES cannot learn this simple task, the tool-augmented architecture has no hope.

### Diagnostic experiments (exp003 series):

Created exp003 to test if ES can learn a simple autoencoder that reconstructs one-hot encoded digits.

**Initial results: FAILURE**
- exp003: 10,000 generations, digit accuracy stuck at 10% (random)
- exp003b: Cross-entropy loss instead of MSE, still failed
- exp003c: Learning rate 1.0, still failed

### Bug discovery #1: Missing gradient normalization

Found incorrect gradient computation in `es.py`:
- **Wrong:** `gradient = np.dot(epsilon.T, ranks) / self.sigma`
- **Fixed:** `gradient = np.dot(epsilon.T, ranks) / (self.population_size * self.sigma)`

However, fixing this bug did NOT resolve the learning failure.

### Bug discovery #2: Antithetic sampling conflicts with rank-based fitness shaping

Created minimal diagnostic tests (exp003d-exp003g) to isolate the issue:

1. **1D quadratic optimization:** PASS (converges to target)
2. **100D quadratic optimization:** FAIL (distance ~8 instead of ~0)
3. **PyTorch linear regression:** PASS

Key finding: **Antithetic sampling breaks rank-based fitness shaping.**
- Antithetic=True: distance 7.46 ± 1.05
- Antithetic=False: distance 4.36 ± 0.16

Root cause: Rank-based fitness shaping destroys the paired information that antithetic sampling relies on for variance reduction. The mirrored perturbations (ε, -ε) should have correlated fitness differences, but ranking ignores this structure.

**Fix:** Changed ES defaults to:
- `antithetic=False`
- `sigma=0.5` (larger perturbations)
- `learning_rate=0.5`

### Hyperparameter sweep results:

Best configuration for 100D quadratic: sigma=0.5, lr=0.5 achieves distance 0.05 (essentially perfect).

### Activation function discovery:

Tested different activations for autoencoders (exp003j):
- ReLU: 10% accuracy (FAIL - dead neurons)
- LeakyReLU: 70-100% accuracy (works with tuning)
- SiLU/Swish: 80-100% accuracy (best performer)
- Tanh: 70% accuracy
- Linear (no activation): 40% accuracy

**ReLU is problematic for ES** because it creates discontinuous gradients and dead neurons that ES cannot recover from.

### Scale limitation discovery:

Systematic testing (exp003k) revealed ES has a parameter count limitation:

| Parameters | Architecture | Accuracy | Status |
|------------|--------------|----------|--------|
| 430 | 10→20→10 | 100% | PASS |
| 430 | 10→20→10 (LeakyReLU) | 100% | PASS |
| 5,224 | 40→64→40 | 25% | FAIL |

**Critical finding:** ES with rank-based fitness can reliably optimize ~400-1000 parameters but struggles beyond that. The 4-digit autoencoder (5,224 params) plateaus at ~25% digit accuracy despite 5,000 generations.

### Updated ES implementation:

```python
# New defaults in src/evolution/es.py
sigma=0.5          # Larger perturbations
learning_rate=0.5  # Higher LR works with larger sigma
antithetic=False   # Conflicts with rank shaping
```

### Implications for main research question:

The tool-augmented model has ~24,000 parameters, which is **far beyond ES's effective range** with the current approach. Options:

1. **Reduce model size** to ~1000 parameters (very limited capacity)
2. **Use better optimizer:** CMA-ES, Natural Evolution Strategies, or hybrid methods
3. **Hierarchical training:** Pre-train components separately, then fine-tune
4. **Different architecture:** Embed tools at inference only, train with gradient methods
5. **Curriculum learning:** Start with 1-digit, gradually increase

**Deliverables:** exp003 series experiments, fixed ES implementation, systematic hyperparameter analysis

**Next steps:**
1. Implement CMA-ES or NES for higher-dimensional optimization
2. Test tool-augmented model with much smaller network (~1000 params)
3. Consider hybrid approach: use backprop where possible, ES only for tool interface

---

## 2026-01-09

**Implemented and tested Genetic Algorithm (GA) for comparison with ES.**

Following the discovery that ES struggles to scale beyond ~500 parameters, implemented a GA with batched GPU evaluation to test whether a different evolutionary approach could overcome this limitation.

### GA Implementation (`src/evolution/ga.py`):

- **BatchedMLP:** Evaluates entire population in parallel using batched matrix operations on GPU
- **GeneticAlgorithm:** Tournament selection, elite preservation (top 10%), Gaussian mutation
- Efficient implementation: ~56 generations/second on RTX 4090

### Experiment 004b: GA on smaller networks

Tested GA across different scales to compare with ES results:

| Digits | Hidden | Pop | Params | Accuracy | Status |
|--------|--------|-----|--------|----------|--------|
| 1 | 20 | 200 | 430 | 100% | PASS |
| 1 | 20 | 500 | 430 | 100% | PASS |
| 2 | 16 | 500 | 676 | 36% | FAIL |
| 2 | 32 | 500 | 1,332 | 32% | FAIL |
| 3 | 16 | 500 | 1,006 | 22% | FAIL |
| 4 | 16 | 1,000 | 1,336 | 24% | FAIL |

### Experiment 004: GA on 4-digit autoencoder

| Configuration | Value |
|---------------|-------|
| Architecture | 40 → 64 → 40 |
| Parameters | 5,224 |
| Population | 500 |
| Generations | 2,000 |
| Best accuracy | 17% |
| Status | FAIL |

### Key Findings:

1. **GA exhibits the same scaling limitation as ES:** Both algorithms achieve 100% accuracy on 1-digit (~430 params) but fail to learn beyond ~500 parameters.

2. **The limitation is not ES-specific:** Since GA uses a completely different optimization mechanism (tournament selection + mutation vs. gradient estimation), the scaling failure appears to be a fundamental property of the optimization landscape, not the algorithm.

3. **Comparison at 430 params:**
   - ES: 100% accuracy (from exp003 series)
   - GA: 100% accuracy (200 generations)
   Both algorithms solve the small-scale problem reliably.

4. **Comparison at ~1000+ params:**
   - ES: ~25% accuracy (from exp003k)
   - GA: ~20-35% accuracy
   Both plateau at near-random performance.

### Analysis:

The fact that both ES and GA fail identically suggests the problem is not the optimization algorithm but rather:

1. **Fitness landscape complexity:** The loss surface for multi-digit autoencoders may have many local minima or saddle points that trap population-based methods.

2. **Search space dimensionality:** The number of generations required may scale exponentially with parameters, making >500-param optimization infeasible within practical time budgets.

3. **Gradient information loss:** Both ES and GA discard gradient information. Backpropagation can efficiently navigate high-dimensional spaces using local gradient structure; evolutionary methods cannot.

### Implications for Main Research:

This confirms that vanilla evolutionary algorithms (both ES and GA) cannot scale to the ~24,000 parameters needed for tool-augmented networks. Viable paths forward:

1. **CMA-ES:** Uses covariance matrix adaptation to learn search distribution shape; known to scale better than simple ES/GA
2. **Hybrid approach:** Use backprop for most parameters, ES only for the non-differentiable tool interface (~100-200 params)
3. **Structured search:** Apply evolutionary optimization only to a small "interface module" connecting learned representations to tools
4. **Neuroevolution of network topology:** NEAT-style approaches that evolve structure alongside weights

**Deliverables:** `src/evolution/ga.py`, exp004 series experiments

**Next steps:**
1. Implement CMA-ES for comparison
2. Design hybrid architecture: differentiable backbone + evolved tool interface
3. Literature review on scaling neuroevolution (OpenAI large-scale ES, Uber's NEAT variants)

---

## 2026-01-09 (continued)

**BREAKTHROUGH: Fitness function was the bottleneck, not parameter count.**

Investigated alternative fitness functions based on the hypothesis that MSE on one-hot outputs creates a flat, difficult-to-navigate fitness landscape. The results were dramatic.

### Experiment 005: Fitness Function Comparison (2-digit, 1,332 params)

| Fitness Function | Accuracy | Generations | Status |
|------------------|----------|-------------|--------|
| MSE | 29% | 2,000 | FAIL |
| CrossEntropy | 97% | 200 | PASS |
| LogSoftmax | 100% | 400 | PASS |
| SoftAccuracy | 99% | 1,600 | PASS |
| RankBased | 99% | 400 | PASS |

**The same network that failed with MSE achieves 100% accuracy with cross-entropy in 5x fewer generations.**

### Why MSE fails:

MSE on one-hot outputs has extremely weak gradient signal for evolutionary methods:
- If network outputs uniform ~0.1 for all classes, MSE ≈ 0.09 regardless of which class is "closest" to correct
- No reward for partial progress (getting the right class to 0.15 vs 0.11)
- All "wrong" solutions look equally bad

Cross-entropy and log-softmax provide exponentially stronger signal for the correct class, creating a much more navigable fitness landscape.

### Experiment 005b: Scaling with Cross-Entropy

| Digits | Params | Pop | Mut | Accuracy | Status |
|--------|--------|-----|-----|----------|--------|
| 2 | 1,332 | 500 | 0.1 | 99.5% | PASS |
| 3 | 1,982 | 500 | 0.1 | 88.5% | PARTIAL |
| 4 | 5,224 | 500 | 0.1 | 42% | FAIL |
| 4 | 5,224 | 1,000 | 0.05 | **95%** | **PASS** |

**Critical result:** The 4-digit autoencoder (5,224 parameters) that previously failed completely now achieves 95% accuracy with appropriate hyperparameters.

### Revised Understanding:

The "~500 parameter ceiling" observed earlier was **not a fundamental limitation of evolutionary optimization** but rather an artifact of poor fitness function choice. With proper fitness design:

1. Cross-entropy >> MSE for classification-style tasks
2. Larger populations compensate for increased dimensionality
3. Smaller mutation rates prevent overshooting in larger search spaces

### Implications:

This substantially changes the outlook for the main research question:

1. **Tool-augmented architecture may be feasible:** The autoencoder (which is the core challenge for the tool pathway) can now be trained with GA at 5,000+ parameters
2. **Fitness function design is critical:** For non-differentiable tool interfaces, the choice of fitness signal may be more important than the optimization algorithm
3. **Scaling recipe:** population ∝ √params, mutation_std ∝ 1/√params appears effective

### Next experiment:

Apply cross-entropy fitness to the actual tool-augmented addition network and test if the tool pathway can now be learned.

**Deliverables:** exp005 series experiments, fitness function analysis

**Next steps:**
1. Rerun tool-augmented addition experiments with cross-entropy fitness
2. Test if the decoder→tool→encoder pathway can be learned
3. Investigate curriculum strategies (1-digit → 2-digit → N-digit)

---

## 2026-01-10

**Tool-augmented network experiments: The decoder bottleneck.**

Applied cross-entropy fitness to the tool-augmented addition architecture. Despite the fitness function breakthrough for autoencoders, the tool-augmented network still fails to learn.

### Experiment 006: Tool-augmented with CE fitness

Tested the dual-pathway architecture (direct + tool pathways with learned gate):

| Digits | Hidden | Params | Digit Acc | Gate | Status |
|--------|--------|--------|-----------|------|--------|
| 2 | 32 | 6,661 | 35% | 0.93 | FAIL |
| 2 | 64 | 15,269 | 36% | 0.96 | FAIL |

**Key observation:** The gate increases to 0.93-0.96, meaning the network learns to *prefer* the tool pathway, but accuracy stays at ~35% (barely above random 33% for 3 output digits).

### Experiment 006b: Skip connections

Added skip connections from original inputs to the tool decoders:

| Digits | Hidden | Params | Digit Acc | Gate | Status |
|--------|--------|--------|-----------|------|--------|
| 2 | 32 | 10,053 | 35% | 0.96 | FAIL |
| 2 | 64 | 26,149 | 37% | 0.95 | FAIL |

No improvement - skip connections don't help.

### Experiment 006c: Simpler task (1-digit)

Tested if reducing to 1-digit addition (only 10 classes per operand) helps:

| Hidden | Params | Digit Acc | Gate | Dec_A | Dec_B | Status |
|--------|--------|-----------|------|-------|-------|--------|
| 32 | 3,049 | 73% | 0.78 | 18% | 20% | PARTIAL |
| 64 | 8,105 | 89% | **0.28** | 8% | 23% | PARTIAL |

**Critical insight:** The larger model achieves 89% accuracy but with gate=0.28 (preferring the *direct* pathway). The network learns to bypass the tool entirely and solve the problem directly. Decoder accuracy stays near random (~10-20%).

### Experiment 006d: Tool-only architecture (forced tool use)

Removed the direct pathway entirely, forcing the network to use the tool:

| Digits | Hidden | Params | Digit Acc | Dec_A | Dec_B | Status |
|--------|--------|--------|-----------|-------|-------|--------|
| 1 | 32 | 5,832 | 61% | 8% | 9% | FAIL |
| 1 | 64 | 17,768 | 64% | 9% | 9% | FAIL |

**The decoders never learn.** Accuracy stays at 8-9%, which is *below* random (10%). Even after 5,000 generations with population 1,000, the network cannot learn to reconstruct the operands.

### Analysis: Why the decoders fail

The fundamental issue is **credit assignment through the discretization barrier**:

1. **Argmax breaks gradients:** The decoder outputs are discretized via argmax before passing to the tool. Evolution can't tell which decoder weights caused the error.

2. **Degenerate solutions:** Many decoder configurations produce identical fitness. If decoder_a always outputs "5" and decoder_b always outputs "3", the tool gives "8" for all inputs. This might match ~10% of targets by chance - the same as random decoders.

3. **No local improvement signal:** Slightly better decoders don't produce slightly better fitness. The decoder must be *exactly right* to help.

4. **Combinatorial explosion:** For 2-digit addition, decoders must learn a 20→20 mapping (each operand). The space of possible mappings is vast, and random search can't find the identity mapping.

### Comparison: Autoencoder vs Tool pathway

| Task | Can GA learn? | Why? |
|------|---------------|------|
| Autoencoder (X→hidden→X) | **Yes** (95%+) | Continuous output, CE provides smooth gradient signal |
| Tool decoder (X→hidden→X→argmax→tool) | **No** (~10%) | Discretization destroys gradient, no signal for partial progress |

### Implications

The core hypothesis - that evolutionary methods can learn to interface with non-differentiable tools - faces a fundamental obstacle. The discretization required to invoke exact computation tools creates a credit assignment barrier that simple evolutionary methods cannot cross.

### Potential solutions (for future work)

1. **Supervised decoder pre-training:** Pre-train decoders as autoencoders with backprop, then evolve only the gate/blending
2. **Soft tool interface:** Use Gumbel-softmax or similar to maintain differentiability during training
3. **Auxiliary fitness terms:** Add explicit decoder reconstruction accuracy to fitness
4. **Neuroevolution of augmenting topologies (NEAT):** Evolve network structure to find minimal tool interface
5. **Reinforcement learning:** Frame as RL problem where tool use is an action

**Deliverables:** exp006 series experiments, analysis of decoder failure mode

**Current status:** The project has identified a fundamental barrier to pure evolutionary learning of tool interfaces. The discretization required for exact computation prevents gradient-like information from reaching the decoder. Future work should explore hybrid approaches that maintain some differentiability or provide auxiliary supervision.
