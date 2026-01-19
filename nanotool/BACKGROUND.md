# Background Literature Review

This document summarizes relevant prior work for the project exploring whether neural networks can learn to use embedded tool modules (such as arithmetic units) when trained via genetic algorithms or evolution strategies instead of backpropagation.

## 1. Neuroevolution: Training Neural Networks with Evolutionary Methods

### 1.1 Deep Neuroevolution with Genetic Algorithms

**Such et al. (2017)** demonstrated that simple genetic algorithms can effectively train deep neural networks with millions of parameters, achieving competitive results on Atari games and continuous control tasks. Their key finding was that a population-based GA without gradients can match or exceed the performance of sophisticated gradient-based methods like DQN and A3C, while being embarrassingly parallel. With sufficient compute, they trained Atari agents in ~4 hours versus ~7-10 days for DQN.

*Relevance: This establishes that gradient-free optimization is viable for large neural networks, which is essential for our approach since tool modules break differentiability.*

### 1.2 Evolution Strategies

**Salimans et al. (2017)** showed that Evolution Strategies (ES) can rival reinforcement learning methods on challenging tasks. ES optimizes by sampling perturbations of network weights and estimating gradients via finite differences. Key advantages include extreme parallelizability (they trained a humanoid walker in 10 minutes using 1,440 CPU cores), invariance to action frequency, tolerance of sparse/delayed rewards, and no need for value function approximation.

*Relevance: ES provides a scalable, gradient-free training method that could optimize networks containing non-differentiable tool modules.*

### 1.3 CMA-ES (Covariance Matrix Adaptation)

**Hansen & Ostermeier (2001)** developed CMA-ES, considered state-of-the-art for continuous black-box optimization. It maintains a covariance matrix that adapts the search distribution to the local fitness landscape. CMA-ES has been applied to neural architecture search and hyperparameter optimization.

*Relevance: CMA-ES offers a sophisticated alternative to simple ES for optimizing the weights that interface with tool modules.*

### 1.4 NEAT and Topology Evolution

**Stanley & Miikkulainen (2002)** introduced NEAT (NeuroEvolution of Augmenting Topologies), which evolves both network topology and weights. NEAT starts with minimal networks and incrementally adds complexity, using speciation to protect structural innovations. It received the Outstanding Paper of the Decade award from ISAL.

*Relevance: NEAT could evolve the connectivity patterns between the main network and tool modules, potentially discovering optimal interfacing architectures.*

### 1.5 Weight Agnostic Neural Networks

**Gaier & Ha (2019)** showed that network architecture alone can encode useful behaviors without weight training. They evolved topologies where a single shared weight value (sampled randomly) produces functional behavior. This demonstrates that structure can carry substantial computational capacity.

*Relevance: This suggests that the interface architecture between network and tool modules may be as important as the learned weights.*

## 2. Neural Networks Learning Arithmetic

### 2.1 Grokking and Mechanistic Interpretability

**Power et al. (2022)** discovered "grokking"—a phenomenon where networks trained on modular arithmetic first memorize training data, then suddenly generalize after extended training.

**Nanda et al. (2023)** reverse-engineered the learned algorithm, finding that transformers learn discrete Fourier transforms and use trigonometric identities to convert modular addition into rotation in 2D space. The embedding layer organizes tokens circularly, like a clock.

*Relevance: These findings reveal the complex internal circuits networks develop to approximate arithmetic. Providing a direct tool interface could bypass this complexity entirely.*

### 2.2 Limitations of Transformers on Arithmetic

**Nogueira et al. (2021)** systematically investigated transformer limitations on arithmetic. Key findings:
- Surface form (tokenization) strongly affects accuracy
- Standard subword tokenization fails on 5+ digit addition
- Position tokens improve accuracy up to 60 digits
- Models cannot extrapolate to longer sequences than training

*Relevance: These fundamental limitations motivate providing external arithmetic modules that guarantee correctness regardless of input size.*

### 2.3 Neural Arithmetic Logic Units (NALU)

**Trask et al. (2018)** proposed NALU—specialized neural modules for arithmetic that use linear activations and learned gates to perform addition, subtraction, multiplication, and division. NALUs can extrapolate to numbers far outside the training range, maintaining near-perfect accuracy even on sequences 1000x longer than training.

*Relevance: NALU represents an attempt to build arithmetic capability into differentiable modules. Our approach differs by using truly non-differentiable, exact arithmetic modules trained via evolution.*

### 2.4 Number Representations and Embeddings

Recent work has explored how neural networks represent numbers internally. **Gurnee et al.** found that LLMs develop helical representations where arithmetic operations correspond to rotations. Various approaches to improve numerical encoding include:
- Scientific notation / exponent embeddings
- Binary/floating-point encodings (BitTokens)
- Learned numeral embeddings

*Relevance: Understanding number representation helps design the encoding scheme for tool module inputs/outputs.*

## 3. Tool Use in Neural Networks

### 3.1 Toolformer

**Schick et al. (2023)** trained language models to insert API calls (calculator, search, Q&A) into generated text. The model learns when tool use reduces perplexity of future tokens. A 6.7B model outperformed much larger GPT-3 on tasks requiring tools.

*Relevance: Toolformer demonstrates LLMs can learn tool use, but operates at the token level. Our approach embeds tools directly at the neural level.*

### 3.2 PAL: Program-Aided Language Models

**Gao et al. (2022)** showed that LLMs can generate Python code to solve reasoning problems, offloading computation to an interpreter. This separation—decomposition by LLM, execution by interpreter—achieves state-of-the-art on 12 benchmarks.

*Relevance: PAL separates understanding from computation, similar to our goal. The key difference is we integrate the computation module directly into the network architecture.*

### 3.3 Scratchpads

**Nye et al. (2021)** trained models to emit intermediate computation steps before final answers. This dramatically improves multi-step arithmetic, enabling computation that single-pass networks cannot perform.

*Relevance: Scratchpads are a precursor to chain-of-thought and show that explicit intermediate steps help. Tool modules could provide guaranteed-correct intermediate computations.*

## 4. Memory-Augmented Neural Networks

### 4.1 Neural Turing Machines

**Graves et al. (2014)** coupled neural networks with external memory via differentiable attention mechanisms. NTMs can learn algorithms like copying, sorting, and recall from examples alone.

*Relevance: NTMs demonstrate that networks can learn to use external resources, though their memory is differentiable unlike our proposed tool modules.*

### 4.2 Differentiable Neural Computers

**Graves et al. (2016)** extended NTMs with improved memory addressing (temporal links, content-based lookup) and dynamic memory allocation. DNCs learned to navigate graphs and solve block puzzles from examples.

*Relevance: DNCs represent the state-of-the-art in differentiable memory. Our approach explores non-differentiable tool modules that require evolutionary training.*

### 4.3 Evolving Differentiable Neural Computers

**Rasekh & Safi-Esfahani (2020)** applied neuroevolution to automatically find optimal DNC controller architectures. Their EDNC method uses specialized encodings (ALNE and M_ALNE) to evolve neural network structures that interface with external memory.

*Relevance: This is the closest prior work—using evolution to optimize networks that interface with external modules. However, their memory is still differentiable; we propose truly non-differentiable tool modules.*

### 4.4 Neural Programmer-Interpreters

**Reed & de Freitas (2015)** created NPI—a recurrent network that learns to execute programs from execution traces. NPI uses a persistent program memory and can compose learned subroutines to solve complex tasks (addition, sorting, 3D canonicalization) with strong generalization.

*Relevance: NPI shows networks can learn compositional use of primitive operations. Our tool modules provide similar primitives but with guaranteed correctness.*

## 5. Handling Non-Differentiable Components

### 5.1 Straight-Through Estimator

**Bengio et al. (2013)** proposed the straight-through estimator (STE)—during forward pass, apply the non-differentiable operation; during backward pass, pass gradients through unchanged. STE is widely used for quantization-aware training and discrete variables.

*Relevance: STE is one approach to train through discrete operations, though it introduces bias. Our evolutionary approach avoids this by not requiring gradients at all.*

### 5.2 Gumbel-Softmax

**Jang et al. (2016); Maddison et al. (2016)** independently proposed Gumbel-Softmax (Concrete distribution)—a continuous relaxation of discrete distributions that enables backpropagation through sampling. Temperature controls the softness/hardness of the approximation.

*Relevance: Gumbel-Softmax enables gradient-based training with discrete choices. For tool modules, this could help select which tool to use, though the tool computation itself remains non-differentiable.*

### 5.3 Reinforcement Learning for Non-Differentiable Objectives

REINFORCE and policy gradient methods can optimize non-differentiable objectives by treating them as reward signals. However, these methods often have high variance and sample complexity.

*Relevance: RL provides an alternative to evolution for training with non-differentiable components, but evolution may be more suitable for purely structural optimization.*

## 6. Modular Neural Networks

### 6.1 Neural Module Networks

**Andreas et al. (2016)** proposed dynamically assembling neural networks from learned modules based on linguistic structure. Each module implements a primitive operation (attend, classify, combine), composed via parsing.

*Relevance: NMNs demonstrate compositional use of specialized modules. Our tool modules are similar but provide exact computation rather than learned approximations.*

### 6.2 Mixture of Experts and Routing

Modern approaches (Switch Transformer, mixture-of-experts) use learned routing to select specialized subnetworks. However, ensuring module specialization remains challenging—modules often fail to specialize as intended.

*Relevance: Routing mechanisms could help networks decide when to use tool modules. The specialization problem suggests careful design of tool interfaces is important.*

## 7. Neuro-Symbolic Integration

The field of neuro-symbolic AI attempts to combine neural learning with symbolic reasoning. Key approaches include:

- **Logic Tensor Networks**: Encode logical formulas as neural networks
- **DeepProbLog**: Combine neural networks with probabilistic logic
- **Scallop**: Differentiable Datalog for relational reasoning

A fundamental challenge is that symbolic logic is discrete (non-differentiable), forcing trade-offs between "softening" logic (losing guarantees) or keeping systems loosely coupled.

*Relevance: Our approach of embedding exact tool modules and training via evolution represents an alternative to differentiable approximations—maintaining tool correctness while using evolution to optimize the interface.*

## 8. Evolution Strategies vs Genetic Algorithms: Critical Distinctions

### 8.1 The RL vs Supervised Learning Divide

A critical finding from the literature is that **ES excels at reinforcement learning but struggles with supervised learning**. OpenAI's experiments revealed that ES can be "1000 times slower" than backpropagation for supervised tasks like MNIST classification, despite scaling to millions of parameters for RL.

**Why ES works for RL:**
- RL already requires sampling to estimate policy gradients—ES's sampling is not additional overhead
- Episode-level rewards are naturally sparse, which ES handles well
- ES treats networks as black boxes, needing only total episode reward
- RL gradients are noisy anyway; ES's gradient estimate has comparable variance

**Why ES struggles with supervised learning:**
- Backprop provides exact gradients at minimal cost (one forward + backward pass)
- Supervised loss provides dense, informative gradient signal at every step
- ES must evaluate N population members to estimate one gradient
- The gradient information in supervised learning is "extremely informative" (Salimans et al.)

### 8.2 GA vs ES for Weight Optimization

**Such et al. (2017)** used a **Genetic Algorithm** (not ES) to train networks with 4+ million parameters. Key differences:

| Aspect | Evolution Strategies | Genetic Algorithms |
|--------|---------------------|-------------------|
| Update | Gaussian perturbation + gradient estimate | Selection + mutation |
| Gradient | Estimates gradient via finite differences | Purely gradient-free |
| Selection | Rank-based, deterministic | Tournament/roulette selection |
| Crossover | Absent or minimal | Can be used (though often omitted for NN weights) |

The literature suggests GA may be better suited for discrete search spaces and architecture optimization, while ES performs better on continuous parameter optimization—but primarily in RL settings.

### 8.3 Scaling ES to Billions of Parameters

Recent work has addressed ES's scalability limitations:

**EGGROLL (2024)** scales ES to billions of parameters for LLM fine-tuning via **low-rank perturbations**. Instead of full perturbation matrices E ∈ ℝ^(m×n), they generate smaller matrices A ∈ ℝ^(m×r) and B ∈ ℝ^(n×r) where r ≪ min(m,n), reducing memory from mn to r(m+n) and achieving O(1/r) convergence to full-rank behavior.

**Key insight**: The success of EGGROLL on LLM fine-tuning may be because:
1. It operates on **pretrained models** (not from scratch)
2. LLM fine-tuning objectives (RLHF, reward models) are more like RL than supervised learning
3. Low-rank structure constrains the search space dramatically

### 8.4 Hybrid Approaches

**Conti et al. (2018)** showed that hybrid methods combining ES with backprop can be effective:
- Use backprop for differentiable parameters (weights)
- Use ES for non-differentiable parameters (sparsity masks, architecture)

This achieves competitive results on CIFAR-10 with only "negligible training time overhead" compared to pure gradient descent.

*Relevance: Our tool-augmented network might benefit from a hybrid approach—using backprop where possible and ES only for the tool interface parameters.*

### 8.5 Population Size and Parallelization

A key factor in ES scalability is massive parallelization:
- Salimans et al. used **1,440 CPU cores** in parallel
- Population sizes of **thousands** of workers
- Communication overhead reduced via shared random seeds

Our experiments used population sizes of 50-200 on a single machine. The literature suggests that ES requires substantial parallelization to be competitive.

## 9. Summary and Research Gaps

The literature reveals several key themes:

1. **Evolution is viable for large networks**: Deep neuroevolution and ES can train networks with millions of parameters.

2. **Arithmetic is hard for neural networks**: Despite sophisticated internal representations, networks struggle with extrapolation and can be unreliable on basic arithmetic.

3. **Tool use helps but is typically token-level**: Toolformer and PAL demonstrate benefits of external computation, but integrate at the language/token level rather than neural level.

4. **Memory augmentation is usually differentiable**: NTMs and DNCs show networks can use external resources, but keep everything differentiable.

5. **Non-differentiable training has solutions**: STE, Gumbel-Softmax, and RL can handle discrete operations, but evolution offers a cleaner approach when the entire training regime is gradient-free.

### Research Gap

No prior work has systematically explored:
- Embedding **non-differentiable, exact computation modules** directly into neural network architectures
- Training such hybrid systems via **pure neuroevolution** (GA/ES)
- Understanding under what conditions networks **discover and utilize** such embedded tools
- Optimal **encoding schemes** for tool inputs/outputs that facilitate evolutionary learning

This project aims to fill this gap by investigating whether evolution can discover effective use of embedded tool modules, starting with arithmetic as a well-understood test case.
