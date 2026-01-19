#!/usr/bin/env python3
"""
Experiment 003e: Minimal ES Test

Single-parameter optimization to isolate the ES bug.
"""

import numpy as np


def es_step(theta, sigma, lr, pop_size, fitness_fn):
    """One step of ES with detailed logging."""
    # Generate perturbations
    half = pop_size // 2
    eps_half = np.random.randn(half)
    epsilon = np.concatenate([eps_half, -eps_half])

    # Evaluate fitness
    population = theta + sigma * epsilon
    fitness = np.array([fitness_fn(p) for p in population])

    # Rank-based fitness shaping
    ranks = np.zeros(len(fitness))
    for rank, idx in enumerate(np.argsort(fitness)):
        ranks[idx] = rank
    ranks = (ranks / (len(fitness) - 1)) - 0.5  # [-0.5, 0.5]

    # Compute gradient
    gradient = np.dot(epsilon, ranks) / (pop_size * sigma)

    # Update
    new_theta = theta + lr * gradient

    return new_theta, np.mean(fitness), gradient


def test_1d_quadratic():
    """Minimize f(x) = (x - 5)^2"""
    print("=" * 60)
    print("1D Quadratic: minimize f(x) = (x - 5)^2")
    print("=" * 60)

    target = 5.0
    fitness_fn = lambda x: -(x - target) ** 2  # Negative because we maximize fitness

    theta = 0.0
    sigma = 0.5
    lr = 1.0
    pop_size = 20

    np.random.seed(42)

    print(f"Target: {target}")
    print(f"Initial theta: {theta}")
    print(f"sigma={sigma}, lr={lr}, pop={pop_size}")
    print("-" * 60)

    for gen in range(50):
        theta, mean_fit, grad = es_step(theta, sigma, lr, pop_size, fitness_fn)
        if gen % 10 == 0:
            print(f"Gen {gen:3d} | theta={theta:8.4f} | target-theta={target - theta:8.4f} | grad={grad:8.4f} | fit={mean_fit:.4f}")

    print(f"\nFinal: theta={theta:.4f}, error={abs(theta - target):.4f}")
    print("PASS" if abs(theta - target) < 0.5 else "FAIL")


def test_1d_without_rank():
    """Same test but with raw fitness (no rank shaping)."""
    print("\n" + "=" * 60)
    print("1D Quadratic (NO rank shaping)")
    print("=" * 60)

    target = 5.0

    def es_step_raw(theta, sigma, lr, pop_size):
        half = pop_size // 2
        eps_half = np.random.randn(half)
        epsilon = np.concatenate([eps_half, -eps_half])

        population = theta + sigma * epsilon
        fitness = np.array([-(p - target) ** 2 for p in population])

        # Z-score normalize fitness
        fitness_norm = (fitness - fitness.mean()) / (fitness.std() + 1e-8)

        gradient = np.dot(epsilon, fitness_norm) / (pop_size * sigma)
        new_theta = theta + lr * gradient

        return new_theta, np.mean(fitness), gradient

    theta = 0.0
    sigma = 0.5
    lr = 1.0
    pop_size = 20

    np.random.seed(42)

    print(f"Target: {target}")
    print(f"Initial theta: {theta}")
    print("-" * 60)

    for gen in range(50):
        theta, mean_fit, grad = es_step_raw(theta, sigma, lr, pop_size)
        if gen % 10 == 0:
            print(f"Gen {gen:3d} | theta={theta:8.4f} | target-theta={target - theta:8.4f} | grad={grad:8.4f}")

    print(f"\nFinal: theta={theta:.4f}, error={abs(theta - target):.4f}")
    print("PASS" if abs(theta - target) < 0.5 else "FAIL")


def test_openai_es_reference():
    """Reference implementation based on OpenAI ES paper."""
    print("\n" + "=" * 60)
    print("OpenAI ES Reference Implementation")
    print("=" * 60)

    target = 5.0
    theta = 0.0
    sigma = 0.5
    lr = 0.5  # Often called alpha in papers
    pop_size = 20

    np.random.seed(42)

    print(f"Target: {target}")
    print(f"Initial theta: {theta}")
    print("-" * 60)

    for gen in range(50):
        # Sample noise
        epsilon = np.random.randn(pop_size)

        # Evaluate
        fitness = np.array([-(theta + sigma * e - target) ** 2 for e in epsilon])

        # Standardize rewards (important for stability)
        A = (fitness - np.mean(fitness)) / (np.std(fitness) + 1e-8)

        # Gradient estimate (this is the key formula from OpenAI)
        # theta += alpha * (1/(n*sigma)) * sum(A_i * epsilon_i)
        gradient = np.dot(A, epsilon) / pop_size
        theta += lr * gradient / sigma

        if gen % 10 == 0:
            print(f"Gen {gen:3d} | theta={theta:8.4f} | error={abs(theta - target):8.4f}")

    print(f"\nFinal: theta={theta:.4f}, error={abs(theta - target):.4f}")
    print("PASS" if abs(theta - target) < 0.5 else "FAIL")


def test_simple_gradient_ascent():
    """Direct gradient computation for comparison."""
    print("\n" + "=" * 60)
    print("ES Gradient Check")
    print("=" * 60)

    target = 5.0
    theta = 0.0
    sigma = 0.1

    # True gradient of f(x) = -(x-5)^2 at x=0 is:
    # df/dx = -2(x-5) = -2(0-5) = 10
    # So gradient points toward target (positive direction)

    print(f"True gradient at x=0: {-2 * (0 - target)}")

    # ES gradient estimate
    np.random.seed(42)
    pop_size = 10000
    epsilon = np.random.randn(pop_size)

    fitness = np.array([-(theta + sigma * e - target) ** 2 for e in epsilon])
    fitness_norm = (fitness - fitness.mean()) / (fitness.std() + 1e-8)

    # Standard ES gradient formula
    es_gradient = np.dot(fitness_norm, epsilon) / (pop_size * sigma)
    print(f"ES gradient estimate (n={pop_size}): {es_gradient:.4f}")

    # Check direction
    print(f"Gradient direction correct: {es_gradient > 0}")


if __name__ == '__main__':
    test_1d_quadratic()
    test_1d_without_rank()
    test_openai_es_reference()
    test_simple_gradient_ascent()
