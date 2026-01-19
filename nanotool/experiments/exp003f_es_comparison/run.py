#!/usr/bin/env python3
"""
Experiment 003f: Compare ES implementations on high-dimensional problem.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evolution.es import EvolutionStrategy


def test_my_es(num_params, target, max_gens=500):
    """Test current ES implementation."""
    es = EvolutionStrategy(
        num_params=num_params,
        sigma=0.1,
        learning_rate=1.0,
        population_size=50,
        weight_decay=0.0,
        seed=42
    )
    es.set_params(np.zeros(num_params))

    for gen in range(max_gens):
        epsilon, population = es.ask()
        fitness = np.array([-np.sum((p - target)**2) for p in population])
        es.tell(epsilon, fitness)

    return np.linalg.norm(es.get_params() - target)


def test_reference_es(num_params, target, max_gens=500):
    """Reference OpenAI ES implementation."""
    theta = np.zeros(num_params)
    sigma = 0.1
    lr = 1.0
    pop_size = 50

    np.random.seed(42)

    for gen in range(max_gens):
        epsilon = np.random.randn(pop_size, num_params)
        population = theta + sigma * epsilon

        fitness = np.array([-np.sum((p - target)**2) for p in population])

        # Standardize
        A = (fitness - fitness.mean()) / (fitness.std() + 1e-8)

        # Gradient: (1/n) * sum(A_i * epsilon_i)
        gradient = np.dot(A, epsilon) / pop_size

        # Update: theta += lr * gradient / sigma
        theta += lr * gradient / sigma

    return np.linalg.norm(theta - target)


def test_reference_es_with_ranks(num_params, target, max_gens=500):
    """Reference ES with rank-based fitness shaping."""
    theta = np.zeros(num_params)
    sigma = 0.1
    lr = 1.0
    pop_size = 50

    np.random.seed(42)

    for gen in range(max_gens):
        epsilon = np.random.randn(pop_size, num_params)
        population = theta + sigma * epsilon

        fitness = np.array([-np.sum((p - target)**2) for p in population])

        # Rank-based shaping (same as my implementation)
        ranks = np.zeros(len(fitness))
        for rank, idx in enumerate(np.argsort(fitness)):
            ranks[idx] = rank
        ranks = (ranks / (len(fitness) - 1)) - 0.5

        # Gradient
        gradient = np.dot(ranks, epsilon) / pop_size

        # Update
        theta += lr * gradient / sigma

    return np.linalg.norm(theta - target)


def debug_single_step():
    """Debug a single ES step to find the discrepancy."""
    print("=" * 60)
    print("Debugging single ES step")
    print("=" * 60)

    np.random.seed(42)
    num_params = 10
    target = np.ones(num_params) * 5.0

    # My ES
    es = EvolutionStrategy(
        num_params=num_params,
        sigma=0.1,
        learning_rate=1.0,
        population_size=10,
        weight_decay=0.0,
        seed=42
    )
    es.set_params(np.zeros(num_params))

    epsilon_my, pop_my = es.ask()
    fitness_my = np.array([-np.sum((p - target)**2) for p in pop_my])

    print(f"My ES - epsilon shape: {epsilon_my.shape}")
    print(f"My ES - population shape: {pop_my.shape}")
    print(f"My ES - fitness: {fitness_my}")

    # Before tell
    theta_before = es.get_params().copy()

    es.tell(epsilon_my, fitness_my)
    theta_after = es.get_params()
    my_update = theta_after - theta_before

    print(f"\nMy ES - theta update (first 3): {my_update[:3]}")
    print(f"My ES - update magnitude: {np.linalg.norm(my_update):.6f}")

    # Reference implementation
    np.random.seed(42)
    theta_ref = np.zeros(num_params)
    sigma = 0.1
    lr = 1.0
    pop_size = 10

    epsilon_ref = np.random.randn(pop_size, num_params)
    pop_ref = theta_ref + sigma * epsilon_ref

    print(f"\nRef ES - epsilon shape: {epsilon_ref.shape}")

    fitness_ref = np.array([-np.sum((p - target)**2) for p in pop_ref])

    # Check if perturbations match
    print(f"\nEpsilon match: {np.allclose(epsilon_my, epsilon_ref)}")
    print(f"Population match: {np.allclose(pop_my, pop_ref)}")
    print(f"Fitness match: {np.allclose(fitness_my, fitness_ref)}")

    # Rank-based shaping
    ranks = np.zeros(len(fitness_ref))
    for rank, idx in enumerate(np.argsort(fitness_ref)):
        ranks[idx] = rank
    ranks = (ranks / (len(fitness_ref) - 1)) - 0.5

    # Compute gradient two ways
    # My way: dot(epsilon.T, ranks) / (pop_size * sigma)
    grad_my_way = np.dot(epsilon_ref.T, ranks) / (pop_size * sigma)

    # Reference way: dot(ranks, epsilon) / pop_size, then divide by sigma in update
    grad_ref_way = np.dot(ranks, epsilon_ref) / pop_size

    print(f"\nGradient (my formula): {grad_my_way[:3]}")
    print(f"Gradient (ref formula, pre-sigma): {grad_ref_way[:3]}")
    print(f"Gradient (ref formula, post-sigma): {(grad_ref_way / sigma)[:3]}")

    # Update reference
    ref_update = lr * grad_ref_way / sigma

    print(f"\nMy update: {my_update[:3]}")
    print(f"Ref update: {ref_update[:3]}")
    print(f"Updates match: {np.allclose(my_update, ref_update)}")

    # What's the expected direction?
    # Target is [5,5,5,...], we start at [0,0,0,...]
    # So we should move in the positive direction
    print(f"\nExpected direction: positive (toward target)")
    print(f"My update mean: {my_update.mean():.6f}")
    print(f"Ref update mean: {ref_update.mean():.6f}")


if __name__ == '__main__':
    debug_single_step()

    print("\n" + "=" * 60)
    print("Comparing implementations on 100D problem")
    print("=" * 60)

    np.random.seed(123)
    num_params = 100
    target = np.random.randn(num_params) * 10

    print(f"Target norm: {np.linalg.norm(target):.4f}")

    dist_my = test_my_es(num_params, target)
    print(f"My ES final distance: {dist_my:.4f}")

    dist_ref = test_reference_es(num_params, target)
    print(f"Reference ES (z-score) final distance: {dist_ref:.4f}")

    dist_ref_rank = test_reference_es_with_ranks(num_params, target)
    print(f"Reference ES (ranks) final distance: {dist_ref_rank:.4f}")
