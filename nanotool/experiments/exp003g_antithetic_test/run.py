#!/usr/bin/env python3
"""
Experiment 003g: Test antithetic sampling impact.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evolution.es import EvolutionStrategy


def test_es_antithetic(antithetic, num_params, target, max_gens=500, trials=3):
    """Test ES with/without antithetic sampling."""
    distances = []

    for trial in range(trials):
        es = EvolutionStrategy(
            num_params=num_params,
            sigma=0.1,
            learning_rate=1.0,
            population_size=50,
            antithetic=antithetic,
            weight_decay=0.0,
            seed=42 + trial
        )
        es.set_params(np.zeros(num_params))

        for gen in range(max_gens):
            epsilon, population = es.ask()
            fitness = np.array([-np.sum((p - target)**2) for p in population])
            es.tell(epsilon, fitness)

        distances.append(np.linalg.norm(es.get_params() - target))

    return np.mean(distances), np.std(distances)


def test_sigma_lr_combinations():
    """Test different sigma and learning rate combinations."""
    print("=" * 60)
    print("Testing sigma/lr combinations on 100D quadratic")
    print("=" * 60)

    np.random.seed(123)
    num_params = 100
    target = np.random.randn(num_params) * 10
    print(f"Target norm: {np.linalg.norm(target):.4f}\n")

    sigmas = [0.01, 0.05, 0.1, 0.5]
    lrs = [0.1, 0.5, 1.0, 2.0]

    print(f"{'sigma':>8} {'lr':>8} {'distance':>12}")
    print("-" * 30)

    best_dist = float('inf')
    best_config = None

    for sigma in sigmas:
        for lr in lrs:
            es = EvolutionStrategy(
                num_params=num_params,
                sigma=sigma,
                learning_rate=lr,
                population_size=50,
                antithetic=True,
                weight_decay=0.0,
                seed=42
            )
            es.set_params(np.zeros(num_params))

            for gen in range(500):
                epsilon, population = es.ask()
                fitness = np.array([-np.sum((p - target)**2) for p in population])
                es.tell(epsilon, fitness)

            dist = np.linalg.norm(es.get_params() - target)
            print(f"{sigma:>8.2f} {lr:>8.1f} {dist:>12.4f}")

            if dist < best_dist:
                best_dist = dist
                best_config = (sigma, lr)

    print(f"\nBest config: sigma={best_config[0]}, lr={best_config[1]}")
    print(f"Best distance: {best_dist:.4f}")


def test_larger_population():
    """Test with larger population sizes."""
    print("\n" + "=" * 60)
    print("Testing population size impact")
    print("=" * 60)

    np.random.seed(123)
    num_params = 100
    target = np.random.randn(num_params) * 10

    pop_sizes = [20, 50, 100, 200, 500]

    print(f"{'pop_size':>10} {'distance':>12}")
    print("-" * 25)

    for pop_size in pop_sizes:
        es = EvolutionStrategy(
            num_params=num_params,
            sigma=0.1,
            learning_rate=1.0,
            population_size=pop_size,
            antithetic=True,
            weight_decay=0.0,
            seed=42
        )
        es.set_params(np.zeros(num_params))

        for gen in range(500):
            epsilon, population = es.ask()
            fitness = np.array([-np.sum((p - target)**2) for p in population])
            es.tell(epsilon, fitness)

        dist = np.linalg.norm(es.get_params() - target)
        print(f"{pop_size:>10} {dist:>12.4f}")


def test_more_generations():
    """Test with more generations."""
    print("\n" + "=" * 60)
    print("Testing convergence over time")
    print("=" * 60)

    np.random.seed(123)
    num_params = 100
    target = np.random.randn(num_params) * 10

    es = EvolutionStrategy(
        num_params=num_params,
        sigma=0.1,
        learning_rate=1.0,
        population_size=100,
        antithetic=True,
        weight_decay=0.0,
        seed=42
    )
    es.set_params(np.zeros(num_params))

    print(f"{'Gen':>8} {'distance':>12}")
    print("-" * 22)

    for gen in range(5001):
        epsilon, population = es.ask()
        fitness = np.array([-np.sum((p - target)**2) for p in population])
        es.tell(epsilon, fitness)

        if gen % 500 == 0:
            dist = np.linalg.norm(es.get_params() - target)
            print(f"{gen:>8} {dist:>12.4f}")


if __name__ == '__main__':
    # Test antithetic
    print("=" * 60)
    print("Testing antithetic sampling")
    print("=" * 60)

    np.random.seed(123)
    num_params = 100
    target = np.random.randn(num_params) * 10

    mean_dist, std_dist = test_es_antithetic(True, num_params, target)
    print(f"Antithetic=True:  {mean_dist:.4f} ± {std_dist:.4f}")

    mean_dist, std_dist = test_es_antithetic(False, num_params, target)
    print(f"Antithetic=False: {mean_dist:.4f} ± {std_dist:.4f}")

    test_sigma_lr_combinations()
    test_larger_population()
    test_more_generations()
