import numpy as np
import random

def swap_mutation_vectorized(x, n_offspring=1, p_m=None):
    size = len(x)
    if size < 2:
        return np.array([x] * n_offspring)
    if p_m is None:
        p_m = 1.0 / size
    offspring = np.array([x.copy() for _ in range(n_offspring)])
    for i in range(n_offspring):
        if np.random.random() < p_m:
            idx1, idx2 = np.random.choice(size, 2, replace=False)
            offspring[i][idx1], offspring[i][idx2] = offspring[i][idx2], offspring[i][idx1]
        assert len(np.unique(offspring[i])) == len(offspring[i]), f"Cromossomo inválido: {offspring[i]}"
    return offspring

def insertion_mutation(x, n_offspring=1, p_m=None):
    size = len(x)
    # Verificação de entrada
    if len(set(x)) != len(x):
        print(f"[ERRO] Cromossomo de entrada inválido para mutação: {x}")
        raise AssertionError(f"Cromossomo de entrada inválido: {x}")
    if size < 2:
        return np.array([x] * n_offspring)
    if p_m is None:
        p_m = 1.0 / size
    offspring = []
    for _ in range(n_offspring):
        child = list(x)
        if np.random.random() < p_m:
            idx1, idx2 = np.random.choice(size, 2, replace=False)
            gene = child.pop(idx1)
            child.insert(idx2, gene)
        if len(set(child)) != len(child):
            print(f"[ERRO] Cromossomo gerado inválido: {child}")
        assert len(set(child)) == len(child), f"Cromossomo inválido: {child}"
        offspring.append(child)
    return np.array(offspring)

def inversion_mutation(x, n_offspring=1, p_m=None):
    size = len(x)
    if size < 2:
        return np.array([x] * n_offspring)
    if p_m is None:
        p_m = 1.0 / size
    offspring = np.array([x.copy() for _ in range(n_offspring)])
    for i in range(n_offspring):
        if np.random.random() < p_m:
            idx1, idx2 = np.random.choice(size, 2, replace=False)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            offspring[i][idx1:idx2+1] = np.flip(offspring[i][idx1:idx2+1])
        assert len(np.unique(offspring[i])) == len(offspring[i]), f"Cromossomo inválido: {offspring[i]}"
    return offspring

def two_opt_mutation(x, n_offspring=1, p_m=None, problem_eval=None):
    size = len(x)
    if size < 4:
        return np.array([x] * n_offspring)
    if p_m is None:
        p_m = 1.0 / size
    offspring = np.array([x.copy() for _ in range(n_offspring)])
    orig_fitness = None
    if problem_eval is not None:
        # Calcula fitness original apenas uma vez
        orig_fitness = problem_eval(np.array([x]))[0]
    for i in range(n_offspring):
        if np.random.random() < p_m:
            idx1, idx2 = np.random.choice(size, 2, replace=False)
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            original = offspring[i].copy()
            offspring[i][idx1:idx2+1] = np.flip(offspring[i][idx1:idx2+1])
            if problem_eval is not None:
                new_fitness = problem_eval(np.array([offspring[i]]))[0]
                # Usa o fitness original calculado fora do loop
                if new_fitness[0] >= orig_fitness[0]:
                    offspring[i] = original
        assert len(np.unique(offspring[i])) == len(offspring[i]), f"Cromossomo inválido: {offspring[i]}"
    return offspring
