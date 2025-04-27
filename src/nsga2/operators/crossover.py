import numpy as np
import random
from copy import deepcopy

def _validate_parents(p1, p2):
    if not (isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray)):
        raise TypeError("p1 e p2 devem ser arrays numpy.")
    if p1.ndim != 1 or p2.ndim != 1:
        raise ValueError("p1 e p2 devem ser arrays 1-D.")
    if len(p1) != len(p2):
        raise ValueError("p1 e p2 devem ter o mesmo tamanho.")
    if len(np.unique(p1)) != len(p1) or len(np.unique(p2)) != len(p2):
        raise ValueError("p1 e p2 não devem conter duplicatas.")

def _repair_offspring(offspring, p1):
    """
    Garante que cada filho em offspring é uma permutação válida de p1,
    corrigindo duplicatas e inserindo elementos faltantes.
    """
    gene_set = set(p1.tolist())
    for k in range(len(offspring)):
        child = list(offspring[k])
        seen = set()
        dup_indices = []
        for i, v in enumerate(child):
            if v in seen:
                dup_indices.append(i)
            else:
                seen.add(v)
        missing = list(gene_set - seen)
        for idx, m in zip(dup_indices, missing):
            child[idx] = m
        offspring[k] = np.array(child, dtype=offspring.dtype)
    return offspring

def _repair_offspring_jssp(offspring, p1, n_jobs, ops_per_job):
    """
    Reparo específico para JSSP: garante que as operações de cada job aparecem na ordem correta.
    offspring: np.ndarray (n_offspring, size)
    p1: cromossomo base (np.ndarray)
    n_jobs: número de jobs
    ops_per_job: número de operações por job
    """
    size = len(p1)
    for k in range(len(offspring)):
        child = list(offspring[k])
        # Para cada job, colete as operações na ordem original
        job_ops = {j: [] for j in range(n_jobs)}
        for op in child:
            job_id = op // 100
            job_ops[job_id].append(op)
        # Corrija a ordem das operações de cada job
        fixed_child = []
        for op in p1:
            job_id = op // 100
            if job_ops[job_id]:
                fixed_child.append(job_ops[job_id].pop(0))
        offspring[k] = np.array(fixed_child, dtype=offspring.dtype)
    return offspring

def ox_crossover_vectorized(p1, p2, n_offspring=2, n_jobs=None, ops_per_job=None):
    _validate_parents(p1, p2)
    size = len(p1)
    offspring = np.zeros((n_offspring, size), dtype=p1.dtype)
    a = np.random.randint(0, size)
    b = np.random.randint(0, size)
    if a > b:
        a, b = b, a
    # Float: usa np.nan como sentinela
    if np.issubdtype(p1.dtype, np.floating):
        sentinela = np.nan
        offspring[0].fill(sentinela)
        offspring[0, a:b] = p1[a:b]
        fill = [item for item in p2 if item not in p1[a:b]]
        pos = b
        for item in fill:
            if pos >= size:
                pos = 0
            while not np.isnan(offspring[0, pos]):
                pos = (pos + 1) % size
            offspring[0, pos] = item
            pos += 1
        offspring[1].fill(sentinela)
        offspring[1, a:b] = p2[a:b]
        fill = [item for item in p1 if item not in p2[a:b]]
        pos = b
        for item in fill:
            if pos >= size:
                pos = 0
            while not np.isnan(offspring[1, pos]):
                pos = (pos + 1) % size
            offspring[1, pos] = item
            pos += 1
    else:
        # Int: usa máscara booleana para slots vazios
        mask0 = np.zeros(size, dtype=bool)
        mask0[a:b] = True
        offspring[0, a:b] = p1[a:b]
        fill = [item for item in p2 if item not in p1[a:b]]
        pos = b
        for item in fill:
            if pos >= size:
                pos = 0
            while mask0[pos]:
                pos = (pos + 1) % size
            offspring[0, pos] = item
            mask0[pos] = True
            pos += 1
        mask1 = np.zeros(size, dtype=bool)
        mask1[a:b] = True
        offspring[1, a:b] = p2[a:b]
        fill = [item for item in p1 if item not in p2[a:b]]
        pos = b
        for item in fill:
            if pos >= size:
                pos = 0
            while mask1[pos]:
                pos = (pos + 1) % size
            offspring[1, pos] = item
            mask1[pos] = True
            pos += 1
    # fim da geração dos filhos
    # garante permutações válidas e ordem dos jobs
    if n_jobs is not None and ops_per_job is not None:
        return _repair_offspring_jssp(offspring, p1, n_jobs, ops_per_job)
    return _repair_offspring(offspring, p1)

def ppx_crossover(p1, p2, n_offspring=2, n_jobs=None, ops_per_job=None):
    _validate_parents(p1, p2)
    size = len(p1)
    offspring = np.zeros((n_offspring, size), dtype=p1.dtype)
    mask = np.random.randint(0, 2, size=size)
    for k in range(n_offspring):
        if k == 1:
            mask = 1 - mask
        pos = 0
        used = set()
        p1_copy = p1.copy()
        p2_copy = p2.copy()
        for i in range(size):
            parent = p1_copy if mask[i] == 0 else p2_copy
            for j in range(size):
                if parent[j] not in used:
                    offspring[k, pos] = parent[j]
                    used.add(parent[j])
                    pos += 1
                    break
    # garante permutações válidas
    if n_jobs is not None and ops_per_job is not None:
        return _repair_offspring_jssp(offspring, p1, n_jobs, ops_per_job)
    return _repair_offspring(offspring, p1)

def ippx_crossover(p1, p2, n_offspring=2, n_jobs=None, ops_per_job=None):
    _validate_parents(p1, p2)
    offspring = ppx_crossover(p1, p2, n_offspring)
    if n_jobs is None or ops_per_job is None:
        return offspring
    for k in range(n_offspring):
        child = offspring[k]
        job_blocks = {}
        for i, op in enumerate(child):
            job_id = op // 100
            if job_id not in job_blocks:
                job_blocks[job_id] = []
            job_blocks[job_id].append((i, op))
        for job_id, block in job_blocks.items():
            if len(block) > 1:
                indices = [b[0] for b in block]
                random.shuffle(indices)
                for idx, (_, op) in zip(indices, block):
                    offspring[k, idx] = op
    # garante permutações válidas e ordem dos jobs
    return _repair_offspring_jssp(offspring, p1, n_jobs, ops_per_job)

def extended_ppx_crossover(p1, p2, n_offspring=2, n_jobs=None, ops_per_job=None):
    _validate_parents(p1, p2)
    size = len(p1)
    offspring = np.zeros((n_offspring, size), dtype=p1.dtype)
    n_cuts = random.randint(2, min(5, size//10 + 2))
    cut_points = sorted(random.sample(range(1, size), n_cuts))
    cut_points = [0] + cut_points + [size]
    for k in range(n_offspring):
        p_order = [p1, p2] if k == 0 else [p2, p1]
        pos = 0
        used = set()
        for i in range(len(cut_points) - 1):
            parent = p_order[i % 2]
            segment_size = cut_points[i+1] - cut_points[i]
            segment = []
            for item in parent:
                if item not in used and len(segment) < segment_size:
                    segment.append(item)
                    used.add(item)
            other_parent = p_order[(i+1) % 2]
            for item in other_parent:
                if item not in used and len(segment) < segment_size:
                    segment.append(item)
                    used.add(item)
            offspring[k, pos:pos+segment_size] = segment
            pos += segment_size
    # garante permutações válidas
    if n_jobs is not None and ops_per_job is not None:
        return _repair_offspring_jssp(offspring, p1, n_jobs, ops_per_job)
    return _repair_offspring(offspring, p1)

def pmx_crossover(p1, p2, n_offspring=2, n_jobs=None, ops_per_job=None):
    _validate_parents(p1, p2)
    size = len(p1)
    offspring = np.zeros((n_offspring, size), dtype=p1.dtype)
    a = np.random.randint(0, size)
    b = np.random.randint(0, size)
    if a > b:
        a, b = b, a
    for k in range(n_offspring):
        parent1 = p1 if k == 0 else p2
        parent2 = p2 if k == 0 else p1
        child = parent2.copy()
        child[a:b] = parent1[a:b]
        mapping = {}
        for i in range(a, b):
            if parent1[i] != parent2[i]:
                mapping[parent2[i]] = parent1[i]
        for i in range(size):
            if i < a or i >= b:
                item = child[i]
                while item in mapping:
                    item = mapping[item]
                child[i] = item
        offspring[k] = child
    # garante permutações válidas
    if n_jobs is not None and ops_per_job is not None:
        return _repair_offspring_jssp(offspring, p1, n_jobs, ops_per_job)
    return _repair_offspring(offspring, p1)
