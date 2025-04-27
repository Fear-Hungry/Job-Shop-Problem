import numpy as np
from copy import deepcopy

def local_search_n7(ind, problem_eval, n_machines, n_jobs, max_iter=30):
    """
    Busca local N7 (Nowicki-Smutnicki) para problemas JSSP.
    Aplica movimentos sobre blocos críticos no caminho crítico.
    Args:
        ind: Indivíduo a ser melhorado (deve ter atributo x)
        problem_eval: função de avaliação (recebe np.array de permutação)
        n_machines: número de máquinas
        n_jobs: número de jobs
        max_iter: número máximo de iterações
    Returns:
        Indivíduo melhorado (atributos x e f atualizados)
    """
    if n_machines is None or n_jobs is None:
        return ind
    current = deepcopy(ind)
    current_x = np.array(current.x)
    current_f = problem_eval(np.array([current_x]))[0]
    best_x = current_x.copy()
    best_f = current_f.copy()
    for _ in range(max_iter):
        improved = False
        machine_ops = {}
        for i, op in enumerate(current_x):
            machine = op % n_machines  # Corrigido para usar n_machines
            if machine not in machine_ops:
                machine_ops[machine] = []
            machine_ops[machine].append((i, op))
        for machine, ops in machine_ops.items():
            if len(ops) < 2:
                continue
            for j in range(len(ops) - 1):
                idx1, op1 = ops[j]
                idx2, op2 = ops[j+1]
                new_x = current_x.copy()
                new_x[idx1], new_x[idx2] = new_x[idx2], new_x[idx1]
                new_f = problem_eval(np.array([new_x]))[0]
                if new_f[0] < best_f[0]:
                    best_x = new_x.copy()
                    best_f = new_f.copy()
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
        current_x = best_x.copy()
        current_f = best_f.copy()
    ind.x = best_x
    ind.f = best_f
    return ind
