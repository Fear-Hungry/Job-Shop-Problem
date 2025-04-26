import numpy as np
from numba import njit, prange
from collections import Counter
import heapq

def validate_permutation(perm, jobs, num_jobs, num_machines):
    # Conta quantas vezes cada job aparece na permutação
    job_counts = Counter([op // num_machines for op in perm])
    for job_id in range(num_jobs):
        expected = len(jobs[job_id])
        if job_counts[job_id] != expected:
            return False
    return True


def decode_permutation(perm, jobs, num_jobs, num_machines):
    # Validação da permutação
    if not validate_permutation(perm, jobs, num_jobs, num_machines):
        # Retorna penalidade alta
        return [(None, None, None, 0, 1e9)]
    job_counters = [0] * num_jobs
    machine_times = [0] * num_machines
    job_times = [0] * num_jobs
    schedule = []
    for op in perm:
        job_id = op // num_machines
        if job_id >= num_jobs:
            continue
        op_idx = job_counters[job_id]
        if op_idx >= len(jobs[job_id]):
            continue
        machine, duration = jobs[job_id][op_idx]
        start = max(machine_times[machine], job_times[job_id])
        schedule.append((job_id, op_idx, machine, start, duration))
        machine_times[machine] = start + duration
        job_times[job_id] = start + duration
        job_counters[job_id] += 1
    return schedule


def decode_giffler_thompson(perm, jobs, num_jobs, num_machines):
    if not validate_permutation(perm, jobs, num_jobs, num_machines):
        return [(None, None, None, 0, 1e9)]
    # Inicialização
    job_next_op = {job_id: 0 for job_id in range(num_jobs) if len(jobs[job_id]) > 0}
    machine_ready = {m: 0 for m in range(num_machines)}
    job_ready = {j: 0 for j in range(num_jobs)}
    perm_order = {}
    for idx, op in enumerate(perm):
        job_id = op // num_machines
        if (job_id, 0) not in perm_order:
            perm_order[(job_id, 0)] = idx
    schedule = []
    heap = []
    # Inicializa heap com a primeira operação de cada job
    for job_id in job_next_op:
        op_idx = 0
        machine, duration = jobs[job_id][op_idx]
        ready_time = max(job_ready[job_id], machine_ready[machine])
        completion_time = ready_time + duration
        heapq.heappush(heap, (completion_time, perm_order.get((job_id, op_idx), float('inf')), job_id, op_idx, machine, duration))
    while heap:
        # Extrai a operação com menor completion_time (e menor ordem na permutação em caso de empate)
        _, _, job_id, op_idx, machine, duration = heapq.heappop(heap)
        start_time = max(job_ready[job_id], machine_ready[machine])
        schedule.append((job_id, op_idx, machine, start_time, duration))
        # Atualiza ready times
        finish_time = start_time + duration
        machine_ready[machine] = finish_time
        job_ready[job_id] = finish_time
        # Próxima operação do job
        job_next_op[job_id] += 1
        next_op_idx = job_next_op[job_id]
        if next_op_idx < len(jobs[job_id]):
            next_machine, next_duration = jobs[job_id][next_op_idx]
            ready_time = max(job_ready[job_id], machine_ready[next_machine])
            completion_time = ready_time + next_duration
            # Atualiza perm_order se necessário
            if (job_id, next_op_idx) not in perm_order:
                for idx, op in enumerate(perm):
                    if op // num_machines == job_id:
                        if idx > perm_order.get((job_id, op_idx), -1):
                            perm_order[(job_id, next_op_idx)] = idx
                            break
            heapq.heappush(heap, (completion_time, perm_order.get((job_id, next_op_idx), float('inf')), job_id, next_op_idx, next_machine, next_duration))
    return schedule

def preprocess_jobs(jobs, num_jobs, num_machines):
    max_ops = max(len(j) for j in jobs)
    machines = np.full((num_jobs, max_ops), -1, dtype=np.int32)
    durations = np.zeros((num_jobs, max_ops), dtype=np.int32)
    for j in range(num_jobs):
        for o, (m, d) in enumerate(jobs[j]):
            machines[j, o] = m
            durations[j, o] = d
    return machines, durations, max_ops

@njit(parallel=True)
def jobshop_objective_numba(X, machines, durations, num_jobs, num_machines, max_ops):
    n = X.shape[0]
    F = np.empty((n, 1), dtype=np.float64)
    for i in prange(n):
        perm = X[i].astype(np.int32)
        job_counters = np.zeros(num_jobs, dtype=np.int32)
        machine_times = np.zeros(num_machines, dtype=np.int32)
        job_times = np.zeros(num_jobs, dtype=np.int32)
        for op in perm:
            job_id = op // num_machines
            op_idx = job_counters[job_id]
            if op_idx >= max_ops or machines[job_id, op_idx] == -1:
                continue
            machine = machines[job_id, op_idx]
            duration = durations[job_id, op_idx]
            start = max(machine_times[machine], job_times[job_id])
            machine_times[machine] = start + duration
            job_times[job_id] = start + duration
            job_counters[job_id] += 1
        F[i, 0] = np.max(job_times)
    return F

def jobshop_objective(X, jobs, num_jobs, num_machines):
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    machines, durations, max_ops = preprocess_jobs(jobs, num_jobs, num_machines)
    return jobshop_objective_numba(X, machines, durations, num_jobs, num_machines, max_ops)

def evaluate_schedule_multi_objective(schedule, jobs, num_jobs, num_machines, due_dates=None):
    if not schedule:
        return [float('inf'), float('inf'), float('inf')]
    makespan = max(start + duration for _, _, _, start, duration in schedule)
    job_completion_times = {}
    for job_id, op_idx, _, start, duration in schedule:
        completion_time = start + duration
        if job_id not in job_completion_times or completion_time > job_completion_times[job_id]:
            job_completion_times[job_id] = completion_time
    if due_dates is None:
        avg_makespan = sum(job_completion_times.values()) / len(job_completion_times)
        due_dates = {job_id: avg_makespan * 0.8 for job_id in range(num_jobs)}
    tardiness = sum(max(0, job_completion_times.get(job_id, 0) - due_dates.get(job_id, 0))
                    for job_id in range(num_jobs))
    machine_loads = [0] * num_machines
    for _, _, machine, start, duration in schedule:
        if machine is None:
            continue  # Ignora operações inválidas
        machine_loads[machine] += duration
    if sum(machine_loads) > 0:
        mean_load = sum(machine_loads) / num_machines
        load_variance = sum((load - mean_load) ** 2 for load in machine_loads) / num_machines
        load_balance = np.sqrt(load_variance) / mean_load if mean_load > 0 else 0
    else:
        load_balance = 0
    return [makespan, tardiness, load_balance]

def multi_objective_evaluation(X, jobs, num_jobs, num_machines, due_dates=None):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    results = []
    for perm in X:
        schedule = decode_giffler_thompson(perm.astype(int), jobs, num_jobs, num_machines)
        results.append(evaluate_schedule_multi_objective(
            schedule, jobs, num_jobs, num_machines, due_dates))
    return np.array(results)

def identify_critical_path(schedule, num_jobs, num_machines):
    if not schedule:
        return []
    # Construção do grafo disjuntivo
    # Cada nó: (job_id, op_idx)
    # Arestas: precedência de job e ordem na máquina
    from collections import defaultdict, deque
    op_nodes = []
    op_info = {}  # (job_id, op_idx) -> (job_id, op_idx, machine, start, duration)
    job_ops = defaultdict(list)
    machine_ops = defaultdict(list)
    for job_id, op_idx, machine, start, duration in schedule:
        node = (job_id, op_idx)
        op_nodes.append(node)
        op_info[node] = (job_id, op_idx, machine, start, duration)
        job_ops[job_id].append((op_idx, node))
        machine_ops[machine].append((start, node))
    # Ordena operações de cada job e máquina
    for job_id in job_ops:
        job_ops[job_id].sort()
    for machine in machine_ops:
        machine_ops[machine].sort()
    # Grafo: arestas e predecessores
    edges = defaultdict(list)
    preds = defaultdict(list)
    for job_id in job_ops:
        ops = [node for _, node in job_ops[job_id]]
        for i in range(1, len(ops)):
            edges[ops[i-1]].append(ops[i])
            preds[ops[i]].append(ops[i-1])
    for machine in machine_ops:
        ops = [node for _, node in machine_ops[machine]]
        for i in range(1, len(ops)):
            edges[ops[i-1]].append(ops[i])
            preds[ops[i]].append(ops[i-1])
    # Longest path via ordenação topológica
    in_degree = {node: 0 for node in op_nodes}
    for node in op_nodes:
        for v in edges[node]:
            in_degree[v] += 1
    topo = []
    queue = deque([node for node in op_nodes if in_degree[node] == 0])
    while queue:
        u = queue.popleft()
        topo.append(u)
        for v in edges[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    # Calcula o longest path
    dist = {node: 0 for node in op_nodes}
    pred_on_path = {node: None for node in op_nodes}
    for node in topo:
        job_id, op_idx, machine, start, duration = op_info[node]
        for v in edges[node]:
            if dist[v] < dist[node] + duration:
                dist[v] = dist[node] + duration
                pred_on_path[v] = node
    # Encontra o nó com maior distância (fim do caminho crítico)
    last_node = max(op_nodes, key=lambda n: dist[n] + op_info[n][4])
    # Reconstrói caminho crítico
    critical_path = []
    node = last_node
    while node is not None:
        critical_path.append(op_info[node])
        node = pred_on_path[node]
    critical_path.reverse()
    return critical_path

def find_critical_blocks(critical_path):
    if not critical_path:
        return []
    blocks = []
    current_block = [critical_path[0]]
    current_machine = critical_path[0][2]
    for i in range(1, len(critical_path)):
        op = critical_path[i]
        if op[2] == current_machine:
            current_block.append(op)
        else:
            if len(current_block) > 1:
                blocks.append(current_block)
            current_block = [op]
            current_machine = op[2]
    if len(current_block) > 1:
        blocks.append(current_block)
    return blocks
