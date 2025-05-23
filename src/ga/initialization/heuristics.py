import heapq
from typing import List, Tuple, Dict

# Tipo para representar uma operação: (job_id, op_id_no_job)
OpIdentifier = Tuple[int, int]
# Tipo para dados dos jobs: [[(id_maquina, duracao), ...], ...]
JobsData = List[List[Tuple[int, int]]]

def _generate_heuristic_chromosome(jobs: JobsData, num_jobs: int, num_machines: int, priority_func: callable) -> List[OpIdentifier]:
    """
    Função genérica para gerar um cromossomo usando uma heurística baseada em prioridade.

    Simula o processo de agendamento, mantendo o controle do tempo e das operações prontas.
    A cada passo, escolhe a próxima operação a ser agendada com base na função de prioridade
    fornecida (que opera sobre as operações prontas em uma máquina).

    Args:
        jobs: Lista de listas, onde jobs[j][k] é (id_maquina, duracao) da k-ésima operação do job j.
        num_jobs: Número total de jobs.
        num_machines: Número total de máquinas.
        priority_func: Função que recebe (id_maquina, duracao, job_id, op_id) e retorna
                       um valor numérico para priorização (menor valor = maior prioridade).

    Returns:
        Um cromossomo (lista de OpIdentifier) representando a ordem de início das operações.
    """
    chromosome: List[OpIdentifier] = []
    num_total_ops = sum(len(job) for job in jobs)

    # Estado do agendamento
    machine_available_time = [0] * num_machines # Tempo em que cada máquina estará disponível
    job_next_op_idx = [0] * num_jobs # Índice da próxima operação a ser agendada para cada job
    job_completion_time = [0] * num_jobs # Tempo de conclusão da última operação agendada do job

    # Operações prontas para serem agendadas
    # (valor_prioridade, job_id, op_id)
    ready_heap: List[Tuple[float, int, int]] = [] 
    # (job_id, op_id) -> (id_maquina, duracao)
    ops_details: Dict[OpIdentifier, Tuple[int, int]] = {} 

    # Inicializa com a primeira operação de cada job
    for j in range(num_jobs):
        if jobs[j]: # Se o job tem operações
            op_id = 0
            machine_id, duration = jobs[j][op_id]
            ops_details[(j, op_id)] = (machine_id, duration)
            priority_val = priority_func(machine_id, duration, j, op_id)
            heapq.heappush(ready_heap, (priority_val, j, op_id))

    scheduled_ops_count = 0
    while scheduled_ops_count < num_total_ops:
        if not ready_heap:
            # Isso não deveria acontecer em um JSSP válido se nem todas as ops foram agendadas.
            print("Aviso: Heurística ficou sem operações prontas antes de concluir.")
            break

        # Para simplificar, pegamos a de maior prioridade geral (menor valor)
        # e agendamos na sua máquina assim que a máquina E o job estiverem prontos.

        best_op_to_schedule = None
        temp_rejected = [] # Guarda temporariamente ops cuja máquina/job não está pronto

        while ready_heap:
            priority_val, job_id, op_id = heapq.heappop(ready_heap)
            machine_id, duration = ops_details[(job_id, op_id)]

            # Tempo de início = max(máquina disponível, job predecessora concluída)
            start_time = max(machine_available_time[machine_id], job_completion_time[job_id])
            completion_time = start_time + duration

            best_op_to_schedule = (job_id, op_id)
            chromosome.append(best_op_to_schedule)
            scheduled_ops_count += 1

            # Atualiza tempos
            machine_available_time[machine_id] = completion_time
            job_completion_time[job_id] = completion_time

            # Libera a próxima operação do mesmo job, se houver
            next_op_id = op_id + 1
            if next_op_id < len(jobs[job_id]):
                next_machine_id, next_duration = jobs[job_id][next_op_id]
                ops_details[(job_id, next_op_id)] = (next_machine_id, next_duration)
                next_priority_val = priority_func(next_machine_id, next_duration, job_id, next_op_id)
                heapq.heappush(ready_heap, (next_priority_val, job_id, next_op_id))

            # Coloca de volta as operações temporariamente rejeitadas
            for rejected_op in temp_rejected:
                heapq.heappush(ready_heap, rejected_op)

            break # Achamos uma operação para agendar nesta iteração

        if best_op_to_schedule is None and ready_heap:
             print("Aviso: Loop interno da heurística não encontrou operação agendável.")
             break
        elif not ready_heap and scheduled_ops_count < num_total_ops:
             print("Erro: Heap vazia mas nem todas as operações foram agendadas.")
             break

    if scheduled_ops_count != num_total_ops:
         print(f"Erro: Heurística gerou cromossomo incompleto ({scheduled_ops_count}/{num_total_ops} ops).")
         return [] # Retorna lista vazia para indicar falha

    return chromosome

# --- Funções Específicas de Prioridade ---

def spt_priority(_machine_id: int, duration: int, _job_id: int, _op_id: int) -> int:
    """Prioridade SPT: Menor duração tem maior prioridade (menor valor)."""
    return duration

def lpt_priority(_machine_id: int, duration: int, _job_id: int, _op_id: int) -> int:
    """Prioridade LPT: Maior duração tem maior prioridade (usamos negativo para caber no min-heap)."""
    return -duration

# --- Funções Públicas para Gerar Cromossomos ---

def generate_spt_chromosome(jobs: JobsData, num_jobs: int, num_machines: int) -> List[OpIdentifier]:
    """Gera um cromossomo usando a heurística Shortest Processing Time (SPT)."""
    print("Gerando cromossomo inicial com heurística SPT...")
    return _generate_heuristic_chromosome(jobs, num_jobs, num_machines, spt_priority)

def generate_lpt_chromosome(jobs: JobsData, num_jobs: int, num_machines: int) -> List[OpIdentifier]:
    """Gera um cromossomo usando a heurística Longest Processing Time (LPT)."""
    print("Gerando cromossomo inicial com heurística LPT...")
    return _generate_heuristic_chromosome(jobs, num_jobs, num_machines, lpt_priority)

def fifo_priority(_machine_id: int, _duration: int, job_id: int, op_id: int) -> Tuple[int, int]:
    """Prioridade FIFO: Menor Job ID, depois menor Op ID (menor valor = maior prioridade)."""
    return (job_id, op_id)

def generate_fifo_chromosome(jobs: JobsData, num_jobs: int, num_machines: int) -> List[OpIdentifier]:
    """Gera um cromossomo usando a heurística First-In, First-Out (FIFO) por Job ID."""
    print("Gerando cromossomo inicial com heurística FIFO...")
    return _generate_heuristic_chromosome(jobs, num_jobs, num_machines, fifo_priority) 