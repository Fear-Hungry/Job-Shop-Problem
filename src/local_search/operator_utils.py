import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

def calculate_schedule_and_critical_path(chrom: list, jobs_data: List[List[Tuple[int, int]]], num_machines: int) -> tuple[dict[tuple[int, int], float], list[tuple[int, int]], float]:
    """Calcula os tempos de término das operações, o makespan e a rota crítica.

    Args:
        chrom: A sequência de operações (lista de tuplas (job_id, op_id)).
        jobs_data: Lista de jobs, onde cada job é uma lista de operações (machine_id, duration).
        num_machines: O número total de máquinas.

    Returns:
        Uma tupla contendo:
        - completion_times: Dicionário mapeando (job_id, op_id) para seu tempo de término.
        - critical_path: Lista de operações (job_id, op_id) na rota crítica.
        - makespan: O tempo de término da última operação.
        Retorna ({}, [], 0.0) em caso de erro ou se o cromossomo estiver vazio.
    """
    if not chrom:
        return {}, [], 0.0

    num_total_ops = len(chrom)
    completion_times = {}  # (job_id, op_id) -> end_time
    # machine_release_times = {m: 0.0 for m in range(num_machines)} # Não usado diretamente no cálculo do makespan ou CP essencial aqui
    # job_release_times = {j: 0.0 for j in range(len(jobs_data))} # Não usado diretamente

    op_details = {(j, i): jobs_data[j][i] for j, job in enumerate(jobs_data) for i in range(len(job))}
    job_predecessors = {}
    for j, job in enumerate(jobs_data):
        for i in range(len(job)):
            if i > 0:
                job_predecessors[(j, i)] = (j, i - 1)

    # Decodificação direta do cronograma: para cada operação em chrom, agendar no início mais cedo
    machine_release_times = {m: 0.0 for m in range(num_machines)}
    # Predecessores de máquina para backtracking do caminho crítico
    machine_predecessors = {}
    last_machine_op: Dict[int, Tuple[int, int]] = {}
    for op_tuple in chrom:
        if op_tuple not in op_details:
            logger.error(f"Operação {op_tuple} do cromossomo não encontrada em op_details. Verifique seus dados de jobs.")
            return {}, [], 0.0
        # tempo de término do predecessor de job
        job_pred = job_predecessors.get(op_tuple)
        job_pred_end = completion_times.get(job_pred, 0.0) if job_pred else 0.0
        machine_id, duration = op_details[op_tuple]
        prev_op = last_machine_op.get(machine_id)
        machine_pred_end = completion_times.get(prev_op, 0.0) if prev_op else 0.0
        start_time = max(job_pred_end, machine_pred_end)
        end_time = start_time + duration
        completion_times[op_tuple] = end_time
        # registrar predecessor de máquina
        if prev_op:
            machine_predecessors[op_tuple] = prev_op
        last_machine_op[machine_id] = op_tuple
    # Calcula makespan
    makespan = max(completion_times.values()) if completion_times else 0.0

    critical_path = []
    if not completion_times:
        return completion_times, critical_path, makespan

    last_ops = [op for op, ct in completion_times.items() if abs(ct - makespan) < 1e-6] # Tolerância para float
    if not last_ops:
        return completion_times, [], makespan

    current_op = last_ops[0]

    while current_op is not None:
        critical_path.append(current_op)

        # Detalhes da operação atual
        current_op_machine, current_op_duration = op_details[current_op]
        current_op_start_time = completion_times[current_op] - current_op_duration

        job_pred = job_predecessors.get(current_op)
        machine_pred = machine_predecessors.get(current_op)

        prev_op_candidate = None

        # Verificar se o predecessor do job define o tempo de início
        if job_pred and abs(completion_times.get(job_pred, -1.0) - current_op_start_time) < 1e-6:
            prev_op_candidate = job_pred

        # Verificar se o predecessor da máquina define o tempo de início
        # Esta condição tem precedência se ambas forem verdadeiras (ou a lógica pode precisar ser mais complexa para múltiplos caminhos críticos)
        if machine_pred and abs(completion_times.get(machine_pred, -1.0) - current_op_start_time) < 1e-6:
            # Se o predecessor da máquina também é crítico, ele é preferido ou o que acontece?
            # A lógica original parecia priorizar job_pred se ambos fossem candidatos.
            # Vamos manter uma lógica que pegue um deles. Se job_pred já foi pego, este o substitui se também for crítico.
            # Para simplificar, se ambos estiverem no caminho crítico, podemos escolher um.
            # No entanto, o caminho crítico é uma sequência.
            if prev_op_candidate == job_pred: # job_pred já foi identificado como crítico
                 # Se machine_pred também é crítico e diferente de job_pred, pode haver um problema ou um ponto de junção.
                 # A heurística aqui é que se job_pred é crítico, ele geralmente é o "motivador".
                 # Mas se machine_pred também leva exatamente a esse start_time, ele também está no caminho.
                 # A questão é qual *precedeu* na formação do schedule.
                 # A lógica original de backtracking era um pouco ambígua aqui.
                 # Vamos dar prioridade ao predecessor que resulta em um tempo de término igual ao início da operação atual.
                 # Se ambos, precisamos de uma regra. A original tinha um if/elif, que implicitamente priorizava job_pred.
                 pass # job_pred já é o candidato

            prev_op_candidate = machine_pred # Se machine_pred é crítico, ele se torna o candidato (ou sobrescreve job_pred)


        if prev_op_candidate is None:
            # Se a operação atual não tem predecessor que justifique seu tempo de início,
            # E seu tempo de início não é 0 (ou seja, não é a primeira op global),
            # então algo está estranho ou é a primeira op.
            if abs(current_op_start_time) > 1e-6: # Não é a primeira operação começando em zero
                 # logger.warning(
                 # f"Backtracking da rota crítica: {current_op} (início {current_op_start_time:.2f}) não tem predecessor crítico claro. "
                 # f"JobPred: {job_pred} (fim {completion_times.get(job_pred, -1):.2f}), "
                 # f"MachPred: {machine_pred} (fim {completion_times.get(machine_pred, -1):.2f}). Path pode ser incompleto."
                 # )
                 pass # Mantém o prev_op_candidate como None, o que terminará o loop.

        current_op = prev_op_candidate

    critical_path.reverse()
    return completion_times, critical_path, makespan
