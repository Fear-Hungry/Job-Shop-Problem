import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

def calculate_schedule_and_critical_path(
    chrom: list, 
    num_machines: int,
    op_details: Dict[Tuple[int, int], Tuple[int, int]], # (job_id, op_id) -> (machine_id, duration)
    job_predecessors: Dict[Tuple[int, int], Tuple[int, int]] # (job_id, op_id) -> (job_id, op_id-1)
) -> tuple[dict[tuple[int, int], float], list[tuple[int, int]], float]:
    """Calcula os tempos de término das operações, o makespan e a rota crítica.

    Args:
        chrom: A sequência de operações (lista de tuplas (job_id, op_id)).
        num_machines: O número total de máquinas.
        op_details: Dicionário pré-calculado mapeando (job_id, op_id) para (machine_id, duration).
        job_predecessors: Dicionário pré-calculado mapeando (job_id, op_id) para seu predecessor de job.

    Returns:
        Uma tupla contendo:
        - completion_times: Dicionário mapeando (job_id, op_id) para seu tempo de término.
        - critical_path: Lista de operações (job_id, op_id) na rota crítica.
        - makespan: O tempo de término da última operação.
        Retorna ({}, [], 0.0) em caso de erro ou se o cromossomo estiver vazio.
    """
    if not chrom:
        return {}, [], 0.0

    completion_times = {}  # (job_id, op_id) -> end_time

    machine_op_predecessors = {} # Armazena o predecessor real na máquina na sequência `chrom`
    last_machine_op: Dict[int, Tuple[int, int]] = {} # Mantém a última operação agendada em cada máquina
    
    for op_tuple in chrom:
        if op_tuple not in op_details:
            logger.error(f"Operação {op_tuple} do cromossomo não encontrada em op_details. Verifique seus dados de jobs.")
            return {}, [], 0.0 # Retorno de erro
        
        job_pred = job_predecessors.get(op_tuple)
        job_pred_end = completion_times.get(job_pred, 0.0) # 0.0 se não houver predecessor (primeira op do job)
        
        machine_id, duration = op_details[op_tuple]
        
        # Predecessor na máquina é a última operação agendada *nessa máquina* na sequência `chrom`
        prev_op_on_machine = last_machine_op.get(machine_id)
        machine_pred_end = completion_times.get(prev_op_on_machine, 0.0) # 0.0 se for a primeira op na máquina
        
        start_time = max(job_pred_end, machine_pred_end)
        end_time = start_time + duration
        completion_times[op_tuple] = end_time
        
        if prev_op_on_machine:
            machine_op_predecessors[op_tuple] = prev_op_on_machine # op_tuple foi precedido por prev_op_on_machine nesta máquina
        
        last_machine_op[machine_id] = op_tuple # Atualiza a última operação nesta máquina

    if not completion_times:
        return {}, [], 0.0
    
    makespan = max(completion_times.values())

    critical_path = []
    last_op_on_critical_path = None
    for op, ct in completion_times.items():
        if abs(ct - makespan) < 1e-6:
            # Se múltiplas operações terminam no makespan, escolhemos uma.
            # Poderia haver uma heurística melhor, mas para um caminho, qualquer uma serve como ponto de partida.
            last_op_on_critical_path = op
            break 

    if last_op_on_critical_path is None:
        # Isso pode acontecer se completion_times estiver vazio, mas já checamos isso.
        # Ou se o makespan for 0 (ex: cromossomo com ops de duração zero), e nenhuma op termina em 0.
        # Se o makespan for > 0, pelo menos uma op deve terminar nele.
        if makespan > 1e-6: # Apenas logar se o makespan for significativamente > 0
            logger.warning(f"Nenhuma operação encontrada terminando no makespan {makespan}. Detalhes: {completion_times}")
        return completion_times, [], makespan

    current_op = last_op_on_critical_path
    while current_op is not None:
        critical_path.append(current_op)
        
        _, current_op_duration = op_details[current_op]
        current_op_start_time = completion_times[current_op] - current_op_duration

        # Se a operação atual começa em 0, ela é o início do caminho crítico (ou um deles)
        if abs(current_op_start_time) < 1e-6:
            break

        job_pred = job_predecessors.get(current_op)
        machine_pred = machine_op_predecessors.get(current_op) # Predecessor na máquina na sequência `chrom`

        chosen_prev_op = None

        # Checar predecessor do job
        if job_pred and abs(completion_times.get(job_pred, -1.0) - current_op_start_time) < 1e-6:
            chosen_prev_op = job_pred
        
        # Checar predecessor da máquina
        if machine_pred and abs(completion_times.get(machine_pred, -1.0) - current_op_start_time) < 1e-6:
            # Se job_pred também era candidato, qual escolher?
            # Priorizar aquele com maior tempo de término (ambos resultam no mesmo start_time para current_op)
            # Se machine_pred é o escolhido, ele se torna o candidato.
            # Se job_pred já era o escolhido, e machine_pred também é crítico, 
            # o machine_pred tem prioridade se seu tempo de término for igual (como ambos são críticos, seus tempos de término serão iguais a current_op_start_time).
            # Essencialmente, se ambos são críticos, a restrição de máquina é frequentemente a mais limitante.
            chosen_prev_op = machine_pred # Simplesmente prioriza machine_pred se ambos forem elegíveis.
                                       # Se apenas machine_pred for elegível, ele será escolhido.

        current_op = chosen_prev_op # Se nenhum for encontrado, current_op se torna None e o loop termina

    critical_path.reverse()
    return completion_times, critical_path, makespan
