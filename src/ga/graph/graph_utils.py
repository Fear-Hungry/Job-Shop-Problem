from typing import Tuple, List

# Formato da tupla de operação: (job_id, op_id_dentro_do_job)
OpTuple = Tuple[int, int]
# Formato dos dados do Job: List[List[Tuple[machine_id, duration]]]
JobData = List[List[Tuple[int, int]]]

def has_path_in_job_graph(op1: OpTuple, op2: OpTuple, jobs: JobData) -> bool:
    """
    Verifica se existe um caminho de precedência de job de op1 para op2.

    Isso é verdadeiro se e somente se op1 e op2 pertencem ao mesmo job,
    e op1 ocorre antes de op2 na sequência desse job.

    Args:
        op1: A primeira operação (job_id, op_id).
        op2: A segunda operação (job_id, op_id).
        jobs: A estrutura de dados do job (não utilizada nesta verificação simples,
              mas mantida para potenciais necessidades futuras do grafo).

    Returns:
        True se op2 segue op1 no mesmo job, False caso contrário.
    """
    job_id1, op_id1 = op1
    job_id2, op_id2 = op2

    # Se as operações são de jobs diferentes, não há caminho de precedência de job direto.
    if job_id1 != job_id2:
        return False

    # Se são a mesma operação, não há caminho de uma para a outra.
    if op_id1 == op_id2:
        return False

    # Se são do mesmo job, um caminho existe de op1 para op2 se e somente se op1 vem antes de op2.
    return op_id1 < op_id2 