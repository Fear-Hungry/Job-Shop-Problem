"""
Módulo que contém a classe Schedule para representar uma agenda de operações do Job Shop.
"""


class Schedule:
    """
    Classe que representa uma agenda de operações para o problema Job Shop.
    """
    def __init__(self, operations=None):
        """
        Inicializa uma agenda de operações.

        Args:
            operations (list[tuple[int, int, int, int, int]], opcional): Lista de operações agendadas, onde
                cada operação é representada como uma tupla: (job_id, operation_index, machine_id, start_time, duration).
        """
        self.operations = operations or []
        self.sort_by_start_time()

    def add_operation(self, job_id, op_idx, machine_id, start_time, duration):
        """
        Adiciona uma operação à agenda.

        Args:
            job_id (int): ID do job.
            op_idx (int): Índice da operação no job.
            machine_id (int): ID da máquina.
            start_time (int): Tempo de início.
            duration (int): Duração da operação.
        """
        self.operations.append((job_id, op_idx, machine_id, start_time, duration))
    
    def sort_by_start_time(self):
        """
        Ordena as operações por tempo de início.
        """
        if self.operations:
            self.operations.sort(key=lambda x: x[3])

    def get_makespan(self):
        """
        Calcula o makespan da agenda.

        Returns:
            int: O tempo máximo de conclusão de todas as operações na agenda (makespan).

        Raises:
            ValueError: Se a agenda estiver vazia.
        """
        if not self.operations:
            raise ValueError("Schedule is empty")
        return max(start + duration for _, _, _, start, duration in self.operations)
    
    def print(self):
        """
        Imprime a agenda de operações em um formato legível por humanos, junto com o makespan total.
        """
        print("Schedule:")
        for job_id, op_idx, machine, start, duration in self.operations:
            print(
                f"Job {job_id}, Op {op_idx} -> Machine {machine} | Start: {start}, Duration: {duration}"
            )
        print(f"\nTotal Makespan: {self.get_makespan()}")