"""
Módulo principal que fornece a classe Solver para resolver o problema Job Shop Scheduling.
"""
from models.schedule import Schedule
from validators.schedule_validator import ScheduleValidator
from solvers.ortools_cpsat_solver import ORToolsCPSATSolver


class Solver:
    """
    Classe principal para resolver o problema Job Shop Scheduling.
    Mantida para compatibilidade com o código existente.
    """
    def __init__(self, jobs, num_jobs, num_machines):
        """
        Inicializa uma instância de Solver para o problema Job Shop Scheduling.

        Args:
            jobs (list[list[tuple[int, int]]]): Lista de jobs, onde cada job é uma lista de pares (máquina, duração).
            num_jobs (int): Número de jobs.
            num_machines (int): Número de máquinas.
        """
        self.solver = ORToolsCPSATSolver(jobs, num_jobs, num_machines)
        self.validator = ScheduleValidator(jobs, num_jobs, num_machines)
        self.jobs = jobs
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.schedule = []

    def solve(self, time_limit=30):
        """
        Resolve o problema Job Shop Scheduling usando Google OR-Tools CP-SAT.
        
        Args:
            time_limit (int): Limite máximo de tempo em segundos para o solver.
            
        Returns:
            list[tuple[int, int, int, int, int]]: Uma lista de operações agendadas no formato
                (job_id, operation_index, machine_id, start_time, duration).
        """
        schedule = self.solver.solve(time_limit=time_limit)
        self.schedule = schedule.operations
        return self.schedule

    def get_makespan(self, schedule):
        """
        Calcula o makespan da agenda fornecida.

        Args:
            schedule (list[tuple[int, int, int, int, int]]): Uma lista de operações agendadas, onde cada operação é representada como uma tupla:
                (job_id, operation_index, machine_id, start_time, duration).

        Returns:
            int: O tempo máximo de conclusão de todas as operações na agenda (makespan).

        Raises:
            ValueError: Se a agenda estiver vazia.
        """
        temp_schedule = Schedule(schedule)
        return temp_schedule.get_makespan()

    def print_schedule(self, schedule):
        """
        Imprime a agenda de operações em um formato legível por humanos, junto com o makespan total.

        Args:
            schedule (list[tuple[int, int, int, int, int]]): Uma lista de operações agendadas, onde
                cada operação é representada como uma tupla: (job_id, operation_index, machine_id,
                start_time, duration).
        """
        temp_schedule = Schedule(schedule)
        temp_schedule.print()

    def is_valid_schedule(self, schedule):
        """
        Valida a agenda fornecida para o problema Job Shop Scheduling.

        Args:
            schedule (list[tuple[int, int, int, int, int]]): Uma lista de operações agendadas,
                onde cada operação é representada como uma tupla:
                (job_id, operation_index, machine_id, start_time, duration).

        Returns:
            bool: True se a agenda é válida, False caso contrário.
        """
        temp_schedule = Schedule(schedule)
        return self.validator.is_valid(temp_schedule)
