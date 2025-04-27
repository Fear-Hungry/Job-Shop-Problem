"""
Módulo que contém a classe abstrata BaseSolver para implementações de solvers do Job Shop.
"""
from abc import ABC, abstractmethod

from validators.schedule_validator import ScheduleValidator
from models.schedule import Schedule


class BaseSolver(ABC):
    """
    Classe base abstrata para solvers do problema Job Shop Scheduling.
    """
    def __init__(self, jobs, num_jobs, num_machines):
        """
        Inicializa um solver para o problema Job Shop Scheduling.

        Args:
            jobs (list[list[tuple[int, int]]]): Lista de jobs, onde cada job é uma lista de pares (máquina, duração).
            num_jobs (int): Número de jobs.
            num_machines (int): Número de máquinas.
        """
        self.jobs = jobs
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.validator = ScheduleValidator(jobs, num_jobs, num_machines)
        self.schedule = Schedule()

    @abstractmethod
    def solve(self, **kwargs):
        """
        Método abstrato para resolver o problema Job Shop Scheduling.
        Deve ser implementado pelas classes concretas.

        Returns:
            Schedule: A agenda de operações resultante.
        """
        pass