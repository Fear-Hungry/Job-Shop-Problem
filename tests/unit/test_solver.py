"""
Testes unitários para a classe Solver.
"""
import pytest
from src.solver import Solver
from src.models.schedule import Schedule

pytestmark = [pytest.mark.unit]

@pytest.fixture
def simple_job_shop_instance():
    """Fixture que fornece uma instância simples do problema."""
    jobs = [
        [(0, 3), (1, 2), (2, 2)],  # Job 0
        [(1, 2), (0, 2), (2, 1)],  # Job 1
    ]
    num_jobs = 2
    num_machines = 3
    return jobs, num_jobs, num_machines

def test_solver_initialization(simple_job_shop_instance):
    """Testa se o Solver é inicializado corretamente."""
    jobs, num_jobs, num_machines = simple_job_shop_instance
    solver = Solver(jobs, num_jobs, num_machines)
    
    assert solver.jobs == jobs
    assert solver.num_jobs == num_jobs
    assert solver.num_machines == num_machines
    assert len(solver.schedule) == 0

def test_solve_method(simple_job_shop_instance):
    """Testa se o método solve retorna uma solução válida."""
    jobs, num_jobs, num_machines = simple_job_shop_instance
    solver = Solver(jobs, num_jobs, num_machines)
    
    schedule = solver.solve(time_limit=5)
    
    assert isinstance(schedule, list)
    assert len(schedule) == sum(len(job) for job in jobs)
    assert solver.is_valid_schedule(schedule)

def test_get_makespan(simple_job_shop_instance):
    """Testa se o cálculo do makespan está correto."""
    jobs, num_jobs, num_machines = simple_job_shop_instance
    solver = Solver(jobs, num_jobs, num_machines)
    
    # Cria uma agenda simples com makespan conhecido
    schedule = [
        (0, 0, 0, 0, 3),  # Job 0, Op 0
        (0, 1, 1, 3, 2),  # Job 0, Op 1
        (0, 2, 2, 5, 2),  # Job 0, Op 2
        (1, 0, 1, 0, 2),  # Job 1, Op 0
        (1, 1, 0, 3, 2),  # Job 1, Op 1
        (1, 2, 2, 7, 1)   # Job 1, Op 2
    ]
    
    makespan = solver.get_makespan(schedule)
    assert makespan == 8  # O último job termina em t=8

def test_is_valid_schedule(simple_job_shop_instance):
    """Testa se a validação da agenda funciona corretamente."""
    jobs, num_jobs, num_machines = simple_job_shop_instance
    solver = Solver(jobs, num_jobs, num_machines)
    
    # Agenda válida
    valid_schedule = [
        (0, 0, 0, 0, 3),
        (0, 1, 1, 3, 2),
        (0, 2, 2, 5, 2),
        (1, 0, 1, 0, 2),
        (1, 1, 0, 3, 2),
        (1, 2, 2, 7, 1)
    ]
    assert solver.is_valid_schedule(valid_schedule)
    
    # Agenda inválida (conflito de máquina)
    invalid_schedule = [
        (0, 0, 0, 0, 3),
        (0, 1, 1, 0, 2),  # Conflito com (1,0)
        (0, 2, 2, 5, 2),
        (1, 0, 1, 0, 2),  # Conflito com (0,1)
        (1, 1, 0, 3, 2),
        (1, 2, 2, 7, 1)
    ]
    assert not solver.is_valid_schedule(invalid_schedule)