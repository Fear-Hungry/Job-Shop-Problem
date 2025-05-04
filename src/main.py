import sys

from common import read_jobshop_instance, write_output
from solver import Solver
from solvers.genetic_solver import GeneticSolver


def main(instance_path):
    """
    Função principal do programa. Lê uma instância do job shop do arquivo em
    `instance_path`, resolve inicialmente com CP-SAT, e então refina a solução
    usando um Algoritmo Genético (GA) inicializado com a solução do CP-SAT.
    Finalmente, escreve a solução final em um arquivo.

    Args:
        instance_path (str): Caminho para o arquivo contendo a instância do job shop.
    """
    jobs, num_jobs, num_machines = read_jobshop_instance(instance_path)

    # 1. Resolve inicialmente com CP-SAT para obter uma boa solução inicial
    print("Executando CP-SAT para obter a solução inicial...")
    solver_cpsat = Solver(jobs, num_jobs, num_machines, solver_type="cpsat")
    initial_solution_cpsat = solver_cpsat.solve()
    if initial_solution_cpsat is None:
        print("CP-SAT não encontrou uma solução inicial viável.")
        sys.exit(1)  # Ou lide com o erro de outra forma

    print("Solução inicial (CP-SAT):")
    solver_cpsat.print_schedule(initial_solution_cpsat)
    initial_schedule_obj = solver_cpsat.solver.schedule  # Acessa o objeto Schedule

    initial_makespan = initial_schedule_obj.get_makespan()
    print(f"Melhor schedule inicial (CP-SAT) Makespan: {initial_makespan}")

    # 2. Refina a solução com GA, passando a solução do CP-SAT no construtor
    print("\nExecutando GA para refinar a solução...")
    # Ajuste population_size e generations conforme necessário
    ga_solver = GeneticSolver(
        jobs, num_jobs, num_machines,
        population_size=50, generations=100,
        initial_schedule=initial_schedule_obj  # Passa a solução aqui
    )

    # Resolve com GA
    # Ajuste time_limit conforme necessário
    best_schedule_ga = ga_solver.solve(time_limit=60)

    if best_schedule_ga is None:
        print("GA não encontrou uma solução ou não melhorou a inicial.")
        # Usa a solução do CP-SAT
        final_solution = initial_solution_cpsat
        final_makespan = initial_makespan
        print(
            f"Usando a solução inicial do CP-SAT. Makespan: {final_makespan}")
    else:
        print("\nSolução final (GA):")
        final_solution = best_schedule_ga.operations  # Acessa as operações do Schedule
        # Calcula o makespan do Schedule final
        final_makespan = best_schedule_ga.get_makespan()
        print(f"Melhor schedule final (GA) Makespan: {final_makespan}")

        # Opcional: Comparar makespans
        if final_makespan >= initial_makespan:
            print("Aviso: GA não melhorou o makespan da solução inicial do CP-SAT.")

    # Escreve a solução final (do GA ou do CP-SAT)
    write_output(final_solution, instance_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:  # Apenas o caminho da instância é necessário
        print("Usage: python main.py <instance_path>")
        sys.exit(1)

    instance_path = sys.argv[1]
    main(instance_path)  # Chama main apenas com o caminho da instância
