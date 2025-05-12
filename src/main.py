import sys
import time  # Adicionado para controle de tempo
import functools # Adicionado para partial
import copy # Adicionado para deepcopy

from common import read_jobshop_instance, write_output
from models.schedule import Schedule # Importa Schedule
from solver import Solver
from solvers.genetic_solver import GeneticSolver
# Importa VND e utils necessários para a fase final de LS
from local_search.strategies import VNDLocalSearch
from common.scheduling_utils import _calculate_schedule_details_static, _calculate_makespan_static


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

    TOTAL_TIME_LIMIT = 600  # Tempo total em segundos (10 minutos)
    start_time = time.time()

    # 1. Resolve inicialmente com CP-SAT
    print("Executando CP-SAT para obter a solução inicial...")
    solver_cpsat = Solver(jobs, num_jobs, num_machines, solver_type="cpsat")
    cpsat_time_limit = TOTAL_TIME_LIMIT / 10 # Aloca 10% do tempo para CP-SAT
    initial_schedule_obj = solver_cpsat.solver.solve(time_limit=cpsat_time_limit)

    cpsat_end_time = time.time()
    elapsed_cpsat_time = cpsat_end_time - start_time
    remaining_time = TOTAL_TIME_LIMIT - elapsed_cpsat_time

    if initial_schedule_obj is None:
        print(f"CP-SAT não encontrou uma solução inicial viável em {elapsed_cpsat_time:.2f} segundos.")
        sys.exit(1)

    print(f"Solução inicial (CP-SAT) encontrada em {elapsed_cpsat_time:.2f} segundos:")
    initial_schedule_obj.print()
    initial_makespan = initial_schedule_obj.get_makespan()
    print(f"Melhor schedule inicial (CP-SAT) Makespan: {initial_makespan}")

    # Inicializa a melhor solução encontrada até agora
    best_overall_schedule = initial_schedule_obj
    best_overall_makespan = initial_makespan
    best_overall_chromosome = None # Não temos o cromossomo do CP-SAT

    # 2. Refina a solução com GA, se houver tempo restante
    if remaining_time > 1: # Precisa de tempo mínimo para rodar GA
        print(f"\nExecutando GA para refinar a solução com tempo restante: {remaining_time:.2f} segundos...")
        ga_time_limit = remaining_time * 0.8 # Aloca 80% do tempo restante para GA
        ga_solver = GeneticSolver(
            jobs, num_jobs, num_machines,
            population_size=200, generations=200, # Aumentar gerações?
            initial_schedule=initial_schedule_obj
        )

        # GA solve agora retorna (best_schedule, best_chromosome)
        ga_result_schedule, ga_result_chromosome = ga_solver.solve(time_limit=int(ga_time_limit))

        ga_end_time = time.time()
        elapsed_ga_time = ga_end_time - cpsat_end_time
        remaining_time_after_ga = TOTAL_TIME_LIMIT - (ga_end_time - start_time)

        if ga_result_schedule is None or not ga_result_schedule.operations:
            print("GA não encontrou uma solução válida dentro do tempo limite.")
            # Mantém a solução do CP-SAT como a melhor por enquanto
        else:
            ga_makespan = ga_result_schedule.get_makespan()
            print(f"\nSolução intermediária (GA) encontrada em {elapsed_ga_time:.2f}s: Makespan: {ga_makespan}")

            # Atualiza a melhor solução geral se o GA melhorou
            if ga_makespan < best_overall_makespan:
                print(f"Melhora do GA sobre CP-SAT: {best_overall_makespan - ga_makespan} ({(best_overall_makespan - ga_makespan) / best_overall_makespan:.2%})")
                best_overall_schedule = ga_result_schedule
                best_overall_makespan = ga_makespan
                best_overall_chromosome = ga_result_chromosome # Guarda o melhor cromossomo do GA
            else:
                print("Aviso: GA não melhorou o makespan da solução inicial do CP-SAT.")
                # best_overall_chromosome ainda é None ou o do melhor GA anterior (se houvesse)
                # Se GA não melhorou, o melhor cromossomo para LS é indefinido. Usar o do GA mesmo que não seja melhor?
                # Ou usar o cromossomo da solução inicial (se pudermos gerar)?
                # Vamos usar o do GA retornado, mesmo que o makespan não seja melhor, como ponto de partida para LS.
                if ga_result_chromosome:
                    best_overall_chromosome = ga_result_chromosome

            best_overall_makespan = ga_makespan
        # --- Fase 3: Busca Local Final (VND com operadores de CP) ---
        if remaining_time_after_ga > 1 and best_overall_chromosome is not None: # Tempo mínimo e cromossomo válido
            print(f"\nExecutando Busca Local Final (VND) com tempo restante: {remaining_time_after_ga:.2f} segundos...")

            # Cria a função de fitness parcial para o VND
            # Usamos a função estática importada
            final_ls_fitness_func = functools.partial(
                 _calculate_makespan_static,
                 jobs=jobs,
                 num_jobs=num_jobs,
                 num_machines=num_machines
            )

            # Instancia VND com operadores avançados habilitados
            final_vnd = VNDLocalSearch(
                fitness_func=final_ls_fitness_func,
                jobs=jobs,
                num_machines=num_machines,
                random_seed=int(time.time()), # Semente diferente para a fase final
                use_critical_path_operators=True, # Habilita operadores de CP
                use_block_operators=True,         # Habilita operadores de bloco
                use_advanced_neighborhoods=True,  # Habilita 2-opt, 3-opt
                max_tries_per_neighborhood=100, # Aumenta tentativas para forçar mais exploração
                # LNS pode ser útil aqui, mas precisa de mais configuração/tempo
                # perform_lns_shake=True, lns_shake_frequency=5, lns_shake_intensity=0.15
            )

            # Executa a busca local partindo do melhor cromossomo encontrado pelo GA
            try:
                 start_vnd_time = time.time()
                 # VND.local_search não tem limite de tempo interno, controlamos externamente (aproximado)
                 # Poderíamos adicionar um loop com verificação de tempo aqui se precisarmos de controle fino
                 improved_chromosome = final_vnd.local_search(best_overall_chromosome)
                 vnd_run_time = time.time() - start_vnd_time
                 print(f"Busca Local Final (VND) executada em {vnd_run_time:.2f}s.")

                 # Decodifica o resultado do VND
                 final_ops_list, final_makespan_vnd = _calculate_schedule_details_static(
                     improved_chromosome, jobs, num_jobs, num_machines
                 )

                 if final_ops_list and final_makespan_vnd < best_overall_makespan:
                     print(f"VND final melhorou o makespan de {best_overall_makespan} para: {final_makespan_vnd}")
                     # Atualiza a melhor solução geral
                     best_overall_schedule = Schedule(final_ops_list)
                     best_overall_makespan = final_makespan_vnd
                     # best_overall_chromosome = improved_chromosome # Atualiza o cromossomo também
                 else:
                     print("VND final não melhorou ou falhou em gerar schedule válido.")
                     # Mantém a solução anterior (do GA ou CP-SAT)
            except Exception as e:
                 print(f"Erro durante a execução do VND final: {e}")
                 # Mantém a solução anterior em caso de erro

        else:
             if best_overall_chromosome is None:
                  print("\nBusca Local Final não executada (nenhum cromossomo válido do GA).")
             else:
                  print(f"\nNão há tempo suficiente ({remaining_time_after_ga:.2f}s) para executar a Busca Local Final.")

    else:
        print("\nNão há tempo restante suficiente após CP-SAT para executar o GA ou Busca Local.")
        # A melhor solução já é a do CP-SAT (best_overall_schedule)

    # --- Finalização ---
    # Escreve a melhor solução encontrada (CP-SAT, GA ou GA+VND)
    total_elapsed_time = time.time() - start_time
    print(f"\nProcesso concluído. Melhor Makespan Final: {best_overall_makespan}")
    print(f"Tempo total de execução: {total_elapsed_time:.2f} segundos.")
    # Garante que está escrevendo a melhor solução final encontrada
    write_output(best_overall_schedule.operations, instance_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:  # Apenas o caminho da instância é necessário
        print("Usage: python main.py <instance_path>")
        sys.exit(1)

    instance_path = sys.argv[1]
    main(instance_path)  # Chama main apenas com o caminho da instância
