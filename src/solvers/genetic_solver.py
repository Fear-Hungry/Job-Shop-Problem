import random
import copy
import time
import functools
import concurrent.futures
import numpy as np
from typing import List, Dict, Callable, Optional, Any
import os # Adicionado para obter a contagem de CPUs

# Use absolute imports from the 'src' directory
from solvers.base_solver import BaseSolver
from models.schedule import Schedule
from solvers.ortools_cpsat_solver import ORToolsCPSATSolver
# Import static helpers from common utils
from common.scheduling_utils import _calculate_schedule_details_static, _calculate_makespan_static

# Importações específicas do GA (também absolutas a partir de 'src')
from ga.genetic_operators.crossover import (
    OrderCrossover, PMXCrossover, CycleCrossover, PositionBasedCrossover, DisjunctiveCrossover
)
from ga.genetic_operators.mutation import StandardMutation, DisjunctiveMutation, CriticalPathSwap
from ga.genetic_operators.base import CrossoverStrategy, MutationStrategy, LocalSearchStrategy
from local_search.strategies import VNDLocalSearch
from ga.population.diversity import population_diversity
from ga.graph.disjunctive_graph import DisjunctiveGraph
from ga.graph.dsu import DSU
# Importar heurísticas de inicialização
from ga.initialization.heuristics import generate_spt_chromosome, generate_lpt_chromosome, generate_fifo_chromosome


class GeneticSolver(BaseSolver):
    def __init__(self, jobs, num_jobs, num_machines, population_size=30, generations=100, crossover_rate=0.8, mutation_rate=0.2, elite_size=1,
                 crossover_strategy=None, mutation_strategy=None, local_search_strategy=None, use_dsu=True,
                 initial_schedule=None):
        super().__init__(jobs, num_jobs, num_machines)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.elitism_type = None
        self.use_dsu = use_dsu

        if local_search_strategy is None:
             vnd_fitness_func = functools.partial(
                 _calculate_makespan_static,
                 jobs=self.jobs,
                 num_jobs=self.num_jobs,
                 num_machines=self.num_machines
             )
             self.local_search_strategy = VNDLocalSearch(
                 fitness_func=vnd_fitness_func,
                 jobs=self.jobs,
                 num_machines=self.num_machines,
                 random_seed=42
             )
        else:
             self.local_search_strategy = local_search_strategy

        if crossover_strategy is None:
            self.crossover_strategy = DisjunctiveCrossover(
                local_search_strategy=self.local_search_strategy)
        else:
            self.crossover_strategy = crossover_strategy

        if mutation_strategy is None:
             self.mutation_strategy = DisjunctiveMutation(
                 local_search_strategy=self.local_search_strategy)
        else:
             self.mutation_strategy = mutation_strategy

        self.crossover_operators = [
            OrderCrossover(self.local_search_strategy),
            PMXCrossover(self.local_search_strategy),
            CycleCrossover(self.local_search_strategy),
            PositionBasedCrossover(self.local_search_strategy)
        ]
        self.mutation_operators = [
            StandardMutation(self.local_search_strategy),
            DisjunctiveMutation(self.local_search_strategy),
            CriticalPathSwap(self.local_search_strategy)
        ]

        self.crossover_scores = [0.0 for _ in self.crossover_operators]
        self.mutation_scores = [0.0 for _ in self.mutation_operators]
        self.crossover_probs = [1.0 / len(self.crossover_operators) for _ in self.crossover_operators]
        self.mutation_probs = [1.0 / len(self.mutation_operators) for _ in self.mutation_operators]
        self.crossover_decay = 0.9
        self.mutation_decay = 0.9
        self.initial_schedule = initial_schedule

        from validators.schedule_validator import ScheduleValidator
        self.validator = ScheduleValidator(self.jobs, self.num_jobs, self.num_machines)

    def _machine_ops_from_chromosome(self, chromosome):
        machine_ops = {m: [] for m in range(self.num_machines)}
        for op_idx, (job_id, op_id) in enumerate(chromosome):
            machine_id, _ = self.jobs[job_id][op_id]
            machine_ops[machine_id].append((job_id, op_id))
        return machine_ops

    def _build_disjunctive_graph(self, chromosome, use_dsu=None):
        if use_dsu is None:
            use_dsu = self.use_dsu
        num_ops = len(chromosome)
        graph = DisjunctiveGraph(num_ops, use_dsu=use_dsu)
        # Create both mappings
        op_to_idx = {op: idx for idx, op in enumerate(chromosome)}
        idx_to_op = {idx: op for idx, op in enumerate(chromosome)} # Create inverse map
        graph.idx_to_op = idx_to_op # Attach to graph object
        graph.op_to_idx = op_to_idx # Attach op_to_idx as well, might be useful

        # Add job edges using op_to_idx
        for job_id, job in enumerate(self.jobs):
            for k in range(len(job) - 1):
                op1 = (job_id, k)
                op2 = (job_id, k+1)
                # Check if ops exist in the current chromosome mapping
                if op1 in op_to_idx and op2 in op_to_idx:
                     idx1 = op_to_idx[op1]
                     idx2 = op_to_idx[op2]
                     graph.add_edge(idx1, idx2) # Use indices

        # Add machine edges using op_to_idx
        machine_ops = self._machine_ops_from_chromosome(chromosome)
        for ops in machine_ops.values():
            for i in range(len(ops) - 1):
                op_u = ops[i]
                op_v = ops[i+1]
                # Check if ops exist in the current chromosome mapping
                if op_u in op_to_idx and op_v in op_to_idx:
                    u = op_to_idx[op_u]
                    v = op_to_idx[op_v]
                    graph.add_edge(u, v) # Use indices
        return graph

    def _decode_chromosome(self, chromosome):
        """Decodes chromosome to a Schedule object using the static helper."""
        scheduled_ops_list, _ = _calculate_schedule_details_static(
            chromosome, self.jobs, self.num_jobs, self.num_machines
        )
        if scheduled_ops_list is None:
            return Schedule([])
        else:
            return Schedule(scheduled_ops_list)

    def _initialize_population(self, initial_schedule):
        population = []
        initial_chromosomes = set() # Usar set para evitar duplicatas exatas iniciais

        # 1. Adicionar cromossomo da solução inicial (CP-SAT)
        if initial_schedule and initial_schedule.operations:
            base_chromosome = tuple((job_id, op_idx) for job_id,
                                op_idx, _, _, _ in initial_schedule.operations)
            if base_chromosome:
                population.append({'chromosome': list(base_chromosome)})
                initial_chromosomes.add(base_chromosome)
            else:
                print("Warning: Solução inicial fornecida não gerou cromossomo base.")
        else:
            print("Warning: Nenhuma solução inicial válida fornecida para _initialize_population.")
            # Poderia gerar um totalmente aleatório aqui como fallback?

        # 2. Adicionar cromossomos de heurísticas
        heuristic_generators = [
            generate_spt_chromosome,
            generate_lpt_chromosome,
            generate_fifo_chromosome
            # Adicionar mais geradores aqui
        ]

        for generator in heuristic_generators:
            if len(population) >= self.population_size:
                break
            try:
                heuristic_chrom_list = generator(self.jobs, self.num_jobs, self.num_machines)
                if heuristic_chrom_list:
                    heuristic_chrom_tuple = tuple(heuristic_chrom_list)
                    if heuristic_chrom_tuple not in initial_chromosomes:
                        population.append({'chromosome': heuristic_chrom_list})
                        initial_chromosomes.add(heuristic_chrom_tuple)
                    else:
                        print(f"Heurística {generator.__name__} gerou cromossomo duplicado.")
            except Exception as e:
                print(f"Erro ao gerar cromossomo com {generator.__name__}: {e}")

        # 3. Preencher o restante da população
        # Estratégia: Usar variações (shuffle) das melhores soluções encontradas até agora
        # Ou adicionar puramente aleatórios?
        # Vamos usar shuffle leve das soluções existentes (CP-SAT + heurísticas)

        num_to_generate = self.population_size - len(population)
        if num_to_generate > 0:
            print(f"Preenchendo {num_to_generate} indivíduos restantes na população inicial...")
            source_chromosomes = [indiv['chromosome'] for indiv in population] if population else []

            # Se não tivermos nenhuma solução inicial válida, geramos aleatórios
            if not source_chromosomes:
                print("Gerando população inicial totalmente aleatória como fallback.")
                base_ops = [(j, k) for j, job in enumerate(self.jobs) for k in range(len(job))]
                for _ in range(self.population_size):
                    random.shuffle(base_ops)
                    population.append({'chromosome': base_ops[:]})
            else:
                # Preenche ciclicamente usando shuffles leves dos existentes
                idx = 0
                while len(population) < self.population_size:
                    base_chrom = source_chromosomes[idx % len(source_chromosomes)][:]
                    # Shuffle leve
                    for _ in range(random.randint(1, max(3, len(base_chrom) // 10))):
                        a, b = random.sample(range(len(base_chrom)), 2)
                        base_chrom[a], base_chrom[b] = base_chrom[b], base_chrom[a]

                    # Verifica duplicata antes de adicionar
                    if tuple(base_chrom) not in initial_chromosomes:
                        population.append({'chromosome': base_chrom})
                        initial_chromosomes.add(tuple(base_chrom))
                    # Se for duplicata, tentamos de novo na próxima iteração do while
                    # (pode gerar loop se a diversidade for muito baixa e pop_size grande,
                    # mas improvável com shuffle)
                    idx += 1 # Garante que usamos fontes diferentes
                    # Adicionar um limite de tentativas para evitar loop infinito?
                    if idx > self.population_size * 5: # Limite arbitrário
                        print("Warning: Atingido limite de tentativas para gerar população inicial diversa. Pode haver duplicatas.")
                        # Adiciona mesmo que seja duplicata para atingir o tamanho
                        if len(population) < self.population_size:
                             population.append({'chromosome': base_chrom})
                             # Não adiciona ao 'initial_chromosomes' set se for duplicata forçada


        # Garante que a população tenha o tamanho exato
        if len(population) > self.population_size:
            population = population[:self.population_size]
        elif len(population) < self.population_size:
            print(f"Warning: População inicial final tem {len(population)} indivíduos, esperado {self.population_size}.")
            # Poderia adicionar mais aleatórios aqui se necessário

        print(f"População inicial gerada com {len(population)} indivíduos.")
        return population

    def _selection(self, population, fitnesses):
        # Tournament selection
        selected = []
        for _ in range(len(population)):
            i, j = random.sample(range(len(population)), 2)
            selected.append(population[i] if fitnesses[i]
                            < fitnesses[j] else population[j])
        return selected

    def _select_operator(self, probs):
        # Roulette wheel selection for operators
        r = random.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        return len(probs) - 1

    def _mutate(self, indiv):
        chromosome = indiv['chromosome']
        op_idx = self._select_operator(self.mutation_probs)
        mutation_op = self.mutation_operators[op_idx]

        kwargs = {
            'chromosome': chromosome,
            'jobs': self.jobs
        }

        if isinstance(mutation_op, (DisjunctiveMutation, CriticalPathSwap)):
            kwargs['graph_builder'] = self._build_disjunctive_graph
            if isinstance(mutation_op, DisjunctiveMutation):
                 kwargs['machine_ops'] = self._machine_ops_from_chromosome(chromosome)
                 kwargs['use_dsu'] = self.use_dsu

        try:
            new_chrom = mutation_op.mutate(**kwargs)
        except TypeError as e:
            print(f"TypeError calling mutate {mutation_op.__class__.__name__}: {e}, Args: {kwargs.keys()}")
            new_chrom = chromosome
        except Exception as e:
            print(f"Exception calling mutate {mutation_op.__class__.__name__}: {e}")
            new_chrom = chromosome

        indiv['mutation_op_idx'] = op_idx
        indiv['chromosome'] = new_chrom
        return indiv

    def _crossover(self, indiv1, indiv2):
        chrom1 = indiv1['chromosome']
        chrom2 = indiv2['chromosome']
        # new_dsu = DSU(len(chrom1)) # Child gets a new DSU

        op_idx = self._select_operator(self.crossover_probs)
        crossover_op = self.crossover_operators[op_idx]

        kwargs = {
            'parent1': chrom1,
            'parent2': chrom2
        }
        if isinstance(crossover_op, DisjunctiveCrossover):
            kwargs['machine_ops_builder'] = self._machine_ops_from_chromosome
            kwargs['graph_builder'] = self._build_disjunctive_graph
            kwargs['use_dsu'] = self.use_dsu
            # kwargs['dsu'] = new_dsu # DSU is built internally by the operator now if needed

        try:
            new_chrom = crossover_op.crossover(**kwargs)
        except TypeError as e:
            print(f"TypeError calling crossover {crossover_op.__class__.__name__}: {e}, Args: {kwargs.keys()}")
            new_chrom = chrom1
        except Exception as e:
            print(f"Exception calling crossover {crossover_op.__class__.__name__}: {e}")
            new_chrom = chrom1

        child_indiv = {
            'chromosome': new_chrom,
            # 'dsu': new_dsu,
            'crossover_op_idx': op_idx
        }
        return child_indiv

    def _fitness_chromosome(self, chromosome):
        """Calculates fitness (makespan) using the static helper."""
        try:
            makespan = _calculate_makespan_static(
                 chromosome, self.jobs, self.num_jobs, self.num_machines
            )
            return makespan
        except Exception as e:
            print(f"Error in _fitness_chromosome: {e}")
            return float('inf')

    def _fitness(self, indiv):
        return self._fitness_chromosome(indiv['chromosome'])

    def _vnd(self, indiv):
        chrom = indiv['chromosome']
        # DSU is not typically modified by VND, pass it along if present.
        dsu = indiv.get('dsu')
        new_chrom = self.local_search_strategy.local_search(chrom)
        return {'chromosome': new_chrom, 'dsu': dsu}

    def _population_diversity(self, chromosomes_list):
        # This method now correctly receives a list of chromosomes
        # and passes it to the external population_diversity function.
        return population_diversity(chromosomes_list)

    def solve(self, time_limit=30):
        random.seed(42)
        np.random.seed(42)
        self.elitism_type = random.choice([
            'generational', 'steady_state', 'local'])

        if self.initial_schedule:
            current_initial_schedule = self.initial_schedule
            print("Using provided initial CP-SAT solution for GA.")
        else:
            print("Warning: No initial solution provided. Calculating new one with CP-SAT...")
            initial_solver = ORToolsCPSATSolver(
                self.jobs, self.num_jobs, self.num_machines)
            initial_time_limit = max(10, time_limit // 10)
            current_initial_schedule_ops = initial_solver.solve(initial_time_limit) # This is Schedule object
            if current_initial_schedule_ops is None or not current_initial_schedule_ops.operations:
                print("Error: Internal CP-SAT failed to find a solution. GA cannot start.")
                return None, None # Return None for schedule and chromosome
            current_initial_schedule = current_initial_schedule_ops

        population = self._initialize_population(current_initial_schedule)
        best_indiv = None
        best_fitness = float('inf')

        min_mut, max_mut = 0.05, 0.7
        min_cross, max_cross = 0.5, 0.95
        target_div, ajuste = 0.3, 0.05
        low_div_count = 0
        start_time_ga = time.time()

        # Variáveis para Early Stopping por Convergência
        generations_without_improvement = 0
        # Permitir configurar externamente? Por agora, fixo.
        NO_IMPROVEMENT_LIMIT = 50

        partial_static_fitness_func = functools.partial(
            _calculate_makespan_static,
            jobs=self.jobs,
            num_jobs=self.num_jobs,
            num_machines=self.num_machines
        )

        # Avalia a população em paralelo
        futures = {}
        # Usar os.cpu_count() como padrão explícito
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for generation in range(self.generations):
                current_time = time.time()
                elapsed_time = current_time - start_time_ga
                if elapsed_time > time_limit:
                    print(f"Tempo limite do GA ({time_limit}s) atingido na geração {generation + 1}. Tempo decorrido: {elapsed_time:.2f}s")
                    break

                fitnesses = [self._fitness(indiv) for indiv in population]

                # Verifica se a lista de fitnesses não está vazia antes de prosseguir
                if not fitnesses:
                     print(f"Warning: Geração {generation + 1} não produziu indivíduos válidos.")
                     # O que fazer aqui? Parar? Continuar?
                     # Vamos pular esta geração, mas isso pode indicar um problema.
                     # Poderíamos incrementar generations_without_improvement?
                     generations_without_improvement += 1 # Considerar a falta de indivíduos como não-melhora
                     # Adicionar verificação de convergência aqui também
                     if generations_without_improvement >= NO_IMPROVEMENT_LIMIT:
                         print(f"GA Parado: Nenhuma melhoria ou indivíduos inválidos por {NO_IMPROVEMENT_LIMIT} gerações.")
                         break
                     continue # Pula para a próxima geração

                current_best_fitness_in_gen = min(fitnesses)

                # Atualiza o melhor indivíduo global e reseta o contador de não-melhora
                if best_indiv is None or current_best_fitness_in_gen < best_fitness:
                    best_fitness = current_best_fitness_in_gen
                    best_indiv_idx = fitnesses.index(best_fitness)
                    best_indiv = copy.deepcopy(population[best_indiv_idx])
                    generations_without_improvement = 0 # Resetar contador
                else:
                    # Incrementa contador se não houve melhora nesta geração
                    generations_without_improvement += 1

                # It expects a list of individuals (dictionaries) or chromosomes
                # Let's pass the list of chromosomes from the current population.
                # The external population_diversity function seems to operate on chromosomes.
                div = -1 # Default value
                if generation % 10 == 0: # Calculate diversity only every 10 generations
                     chromosomes_for_diversity = [indiv['chromosome'] for indiv in population]
                     div = self._population_diversity(chromosomes_for_diversity)

                # Calculate avg_fitness carefully to avoid division by zero with inf values
                valid_fitnesses = [f for f in fitnesses if f != float('inf')]
                avg_fitness = sum(valid_fitnesses) / len(valid_fitnesses) if valid_fitnesses else float('inf')
                gen_time = time.time() - current_time
                # Include generation number starting from 1 in the printout
                # Print diversity value even if it wasn't calculated this generation (shows last calculated or -1)
                # Limit logging frequency
                if generation == 0 or (generation + 1) % 20 == 0 or generation == self.generations - 1:
                     # Adicionar contagem de não melhoria ao log
                     print(f"Gen {generation+1}/{self.generations} | Best(G): {current_best_fitness_in_gen:.2f} | Best(O): {best_fitness:.2f} | Avg: {avg_fitness:.2f} | Div: {div:.3f} | NoImprove: {generations_without_improvement} | Time: {gen_time:.2f}s")

                # Verificação de Convergência após atualização do contador e log
                if generations_without_improvement >= NO_IMPROVEMENT_LIMIT:
                    print(f"GA Convergido: Sem melhoria por {NO_IMPROVEMENT_LIMIT} gerações consecutivas.")
                    break # Sair do loop

                if div != -1 and div < 0.15: # Check diversity only if calculated
                    low_div_count += 1
                else:
                    low_div_count = 0
                
                if low_div_count >= 5:
                    num_new = max(1, self.population_size // 10) # Inject fewer individuals
                    # Get chromosome from current best_indiv if available, else from population[0]
                    base_chrom_for_injection = best_indiv['chromosome'][:] if best_indiv else population[0]['chromosome'][:]

                    print(f"Low diversity for {low_div_count} gens. Injecting {num_new} new individuals.")
                    worst_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
                    for i in range(num_new):
                        chrom = base_chrom_for_injection[:]
                        random.shuffle(chrom)
                        replace_idx = worst_indices[i % len(worst_indices)] # Ensure valid index
                        population[replace_idx] = {'chromosome': chrom}
                    low_div_count = 0 
                    # Fitness will be re-evaluated at the start of the next generation

                if div < target_div:
                    self.mutation_rate = min(self.mutation_rate + ajuste, max_mut)
                    self.crossover_rate = max(self.crossover_rate - ajuste, min_cross)
                else:
                    self.mutation_rate = max(self.mutation_rate - ajuste, min_mut)
                    self.crossover_rate = min(self.crossover_rate + ajuste, max_cross)

                selected = self._selection(population, fitnesses)
                next_population = []
                crossover_rewards = [0.0 for _ in self.crossover_operators]
                mutation_rewards = [0.0 for _ in self.mutation_operators]
                crossover_counts = [0 for _ in self.crossover_operators]
                mutation_counts = [0 for _ in self.mutation_operators]

                # Cache fitness of parents for reward calculation
                # Ensure keys are hashable (convert list chromosome to tuple)
                parent_fitness_map = {}
                for indiv, fit in zip(population, fitnesses):
                    if isinstance(indiv.get('chromosome'), list):
                         parent_fitness_map[tuple(indiv['chromosome'])] = fit
                    # Handle cases where chromosome might not be a list? Unlikely here.

                for i in range(0, self.population_size, 2):
                    if i + 1 >= len(selected): # Handle odd population size if selection returns less
                         # If odd population, the last selected individual might be processed alone
                         # or duplicated. Let's just break or handle appropriately.
                         # For now, assume selection returns population_size individuals.
                         if len(selected) > i:
                              p1 = selected[i]
                              # Option 1: Mutate only
                              # Option 2: Add directly to next_population (less common)
                              # Option 3: Pair with a random individual?
                              # Let's skip the last odd individual for simplicity, consistent with range step 2.
                              # Or ensure selection/population size avoids this.
                         break

                    p1 = selected[i]
                    p2 = selected[(i+1)] # Use i+1 directly

                    # --- Optimized Copying ---
                    c1, c2 = None, None
                    crossover_op_idx_c1, crossover_op_idx_c2 = -1, -1
                    mutation_op_idx_c1, mutation_op_idx_c2 = -1, -1

                    # Get parent fitness for reward calculation (use cache)
                    fitness_p1 = parent_fitness_map.get(tuple(p1.get('chromosome', [])), float('inf'))
                    fitness_p2 = parent_fitness_map.get(tuple(p2.get('chromosome', [])), float('inf'))
                    if fitness_p1 == float('inf'): fitness_p1 = self._fitness(p1) # Recalculate if not in map
                    if fitness_p2 == float('inf'): fitness_p2 = self._fitness(p2) # Recalculate if not in map

                    if random.random() < self.crossover_rate:
                        # --- Crossover applied ---
                        res1 = self._crossover(p1, p2) # Returns a new dict with chromosome and new DSU
                        res2 = self._crossover(p2, p1) # Returns a new dict with chromosome and new DSU
                        c1, c2 = res1, res2
                        # Record which crossover operator was used (assuming _crossover adds this key)
                        crossover_op_idx_c1 = res1.get('crossover_op_idx', -1)
                        crossover_op_idx_c2 = res2.get('crossover_op_idx', -1)
                        # DSU deepcopy avoided here
                    else:
                        # --- Crossover skipped ---
                        # Prepare c1, c2 as copies of p1, p2 for potential mutation
                        # Shallow copy dict, deep copy chromosome, deep copy DSU (only now)
                        c1 = p1.copy()
                        c1['chromosome'] = copy.deepcopy(p1.get('chromosome'))
                        if 'dsu' in p1 and p1['dsu'] is not None:
                            c1['dsu'] = copy.deepcopy(p1['dsu']) # Deep copy DSU only if no crossover

                        c2 = p2.copy()
                        c2['chromosome'] = copy.deepcopy(p2.get('chromosome'))
                        if 'dsu' in p2 and p2['dsu'] is not None:
                            c2['dsu'] = copy.deepcopy(p2['dsu']) # Deep copy DSU only if no crossover
                        # crossover_op_idx remains -1

                    # --- End Optimized Copying ---

                    # Calculate fitness after potential crossover (before mutation)
                    fitness_c1_after_cross = self._fitness(c1)
                    fitness_c2_after_cross = self._fitness(c2)

                    # Apply Mutation
                    if random.random() < self.mutation_rate:
                        # Pass the current child (c1) to mutate.
                        # _mutate modifies the dictionary in-place (chromosome) and returns it.
                        c1_mutated = self._mutate(c1)
                        mutation_op_idx_c1 = c1_mutated.get('mutation_op_idx', -1)
                        c1 = c1_mutated # Update c1 reference if _mutate returns a new object (though current impl modifies in-place)

                    if random.random() < self.mutation_rate:
                        c2_mutated = self._mutate(c2)
                        mutation_op_idx_c2 = c2_mutated.get('mutation_op_idx', -1)
                        c2 = c2_mutated

                    # Calculate final fitness after potential mutation
                    fitness_c1_final = self._fitness(c1)
                    fitness_c2_final = self._fitness(c2)

                    # --- Reward Calculation ---
                    # Crossover reward: Improvement over the *average* or *best* parent? Let's use min(parent_fitness)
                    parent_fit_baseline = min(fitness_p1, fitness_p2)
                    if crossover_op_idx_c1 != -1:
                        # Reward is how much crossover improved fitness compared to parents baseline
                        reward_cross1 = max(0, parent_fit_baseline - fitness_c1_after_cross)
                        crossover_rewards[crossover_op_idx_c1] += reward_cross1
                        crossover_counts[crossover_op_idx_c1] += 1
                    # Should we calculate reward for c2 crossover too? Assume symmetrical operators for now.
                    # If c2 used same operator (or _crossover returns pairs), account for it.
                    # Let's assume _crossover call implies one operator use for the pair for simplicity.

                    # Mutation reward: Improvement from state *after* crossover to *final* state
                    if mutation_op_idx_c1 != -1:
                        reward_mut1 = max(0, fitness_c1_after_cross - fitness_c1_final)
                        mutation_rewards[mutation_op_idx_c1] += reward_mut1
                        mutation_counts[mutation_op_idx_c1] += 1
                    if mutation_op_idx_c2 != -1:
                         # Fitness before mutation for c2 was fitness_c2_after_cross
                         reward_mut2 = max(0, fitness_c2_after_cross - fitness_c2_final)
                         mutation_rewards[mutation_op_idx_c2] += reward_mut2
                         mutation_counts[mutation_op_idx_c2] += 1

                    # Add the possibly mutated children to the next population
                    if c1 is not None: next_population.append(c1)
                    if c2 is not None: next_population.append(c2)

                # Ensure next_population has the right size (handle odd sizes if necessary)
                # If next_population is slightly off due to odd numbers, trim or duplicate?
                # For now, assume it's roughly population_size. Elitism will adjust size.

                # --- Operator Score Update ---
                # Update crossover scores/probabilities
                for i in range(len(self.crossover_operators)):
                    avg_reward = (crossover_rewards[i] / crossover_counts[i]) if crossover_counts[i] > 0 else 0.0
                    # Use exponential moving average
                    self.crossover_scores[i] = self.crossover_scores[i] * self.crossover_decay + (1 - self.crossover_decay) * avg_reward
                total_score = sum(self.crossover_scores)
                if total_score > 1e-6: # Avoid division by zero
                    self.crossover_probs = [s / total_score for s in self.crossover_scores]
                else: # Reinitialize if all scores are zero
                    self.crossover_probs = [1.0 / len(self.crossover_operators)] * len(self.crossover_operators)

                # Update mutation scores/probabilities
                for i in range(len(self.mutation_operators)):
                     avg_reward = (mutation_rewards[i] / mutation_counts[i]) if mutation_counts[i] > 0 else 0.0
                     self.mutation_scores[i] = self.mutation_scores[i] * self.mutation_decay + (1 - self.mutation_decay) * avg_reward
                total_score = sum(self.mutation_scores)
                if total_score > 1e-6:
                    self.mutation_probs = [s / total_score for s in self.mutation_scores]
                else: # Reinitialize
                    self.mutation_probs = [1.0 / len(self.mutation_operators)] * len(self.mutation_operators)

                # --- Elitism ---
                # Get elites from the *current* population before replacement
                elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:self.elite_size]
                # Elites need deep copies to survive replacement if they are not selected again
                elites = [copy.deepcopy(population[i]) for i in elite_indices]

                # Replace worst individuals in next_population with elites
                if self.elite_size > 0 and len(next_population) >= self.population_size:
                    # Ensure next_population has fitness calculated for sorting
                    next_fitnesses = [self._fitness(indiv) for indiv in next_population]
                    worst_indices_next = sorted(range(len(next_fitnesses)), key=lambda i: next_fitnesses[i], reverse=True)

                    # Replace the worst 'elite_size' individuals of the new generation with the best 'elite_size' from the previous one
                    num_replaced = 0
                    replaced_indices = set() # Avoid replacing the same slot twice if elites overlap with worst
                    for elite_idx in range(self.elite_size):
                         if elite_idx < len(worst_indices_next):
                              replace_idx = worst_indices_next[elite_idx]
                              if replace_idx not in replaced_indices:
                                   if elite_idx < len(elites): # Ensure we have an elite to place
                                        next_population[replace_idx] = elites[elite_idx]
                                        replaced_indices.add(replace_idx)
                                        num_replaced += 1
                         # Ensure population size is maintained if elites were already part of next_pop
                         # This simple replacement assumes next_population size >= population_size

                    # Trim next_population back to population_size if it grew
                    population = next_population[:self.population_size]
                elif len(next_population) > 0 : # Handle case if elitism didn't run but we have a population
                    population = next_population[:self.population_size]
                    # If population is smaller than expected, pad or adjust? For now, just take what we have.
                    while len(population) < self.population_size:
                         # Add duplicates of best? Or random immigrants? Let's duplicate best for now.
                         if population:
                              population.append(copy.deepcopy(population[0])) # Duplicate best (assuming sorted/first is good)
                         else:
                              break # Cannot pad if empty

        total_ga_time = time.time() - start_time_ga
        print(f"GA finalizado. Tempo total GA: {total_ga_time:.2f}s. Melhor Makespan GA: {best_fitness}")

        if best_indiv is None:
            print("GA não encontrou nenhuma solução viável.")
            # Tentar retornar a solução inicial se disponível?
            if self.initial_schedule:
                 print("Retornando a solução inicial fornecida.")
                 # Precisamos converter Schedule para formato chromosome? Sim.
                 # A inicialização já faz isso, mas precisamos do cromossomo aqui.
                 initial_chromosome = [(job_id, op_idx) for job_id, op_idx, _, _, _ in self.initial_schedule.operations]
                 return self.initial_schedule, initial_chromosome # Retorna Schedule e Chromosome inicial
            else:
                 return None, None # Sem solução inicial e GA falhou

        # Retorna a melhor solução encontrada (Schedule e Chromosome)
        # Usa _decode_chromosome para obter o objeto Schedule a partir do melhor cromossomo
        best_schedule_obj = self._decode_chromosome(best_indiv['chromosome'])
        return best_schedule_obj, best_indiv['chromosome']

    def __str__(self):
        return (f"GeneticSolver(pop={self.population_size}, gen={self.generations}, "
                f"cx={self.crossover_rate:.2f}, mut={self.mutation_rate:.2f}, elite={self.elite_size})")
