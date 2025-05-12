import random
import copy
import time
import functools
import concurrent.futures
import numpy as np
from typing import List, Dict, Callable, Optional, Any

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
        op_to_idx = {op: idx for idx, op in enumerate(chromosome)}

        for job_id, job in enumerate(self.jobs):
            for k in range(len(job) - 1):
                op1 = (job_id, k)
                op2 = (job_id, k+1)
                if op1 in op_to_idx and op2 in op_to_idx:
                    graph.add_edge(op_to_idx[op1], op_to_idx[op2])

        machine_ops = self._machine_ops_from_chromosome(chromosome)
        for ops in machine_ops.values():
            for i in range(len(ops) - 1):
                u = op_to_idx[ops[i]]
                v = op_to_idx[ops[i+1]]
                graph.add_edge(u, v)
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
        base_chromosome = [(job_id, op_idx) for job_id,
                           op_idx, _, _, _ in initial_schedule.operations]
        population = [{'chromosome': base_chromosome[:],
                       'dsu': DSU(len(base_chromosome))}]
        for _ in range(self.population_size - 1):
            chrom = base_chromosome[:]
            # Small shuffle to create initial diversity
            for _ in range(random.randint(1, max(3, len(chrom) // 10))):
                a, b = random.sample(range(len(chrom)), 2)
                chrom[a], chrom[b] = chrom[b], chrom[a]
            population.append({'chromosome': chrom, 'dsu': DSU(len(chrom))})
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
        dsu = indiv.get('dsu')
        op_idx = self._select_operator(self.mutation_probs)
        mutation_op = self.mutation_operators[op_idx]

        kwargs = {
            'chromosome': chromosome,
            'dsu': dsu,
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
        new_dsu = DSU(len(chrom1)) # Child gets a new DSU

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
            kwargs['dsu'] = new_dsu

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
            'dsu': new_dsu,
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

    def _population_diversity(self, population):
        return population_diversity(population)

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

        partial_static_fitness_func = functools.partial(
            _calculate_makespan_static,
            jobs=self.jobs,
            num_jobs=self.num_jobs,
            num_machines=self.num_machines
        )

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for gen in range(self.generations):
                gen_start_time = time.time()
                chromosomes_to_eval = [indiv['chromosome'] for indiv in population]

                try:
                    fitnesses = list(executor.map(partial_static_fitness_func, chromosomes_to_eval))
                except Exception as e:
                    print(f"Error during parallel fitness (Gen {gen+1}): {e}. Falling back to sequential.")
                    fitnesses = [self._fitness(indiv) for indiv in population]

                current_best_fitness_in_gen = min(fitnesses) if fitnesses else float('inf')
                if best_indiv is None or current_best_fitness_in_gen < best_fitness:
                     if fitnesses:
                        best_fitness = current_best_fitness_in_gen
                        best_indiv_idx = fitnesses.index(best_fitness)
                        best_indiv = copy.deepcopy(population[best_indiv_idx])

                div = self._population_diversity(chromosomes_to_eval)
                # Calculate avg_fitness carefully to avoid division by zero with inf values
                valid_fitnesses = [f for f in fitnesses if f != float('inf')]
                avg_fitness = sum(valid_fitnesses) / len(valid_fitnesses) if valid_fitnesses else float('inf')
                gen_time = time.time() - gen_start_time
                print(f"Gen {gen+1}/{self.generations} | Best(G): {current_best_fitness_in_gen:.2f} | Best(O): {best_fitness:.2f} | Avg: {avg_fitness:.2f} | Div: {div:.3f} | Time: {gen_time:.2f}s")

                if div < 0.15:
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
                        population[replace_idx] = {'chromosome': chrom, 'dsu': DSU(len(chrom))}
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

                parent_fitness_map = {tuple(indiv['chromosome']): fit for indiv, fit in zip(population, fitnesses) if isinstance(indiv['chromosome'], list)}

                for i in range(0, self.population_size, 2):
                    p1 = selected[i]
                    p2 = selected[(i+1) % len(selected)]
                    fitness_p1 = parent_fitness_map.get(tuple(p1['chromosome']), self._fitness(p1))
                    fitness_p2 = parent_fitness_map.get(tuple(p2['chromosome']), self._fitness(p2))

                    c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
                    crossover_op_idx_c1, crossover_op_idx_c2 = -1, -1
                    mutation_op_idx_c1, mutation_op_idx_c2 = -1, -1

                    if random.random() < self.crossover_rate:
                        res1 = self._crossover(p1, p2)
                        res2 = self._crossover(p2, p1)
                        c1, c2 = res1, res2
                        crossover_op_idx_c1 = res1.get('crossover_op_idx', -1)
                        crossover_op_idx_c2 = res2.get('crossover_op_idx', -1)

                    fitness_c1_after_cross = self._fitness(c1)
                    fitness_c2_after_cross = self._fitness(c2)

                    if random.random() < self.mutation_rate:
                        c1 = self._mutate(c1)
                        mutation_op_idx_c1 = c1.get('mutation_op_idx', -1)
                    if random.random() < self.mutation_rate:
                        c2 = self._mutate(c2)
                        mutation_op_idx_c2 = c2.get('mutation_op_idx', -1)

                    fitness_c1_final = self._fitness(c1)
                    fitness_c2_final = self._fitness(c2)

                    if crossover_op_idx_c1 != -1:
                        parent_fit = min(fitness_p1, fitness_p2)
                        reward = max(0, parent_fit - fitness_c1_after_cross)
                        crossover_rewards[crossover_op_idx_c1] += reward
                        crossover_counts[crossover_op_idx_c1] += 1

                    if mutation_op_idx_c1 != -1:
                        reward = max(0, fitness_c1_after_cross - fitness_c1_final)
                        mutation_rewards[mutation_op_idx_c1] += reward
                        mutation_counts[mutation_op_idx_c1] += 1
                    if mutation_op_idx_c2 != -1:
                         reward = max(0, fitness_c2_after_cross - fitness_c2_final)
                         mutation_rewards[mutation_op_idx_c2] += reward
                         mutation_counts[mutation_op_idx_c2] += 1

                    next_population.extend([c1, c2])

                for i in range(len(self.crossover_operators)):
                    avg_reward = (crossover_rewards[i] / crossover_counts[i]) if crossover_counts[i] > 0 else 0.0
                    self.crossover_scores[i] = self.crossover_scores[i] * self.crossover_decay + avg_reward
                total_score = sum(self.crossover_scores)
                self.crossover_probs = [s / total_score for s in self.crossover_scores] if total_score > 0 else [1.0 / len(self.crossover_operators)] * len(self.crossover_operators)

                for i in range(len(self.mutation_operators)):
                     avg_reward = (mutation_rewards[i] / mutation_counts[i]) if mutation_counts[i] > 0 else 0.0
                     self.mutation_scores[i] = self.mutation_scores[i] * self.mutation_decay + avg_reward
                total_score = sum(self.mutation_scores)
                self.mutation_probs = [s / total_score for s in self.mutation_scores] if total_score > 0 else [1.0 / len(self.mutation_operators)] * len(self.mutation_operators)

                elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:self.elite_size]
                elites = [copy.deepcopy(population[i]) for i in elite_indices]

                for idx, elite_indiv in enumerate(elites):
                     try:
                         elites[idx] = self._vnd(elite_indiv)
                     except Exception as e:
                         print(f"Error during VND on elite (Gen {gen+1}): {e}")
                         pass

                num_non_elites = self.population_size - self.elite_size
                if num_non_elites < 0: num_non_elites = 0 # Ensure not negative
                next_population = elites + next_population[:num_non_elites]
                population = next_population[:self.population_size]

                elapsed_time_ga = time.time() - start_time_ga
                if elapsed_time_ga > time_limit:
                    print(f"GA time limit ({time_limit}s) reached at Gen {gen+1}.")
                    break

        final_fitnesses = [self._fitness(indiv) for indiv in population]
        best_chromosome_to_return = None

        if not final_fitnesses:
             print("Warning: Final population is empty or fitnesses are invalid.")
             if best_indiv:
                 print(f"Returning best solution from generations (Fitness: {best_fitness}).")
                 best_schedule = self._decode_chromosome(best_indiv['chromosome'])
                 best_chromosome_to_return = best_indiv['chromosome']
             elif current_initial_schedule:
                 print("Returning initial solution as no GA improvement was found.")
                 best_schedule = current_initial_schedule
                 # No chromosome for initial CP-SAT solution typically
             else:
                 print("Error: No valid solution found at all.")
                 return None, None # Should not happen if CP-SAT worked
        else:
            best_final_idx = final_fitnesses.index(min(final_fitnesses))
            best_indiv_final_pop = population[best_final_idx]
            best_fitness_final_pop = final_fitnesses[best_final_idx]

            if best_indiv and best_fitness < best_fitness_final_pop:
                 print(f"Returning best solution from generations (Fitness: {best_fitness}).")
                 best_schedule = self._decode_chromosome(best_indiv['chromosome'])
                 best_chromosome_to_return = best_indiv['chromosome']
            else:
                 print(f"Returning best solution from final population (Fitness: {best_fitness_final_pop}).")
                 best_schedule = self._decode_chromosome(best_indiv_final_pop['chromosome'])
                 best_chromosome_to_return = best_indiv_final_pop['chromosome']
        
        # Ensure best_schedule is always assigned if possible
        if 'best_schedule' not in locals():
            if current_initial_schedule:
                best_schedule = current_initial_schedule
            else: # Should be very rare, means CP-SAT failed and GA somehow also failed to produce even one valid indiv
                return None, None 

        self.schedule = best_schedule
        return best_schedule, best_chromosome_to_return
