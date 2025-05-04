import random
import copy
import time
import functools
import concurrent.futures
import numpy as np

# Use absolute imports from the 'src' directory
from solvers.base_solver import BaseSolver
from models.schedule import Schedule
from solvers.ortools_cpsat_solver import ORToolsCPSATSolver

# Importações específicas do GA (também absolutas a partir de 'src')
from solvers.ga.genetic_operators import (
    CrossoverStrategy, MutationStrategy, LocalSearchStrategy,
    OrderCrossover, PMXCrossover, CycleCrossover, PositionBasedCrossover, DisjunctiveCrossover
)
from solvers.ga.local_search.strategies import VNDLocalSearch
from solvers.ga.mutation.strategies import StandardMutation, DisjunctiveMutation
from solvers.ga.population.diversity import population_diversity
from solvers.ga.graph.disjunctive_graph import DisjunctiveGraph
from solvers.ga.graph.dsu import DSU


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
        self.elitism_type = None  # Será definido aleatoriamente no solve
        self.use_dsu = use_dsu
        # Estratégias
        if local_search_strategy is None:
            self.local_search_strategy = VNDLocalSearch(
                self._fitness_chromosome)
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
        # --- AOS: lista de operadores disponíveis ---
        self.crossover_operators = [
            OrderCrossover(self.local_search_strategy),
            PMXCrossover(self.local_search_strategy),
            CycleCrossover(self.local_search_strategy),
            PositionBasedCrossover(self.local_search_strategy)
        ]
        self.mutation_operators = [
            StandardMutation(self.local_search_strategy),
            DisjunctiveMutation(self.local_search_strategy)
        ]
        self.crossover_scores = [0.0 for _ in self.crossover_operators]
        self.mutation_scores = [0.0 for _ in self.mutation_operators]
        self.crossover_probs = [1/len(self.crossover_operators)
                                for _ in self.crossover_operators]
        self.mutation_probs = [1/len(self.mutation_operators)
                               for _ in self.mutation_operators]
        self.crossover_decay = 0.9
        self.mutation_decay = 0.9
        self.initial_schedule = initial_schedule

    def _machine_ops_from_chromosome(self, chromosome):
        # Converte cromossomo em dict: máquina -> lista de operações
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
        # Arestas de precedência (dentro do job)
        for job_id, job in enumerate(self.jobs):
            for k in range(len(job) - 1):
                op1 = (job_id, k)
                op2 = (job_id, k+1)
                if op1 in op_to_idx and op2 in op_to_idx:
                    graph.add_edge(op_to_idx[op1], op_to_idx[op2])
        # Arestas disjuntivas (ordem nas máquinas)
        machine_ops = self._machine_ops_from_chromosome(chromosome)
        for ops in machine_ops.values():
            for i in range(len(ops) - 1):
                u = op_to_idx[ops[i]]
                v = op_to_idx[ops[i+1]]
                graph.add_edge(u, v)
        return graph

    def _decode_chromosome(self, chromosome):
        # Decodifica cromossomo via grafo disjuntivo e ordenação topológica
        graph = self._build_disjunctive_graph(chromosome)
        order = graph.topological_sort()
        op_list = [chromosome[i] for i in order]
        job_next_op = [0] * self.num_jobs
        machine_available = [0] * self.num_machines
        job_end_time = [0] * self.num_jobs
        operations = []
        for job_id, op_idx in op_list:
            machine_id, duration = self.jobs[job_id][op_idx]
            start_time = max(
                machine_available[machine_id], job_end_time[job_id])
            operations.append(
                (job_id, op_idx, machine_id, start_time, duration))
            machine_available[machine_id] = start_time + duration
            job_end_time[job_id] = start_time + duration
        return Schedule(operations)

    def _initialize_population(self, initial_schedule):
        base_chromosome = [(job_id, op_idx) for job_id,
                           op_idx, _, _, _ in initial_schedule.operations]
        population = [{'chromosome': base_chromosome[:],
                       'dsu': DSU(len(base_chromosome))}]
        for _ in range(self.population_size - 1):
            chrom = base_chromosome[:]
            for _ in range(random.randint(1, 3)):
                a, b = random.sample(range(len(chrom)), 2)
                chrom[a], chrom[b] = chrom[b], chrom[a]
            population.append({'chromosome': chrom, 'dsu': DSU(len(chrom))})
        return population

    def _selection(self, population, fitnesses):
        selected = []
        for _ in range(len(population)):
            i, j = random.sample(range(len(population)), 2)
            selected.append(population[i] if fitnesses[i]
                            < fitnesses[j] else population[j])
        return selected

    def _select_operator(self, probs):
        r = random.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        return len(probs) - 1

    def _mutate(self, indiv):
        chromosome = indiv['chromosome']
        dsu = indiv['dsu']
        op_idx = self._select_operator(self.mutation_probs)
        mutation_op = self.mutation_operators[op_idx]
        # Prepara argumentos específicos apenas se o operador os necessitar
        kwargs = {
            'chromosome': chromosome,
            'dsu': dsu
        }
        if isinstance(mutation_op, DisjunctiveMutation):
            # DisjunctiveMutation requer machine_ops, graph_builder, use_dsu
            kwargs['machine_ops'] = self._machine_ops_from_chromosome(
                chromosome)
            kwargs['graph_builder'] = self._build_disjunctive_graph
            kwargs['use_dsu'] = self.use_dsu
        else:
            # Operadores padrão podem não precisar de todos os args, mas a assinatura aceita
            kwargs['machine_ops'] = None
            kwargs['graph_builder'] = None
            kwargs['use_dsu'] = False
        try:
            new_chrom = mutation_op.mutate(**kwargs)
        except TypeError as e:
            print(
                f"Erro ao chamar mutate do operador {mutation_op.__class__.__name__}: {e}")
            print(f"Argumentos passados: {kwargs.keys()}")
            # Retorna o cromossomo original em caso de erro na chamada
            new_chrom = chromosome
        # Atualiza o indivíduo com o novo cromossomo e o índice do operador usado
        indiv['mutation_op_idx'] = op_idx
        indiv['chromosome'] = new_chrom
        # O DSU é modificado in-place por DisjunctiveMutation, ou não usado por outros.
        # Não precisamos necessariamente retorná-lo explicitamente se o objeto é o mesmo.
        return indiv

    def _crossover(self, indiv1, indiv2):
        chrom1 = indiv1['chromosome']
        chrom2 = indiv2['chromosome']
        # Para o filho, sempre cria um novo DSU
        new_dsu = DSU(len(chrom1))
        op_idx = self._select_operator(self.crossover_probs)
        crossover_op = self.crossover_operators[op_idx]
        # Prepara argumentos específicos
        kwargs = {
            'parent1': chrom1,
            'parent2': chrom2
        }
        if isinstance(crossover_op, DisjunctiveCrossover):
            # DisjunctiveCrossover requer builders e DSU
            kwargs['machine_ops_builder'] = self._machine_ops_from_chromosome
            kwargs['graph_builder'] = self._build_disjunctive_graph
            kwargs['use_dsu'] = self.use_dsu
            # Passa o novo DSU para ser usado internamente
            kwargs['dsu'] = new_dsu
        try:
            new_chrom = crossover_op.crossover(**kwargs)
        except TypeError as e:
            print(
                f"Erro ao chamar crossover do operador {crossover_op.__class__.__name__}: {e}")
            print(f"Argumentos passados: {kwargs.keys()}")
            # Retorna o primeiro pai em caso de erro
            new_chrom = chrom1
        # Cria um novo dicionário para o filho
        child_indiv = {
            'chromosome': new_chrom,
            'dsu': new_dsu,  # Associa o novo DSU ao filho
            'crossover_op_idx': op_idx
        }
        return child_indiv

    def _fitness_chromosome(self, chromosome):
        try:
            schedule = self._decode_chromosome(chromosome)
            if not self.validator.is_valid(schedule):
                return float('inf')
            return schedule.get_makespan()
        except ValueError:
            return float('inf')

    def _fitness(self, indiv):
        return self._fitness_chromosome(indiv['chromosome'])

    def _get_fitness_cached(self, indiv, cache):
        """ Calcula ou recupera o fitness do cache da geração atual. """
        # Usa uma representação imutável do cromossomo como chave
        chrom_tuple = tuple(indiv['chromosome'])
        if chrom_tuple in cache:
            return cache[chrom_tuple]
        else:
            start_time = time.time()
            fitness = self._fitness(indiv)
            end_time = time.time()
            # Log do tempo de cálculo do fitness apenas quando não está em cache
            # print(f"    [Debug] Fitness Calc Time: {end_time - start_time:.4f}s")
            cache[chrom_tuple] = fitness
            return fitness

    def _vnd(self, indiv):
        chrom = indiv['chromosome']
        dsu = indiv['dsu']
        new_chrom = self.local_search_strategy.local_search(chrom)
        return {'chromosome': new_chrom, 'dsu': dsu}

    def _population_diversity(self, population):
        return population_diversity(population)

    def solve(self, time_limit=30):
        random.seed(42)
        np.random.seed(42)
        self.elitism_type = random.choice(
            ['generational', 'steady_state', 'local'])

        # Usa a solução inicial fornecida ou calcula uma se não houver
        if self.initial_schedule:
            current_initial_schedule = self.initial_schedule
            print("Usando a solução inicial CP-SAT fornecida para o GA.")
        else:
            print(
                "Aviso: Nenhuma solução inicial fornecida. Calculando uma nova com CP-SAT...")
            initial_solver = ORToolsCPSATSolver(
                self.jobs, self.num_jobs, self.num_machines)
            current_initial_schedule = initial_solver.solve(time_limit)
            if current_initial_schedule is None:
                print(
                    "Erro: CP-SAT interno não encontrou solução. O GA não pode começar.")
                return None  # Retorna None se não conseguir inicializar

        population = self._initialize_population(current_initial_schedule)
        best_indiv = None
        best_fitness = float('inf')
        min_mut = 0.05
        max_mut = 0.7
        min_cross = 0.5
        max_cross = 0.95
        target_div = 0.3
        ajuste = 0.05
        low_div_count = 0

        # Cria o executor fora do loop para reutilizar os processos
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for gen in range(self.generations):
                # --- Cache de Fitness para a geração atual ---
                # Este cache NÃO é compartilhado entre processos, cada um terá sua cópia.
                fitness_cache = {}

                # Cria uma função parcial com o cache atual fixado
                partial_get_fitness = functools.partial(
                    self._get_fitness_cached, cache=fitness_cache)

                # Calcula fitness em paralelo
                # Nota: O objeto 'indiv' e seus conteúdos (cromossomo, DSU) devem ser 'picklable'
                try:
                    fitnesses = list(executor.map(
                        partial_get_fitness, population))
                except Exception as e:
                    print(
                        f"Erro durante a execução paralela do fitness na Geração {gen+1}: {e}")
                    # Fallback para execução sequencial em caso de erro
                    fitnesses = [partial_get_fitness(
                        indiv) for indiv in population]

                div = self._population_diversity(
                    [indiv['chromosome'] for indiv in population])

                # --- Logging de Convergência ---
                current_best_fitness = min(fitnesses)
                avg_fitness = sum(fitnesses) / len(fitnesses)
                if best_indiv is None or current_best_fitness < best_fitness:
                    best_fitness = current_best_fitness
                    best_indiv = population[fitnesses.index(
                        current_best_fitness)]
                print(f"Gen {gen+1}/{self.generations} | Melhor Fit (Gen): {current_best_fitness:.2f} | Melhor Fit (Global): {best_fitness:.2f} | Média Fit: {avg_fitness:.2f} | Diversidade: {div:.3f}")
                # Opcional: Logar probabilidades dos operadores AOS
                # crossover_probs_str = ", ".join([f"{p:.2f}" for p in self.crossover_probs])
                # mutation_probs_str = ", ".join([f"{p:.2f}" for p in self.mutation_probs])
                # print(f"  Probs Crossover: [{crossover_probs_str}] | Probs Mutação: [{mutation_probs_str}]")
                # -----------------------------

                if div < 0.15:
                    low_div_count += 1
                else:
                    low_div_count = 0
                if low_div_count >= 5:
                    num_new = max(1, self.population_size // 5)
                    base_chromosome = population[0]['chromosome'][:]
                    for _ in range(num_new):
                        chrom = base_chromosome[:]
                        random.shuffle(chrom)
                        population[random.randint(
                            0, self.population_size-1)] = {'chromosome': chrom, 'dsu': DSU(len(chrom))}
                    low_div_count = 0
                if div < target_div:
                    self.mutation_rate = min(
                        self.mutation_rate + ajuste, max_mut)
                    self.crossover_rate = max(
                        self.crossover_rate - ajuste, min_cross)
                else:
                    self.mutation_rate = max(
                        self.mutation_rate - ajuste, min_mut)
                    self.crossover_rate = min(
                        self.crossover_rate + ajuste, max_cross)
                selected = self._selection(population, fitnesses)
                next_population = []
                crossover_rewards = [0.0 for _ in self.crossover_operators]
                mutation_rewards = [0.0 for _ in self.mutation_operators]
                crossover_counts = [0 for _ in self.crossover_operators]
                mutation_counts = [0 for _ in self.mutation_operators]
                for i in range(0, self.population_size, 2):
                    p1, p2 = selected[i], selected[(
                        i+1) % self.population_size]
                    # Cria cópias para evitar modificar os pais selecionados diretamente
                    c1, c2 = p1.copy(), p2.copy()
                    # Garante DSU independente
                    c1['dsu'] = copy.deepcopy(p1['dsu'])
                    # Garante DSU independente
                    c2['dsu'] = copy.deepcopy(p2['dsu'])
                    # Guarda fitness antes das operações genéticas (usando cache)
                    fitness_p1 = self._get_fitness_cached(p1, fitness_cache)
                    fitness_p2 = self._get_fitness_cached(p2, fitness_cache)
                    fitness_c1_before = self._get_fitness_cached(
                        c1, fitness_cache)
                    fitness_c2_before = self._get_fitness_cached(
                        c2, fitness_cache)
                    # Guarda índices dos operadores que serão usados (se aplicável)
                    crossover_op_idx_c1 = -1
                    crossover_op_idx_c2 = -1
                    mutation_op_idx_c1 = -1
                    mutation_op_idx_c2 = -1
                    if random.random() < self.crossover_rate:
                        # _crossover retorna novos indivíduos
                        res1 = self._crossover(p1, p2)
                        res2 = self._crossover(p2, p1)
                        c1 = res1  # Substitui c1 pelo filho do crossover
                        c2 = res2  # Substitui c2 pelo filho do crossover
                        crossover_op_idx_c1 = res1.get('crossover_op_idx', -1)
                        crossover_op_idx_c2 = res2.get('crossover_op_idx', -1)
                    # Guarda fitness *após* crossover (usando cache)
                    fitness_c1_after_cross = self._get_fitness_cached(
                        c1, fitness_cache)
                    fitness_c2_after_cross = self._get_fitness_cached(
                        c2, fitness_cache)
                    if random.random() < self.mutation_rate:
                        # _mutate modifica o indivíduo in-place e retorna-o
                        c1 = self._mutate(c1)
                        mutation_op_idx_c1 = c1.get('mutation_op_idx', -1)
                    if random.random() < self.mutation_rate:
                        c2 = self._mutate(c2)
                        mutation_op_idx_c2 = c2.get('mutation_op_idx', -1)
                    # Guarda fitness *após* mutação (usando cache)
                    fitness_c1_final = self._get_fitness_cached(
                        c1, fitness_cache)
                    fitness_c2_final = self._get_fitness_cached(
                        c2, fitness_cache)
                    # --- AOS: calcular recompensas ---
                    # Recompensa Crossover (melhora em relação ao melhor pai)
                    if crossover_op_idx_c1 != -1:
                        parent_fitness = min(fitness_p1, fitness_p2)
                        # Compara fitness pós-crossover com o do melhor pai
                        reward = max(0, parent_fitness -
                                     fitness_c1_after_cross)
                        crossover_rewards[crossover_op_idx_c1] += reward
                        crossover_counts[crossover_op_idx_c1] += 1
                    if crossover_op_idx_c2 != -1:
                        parent_fitness = min(fitness_p1, fitness_p2)
                        reward = max(0, parent_fitness -
                                     fitness_c2_after_cross)
                        crossover_rewards[crossover_op_idx_c2] += reward
                        crossover_counts[crossover_op_idx_c2] += 1
                    # Recompensa Mutação (melhora em relação ao estado antes da mutação)
                    if mutation_op_idx_c1 != -1:
                        # Compara fitness final com o fitness antes da mutação (pós-crossover)
                        reward = max(0, fitness_c1_after_cross -
                                     fitness_c1_final)
                        mutation_rewards[mutation_op_idx_c1] += reward
                        mutation_counts[mutation_op_idx_c1] += 1
                    if mutation_op_idx_c2 != -1:
                        reward = max(0, fitness_c2_after_cross -
                                     fitness_c2_final)
                        mutation_rewards[mutation_op_idx_c2] += reward
                        mutation_counts[mutation_op_idx_c2] += 1
                    next_population.extend([c1, c2])
                # --- Atualiza scores e probabilidades (DSOS) ---
                for i in range(len(self.crossover_operators)):
                    avg_reward = (
                        crossover_rewards[i] / crossover_counts[i]) if crossover_counts[i] > 0 else 0.0
                    self.crossover_scores[i] = self.crossover_scores[i] * \
                        self.crossover_decay + avg_reward
                total_score = sum(self.crossover_scores)
                if total_score > 0:
                    self.crossover_probs = [
                        s/total_score for s in self.crossover_scores]
                else:
                    self.crossover_probs = [
                        1/len(self.crossover_operators) for _ in self.crossover_operators]
                for i in range(len(self.mutation_operators)):
                    avg_reward = (
                        mutation_rewards[i] / mutation_counts[i]) if mutation_counts[i] > 0 else 0.0
                    self.mutation_scores[i] = self.mutation_scores[i] * \
                        self.mutation_decay + avg_reward
                total_score = sum(self.mutation_scores)
                if total_score > 0:
                    self.mutation_probs = [
                        s/total_score for s in self.mutation_scores]
                else:
                    self.mutation_probs = [
                        1/len(self.mutation_operators) for _ in self.mutation_operators]
                elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[
                    :self.elite_size]
                elites = [population[i] for i in elite_indices]
                # Aplica VND avançado apenas nos elites
                for idx, elite in enumerate(elites):
                    vnd = VNDLocalSearch(
                        self._fitness_chromosome, use_advanced_neighborhoods=True)
                    elites[idx] = {'chromosome': vnd.local_search(
                        elite['chromosome']), 'dsu': elite['dsu']}
                next_population = elites + \
                    next_population[:self.population_size - self.elite_size]
                population = next_population
        # --- Fim do loop de gerações ---
        # Calcula fitness final da última população (pode usar um cache final se precisar)
        final_fitness_cache = {}
        fitnesses = [self._get_fitness_cached(
            indiv, final_fitness_cache) for indiv in population]

        # Verifica se a lista de fitness não está vazia antes de encontrar o mínimo
        if not fitnesses:
            print("Erro: População final vazia ou inválida.")
            # Retorna a melhor solução global encontrada durante as gerações, se houver
            if best_indiv:
                best_schedule = self._decode_chromosome(
                    best_indiv['chromosome'])
                self.schedule = best_schedule
                return best_schedule
            else:
                # Se nem best_indiv existe, retorna a inicial que garantidamente existe
                self.schedule = current_initial_schedule
                return current_initial_schedule

        best_idx = fitnesses.index(min(fitnesses))
        best_indiv_final = population[best_idx]

        # Compara a melhor solução final com a melhor global encontrada (incluindo a inicial)
        final_decoded_schedule = self._decode_chromosome(
            best_indiv_final['chromosome'])
        final_fitness = final_decoded_schedule.get_makespan()

        # Determina qual é a melhor solução global (pode ser a inicial se nenhuma melhora ocorreu)
        if best_indiv:  # Se best_indiv foi atualizado durante as gerações
            global_best_chrom = best_indiv['chromosome']
            global_best_fitness = self._fitness_chromosome(global_best_chrom)
        else:  # Se nenhuma melhora ocorreu, a melhor global é a inicial
            global_best_chrom = [(j, op) for j, op, _, _,
                                 _ in current_initial_schedule.operations]
            global_best_fitness = current_initial_schedule.get_makespan()

        # Decide qual schedule retornar
        if global_best_fitness <= final_fitness:
            # A melhor solução encontrada durante as gerações (ou a inicial) é a melhor
            best_schedule = self._decode_chromosome(global_best_chrom)
            print(
                f"Retornando melhor solução global encontrada (Fitness: {global_best_fitness})")
        else:
            # A melhor solução da última geração é a melhor global
            best_schedule = final_decoded_schedule
            print(
                f"Retornando melhor solução da última geração (Fitness: {final_fitness})")

        self.schedule = best_schedule
        return best_schedule
