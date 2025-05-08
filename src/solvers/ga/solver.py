import random
import copy
import time
import functools
import concurrent.futures
import numpy as np

from solvers.base_solver import BaseSolver
from models.schedule import Schedule
from solvers.ortools_cpsat_solver import ORToolsCPSATSolver
from .genetic_operators import (
    CrossoverStrategy, MutationStrategy, LocalSearchStrategy,
    OrderCrossover, PMXCrossover, CycleCrossover, PositionBasedCrossover, DisjunctiveCrossover
)
from local_search.strategies import VNDLocalSearch
from .mutation.strategies import StandardMutation, DisjunctiveMutation
from .population.diversity import population_diversity
from .graph.disjunctive_graph import DisjunctiveGraph
from .graph.dsu import DSU


class GeneticSolver(BaseSolver):
    def __init__(self, jobs, num_jobs, num_machines, population_size=30, generations=100, crossover_rate=0.8, mutation_rate=0.2, elite_size=1,
                 crossover_strategy=None, mutation_strategy=None, local_search_strategy=None, use_dsu=True,
                 initial_schedule=None, ucb_exploration_factor=2.0):
        super().__init__(jobs, num_jobs, num_machines)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.elitism_type = None  # Será definido aleatoriamente no solve
        self.use_dsu = use_dsu

        # Define a função de fitness como método para evitar repetição
        self._fitness_func = self._fitness_chromosome

        # Estratégias com injeção de dependência da função de fitness
        if local_search_strategy is None:
            # Passa a função de fitness para o VNDLocalSearch
            self.local_search_strategy = VNDLocalSearch(self._fitness_func)
        else:
            # Assume que a estratégia fornecida já tem uma forma de calcular o fitness
            # ou que será configurada externamente. Se ela precisar da função,
            # poderia ser injetada aqui se a classe permitir.
            self.local_search_strategy = local_search_strategy

        # Crossover Strategy
        if crossover_strategy is None:
            # Passa a estratégia de busca local (que já tem o fitness) para o DisjunctiveCrossover
            self.crossover_strategy = DisjunctiveCrossover(
                local_search_strategy=self.local_search_strategy)
        else:
            self.crossover_strategy = crossover_strategy

        # Mutation Strategy
        if mutation_strategy is None:
            # Passa a estratégia de busca local para DisjunctiveMutation
            self.mutation_strategy = DisjunctiveMutation(
                local_search_strategy=self.local_search_strategy)
        else:
            self.mutation_strategy = mutation_strategy

        # --- AOS: lista de operadores disponíveis ---
        # As estratégias aqui também precisam da função de fitness se usarem busca local interna
        # Se OrderCrossover, PMXCrossover, etc., usam a busca local passada,
        # então elas implicitamente terão acesso à função de fitness através dela.
        # Ajuste: Passa a função de fitness diretamente para estratégias que *não* recebem
        # uma local_search_strategy ou cuja local_search_strategy não é usada para fitness.
        # Assumindo que a local_search_strategy *sempre* é usada se presente:
        self.crossover_operators = [
            OrderCrossover(self.local_search_strategy),
            PMXCrossover(self.local_search_strategy),
            CycleCrossover(self.local_search_strategy),
            PositionBasedCrossover(self.local_search_strategy),
            # Adiciona DisjunctiveCrossover à lista se for desejado usá-lo no AOS
            # DisjunctiveCrossover(self.local_search_strategy) # Já é o padrão, mas pode ser incluído
        ]
        self.mutation_operators = [
            # Assume que StandardMutation usa local_search
            StandardMutation(self.local_search_strategy),
            # DisjunctiveMutation já é o padrão, mas pode ser incluído na lista AOS
            # DisjunctiveMutation(self.local_search_strategy)
        ]

        # --- AOS: Variáveis UCB1 ---
        self.ucb_exploration_factor = ucb_exploration_factor
        # Contagens de seleção para cada operador
        self.crossover_counts = [0] * len(self.crossover_operators)
        self.mutation_counts = [0] * len(self.mutation_operators)
        # Soma das recompensas (para calcular a média/valor)
        self.crossover_rewards_sum = [0.0] * len(self.crossover_operators)
        self.mutation_rewards_sum = [0.0] * len(self.mutation_operators)
        # Contagem total de seleções (para o termo de exploração)
        self.total_crossover_selections = 0
        self.total_mutation_selections = 0
        # --- Fim AOS ---
        # Mantem as variáveis antigas por enquanto, podem ser removidas depois
        self.crossover_scores = [0.0 for _ in self.crossover_operators]
        self.mutation_scores = [0.0 for _ in self.mutation_operators]
        self.crossover_probs = [1/len(self.crossover_operators)
                                for _ in self.crossover_operators]
        self.mutation_probs = [1/len(self.mutation_operators)
                               for _ in self.mutation_operators]
        self.crossover_decay = 0.9
        self.mutation_decay = 0.9
        self.initial_schedule = initial_schedule

        # Cache para fitness
        self.fitness_cache = {}

        # Inicializa DSU se necessário (pode ser útil ter um DSU 'mestre' se aplicável)
        # self.master_dsu = DSU(self.num_jobs * self.num_machines) # Exemplo

    # --- Funções Auxiliares ---

    @functools.lru_cache(maxsize=None)
    def _get_job_op_details(self, job_id, op_id):
        """Retorna (machine_id, duration) para uma operação, com cache."""
        return self.jobs[job_id][op_id]

    # @functools.lru_cache(maxsize=128) # Cache pode ser perigoso se self.jobs mudar
    def _machine_ops_from_chromosome(self, chromosome_tuple):
        """Converte cromossomo (tupla imutável) em dict: máquina -> lista de operações."""
        chromosome = list(
            chromosome_tuple)  # Converte para lista para uso interno se necessário
        machine_ops = {m: [] for m in range(self.num_machines)}
        # Itera sobre o índice e valor do cromossomo
        for op_idx_in_chrom, (job_id, op_id_in_job) in enumerate(chromosome):
            # Obtem detalhes da operação (máquina, duração)
            # Usar _get_job_op_details pode otimizar se self.jobs for grande e acessado frequentemente
            # machine_id, _ = self._get_job_op_details(job_id, op_id_in_job)
            machine_id, _ = self.jobs[job_id][op_id_in_job]
            # Adiciona a operação (job_id, op_id_in_job) à lista da máquina correspondente
            machine_ops[machine_id].append((job_id, op_id_in_job))
        return machine_ops

    # @functools.lru_cache(maxsize=128) # Cache pode ser perigoso
    def _build_disjunctive_graph(self, chromosome_tuple, op_to_idx, use_dsu=None):
        """Constrói o grafo disjuntivo a partir de um cromossomo (tupla) e mapeamento op->idx."""
        chromosome = list(chromosome_tuple)
        if use_dsu is None:
            use_dsu = self.use_dsu

        num_total_ops = len(chromosome)
        graph = DisjunctiveGraph(num_total_ops, use_dsu=use_dsu)

        # Mapeamento op_to_idx é passado como argumento agora
        # op_to_idx = {op: idx for idx, op in enumerate(chromosome)} # Removido

        # 1. Adicionar arestas de precedência (dentro de cada job)
        for job_id, job_ops in enumerate(self.jobs):
            for op_id_in_job in range(len(job_ops) - 1):
                # Operação atual e próxima operação no mesmo job
                op1 = (job_id, op_id_in_job)
                op2 = (job_id, op_id_in_job + 1)

                # Adiciona aresta apenas se ambas operações estão presentes no cromossomo
                # (Isso geralmente é verdade para representações completas)
                if op1 in op_to_idx and op2 in op_to_idx:
                    u = op_to_idx[op1]
                    v = op_to_idx[op2]
                    graph.add_edge(u, v)  # Adiciona aresta de precedência

        # 2. Adicionar arestas disjuntivas (ordem nas máquinas)
        # Obtem a sequência de operações para cada máquina a partir do cromossomo
        machine_ops = self._machine_ops_from_chromosome(
            chromosome_tuple)  # Passa a tupla

        for machine_id, ops_on_machine in machine_ops.items():
            # Para cada par consecutivo de operações na mesma máquina
            for i in range(len(ops_on_machine) - 1):
                op_u = ops_on_machine[i]
                op_v = ops_on_machine[i+1]

                # Obtem os índices no grafo/cromossomo
                u = op_to_idx[op_u]
                v = op_to_idx[op_v]
                # Adiciona aresta disjuntiva (ordem na máquina)
                graph.add_edge(u, v)

        return graph

    # @functools.lru_cache(maxsize=256) # Cache no decode pode ser muito útil
    def _decode_chromosome(self, chromosome_tuple):
        """Decodifica um cromossomo (tupla) para um objeto Schedule, calculando tempos de início."""
        chromosome = list(chromosome_tuple)
        start_decode_time = time.time()

        # Mapeamento op -> idx (criado aqui agora)
        op_to_idx = {op: idx for idx, op in enumerate(chromosome)}

        # 1. Construir o grafo disjuntivo (passando op_to_idx)
        graph = self._build_disjunctive_graph(
            chromosome_tuple, op_to_idx, use_dsu=False)

        # 2. Verificar se há ciclos
        if graph.has_cycle():
            # print(f"ALERTA: Ciclo detectado no cromossomo {chromosome_tuple}")
            # Retornar um Schedule inválido
            # schedule = Schedule([])
            # schedule.makespan = float('inf') # Atributo não existe
            # return schedule
            # Simplesmente retorna um schedule vazio. O fitness será inf por causa da validação.
            return Schedule([])

        # 3. Calcular os caminhos mais longos (tempos de início mais cedo)
        # O número de nós no grafo corresponde ao número total de operações
        num_total_ops = len(chromosome)
        # Os pesos dos nós são as durações das operações
        # Precisamos mapear o índice do nó de volta para a operação (job_id, op_id) para obter a duração
        # Criamos um mapeamento inverso: índice -> operação (Usando o op_to_idx local)
        idx_to_op = {idx: op for op, idx in op_to_idx.items()}

        node_weights = {}
        for i in range(num_total_ops):
            op = idx_to_op[i]
            job_id, op_id = op
            _, duration = self.jobs[job_id][op_id]
            node_weights[i] = duration

        # TODO: Implementar ou importar lógica para calcular caminho mais longo (longest_path)
        # start_times, makespan = graph.longest_path(node_weights)
        # Substituição temporária para evitar erro, mas a lógica real é necessária aqui:
        print("ALERTA: Lógica de cálculo de caminho mais longo (makespan) ausente em _decode_chromosome.")
        start_times = {idx: 0 for idx in range(num_total_ops)}  # Placeholder
        # Placeholder - Este valor não é mais retornado explicitamente
        makespan = float('inf')

        # 4. Construir a lista de operações para o Schedule
        operations = []
        # Verifica se longest_path retornou sucesso (sem ciclos negativos, etc.)
        if start_times is not None:
            for node_index, start_time in start_times.items():
                job_id, op_id = idx_to_op[node_index]
                machine_id, duration = self.jobs[job_id][op_id]
                operations.append(
                    (job_id, op_id, machine_id, start_time, duration)
                )
            # Ordena as operações por tempo de início para o objeto Schedule (opcional, mas bom)
            operations.sort(key=lambda x: x[3])
        else:
            # Se longest_path falhou (pode indicar ciclo não detectado antes, ou outro erro)
            # print(f"ALERTA: Falha ao calcular longest_path para {chromosome_tuple}")
            makespan = float('inf')  # Penaliza

        end_decode_time = time.time()
        # print(f"    Decodificação: {end_decode_time - start_decode_time:.4f}s")
        # Retorna apenas o schedule. O makespan será calculado via get_makespan() se necessário.
        # schedule = Schedule(operations)
        # schedule.makespan = makespan # Atributo não existe
        # return schedule
        return Schedule(operations)

    def _initialize_population(self, initial_schedule=None):
        """Inicializa a população, usando CP-SAT se disponível ou heurísticas."""
        population = []
        initial_chromosomes = set()  # Para evitar duplicatas exatas na inicialização

        # --- Estratégia 1: Usar CP-SAT para uma solução inicial de alta qualidade ---
        best_initial_chromosome = None
        best_initial_fitness = float('inf')

        if initial_schedule:
            # print("Usando solução inicial fornecida.")
            # Converte para tupla
            chrom = tuple([(op[0], op[1])
                          for op in initial_schedule.operations])
            # Valida rapidamente (ex: tamanho)
            # Adapte esta validação se necessário
            if len(chrom) == self.num_jobs * self.num_machines:
                initial_chromosomes.add(chrom)
                # Calcula fitness para referência
                fit = self._fitness_chromosome(chrom)
                if fit < best_initial_fitness:
                    best_initial_fitness = fit
                    best_initial_chromosome = chrom
                    # Adiciona ao DSU cache se necessário
                    # self.fitness_cache[chrom] = fit # Cache inicial
            else:
                print(
                    "Aviso: Schedule inicial fornecido parece inválido (tamanho incorreto?). Ignorando.")

        # Se não houver schedule inicial ou for inválido, tenta gerar um com CP-SAT
        if not best_initial_chromosome:
            # print("Tentando gerar solução inicial com CP-SAT...")
            try:
                # Roda CP-SAT com um limite de tempo curto
                cp_solver = ORToolsCPSATSolver(
                    self.jobs, self.num_jobs, self.num_machines)
                # Limite de tempo baixo para inicialização rápida (Convertido para int)
                cp_schedule = cp_solver.solve(time_limit=int(5.0))
                if cp_schedule and cp_schedule.operations:
                    chrom = tuple([(op[0], op[1])
                                  for op in cp_schedule.operations])
                    # Valida tamanho novamente
                    if len(chrom) == self.num_jobs * self.num_machines:
                        initial_chromosomes.add(chrom)
                        # Usa a função de fitness do GA
                        fit = self._fitness_chromosome(chrom)
                        if fit < best_initial_fitness:
                            best_initial_fitness = fit
                            best_initial_chromosome = chrom
                            # self.fitness_cache[chrom] = fit # Cache
                        # print(f"Solução inicial CP-SAT encontrada com makespan (calculado pelo GA): {fit:.2f}")
                    else:
                        print(
                            "Aviso: Schedule do CP-SAT parece inválido (tamanho incorreto?).")

                else:
                    # print("CP-SAT não encontrou solução inicial no tempo limite.")
                    pass  # Prossegue para heurísticas

            except Exception as e:
                # print(f"Erro ao executar CP-SAT para solução inicial: {e}")
                pass  # Prossegue para heurísticas

        # --- Estratégia 2: Heurísticas Simples (se CP-SAT falhou ou não foi usado) ---
        if not best_initial_chromosome:
            # print("Gerando solução inicial com heurística SPT (Shortest Processing Time)...")
            # Cria um cromossomo baseado em SPT (exemplo simples, pode não ser o melhor)
            ops_with_duration = []
            for j, job in enumerate(self.jobs):
                for o, (_, duration) in enumerate(job):
                    ops_with_duration.append(((j, o), duration))
            # Ordena por duração (SPT)
            ops_with_duration.sort(key=lambda x: x[1])
            base_chromosome_tuple = tuple([op[0] for op in ops_with_duration])
            if len(base_chromosome_tuple) == self.num_jobs * self.num_machines:
                initial_chromosomes.add(base_chromosome_tuple)
                fit = self._fitness_chromosome(base_chromosome_tuple)
                if fit < best_initial_fitness:
                    best_initial_fitness = fit
                    best_initial_chromosome = base_chromosome_tuple
                    # self.fitness_cache[base_chromosome_tuple] = fit # Cache
                # print(f"Solução inicial SPT gerada com makespan: {fit:.2f}")
            else:
                print("Erro: Cromossomo SPT gerado tem tamanho incorreto.")
                # Fallback extremo: gerar aleatório? Ou lançar erro?
                # Lançar erro é mais seguro para indicar problema na lógica.
                raise ValueError("Falha ao gerar cromossomo inicial válido.")

        # Garante que temos pelo menos uma solução inicial
        if not best_initial_chromosome:
            raise RuntimeError(
                "Não foi possível gerar nenhuma solução inicial válida!")

        # Adiciona a melhor solução inicial encontrada à população
        population.append({'chromosome': best_initial_chromosome,
                          'dsu': DSU(len(best_initial_chromosome))})
        # print(f"Melhor solução inicial adicionada à população (Makespan: {best_initial_fitness:.2f})")

        # --- Estratégia 3: Gerar Diversidade a partir da Melhor Inicial ---
        # Gera o restante da população perturbando a melhor solução inicial
        attempts = 0
        max_attempts = self.population_size * 5  # Evita loop infinito

        while len(population) < self.population_size and attempts < max_attempts:
            attempts += 1
            # Cria uma cópia mutável (lista)
            mutated_chrom_list = list(best_initial_chromosome)

            # Aplica N trocas aleatórias (mutação simples para diversificar)
            # Mais trocas para diversificar
            num_swaps = random.randint(
                1, max(2, len(mutated_chrom_list) // 10))
            for _ in range(num_swaps):
                if len(mutated_chrom_list) >= 2:
                    a, b = random.sample(range(len(mutated_chrom_list)), 2)
                    mutated_chrom_list[a], mutated_chrom_list[b] = mutated_chrom_list[b], mutated_chrom_list[a]

            # Converte de volta para tupla para ser hasheável
            mutated_chrom_tuple = tuple(mutated_chrom_list)

            # Adiciona apenas se for um cromossomo novo
            if mutated_chrom_tuple not in initial_chromosomes:
                population.append(
                    {'chromosome': mutated_chrom_tuple, 'dsu': DSU(len(mutated_chrom_tuple))})
                initial_chromosomes.add(mutated_chrom_tuple)
                # print(f"  Adicionado indivíduo {len(population)} por mutação da melhor inicial.")

        # Se ainda faltarem indivíduos, preenche com totalmente aleatórios (menos provável)
        while len(population) < self.population_size:
            # print("Completando população com indivíduos aleatórios...")
            # Começa com uma base válida
            random_chrom_list = list(best_initial_chromosome)
            random.shuffle(random_chrom_list)
            random_chrom_tuple = tuple(random_chrom_list)
            if random_chrom_tuple not in initial_chromosomes:
                population.append(
                    {'chromosome': random_chrom_tuple, 'dsu': DSU(len(random_chrom_tuple))})
                initial_chromosomes.add(random_chrom_tuple)

        # print(f"População inicializada com {len(population)} indivíduos.")
        return population

    # --- Operações Genéticas e Fitness ---

    def _selection(self, population, fitnesses):
        """Seleção por torneio binário."""
        selected = []
        pop_indices = list(range(len(population)))
        for _ in range(len(population)):
            idx1, idx2 = random.sample(pop_indices, 2)
            # Seleciona o indivíduo com menor fitness (makespan)
            winner_idx = idx1 if fitnesses[idx1] <= fitnesses[idx2] else idx2
            # Adiciona uma cópia para evitar modificar o original na população
            selected.append(copy.deepcopy(population[winner_idx]))
        return selected

    def _select_operator_ucb1(self, counts, rewards_sum, total_selections):
        """Seleciona um operador usando o algoritmo UCB1."""
        num_operators = len(counts)

        # Fase de exploração inicial: Garante que cada operador seja testado pelo menos uma vez.
        not_tried_indices = [i for i, count in enumerate(counts) if count == 0]
        if not_tried_indices:
            return random.choice(not_tried_indices)

        # Calcula os scores UCB1 para cada operador
        ucb_scores = []
        # Evita log(0) ou divisão por zero se total_selections for 1 e counts[i] for 1
        log_total_selections = np.log(
            max(1, total_selections))  # Usa max(1,...)

        for i in range(num_operators):
            count = counts[i]
            # Termo de Exploração (Exploitation Term) - Recompensa média
            # Adiciona um pequeno epsilon para evitar divisão por zero se count for 0 (embora já tratado acima)
            average_reward = rewards_sum[i] / count

            # Termo de Exploração (Exploration Term)
            exploration_term = self.ucb_exploration_factor * np.sqrt(
                log_total_selections / count
            )

            ucb_scores.append(average_reward + exploration_term)

        # Retorna o índice do operador com o maior score UCB1
        # Usa np.argmax que lida bem com listas de floats
        return np.argmax(ucb_scores)

    def _apply_mutation(self, indiv, mutation_operators, counts, rewards_sum, total_selections):
        """Aplica mutação selecionando um operador via UCB1."""
        chromosome_tuple = indiv['chromosome']  # Assume que é tupla
        # O DSU pode ou não ser necessário dependendo do operador
        dsu = indiv.get('dsu')  # Usa get para caso não exista

        # Seleciona o índice do operador de mutação usando UCB1
        op_idx = self._select_operator_ucb1(
            counts, rewards_sum, total_selections)
        mutation_op = mutation_operators[op_idx]
        # print(f"    Mutation: Selected operator {mutation_op.__class__.__name__} (Index: {op_idx})")

        # Prepara os argumentos. Passa o cromossomo como tupla se o operador puder lidar com isso,
        # ou converte para lista se necessário. Assume que operadores esperam lista por padrão.
        kwargs = {
            'chromosome': list(chromosome_tuple),  # Passa como lista
            # Adiciona outros argumentos conforme necessário para operadores específicos
        }
        if isinstance(mutation_op, DisjunctiveMutation):
            kwargs['machine_ops'] = self._machine_ops_from_chromosome(
                chromosome_tuple)  # Requer tupla
            # Passa a função builder diretamente, criando op_to_idx dentro do lambda
            kwargs['graph_builder'] = lambda chrom_list: self._build_disjunctive_graph(
                tuple(chrom_list), {op: idx for idx,
                                    op in enumerate(chrom_list)}, self.use_dsu
            )  # Correção: Adicionado op_to_idx e use_dsu
            # kwargs['use_dsu'] = self.use_dsu # Removido pois já passado acima
            # Passa o DSU se existir e for necessário
            if self.use_dsu and dsu:
                kwargs['dsu'] = dsu

        try:
            # Chama o método mutate do operador selecionado
            mutated_chrom_list = mutation_op.mutate(
                **kwargs)  # Espera lista de volta
            mutated_chrom_tuple = tuple(
                mutated_chrom_list)  # Converte para tupla
            # print(f"      Mutated chromosome: {mutated_chrom_tuple[:10]}...") # Debug
        except TypeError as e:
            print(
                f"      Erro de TypeError ao chamar mutate de {mutation_op.__class__.__name__}: {e}")
            print(f"      Argumentos passados: {kwargs.keys()}")
            # Fallback: retorna o cromossomo original como tupla
            mutated_chrom_tuple = chromosome_tuple
        except Exception as e:
            print(
                f"      Erro inesperado durante a mutação com {mutation_op.__class__.__name__}: {e}")
            # Fallback: retorna o cromossomo original
            mutated_chrom_tuple = chromosome_tuple

        # Retorna um NOVO dicionário representando o indivíduo mutado
        # Mantém outros dados do indivíduo original se houver (ex: fitness antigo, etc.)
        # mas atualiza o cromossomo e adiciona o índice do operador usado.
        new_indiv = indiv.copy()  # Cria cópia rasa
        new_indiv['chromosome'] = mutated_chrom_tuple
        # Armazena qual operador foi usado
        new_indiv['mutation_op_idx'] = op_idx
        # Atualiza DSU se necessário (DisjunctiveMutation pode modificar o DSU passado)
        # Se outros operadores pudessem criar um DSU novo, teria que ser tratado.
        # Como estamos passando como kwarg, a modificação é in-place se ocorrer.
        # Se a mutação invalidou o DSU ou criou um novo, precisa atualizar new_indiv['dsu']

        return new_indiv

    def _apply_crossover(self, indiv1, indiv2, crossover_operators, counts, rewards_sum, total_selections):
        """Aplica crossover selecionando um operador via UCB1."""
        parent1_tuple = indiv1['chromosome']
        parent2_tuple = indiv2['chromosome']

        # Seleciona o índice do operador de crossover usando UCB1
        op_idx = self._select_operator_ucb1(
            counts, rewards_sum, total_selections)
        crossover_op = crossover_operators[op_idx]
        # print(f"    Crossover: Selected operator {crossover_op.__class__.__name__} (Index: {op_idx})")

        # Prepara argumentos. Assume que operadores esperam listas.
        kwargs = {
            'parent1': list(parent1_tuple),
            'parent2': list(parent2_tuple),
        }
        if isinstance(crossover_op, DisjunctiveCrossover):
            # Passa builders que aceitam tuplas (otimizado com cache)
            kwargs['machine_ops_builder'] = self._machine_ops_from_chromosome
            # Passa o graph_builder criando op_to_idx
            kwargs['graph_builder'] = lambda chrom_tuple: self._build_disjunctive_graph(
                chrom_tuple, {op: idx for idx, op in enumerate(
                    chrom_tuple)}, self.use_dsu
            )  # Correção: Adicionado op_to_idx e use_dsu
            # kwargs['use_dsu'] = self.use_dsu # Removido
            # Cria um NOVO DSU para o filho, DisjunctiveCrossover pode usá-lo/modificá-lo
            child_dsu = DSU(len(parent1_tuple))
            kwargs['dsu'] = child_dsu  # Passa o novo DSU

        try:
            # Chama o método crossover do operador selecionado
            child_chrom_list = crossover_op.crossover(
                **kwargs)  # Espera lista de volta
            child_chrom_tuple = tuple(child_chrom_list)  # Converte para tupla
            # print(f"      Child chromosome: {child_chrom_tuple[:10]}...") # Debug

        except TypeError as e:
            print(
                f"      Erro de TypeError ao chamar crossover de {crossover_op.__class__.__name__}: {e}")
            print(f"      Argumentos passados: {kwargs.keys()}")
            # Fallback: retorna o primeiro pai como tupla
            child_chrom_tuple = parent1_tuple
            # Tenta pegar DSU do pai1 como fallback
            child_dsu = indiv1.get('dsu')
        except Exception as e:
            print(
                f"      Erro inesperado durante o crossover com {crossover_op.__class__.__name__}: {e}")
            # Fallback: retorna o primeiro pai
            child_chrom_tuple = parent1_tuple
            child_dsu = indiv1.get('dsu')

        # Cria um NOVO dicionário para o filho
        child_indiv = {
            'chromosome': child_chrom_tuple,
            'crossover_op_idx': op_idx,  # Armazena qual operador foi usado
            # Associa o DSU (novo ou fallback) ao filho, se aplicável
            'dsu': child_dsu if 'child_dsu' in locals() and child_dsu is not None else DSU(len(child_chrom_tuple))
        }
        return child_indiv

    def _update_operator_rewards(self, population, fitnesses, original_fitnesses):
        """Atualiza as contagens e recompensas UCB1 com base na melhoria de fitness."""
        for i, indiv in enumerate(population):
            original_fitness = original_fitnesses[i]
            current_fitness = fitnesses[i]
            # Maior é melhor (redução do makespan)
            improvement = original_fitness - current_fitness

            # Se o indivíduo foi resultado de crossover
            if 'crossover_op_idx' in indiv:
                op_idx = indiv['crossover_op_idx']
                self.crossover_counts[op_idx] += 1
                # Recompensa pode ser a melhoria absoluta, relativa, ou binária (melhorou ou não)
                # Usando melhoria absoluta:
                # Não penaliza piora, mas também não recompensa
                reward = max(0, improvement)
                # Ou recompensa binária: reward = 1 if improvement > 0 else 0
                # Ou recompensa relativa: reward = improvement / original_fitness if original_fitness > 0 else 0
                self.crossover_rewards_sum[op_idx] += reward
                self.total_crossover_selections += 1
                # Remove o índice para não contar de novo na mutação
                del indiv['crossover_op_idx']  # Ou marca como contado

            # Se o indivíduo foi resultado de mutação (pode ocorrer após crossover)
            if 'mutation_op_idx' in indiv:
                op_idx = indiv['mutation_op_idx']
                self.mutation_counts[op_idx] += 1
                # Usa a mesma lógica de recompensa
                reward = max(0, improvement)
                self.mutation_rewards_sum[op_idx] += reward
                self.total_mutation_selections += 1
                # Remove o índice
                del indiv['mutation_op_idx']  # Ou marca como contado

    # --- Função de Fitness ---

    # Cache para armazenar resultados de fitness calculados
    _fitness_cache = {}

    def _get_fitness_cached(self, chromosome_tuple):
        """Obtém o fitness do cache ou calcula e armazena."""
        if chromosome_tuple in self._fitness_cache:
            # print(f"    Cache HIT para fitness de {chromosome_tuple[:5]}...")
            return self._fitness_cache[chromosome_tuple]
        else:
            # print(f"    Cache MISS para fitness de {chromosome_tuple[:5]}... Calculando.")
            start_fit_time = time.time()
            # Calcula o fitness (makespan)
            schedule = self._decode_chromosome(
                chromosome_tuple)  # Passa tupla
            fitness = schedule.get_makespan() if schedule else float('inf')

            # Validação adicional (opcional mas recomendado)
            if fitness != float('inf') and not self.validator.is_valid(schedule):
                # print(f"ALERTA: Cromossomo {chromosome_tuple[:10]}... decodificado para schedule inválido, mas makespan não era inf.")
                fitness = float('inf')  # Penaliza se a validação falhar

            end_fit_time = time.time()
            # print(f"      Fitness calculado: {fitness:.2f} (Tempo: {end_fit_time - start_fit_time:.4f}s)")

            # Armazena no cache
            self._fitness_cache[chromosome_tuple] = fitness
            return fitness

    def _fitness_chromosome(self, chromosome):
        """Função de fitness principal que usa o cache. Aceita lista ou tupla."""
        # Converte para tupla para ser hasheável (chave do cache)
        if isinstance(chromosome, list):
            chromosome_tuple = tuple(chromosome)
        elif isinstance(chromosome, tuple):
            chromosome_tuple = chromosome
        else:
            raise TypeError("Cromossomo deve ser uma lista ou tupla.")

        return self._get_fitness_cached(chromosome_tuple)

    def _calculate_population_fitness(self, population):
        """Calcula o fitness para toda a população, usando cache."""
        fitnesses = [self._fitness_chromosome(
            indiv['chromosome']) for indiv in population]
        return fitnesses

    # --- Lógica Principal do Algoritmo Genético ---

    def solve(self, time_limit=None, verbose=True):
        start_time = time.time()
        # print(f"Iniciando Algoritmo Genético para {self.num_jobs} jobs e {self.num_machines} máquinas.")
        # print(f"Parâmetros: Pop={self.population_size}, Gen={self.generations}, CX={self.crossover_rate}, Mut={self.mutation_rate}, Elite={self.elite_size}, UseDSU={self.use_dsu}")

        # 0. Limpar caches do início da execução
        self._fitness_cache.clear()
        # Limpa caches das funções auxiliares se existirem e forem problemáticas entre runs
        # self._machine_ops_from_chromosome.cache_clear() # Exemplo
        # self._build_disjunctive_graph.cache_clear()    # Exemplo
        # self._decode_chromosome.cache_clear()          # Exemplo
        # Resetar UCB stats
        self.crossover_counts = [0] * len(self.crossover_operators)
        self.mutation_counts = [0] * len(self.mutation_operators)
        self.crossover_rewards_sum = [0.0] * len(self.crossover_operators)
        self.mutation_rewards_sum = [0.0] * len(self.mutation_operators)
        self.total_crossover_selections = 0
        self.total_mutation_selections = 0

        # 1. Inicialização
        # print("Inicializando população...")
        population = self._initialize_population(self.initial_schedule)
        # print(f"População inicializada com {len(population)} indivíduos.")
        # Calcula fitness inicial
        # print("Calculando fitness inicial...")
        fitnesses = self._calculate_population_fitness(population)
        # print("Fitness inicial calculado.")

        best_fitness_overall = min(fitnesses)
        best_chromosome_overall = population[fitnesses.index(
            best_fitness_overall)]['chromosome']
        # print(f"Melhor fitness inicial: {best_fitness_overall:.2f}")

        # Armazena histórico de fitness para análise
        fitness_history = [best_fitness_overall]

        # --- Loop Principal por Gerações ---
        for generation in range(self.generations):
            gen_start_time = time.time()
            # print(f"\n--- Geração {generation + 1}/{self.generations} ---")

            # Armazena fitness da população atual para cálculo de recompensa UCB1
            original_fitnesses = list(fitnesses)

            # 2. Elitismo: Seleciona os melhores indivíduos para passar diretamente
            elite_indices = sorted(range(len(population)), key=lambda k: fitnesses[k])[
                :self.elite_size]
            # Cria cópias profundas para evitar problemas com mutações posteriores
            next_population = [copy.deepcopy(
                population[i]) for i in elite_indices]
            # print(f"  Elitismo: {self.elite_size} melhores indivíduos mantidos (Best fit: {fitnesses[elite_indices[0]]:.2f})")

            # 3. Seleção (para reprodução)
            # print("  Seleção (Torneio)...")
            selected_parents = self._selection(population, fitnesses)
            # print(f"  {len(selected_parents)} pais selecionados.")

            # 4. Geração de Filhos (Crossover e Mutação)
            # print("  Reprodução (Crossover & Mutação)...")
            num_offspring_needed = self.population_size - self.elite_size
            offspring = []

            # Paralelização (opcional, pode adicionar overhead se as tarefas forem rápidas)
            use_parallel = False
            if use_parallel and num_offspring_needed > 0:
                # print("    (Usando processamento paralelo para gerar filhos)")
                # tasks = []
                # for i in range(0, num_offspring_needed, 2): # Processa em pares para crossover
                #       if i + 1 < len(selected_parents):
                #            parent1 = selected_parents[i]
                #            parent2 = selected_parents[i+1]
                #            tasks.append((parent1, parent2))

                # with concurrent.futures.ProcessPoolExecutor() as executor: # Ou ThreadPoolExecutor
                #      # Mapeia a função de gerar filho para as tarefas
                #      # Precisa de uma função wrapper que faça crossover E mutação
                #      # Exemplo: def generate_offspring_pair(p1, p2, ga_solver): ... return child1, child2
                #      # results = executor.map(lambda p: generate_offspring_pair(p[0], p[1], self), tasks)
                #      # offspring.extend(child for pair_result in results for child in pair_result)
                #      # A implementação exata da paralelização aqui é complexa devido ao estado compartilhado (UCB stats)
                #      # e necessidade de passar métodos/estado do solver.
                #      # Por simplicidade, mantemos serial por enquanto.
                pass  # Manter serial por enquanto

            # Geração serial
            while len(offspring) < num_offspring_needed:
                # Seleciona pais do pool selecionado
                parent1 = random.choice(selected_parents)
                parent2 = random.choice(selected_parents)

                # Crossover com probabilidade
                if random.random() < self.crossover_rate:
                    child = self._apply_crossover(
                        parent1, parent2, self.crossover_operators,
                        self.crossover_counts, self.crossover_rewards_sum, self.total_crossover_selections
                    )
                else:
                    # Se não houver crossover, um dos pais passa adiante (ou uma cópia)
                    # Copia para evitar modificar original
                    child = copy.deepcopy(parent1)
                    # Limpa/inicializa campos que seriam definidos pelo crossover se necessário
                    child['crossover_op_idx'] = -1  # Indica sem crossover

                # Mutação com probabilidade
                if random.random() < self.mutation_rate:
                    # Aplica mutação no filho (resultado do crossover ou cópia do pai)
                    child = self._apply_mutation(
                        child, self.mutation_operators,
                        self.mutation_counts, self.mutation_rewards_sum, self.total_mutation_selections
                    )
                else:
                    # Limpa/inicializa campos que seriam definidos pela mutação se necessário
                    child['mutation_op_idx'] = -1  # Indica sem mutação

                offspring.append(child)

            # Garante que temos o número certo de filhos (pode ter um a mais se num_offspring_needed for ímpar)
            offspring = offspring[:num_offspring_needed]
            # print(f"    {len(offspring)} filhos gerados.")

            # 5. Nova População
            next_population.extend(offspring)
            population = next_population
            # print(f"  Nova população formada com {len(population)} indivíduos.")

            # 6. Avaliação da Nova População
            # print("  Avaliando nova população...")
            start_eval_time = time.time()
            # Calcula fitness para os *novos* indivíduos (filhos)
            # Os indivíduos da elite já têm fitness calculado (mas recalcular pode ser mais simples)
            fitnesses = self._calculate_population_fitness(population)
            end_eval_time = time.time()
            # print(f"  Avaliação concluída (Tempo: {end_eval_time - start_eval_time:.4f}s).")

            # 7. Atualizar Recompensas UCB1
            self._update_operator_rewards(
                population, fitnesses, original_fitnesses)
            # print("  Recompensas UCB1 atualizadas.")
            # Opcional: Imprimir status UCB1 periodicamente
            # if (generation + 1) % 10 == 0:
            #      self.print_ucb_status()

            # 8. Atualizar Melhor Solução Geral
            current_best_fitness_gen = min(fitnesses)
            best_idx_gen = fitnesses.index(current_best_fitness_gen)
            if current_best_fitness_gen < best_fitness_overall:
                best_fitness_overall = current_best_fitness_gen
                # Guarda uma cópia profunda do melhor cromossomo
                best_chromosome_overall = copy.deepcopy(
                    population[best_idx_gen]['chromosome'])
                # print(f"  *** Novo melhor fitness geral encontrado: {best_fitness_overall:.2f} ***")

            fitness_history.append(best_fitness_overall)

            gen_end_time = time.time()
            # print(f"  Tempo da Geração: {gen_end_time - gen_start_time:.4f}s")
            # print(f"  Melhor Fitness da Geração: {current_best_fitness_gen:.2f}")
            # print(f"  Melhor Fitness Geral: {best_fitness_overall:.2f}")

            # 9. Critério de Parada (Tempo)
            elapsed_time = time.time() - start_time
            if time_limit is not None and elapsed_time > time_limit:
                # print(f"\nLimite de tempo ({time_limit}s) atingido na geração {generation + 1}.")
                break

        # --- Fim do Loop ---

        # print("--- Algoritmo Genético Concluído ---")
        final_elapsed_time = time.time() - start_time
        # print(f"Tempo total de execução: {final_elapsed_time:.2f}s")
        # print(f"Número de gerações executadas: {generation + 1}")
        # print(f"Melhor fitness encontrado: {best_fitness_overall:.2f}")

        # Decodifica o melhor cromossomo encontrado para obter o Schedule final
        # print("Decodificando melhor cromossomo...")
        final_schedule = self._decode_chromosome(
            best_chromosome_overall)  # Passa tupla

        # Validação final
        if not self.validator.is_valid(final_schedule):
            print("ALERTA: O melhor schedule final encontrado é inválido!")
            # O que fazer? Retornar None? Tentar decodificar o segundo melhor?
            # Por ora, retorna o schedule inválido com seu makespan (provavelmente inf)
        else:
            # Verifica se o makespan calculado pelo decode confere com o fitness armazenado
            final_makespan = final_schedule.get_makespan()
            if abs(final_makespan - best_fitness_overall) > 1e-6:  # Tolerância para float
                print(
                    f"ALERTA: Discrepância entre fitness armazenado ({best_fitness_overall:.2f}) e makespan decodificado ({final_makespan:.2f}). Usando o makespan decodificado.")
                best_fitness_overall = final_makespan

        # print(f"Makespan final (recalculado): {best_fitness_overall:.2f}")
        # self.print_ucb_final_status() # Imprime status final dos operadores
        # Retorna o melhor schedule encontrado
        return final_schedule

    def print_ucb_status(self):
        """Imprime o status atual das contagens e valores médios de recompensa UCB1."""
        print("  --- Status UCB1 ---")
        print("  Crossover Operators:")
        for i, op in enumerate(self.crossover_operators):
            count = self.crossover_counts[i]
            avg_reward = self.crossover_rewards_sum[i] / \
                count if count > 0 else 0
            print(
                f"    - {op.__class__.__name__}: Count={count}, AvgReward={avg_reward:.4f}")

        print("  Mutation Operators:")
        for i, op in enumerate(self.mutation_operators):
            count = self.mutation_counts[i]
            avg_reward = self.mutation_rewards_sum[i] / \
                count if count > 0 else 0
            print(
                f"    - {op.__class__.__name__}: Count={count}, AvgReward={avg_reward:.4f}")
        print("  --------------------")

    def print_ucb_final_status(self):
        """Imprime o status final UCB1 após a execução."""
        print("--- Status Final UCB1 ---")
        print("Crossover Operators (Total Selections: {}):".format(
            self.total_crossover_selections))
        # Ordena por contagem para ver os mais usados
        crossover_stats = sorted(
            [
                (op.__class__.__name__, self.crossover_counts[i], self.crossover_rewards_sum[i] /
                 self.crossover_counts[i] if self.crossover_counts[i] > 0 else 0)
                for i, op in enumerate(self.crossover_operators)
            ],
            key=lambda x: x[1], reverse=True  # Ordena por contagem decrescente
        )
        for name, count, avg_reward in crossover_stats:
            print(f"  - {name}: Count={count}, AvgReward={avg_reward:.4f}")

        print("Mutation Operators (Total Selections: {}):".format(
            self.total_mutation_selections))
        mutation_stats = sorted(
            [
                (op.__class__.__name__, self.mutation_counts[i], self.mutation_rewards_sum[i] /
                 self.mutation_counts[i] if self.mutation_counts[i] > 0 else 0)
                for i, op in enumerate(self.mutation_operators)
            ],
            key=lambda x: x[1], reverse=True  # Ordena por contagem decrescente
        )
        for name, count, avg_reward in mutation_stats:
            print(f"  - {name}: Count={count}, AvgReward={avg_reward:.4f}")
        print("------------------------")


# Exemplo de uso (requer a definição das classes Job, etc.)
if __name__ == '__main__':
    # Exemplo simplificado de dados de jobs
    # Formato: [[(machine_id, duration), ...], ...]
    jobs_data = [
        [(0, 3), (1, 2), (2, 2)],  # Job 0
        [(0, 2), (2, 1), (1, 4)],  # Job 1
        [(1, 4), (0, 3)]           # Job 2
    ]
    num_jobs = len(jobs_data)
    num_machines = 3

    print("Criando instância do GeneticSolver...")
    # Exemplo com configuração padrão e busca local VND
    ga_solver = GeneticSolver(jobs_data, num_jobs, num_machines,
                              population_size=50, generations=50, crossover_rate=0.85,
                              mutation_rate=0.15, elite_size=2, use_dsu=True)

    # Para usar uma estratégia diferente, passe como argumento:
    # from solvers.ga.genetic_operators import StandardMutation, PMXCrossover
    # ga_solver = GeneticSolver(jobs_data, num_jobs, num_machines,
    #                           crossover_strategy=PMXCrossover(),
    #                           mutation_strategy=StandardMutation())

    print("Resolvendo o problema...")
    best_schedule = ga_solver.solve(time_limit=10)  # Limite de 10 segundos

    if best_schedule:
        print("--- Melhor Schedule Encontrado ---")
        makespan = best_schedule.get_makespan()
        print(f"Makespan: {makespan}")
        # print("Operações:")
        # for op in best_schedule.operations:
        #     print(f"  Job {op[0]}, Op {op[1]} -> Machine {op[2]}, Start {op[3]}, Duration {op[4]}")
    else:
        print("Não foi possível encontrar uma solução.")
