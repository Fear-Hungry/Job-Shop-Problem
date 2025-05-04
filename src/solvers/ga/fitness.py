import functools
import time
from models.schedule import Schedule


class FitnessEvaluator:
    """
    Classe responsável por calcular o fitness (makespan) dos cromossomos.
    """

    def __init__(self, jobs, num_jobs, num_machines, build_disjunctive_graph_func,
                 machine_ops_from_chromosome_func, use_dsu=True):
        """
        Inicializa o avaliador de fitness.

        Args:
            jobs: Lista de jobs com suas operações
            num_jobs: Número de jobs
            num_machines: Número de máquinas
            build_disjunctive_graph_func: Função para construir o grafo disjuntivo
            machine_ops_from_chromosome_func: Função para obter as operações de cada máquina
            use_dsu: Se deve usar DSU para verificação de ciclos
        """
        self.jobs = jobs
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self._build_disjunctive_graph = build_disjunctive_graph_func
        self._machine_ops_from_chromosome = machine_ops_from_chromosome_func
        self.use_dsu = use_dsu

        # Cache para armazenar resultados de fitness calculados
        self.fitness_cache = {}

    def get_fitness_cached(self, chromosome_tuple):
        """
        Obtém o fitness (makespan) de um cromossomo, usando cache quando possível.

        Args:
            chromosome_tuple: Cromossomo no formato de tupla

        Returns:
            Fitness (makespan) do cromossomo
        """
        # Se o cromossomo já estiver no cache, retorna o fitness armazenado
        if chromosome_tuple in self.fitness_cache:
            return self.fitness_cache[chromosome_tuple]

        # Senão, calcula o fitness
        fitness = self.fitness_chromosome(chromosome_tuple)

        # Armazena no cache e retorna
        self.fitness_cache[chromosome_tuple] = fitness
        return fitness

    def fitness_chromosome(self, chromosome):
        """
        Calcula o fitness de um cromossomo.

        Args:
            chromosome: Cromossomo a ser avaliado

        Returns:
            Fitness (makespan) do cromossomo
        """
        # Primeiro, decodifica o cromossomo para um objeto Schedule
        schedule = self._decode_chromosome(chromosome)

        # Retorna a duração total (makespan) do schedule
        return schedule.get_makespan() if schedule.operations else float('inf')

    def _decode_chromosome(self, chromosome_tuple):
        """
        Decodifica um cromossomo (tupla) para um objeto Schedule, calculando tempos de início.

        Args:
            chromosome_tuple: Cromossomo no formato de tupla

        Returns:
            Objeto Schedule correspondente ao cromossomo
        """
        chromosome = list(chromosome_tuple)
        start_decode_time = time.time()

        # Mapeamento op -> idx
        op_to_idx = {op: idx for idx, op in enumerate(chromosome)}

        # 1. Construir o grafo disjuntivo
        graph = self._build_disjunctive_graph(
            chromosome_tuple, op_to_idx, use_dsu=False)

        # 2. Verificar se há ciclos
        if graph.has_cycle():
            # Retorna um Schedule vazio, cujo fitness será inf
            return Schedule([])

        # 3. Calcular os caminhos mais longos (tempos de início mais cedo)
        num_total_ops = len(chromosome)
        # Criamos um mapeamento inverso: índice -> operação
        idx_to_op = {idx: op for op, idx in op_to_idx.items()}

        node_weights = {}
        for i in range(num_total_ops):
            op = idx_to_op[i]
            job_id, op_id = op
            _, duration = self.jobs[job_id][op_id]
            node_weights[i] = duration

        # Implementar lógica de cálculo de caminho mais longo
        # Esta é uma função que deve ser implementada no grafo disjuntivo
        # Por ora, usando placeholders
        start_times = {idx: 0 for idx in range(num_total_ops)}

        # 4. Construir a lista de operações para o Schedule
        operations = []
        if start_times is not None:
            for node_index, start_time in start_times.items():
                job_id, op_id = idx_to_op[node_index]
                machine_id, duration = self.jobs[job_id][op_id]
                operations.append(
                    (job_id, op_id, machine_id, start_time, duration)
                )
            # Ordena as operações por tempo de início para o objeto Schedule
            operations.sort(key=lambda x: x[3])

        return Schedule(operations)

    def calculate_population_fitness(self, population):
        """
        Calcula o fitness para uma população.

        Args:
            population: Lista de indivíduos (dicionários com chave 'chromosome')

        Returns:
            Lista de valores de fitness para cada indivíduo na população
        """
        return [self.get_fitness_cached(indiv['chromosome']) for indiv in population]
