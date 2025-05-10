import random
from .graph.dsu import DSU
from solvers.ortools_cpsat_solver import ORToolsCPSATSolver


class PopulationInitializer:
    """
    Classe responsável por inicializar a população do algoritmo genético.
    """

    def __init__(self, jobs, num_jobs, num_machines, fitness_func):
        """
        Inicializa o inicializador de população.

        Args:
            jobs: Lista de jobs com suas operações
            num_jobs: Número de jobs
            num_machines: Número de máquinas
            fitness_func: Função para calcular o fitness de um cromossomo
        """
        self.jobs = jobs
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.fitness_func = fitness_func

    def initialize_population(self, population_size, initial_schedule=None):
        """
        Inicializa a população, usando CP-SAT se disponível ou heurísticas.

        Args:
            population_size: Tamanho da população
            initial_schedule: Schedule inicial para ser incluído na população

        Returns:
            Lista de indivíduos (dicionários com chave 'chromosome')
        """
        population = []
        initial_chromosomes = set()  # Para evitar duplicatas exatas na inicialização

        # --- Estratégia 1: Usar CP-SAT para uma solução inicial de alta qualidade ---
        best_initial_chromosome = None
        best_initial_fitness = float('inf')

        # Se houver um schedule inicial fornecido
        if initial_schedule:
            chrom = tuple([(op[0], op[1])
                          for op in initial_schedule.operations])
            # Valida rapidamente o tamanho
            if len(chrom) == self.num_jobs * self.num_machines:
                initial_chromosomes.add(chrom)
                # Calcula fitness para referência
                fit = self.fitness_func(chrom)
                if fit < best_initial_fitness:
                    best_initial_fitness = fit
                    best_initial_chromosome = chrom
            else:
                print(
                    "Aviso: Schedule inicial fornecido parece inválido (tamanho incorreto?). Ignorando.")

        # Se não houver schedule inicial ou for inválido, tenta gerar um com CP-SAT
        if not best_initial_chromosome:
            try:
                # Roda CP-SAT com um limite de tempo curto
                cp_solver = ORToolsCPSATSolver(
                    self.jobs, self.num_jobs, self.num_machines)
                # Limite de tempo baixo para inicialização rápida
                cp_schedule = cp_solver.solve(time_limit=int(5.0))
                if cp_schedule and cp_schedule.operations:
                    chrom = tuple([(op[0], op[1])
                                  for op in cp_schedule.operations])
                    # Valida tamanho novamente
                    if len(chrom) == self.num_jobs * self.num_machines:
                        initial_chromosomes.add(chrom)
                        # Usa a função de fitness do GA
                        fit = self.fitness_func(chrom)
                        if fit < best_initial_fitness:
                            best_initial_fitness = fit
                            best_initial_chromosome = chrom
                    else:
                        print(
                            "Aviso: Schedule do CP-SAT parece inválido (tamanho incorreto?).")

            except Exception as e:
                print(f"Erro ao executar CP-SAT para solução inicial: {e}")
                # Prossegue para heurísticas

        # --- Estratégia 2: Heurísticas Simples (se CP-SAT falhou ou não foi usado) ---
        if not best_initial_chromosome:
            # Cria um cromossomo baseado em SPT (Shortest Processing Time)
            ops_with_duration = []
            for j, job in enumerate(self.jobs):
                for o, (_, duration) in enumerate(job):
                    ops_with_duration.append(((j, o), duration))
            # Ordena por duração (SPT)
            ops_with_duration.sort(key=lambda x: x[1])
            base_chromosome_tuple = tuple([op[0] for op in ops_with_duration])
            if len(base_chromosome_tuple) == self.num_jobs * self.num_machines:
                initial_chromosomes.add(base_chromosome_tuple)
                fit = self.fitness_func(base_chromosome_tuple)
                if fit < best_initial_fitness:
                    best_initial_fitness = fit
                    best_initial_chromosome = base_chromosome_tuple
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

        # --- Estratégia 3: Gerar Diversidade a partir da Melhor Inicial ---
        # Gera o restante da população perturbando a melhor solução inicial
        attempts = 0
        max_attempts = population_size * 5  # Evita loop infinito

        while len(population) < population_size and attempts < max_attempts:
            attempts += 1
            # Cria uma cópia mutável (lista)
            mutated_chrom_list = list(best_initial_chromosome)

            # Aplica N trocas aleatórias (mutação simples para diversificar)
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

        # Se ainda faltarem indivíduos, preenche com totalmente aleatórios
        while len(population) < population_size:
            # Começa com uma base válida
            random_chrom_list = list(best_initial_chromosome)
            random.shuffle(random_chrom_list)
            random_chrom_tuple = tuple(random_chrom_list)
            if random_chrom_tuple not in initial_chromosomes:
                population.append(
                    {'chromosome': random_chrom_tuple, 'dsu': DSU(len(random_chrom_tuple))})
                initial_chromosomes.add(random_chrom_tuple)

        return population
