import random
import numpy as np
from copy import deepcopy
from .NSGA2 import NSGA2
from .Individual import Individual
from .operators import crossover as crossover_ops
from .operators import mutation as mutation_ops
from .operators import local_search as local_search_ops

class NSGA2Perm(NSGA2):
    """
    Implementação especializada do NSGA-II para problemas de permutação.
    """
    def __init__(self, pop_size, n_gen, problem, xl, xu, op_ids, n_jobs=None, n_machines=None, initial_population=None):
        super().__init__(pop_size, n_gen, problem, xl, xu, initial_population=initial_population)
        self.op_ids = op_ids
        self.IndividualClass = Individual  # Define a classe de indivíduo a ser usada
        self.n_jobs = n_jobs
        self.n_machines = n_machines

        # Inicialização dos pesos para AOS (Adaptive Operator Selection)
        self.op_weights = {
            'crossover': {'PPX': 0.4, 'PMX': 0.3, 'OX': 0.3, 'IPPX': 0.0, 'ExtPPX': 0.0},
            'mutation': {'Swap': 0.4, 'Insertion': 0.3, 'Inversion': 0.2, '2Opt': 0.1}
        }

        # Estatísticas de melhoria dos operadores para AOS
        self.op_stats = {
            'crossover': {'PPX': [], 'PMX': [], 'OX': [], 'IPPX': [], 'ExtPPX': []},
            'mutation': {'Swap': [], 'Insertion': [], 'Inversion': [], '2Opt': []}
        }

        # Contador para atualização de pesos AOS
        self.generations_since_update = 0

        # constante de exploração para UCB
        self.aos_alpha = 0.1  # exploration constant for UCB
        # Contador de uso dos operadores para AOS (UCB)
        self.op_counts = {
            'crossover': {op_name: 0 for op_name in self.op_weights['crossover']},
            'mutation': {op_name: 0 for op_name in self.op_weights['mutation']}
        }

        # Parâmetros de busca local
        self.local_search_prob = 0.1  # Probabilidade de aplicar busca local em um indivíduo
        self.max_local_iterations = 30  # Número máximo de iterações da busca local

    def initialize_population(self):
        if self.initial_population is not None:
            self.population = [self.IndividualClass(x) for x in self.initial_population]
            n_missing = self.pop_size - len(self.population)
            if n_missing > 0:
                perm_base = np.array(self.op_ids)
                for _ in range(n_missing):
                    perm = perm_base.copy()
                    np.random.shuffle(perm)
                    self.population.append(self.IndividualClass(perm))
            self.population = self.population[:self.pop_size]
        else:
            self.population = []
            perm_base = np.array(self.op_ids)
            for i in range(self.pop_size):
                perm = perm_base.copy()
                np.random.shuffle(perm)
                # Checagem de permutação válida
                if len(set(perm)) != len(perm):
                    print(f"[ERRO] Permutação inválida na inicialização (ind {i}): {perm}")
                ind = self.IndividualClass(perm)
                self.population.append(ind)

    def sbx(self, parent1, parent2, eta=15):
        """
        Operador de cruzamento para permutações.
        Seleciona um operador de crossover com base nos pesos adaptativos.

        Args:
            parent1, parent2: Indivíduos pais
            eta: Parâmetro de distribuição (não utilizado na maioria dos crossovers de permutação)

        Returns:
            Dois novos indivíduos filhos
        """
        # Seleciona o operador usando AOS
        operators = list(self.op_weights['crossover'].keys())
        weights = list(self.op_weights['crossover'].values())
        op = random.choices(operators, weights=weights, k=1)[0]
        # incrementa contador de uso do operador
        self.op_counts['crossover'][op] += 1

        x1, x2 = np.array(parent1.x), np.array(parent2.x)

        # Aplica o operador selecionado
        if op == 'OX':
            children = crossover_ops.ox_crossover_vectorized(x1, x2, 2)
        elif op == 'PPX':
            children = crossover_ops.ppx_crossover(x1, x2, 2)
        elif op == 'IPPX':
            children = crossover_ops.ippx_crossover(x1, x2, 2, self.n_jobs)
        elif op == 'ExtPPX':
            children = crossover_ops.extended_ppx_crossover(x1, x2, 2)
        elif op == 'PMX':
            children = crossover_ops.pmx_crossover(x1, x2, 2)
        else:
            # Fallback para OX
            children = crossover_ops.ox_crossover_vectorized(x1, x2, 2)

        # Registra o operador usado para atualizar estatísticas depois
        child1, child2 = self.IndividualClass(children[0]), self.IndividualClass(children[1])
        child1.crossover_op = op
        child2.crossover_op = op
        # Checagem de permutação válida após crossover
        if len(set(child1.x)) != len(child1.x):
            print(f"[ERRO] Permutação inválida após crossover (child1): {child1.x}")
        if len(set(child2.x)) != len(child2.x):
            print(f"[ERRO] Permutação inválida após crossover (child2): {child2.x}")
        return child1, child2

    def poly_mutation(
        self,
        ind,
        eta: int = 20,
        p_m: float | None = None,
        xl=None, xu=None,
        n_offspring: int = 1,
        p_m_min: float = 1e-3,):
        """
        Mutação para indivíduos permutação-baseados.

        Parameters
        ----------
        ind : object
            Deve possuir atributo `x` (sequência 1-D) e aceitar reatribuição.
        eta : int
            Mantido por compatibilidade (não usado).
        p_m : float, optional
            Probab. de mutar cada gene. Se None => 1/len(ind).
        xl, xu
            Mantido por compatibilidade (não usado).
        n_offspring : int
            Número de filhos a devolver pelos operadores de mutação.
        p_m_min : float
            Limite mínimo para p_m caso o cromossomo seja longo.
        """

        x = np.asarray(ind.x)            # não copia ainda
        L = len(x)
        if p_m is None:
            p_m = max(1.0 / L, p_m_min)

        # ==== Adaptive-Operator-Selection ====
        ops_dict = self.op_weights['mutation']
        operators, weights = zip(*ops_dict.items())
        if sum(weights) == 0:
            weights = [1] * len(operators)

        op = random.choices(operators, weights=weights, k=1)[0]
        # incrementa contador de uso do operador
        self.op_counts['mutation'][op] += 1

        # ==== Aplica operador escolhido ====
        if op == 'Swap':
            offspring = mutation_ops.swap_mutation_vectorized(x, n_offspring, p_m)
        elif op == 'Insertion':
            offspring = mutation_ops.insertion_mutation(x, n_offspring, p_m)
        elif op == 'Inversion':
            offspring = mutation_ops.inversion_mutation(x, n_offspring, p_m)
        elif op == '2Opt':
            offspring = mutation_ops.two_opt_mutation(x, n_offspring, p_m, self.problem)
        else:  # fallback
            offspring = mutation_ops.swap_mutation_vectorized(x, n_offspring, p_m)

        # Assume que cada operador retorna ndarray shape (n_offspring, L)
        ind.x = offspring[0].tolist()     # ou .copy() se quiser ndarray
        ind.mutation_op = op             # para logging/AOS

        return offspring


    def update_aos_weights(self):
        """
        Atualiza os pesos dos operadores de crossover e mutação
        com base em seu desempenho recente (Adaptive Operator Selection) usando UCB.
        """
        # Atualiza apenas a cada N gerações
        if self.generations_since_update < 10:  # A cada 10 gerações
            self.generations_since_update += 1
            return

        # reinicia contador de gerações para próximo ciclo
        self.generations_since_update = 0

        # Para cada tipo de operador, atualiza pesos usando UCB
        for op_type in ['crossover', 'mutation']:
            # Calcula a média de melhoria para cada operador
            improvements = {}
            for op_name, stats in self.op_stats[op_type].items():
                improvements[op_name] = np.mean(stats) if stats else 0.0
                # limpa estatísticas
                self.op_stats[op_type][op_name] = []

            # total de aplicações de todos operadores
            T = sum(self.op_counts[op_type].values())

            # calcula valor UCB para cada operador
            ucb_values = {}
            for op_name, rbar in improvements.items():
                n = self.op_counts[op_type].get(op_name, 0)
                # bônus de exploração: ln(T)/n ou ln(T+1) se nunca usado
                bonus = self.aos_alpha * (np.log(T) / n if n > 0 else np.log(T + 1))
                ucb_values[op_name] = max(0.0, rbar) + bonus

            # normaliza para somar 1.0
            total_ucb = sum(ucb_values.values())
            if total_ucb > 0:
                for op_name, ucb in ucb_values.items():
                    self.op_weights[op_type][op_name] = ucb / total_ucb

    def evaluate_offspring(self, offspring):
        """
        Avalia os filhos e registra a melhoria dos operadores para AOS.

        Args:
            offspring: Lista de indivíduos filhos a serem avaliados

        Returns:
            Os filhos avaliados
        """
        # Avalia os filhos
        X = np.array([ind.x for ind in offspring])
        F = self.problem(X)

        # Para cada filho, registra as estatísticas
        for i, (ind, f) in enumerate(zip(offspring, F)):
            ind.f = f

            # Registra melhoria para operadores de crossover e mutação
            if hasattr(ind, 'crossover_op'):
                # Calcula melhoria em relação aos pais (se disponível)
                if hasattr(ind, 'parent1_f') and hasattr(ind, 'parent2_f'):
                    # Melhoria percentual em relação ao melhor pai (primeiro objetivo - makespan)
                    parent_best = min(ind.parent1_f[0], ind.parent2_f[0])
                    improvement = (parent_best - f[0]) / parent_best if parent_best > 0 else 0.0
                    self.op_stats['crossover'][ind.crossover_op].append(improvement)

            if hasattr(ind, 'mutation_op'):
                # Registra o operador usado, estatística será calculada após a avaliação
                if hasattr(ind, 'pre_mutation_f'):
                    denom = ind.pre_mutation_f[0]
                    if denom is not None and np.isfinite(denom) and denom > 0:
                        improvement = (denom - f[0]) / denom
                    else:
                        improvement = 0.0
                    self.op_stats['mutation'][ind.mutation_op].append(improvement)

        return offspring

    def make_offspring(self, pop, N, crossover_prob=0.9, eta_c=15, eta_m=20):
        """
        Cria nova população de filhos usando seleção, crossover e mutação.
        Inclui busca local memética em uma fração dos filhos.

        Args:
            pop: População atual
            N: Tamanho da população de filhos
            crossover_prob: Probabilidade de aplicar crossover
            eta_c: Parâmetro de distribuição para crossover (não usado para permutações)
            eta_m: Parâmetro de distribuição para mutação (não usado para permutações)

        Returns:
            Lista de N filhos
        """
        n_needed = N
        kids = []

        # Diversidade adaptativa: aumenta taxa de mutação se diversidade é baixa
        diversity = self.calculate_diversity(pop)
        adaptive_mut_rate = max(0.05, min(0.5, 0.5 * (1.0 / max(0.001, diversity))))

        while len(kids) < n_needed:
            # Selecione pais usando seleção por torneio
            parents_idx = np.random.choice(len(pop), size=(2, 2))
            p1 = self.tournament(pop[parents_idx[0, 0]], pop[parents_idx[0, 1]])
            p2 = self.tournament(pop[parents_idx[1, 0]], pop[parents_idx[1, 1]])

            # Aplique crossover com probabilidade crossover_prob
            if np.random.random() < crossover_prob:
                c1, c2 = self.sbx(p1, p2, eta=eta_c)
                # Guarde referência aos pais para cálculo de melhoria
                c1.parent1_f, c1.parent2_f = p1.f.copy(), p2.f.copy()
                c2.parent1_f, c2.parent2_f = p1.f.copy(), p2.f.copy()
            else:
                c1, c2 = deepcopy(p1), deepcopy(p2)

            # Guarde fitness antes da mutação para comparação
            c1.pre_mutation_f, c2.pre_mutation_f = c1.f.copy() if hasattr(c1, 'f') else None, c2.f.copy() if hasattr(c2, 'f') else None

            # Aplique mutação com taxa adaptativa
            self.poly_mutation(c1, eta=eta_m, p_m=adaptive_mut_rate)
            self.poly_mutation(c2, eta=eta_m, p_m=adaptive_mut_rate)

            # Avalie os filhos
            kids.extend([c1, c2])

        # Aplique busca local em uma fração dos filhos (abordagem memética)
        kids = self.evaluate_offspring(kids[:N])  # Limite de N filhos e avalie

        # Aplica busca local N7 em alguns indivíduos (10% da população, selecionados entre os melhores)
        n_local_search = max(1, int(N * self.local_search_prob))

        # Ordena por rank e seleciona os melhores
        sorted_kids = sorted(kids, key=lambda ind: (ind.f[0], -ind.crowding) if hasattr(ind, 'crowding') else ind.f[0])

        # Aplica busca local nos melhores indivíduos
        for i in range(n_local_search):
            if i < len(sorted_kids):
                sorted_kids[i] = local_search_ops.local_search_n7(
                    sorted_kids[i], self.problem, self.n_machines, self.n_jobs, self.max_local_iterations)

        # Atualiza pesos dos operadores
        self.update_aos_weights()

        return kids[:N]

    def calculate_diversity(self, population):
        """
        Calcula a diversidade da população usando distância média entre soluções.

        Args:
            population: Lista de indivíduos

        Returns:
            Valor de diversidade normalizado (0-1)
        """
        if len(population) <= 1:
            return 0.0

        # Amostra para performance se população for grande
        sample_size = min(len(population), 30)
        if len(population) > sample_size:
            sample = random.sample(population, sample_size)
        else:
            sample = population

        # Calcula distância média par a par (distância de Hamming para permutações)
        total_distance = 0.0
        count = 0

        for i in range(len(sample)):
            for j in range(i+1, len(sample)):
                # Distância de Hamming: número de posições diferentes
                x1, x2 = np.array(sample[i].x), np.array(sample[j].x)
                distance = np.sum(x1 != x2) / len(x1)  # Normalizado por tamanho
                total_distance += distance
                count += 1

        # Diversidade média
        if count > 0:
            return total_distance / count  # Valor entre 0 e 1
        else:
            return 0.0

    def run(self, N=None, N_GEN=None, M=None, verbose=True):
        """
        Executa o NSGA-II para problemas de permutação.
        Sobrescreve o método run da classe base para incluir AOS e busca local.

        Args:
            N: Tamanho da população
            N_GEN: Número de gerações
            M: Número de objetivos
            verbose: Se True, exibe informações durante a execução

        Returns:
            População final após N_GEN gerações
        """
        if N is None:
            N = self.pop_size
        if N_GEN is None:
            N_GEN = self.n_gen
        if M is None:
            # Detecta número de objetivos a partir do primeiro indivíduo
            self.initialize_population()
            self.evaluate_population()
            M = len(self.population[0].f)
        else:
            self.initialize_population()
            self.evaluate_population()

        P = self.population
        # atribui rank e crowding iniciais
        fronts = self.fast_non_dominated_sort(P)
        for front in fronts:
            self.crowding_distance(front, M)

        # Estatísticas para acompanhamento
        best_makespan = float('inf')

        for g in range(N_GEN):
            Q = self.make_offspring(P, N)

            # Já avaliados em make_offspring
            R = P + Q
            fronts = self.fast_non_dominated_sort(R)
            next_P = []

            # Adicione frentes completas enquanto há espaço
            i = 0
            while i < len(fronts) and len(next_P) + len(fronts[i]) <= N:
                # Calcule a crowding distance para cada frente adicionada
                self.crowding_distance(fronts[i], M)
                next_P.extend(fronts[i])
                i += 1

            # Se ainda precisamos de mais indivíduos e há mais frentes disponíveis
            if i < len(fronts) and len(next_P) < N:
                # Calcule crowding distance para a frente parcialmente adicionada
                self.crowding_distance(fronts[i], M)

                # Ordenação deve ser por rank (crescente) e crowding (decrescente)
                last_front = fronts[i]
                crowding_values = np.array([ind.crowding for ind in last_front])
                # Como queremos ordenação decrescente por crowding, usamos o negativo
                sorted_indices = np.argsort(-crowding_values)

                # Adicione os melhores indivíduos da última frente
                needed = N - len(next_P)
                for j in sorted_indices[:needed]:
                    next_P.append(last_front[j])

            P = next_P

            # Atualiza estatísticas
            current_best = min(P, key=lambda i: i.f[0])
            current_makespan = current_best.f[0]
            best_makespan = min(best_makespan, current_makespan)

            if verbose and (g % 50 == 0 or g == N_GEN-1):
                print(f"G {g}/{N_GEN}  |  best makespan: {best_makespan:.2f}  |  current: {current_makespan:.2f}  |  diversity: {self.calculate_diversity(P):.4f}")
                # Se for a última iteração ou cada 100 gerações, mostra os pesos dos operadores
                if g % 100 == 0 or g == N_GEN-1:
                    print(f"  Crossover weights: {self.op_weights['crossover']}")
                    print(f"  Mutation weights: {self.op_weights['mutation']}")

        self.population = P
        return P
