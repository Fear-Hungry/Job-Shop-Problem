import random
import numpy as np
from copy import deepcopy
from .NSGA2 import NSGA2
from .Individual import Individual

class NSGA2Perm(NSGA2):
    """
    Implementação especializada do NSGA-II para problemas de permutação.
    """
    def __init__(self, pop_size, n_gen, problem, xl, xu, op_ids, n_jobs=None, n_machines=None):
        """
        Inicializa o NSGA-II para problemas de permutação.

        Args:
            pop_size: Tamanho da população
            n_gen: Número de gerações
            problem: Função de avaliação que aceita uma matriz de permutações
            xl: Limite inferior (não usado na versão de permutação)
            xu: Limite superior (não usado na versão de permutação)
            op_ids: Lista de IDs de operações que serão permutados
            n_jobs: Número de jobs (para decodificar representações job-based)
            n_machines: Número de máquinas (para busca local)
        """
        super().__init__(pop_size, n_gen, problem, xl, xu)
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

        # Parâmetros de busca local
        self.local_search_prob = 0.1  # Probabilidade de aplicar busca local em um indivíduo
        self.max_local_iterations = 30  # Número máximo de iterações da busca local

    def initialize_population(self):
        """
        Inicializa a população com permutações aleatórias de forma vetorizada.
        """
        self.population = []
        # Cria múltiplas permutações de uma vez
        perm_base = np.array(self.op_ids)

        # Para cada indivíduo na população
        for _ in range(self.pop_size):
            # Cria uma cópia da permutação base e randomiza
            perm = perm_base.copy()
            np.random.shuffle(perm)
            ind = self.IndividualClass(perm)
            self.population.append(ind)

    @staticmethod
    def _ox_crossover_vectorized(p1, p2, n_offspring=2):
        """
        Implementação correta do Order Crossover (OX) para permutações.
        Preenche circularmente as posições restantes.

        Args:
            p1, p2: Arrays NumPy representando as permutações dos pais
            n_offspring: Número de filhos a gerar (tipicamente 2)

        Returns:
            Array de permutações dos filhos
        """
        size = len(p1)
        offspring = np.zeros((n_offspring, size), dtype=p1.dtype)
        a = np.random.randint(0, size)
        b = np.random.randint(0, size)
        if a > b:
            a, b = b, a
        # Filho 1
        offspring[0] = -1
        offspring[0, a:b] = p1[a:b]
        fill = [item for item in p2 if item not in p1[a:b]]
        pos = b
        for item in fill:
            if pos >= size:
                pos = 0
            if offspring[0, pos] == -1:
                offspring[0, pos] = item
                pos += 1
        # Filho 2
        offspring[1] = -1
        offspring[1, a:b] = p2[a:b]
        fill = [item for item in p1 if item not in p2[a:b]]
        pos = b
        for item in fill:
            if pos >= size:
                pos = 0
            if offspring[1, pos] == -1:
                offspring[1, pos] = item
                pos += 1
        return offspring

    @staticmethod
    def _ppx_crossover(p1, p2, n_offspring=2):
        """
        Precedence Preserving Crossover (PPX) para permutações.
        Preserva a precedência copiando genes na ordem em que são "extraídos".

        Args:
            p1, p2: Arrays NumPy representando as permutações dos pais
            n_offspring: Número de filhos a gerar

        Returns:
            Array de permutações dos filhos
        """
        size = len(p1)
        offspring = np.zeros((n_offspring, size), dtype=p1.dtype)

        # Cria uma máscara aleatória para selecionar de qual pai copiar
        mask = np.random.randint(0, 2, size=size)

        # Para os dois filhos
        for k in range(n_offspring):
            # Inverte a máscara para o segundo filho
            if k == 1:
                mask = 1 - mask

            # Posição atual nos filhos
            pos = 0
            # Elementos já copiados
            used = set()

            # Cópias dos pais para manipular
            p1_copy = p1.copy()
            p2_copy = p2.copy()

            # Para cada posição do filho
            for i in range(size):
                # Seleciona o pai conforme a máscara
                parent = p1_copy if mask[i] == 0 else p2_copy

                # Encontra o primeiro elemento do pai que ainda não foi usado
                for j in range(size):
                    if parent[j] not in used:
                        offspring[k, pos] = parent[j]
                        used.add(parent[j])
                        pos += 1
                        break

        return offspring

    @staticmethod
    def _ippx_crossover(p1, p2, n_offspring=2, n_jobs=None):
        """
        Improved Precedence Preserving Crossover (IPPX) para permutações.
        Estende o PPX com um passo adicional que reordena blocos redundantes.

        Args:
            p1, p2: Arrays NumPy representando as permutações dos pais
            n_offspring: Número de filhos a gerar
            n_jobs: Número de jobs na instância (para identificar operações do mesmo job)

        Returns:
            Array de permutações dos filhos
        """
        # Primeiro aplicamos PPX padrão
        offspring = NSGA2Perm._ppx_crossover(p1, p2, n_offspring)

        # Se não temos informação sobre número de jobs, retornamos o PPX normal
        if n_jobs is None:
            return offspring

        # Para cada filho, aplicamos o segundo estágio do IPPX
        for k in range(n_offspring):
            child = offspring[k]
            # Identificamos os blocos de job (operações do mesmo job)
            job_blocks = {}

            # Assumindo que op_id começa com identificador de job
            # Por exemplo: op_id = job_id * 100 + op_seq
            for i, op in enumerate(child):
                job_id = op // 100  # Adequar com a sua codificação de operações
                if job_id not in job_blocks:
                    job_blocks[job_id] = []
                job_blocks[job_id].append((i, op))

            # Reordena internamente cada bloco para minimizar makespan
            # (Simplificação: aqui apenas embaralha para aumentar diversidade)
            for job_id, block in job_blocks.items():
                if len(block) > 1:
                    indices = [b[0] for b in block]
                    random.shuffle(indices)
                    for idx, (_, op) in zip(indices, block):
                        offspring[k, idx] = op

        return offspring

    @staticmethod
    def _extended_ppx_crossover(p1, p2, n_offspring=2):
        """
        Extended Precedence Preserving Crossover para permutações.
        Estende o PPX com pontos de corte adaptativos para mais diversidade.

        Args:
            p1, p2: Arrays NumPy representando as permutações dos pais
            n_offspring: Número de filhos a gerar

        Returns:
            Array de permutações dos filhos
        """
        size = len(p1)
        offspring = np.zeros((n_offspring, size), dtype=p1.dtype)

        # Seleciona número de pontos de corte (entre 2-5 normalmente)
        n_cuts = random.randint(2, min(5, size//10 + 2))

        # Gera os pontos de corte
        cut_points = sorted(random.sample(range(1, size), n_cuts))
        cut_points = [0] + cut_points + [size]

        # Para cada filho
        for k in range(n_offspring):
            # Seleciona ordem dos pais (inverte para o segundo filho)
            p_order = [p1, p2] if k == 0 else [p2, p1]
            pos = 0
            used = set()

            # Para cada segmento entre pontos de corte
            for i in range(len(cut_points) - 1):
                # Alterna entre pais
                parent = p_order[i % 2]
                segment_size = cut_points[i+1] - cut_points[i]

                # Encontra elementos não utilizados do pai atual
                segment = []
                for item in parent:
                    if item not in used and len(segment) < segment_size:
                        segment.append(item)
                        used.add(item)

                # Se não há elementos suficientes do pai atual, complete com o outro pai
                other_parent = p_order[(i+1) % 2]
                for item in other_parent:
                    if item not in used and len(segment) < segment_size:
                        segment.append(item)
                        used.add(item)

                # Copia o segmento para o filho
                offspring[k, pos:pos+segment_size] = segment
                pos += segment_size

        return offspring

    @staticmethod
    def _pmx_crossover(p1, p2, n_offspring=2):
        """
        Partially Mapped Crossover (PMX) para permutações.

        Args:
            p1, p2: Arrays NumPy representando as permutações dos pais
            n_offspring: Número de filhos a gerar

        Returns:
            Array de permutações dos filhos
        """
        size = len(p1)
        offspring = np.zeros((n_offspring, size), dtype=p1.dtype)

        # Pontos de corte
        a = np.random.randint(0, size)
        b = np.random.randint(0, size)
        if a > b:
            a, b = b, a

        # Para os dois filhos
        for k in range(n_offspring):
            # Seleciona os pais (inverte para o segundo filho)
            parent1 = p1 if k == 0 else p2
            parent2 = p2 if k == 0 else p1

            # Inicializa o filho como cópia do segundo pai
            child = parent2.copy()

            # Copia o segmento do primeiro pai
            child[a:b] = parent1[a:b]

            # Constrói o mapeamento
            mapping = {}
            for i in range(a, b):
                if parent1[i] != parent2[i]:
                    mapping[parent2[i]] = parent1[i]

            # Aplica o mapeamento para resolver conflitos
            for i in range(size):
                if i < a or i >= b:
                    item = child[i]
                    while item in mapping:
                        item = mapping[item]
                    child[i] = item

            offspring[k] = child

        return offspring

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

        x1, x2 = np.array(parent1.x), np.array(parent2.x)

        # Aplica o operador selecionado
        if op == 'OX':
            children = self._ox_crossover_vectorized(x1, x2, 2)
        elif op == 'PPX':
            children = self._ppx_crossover(x1, x2, 2)
        elif op == 'IPPX':
            children = self._ippx_crossover(x1, x2, 2, self.n_jobs)
        elif op == 'ExtPPX':
            children = self._extended_ppx_crossover(x1, x2, 2)
        elif op == 'PMX':
            children = self._pmx_crossover(x1, x2, 2)
        else:
            # Fallback para OX
            children = self._ox_crossover_vectorized(x1, x2, 2)

        # Registra o operador usado para atualizar estatísticas depois
        child1, child2 = self.IndividualClass(children[0]), self.IndividualClass(children[1])
        child1.crossover_op = op
        child2.crossover_op = op

        return child1, child2

    @staticmethod
    def _swap_mutation_vectorized(x, n_offspring=1, p_m=None):
        """
        Implementação vetorizada da mutação Swap para permutações.

        Args:
            x: Array NumPy representando a permutação
            n_offspring: Número de cópias a mutar
            p_m: Probabilidade de mutação

        Returns:
            Array de permutações mutadas
        """
        size = len(x)
        if size < 2:
            return np.array([x] * n_offspring)

        if p_m is None:
            p_m = 1.0 / size

        # Crie cópias do indivíduo original
        offspring = np.array([x.copy() for _ in range(n_offspring)])

        # Para cada cópia
        for i in range(n_offspring):
            # Decida se aplica mutação
            if np.random.random() < p_m:
                # Selecione dois índices aleatórios
                idx1, idx2 = np.random.choice(size, 2, replace=False)
                # Troque os valores
                offspring[i][idx1], offspring[i][idx2] = offspring[i][idx2], offspring[i][idx1]

        return offspring

    @staticmethod
    def _insertion_mutation(x, n_offspring=1, p_m=None):
        """
        Mutação por Inserção para permutações.
        Remove um elemento e o insere em outra posição.

        Args:
            x: Array NumPy representando a permutação
            n_offspring: Número de cópias a mutar
            p_m: Probabilidade de mutação

        Returns:
            Array de permutações mutadas
        """
        size = len(x)
        if size < 2:
            return np.array([x] * n_offspring)

        if p_m is None:
            p_m = 1.0 / size

        # Crie cópias do indivíduo original
        offspring = np.array([x.copy() for _ in range(n_offspring)])

        # Para cada cópia
        for i in range(n_offspring):
            # Decida se aplica mutação
            if np.random.random() < p_m:
                # Selecione dois índices aleatórios
                idx1, idx2 = np.random.choice(size, 2, replace=False)

                # Garante que idx1 < idx2 para simplificar
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1

                # Remove o elemento da posição idx1 e o insere antes da posição idx2
                value = offspring[i][idx1]
                # Desloca os elementos entre idx1 e idx2-1
                offspring[i][idx1:idx2] = offspring[i][idx1+1:idx2+1]
                # Insere o valor na posição idx2-1
                offspring[i][idx2-1] = value

        return offspring

    @staticmethod
    def _inversion_mutation(x, n_offspring=1, p_m=None):
        """
        Mutação por Inversão para permutações.
        Inverte uma subsequência da permutação.

        Args:
            x: Array NumPy representando a permutação
            n_offspring: Número de cópias a mutar
            p_m: Probabilidade de mutação

        Returns:
            Array de permutações mutadas
        """
        size = len(x)
        if size < 2:
            return np.array([x] * n_offspring)

        if p_m is None:
            p_m = 1.0 / size

        # Crie cópias do indivíduo original
        offspring = np.array([x.copy() for _ in range(n_offspring)])

        # Para cada cópia
        for i in range(n_offspring):
            # Decida se aplica mutação
            if np.random.random() < p_m:
                # Selecione dois índices aleatórios
                idx1, idx2 = np.random.choice(size, 2, replace=False)

                # Garante que idx1 < idx2
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1

                # Inverte a subsequência
                offspring[i][idx1:idx2+1] = np.flip(offspring[i][idx1:idx2+1])

        return offspring

    @staticmethod
    def _two_opt_mutation(x, n_offspring=1, p_m=None, problem_eval=None):
        """
        Mutação 2-Opt para permutações, voltada para redução de makespan.
        Remove dois arcos e reconecta de forma a potencialmente reduzir makespan.

        Args:
            x: Array NumPy representando a permutação
            n_offspring: Número de cópias a mutar
            p_m: Probabilidade de mutação
            problem_eval: Função para avaliar a qualidade da solução (opcional)

        Returns:
            Array de permutações mutadas
        """
        size = len(x)
        if size < 4:  # Precisa de ao menos 4 elementos para ter 2 arcos
            return np.array([x] * n_offspring)

        if p_m is None:
            p_m = 1.0 / size

        # Crie cópias do indivíduo original
        offspring = np.array([x.copy() for _ in range(n_offspring)])

        # Para cada cópia
        for i in range(n_offspring):
            # Decida se aplica mutação
            if np.random.random() < p_m:
                # Versão simples do 2-opt: seleciona duas posições e inverte a subsequência
                idx1, idx2 = np.random.choice(size, 2, replace=False)

                # Garante que idx1 < idx2
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1

                # Cria uma cópia antes da mutação para comparar depois
                original = offspring[i].copy()

                # Inverte a subsequência (equivalente a reconectar os arcos)
                offspring[i][idx1:idx2+1] = np.flip(offspring[i][idx1:idx2+1])

                # Se temos uma função de avaliação, verificamos se houve melhoria
                if problem_eval is not None:
                    # Avalia a nova solução
                    new_fitness = problem_eval(np.array([offspring[i]]))[0]
                    # Avalia a solução original
                    orig_fitness = problem_eval(np.array([original]))[0]

                    # Se não houve melhoria, reverte a mutação
                    if new_fitness[0] > orig_fitness[0]:  # Assumindo que o primeiro objetivo é makespan
                        offspring[i] = original

        return offspring

    def poly_mutation(self, ind, eta=20, p_m=None, xl=None, xu=None):
        """
        Operador de mutação para permutações.
        Seleciona um operador de mutação com base nos pesos adaptativos.

        Args:
            ind: Indivíduo a ser mutado
            eta: Parâmetro de distribuição (não utilizado em mutações de permutação)
            p_m: Probabilidade de mutação
            xl, xu: Limites inferior e superior (não utilizados em mutações de permutação)
        """
        # Ajusta probabilidade de mutação com base na diversidade da população
        if p_m is None:
            p_m = 1.0 / len(ind.x)

        # Seleciona o operador usando AOS
        operators = list(self.op_weights['mutation'].keys())
        weights = list(self.op_weights['mutation'].values())
        op = random.choices(operators, weights=weights, k=1)[0]

        x = np.array(ind.x)

        # Aplica o operador selecionado
        if op == 'Swap':
            mutated = self._swap_mutation_vectorized(x, 1, p_m)
        elif op == 'Insertion':
            mutated = self._insertion_mutation(x, 1, p_m)
        elif op == 'Inversion':
            mutated = self._inversion_mutation(x, 1, p_m)
        elif op == '2Opt':
            mutated = self._two_opt_mutation(x, 1, p_m, self.problem)
        else:
            # Fallback para Swap
            mutated = self._swap_mutation_vectorized(x, 1, p_m)

        # Registra o operador usado para atualizar estatísticas depois
        ind.x = mutated[0]
        ind.mutation_op = op

    def local_search_n7(self, ind, max_iter=None):
        """
        Busca local N7 (Nowicki-Smutnicki) para problemas JSSP.
        Aplica movimentos sobre blocos críticos no caminho crítico.

        Args:
            ind: Indivíduo a ser melhorado
            max_iter: Número máximo de iterações (default: self.max_local_iterations)

        Returns:
            Indivíduo melhorado
        """
        if max_iter is None:
            max_iter = self.max_local_iterations

        # Se não temos informação sobre máquinas, não é possível aplicar N7
        if self.n_machines is None or self.n_jobs is None:
            return ind

        # Cria cópia do indivíduo para não alterar o original durante a busca
        current = deepcopy(ind)
        current_x = np.array(current.x)

        # Avaliação inicial
        current_f = self.problem(np.array([current_x]))[0]
        best_x = current_x.copy()
        best_f = current_f.copy()

        # Iterações da busca local
        for _ in range(max_iter):
            improved = False

            # Identificar o caminho crítico (simulação simplificada)
            # Aqui, assumimos que operações consecutivas na mesma máquina são candidatas a críticas
            # Para uma implementação real de N7, é necessário identificar o caminho crítico completo

            # Simulamos identificando pares de operações consecutivas em máquinas
            machine_ops = {}
            for i, op in enumerate(current_x):
                machine = op % 100  # Adequar com a sua codificação de operações
                if machine not in machine_ops:
                    machine_ops[machine] = []
                machine_ops[machine].append((i, op))

            # Para cada par de operações consecutivas em uma máquina, tente trocar
            for machine, ops in machine_ops.items():
                if len(ops) < 2:
                    continue

                # Para cada par consecutivo de operações na máquina
                for j in range(len(ops) - 1):
                    idx1, op1 = ops[j]
                    idx2, op2 = ops[j+1]

                    # Tenta o movimento N7: troca operações consecutivas no caminho crítico
                    new_x = current_x.copy()
                    new_x[idx1], new_x[idx2] = new_x[idx2], new_x[idx1]

                    # Avalia o movimento
                    new_f = self.problem(np.array([new_x]))[0]

                    # Se o movimento melhora o makespan (primeiro objetivo), aceita
                    if new_f[0] < best_f[0]:
                        best_x = new_x.copy()
                        best_f = new_f.copy()
                        improved = True
                        break

                if improved:
                    break

            # Se não houve melhoria nessa iteração, encerra a busca
            if not improved:
                break

            # Atualiza solução atual
            current_x = best_x.copy()
            current_f = best_f.copy()

        # Atualiza o indivíduo com a melhor solução encontrada
        ind.x = best_x
        ind.f = best_f

        return ind

    def update_aos_weights(self):
        """
        Atualiza os pesos dos operadores de crossover e mutação
        com base em seu desempenho recente (Adaptive Operator Selection).
        """
        # Atualiza apenas a cada N gerações
        if self.generations_since_update < 10:  # A cada 10 gerações
            self.generations_since_update += 1
            return

        self.generations_since_update = 0

        # Processando estatísticas de crossover
        for op_type in ['crossover', 'mutation']:
            # Calcula a média de melhoria para cada operador
            improvements = {}
            for op_name, stats in self.op_stats[op_type].items():
                if len(stats) > 0:
                    # Média da melhoria percentual
                    improvements[op_name] = np.mean(stats)
                else:
                    # Se não há dados, mantém peso atual
                    improvements[op_name] = 0.0

                # Limpa estatísticas para o próximo ciclo
                self.op_stats[op_type][op_name] = []

            # Se não há dados suficientes, mantém os pesos atuais
            if all(imp == 0.0 for imp in improvements.values()):
                continue

            # Normaliza as melhorias para somar 1.0
            total_improvement = sum(max(0.0, imp) for imp in improvements.values())
            if total_improvement > 0:
                # Atualiza os pesos - UCB (Upper Confidence Bound) simplificado
                # Mantém um mínimo de 0.05 para exploração
                for op_name in improvements:
                    # 80% baseado no desempenho, 20% para exploração
                    if total_improvement > 0:
                        new_weight = 0.8 * max(0.0, improvements[op_name]) / total_improvement + 0.05
                    else:
                        new_weight = 0.05

                    # Atualiza gradualmente (suavização)
                    self.op_weights[op_type][op_name] = 0.7 * self.op_weights[op_type][op_name] + 0.3 * new_weight

                # Normaliza para somar 1.0
                weight_sum = sum(self.op_weights[op_type].values())
                for op_name in self.op_weights[op_type]:
                    self.op_weights[op_type][op_name] /= weight_sum

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
                sorted_kids[i] = self.local_search_n7(sorted_kids[i])

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
