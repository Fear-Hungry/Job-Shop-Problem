import random
import numpy as np
from copy import deepcopy
from .Individual import Individual

class NSGA2:
    def __init__(self, pop_size, n_gen, problem, xl, xu, initial_population=None):
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.problem = problem  # função de avaliação ou classe do problema
        self.xl = xl            # limites inferiores
        self.xu = xu            # limites superiores
        self.population = []    # lista de indivíduos
        self.history = []       # histórico de populações
        self.initial_population = initial_population

    def initialize_population(self):
        if self.initial_population is not None:
            # Usa a população inicial fornecida
            self.population = [Individual(x) for x in self.initial_population]
            # Completa com indivíduos aleatórios se necessário
            n_missing = self.pop_size - len(self.population)
            if n_missing > 0:
                X = np.random.uniform(self.xl, self.xu, size=(n_missing, len(self.xl) if hasattr(self.xl, '__len__') else 1))
                self.population += [Individual(x) for x in X]
            self.population = self.population[:self.pop_size]
        else:
            # Vetorização da inicialização da população
            X = np.random.uniform(self.xl, self.xu, size=(self.pop_size, len(self.xl) if hasattr(self.xl, '__len__') else 1))
            # Criação vetorizada dos indivíduos
            self.population = [Individual(x) for x in X]

    def evaluate_population(self):
        # Vetorizado: avalia todos os indivíduos de uma vez
        X = np.stack([ind.x for ind in self.population])
        F = self.problem(X)  # problem deve aceitar matriz (N, n_var) e retornar (N, n_obj)
        F = np.asarray(F)
        # Atribuição vetorizada dos objetivos
        for i, ind in enumerate(self.population):
            ind.f = F[i]

    def fast_non_dominated_sort(self, population=None):
        """
        Implementa o Fast Non-Dominated Sorting (Deb et al., 2002).
        Entrada: população (lista de Individuos)
        Saída: lista de frentes F = [F1, F2, ...], cada frente é uma lista de Individuos
        """
        if population is None:
            population = self.population

        # Vetorização da dominação usando numpy
        n_pop = len(population)
        F_values = np.array([ind.f for ind in population])

        # Matriz de dominação - dominates[i, j] = 1 se i domina j
        dominates = np.zeros((n_pop, n_pop), dtype=bool)
        for i in range(n_pop):
            for j in range(n_pop):
                if i != j:
                    # p domina q se todos p_i <= q_i e pelo menos um p_i < q_i
                    dominates[i, j] = np.all(F_values[i] <= F_values[j]) and np.any(F_values[i] < F_values[j])

        # Número de indivíduos que dominam cada indivíduo
        n_dominated_by = np.sum(dominates, axis=0)

        # Conjunto de indivíduos dominados por cada indivíduo
        S = [[] for _ in range(n_pop)]
        for i in range(n_pop):
            for j in range(n_pop):
                if dominates[i, j]:
                    S[i].append(population[j])

        # Primeira frente - indivíduos não dominados
        F = [[]]
        for i in range(n_pop):
            if n_dominated_by[i] == 0:
                population[i].rank = 0
                F[0].append(population[i])

        # Construindo as demais frentes
        i = 0
        while F[i]:
            Q = []
            for p in F[i]:
                p_idx = population.index(p)
                for q in S[p_idx]:
                    q_idx = population.index(q)
                    n_dominated_by[q_idx] -= 1
                    if n_dominated_by[q_idx] == 0:
                        q.rank = i + 1
                        Q.append(q)
            i += 1
            F.append(Q)

        if not F[-1]:
            F.pop()
        return F

    @staticmethod
    def dominates(p, q):
        """Retorna True se p domina q (minimização)."""
        return all(p_i <= q_i for p_i, q_i in zip(p.f, q.f)) and any(p_i < q_i for p_i, q_i in zip(p.f, q.f))

    @staticmethod
    def crowding_distance(front, M):
        L = len(front)
        if L <= 1:
            if L == 1:
                front[0].crowding = float("inf")
            return

        # Vetorização do cálculo de crowding distance
        F = np.array([ind.f for ind in front])
        crowding = np.zeros(L)

        # Para cada objetivo
        for m in range(M):
            # Ordene os índices pelos valores do objetivo m
            idx = np.argsort(F[:, m])
            # Atribua infinito aos extremos
            crowding[idx[0]] = float("inf")
            crowding[idx[-1]] = float("inf")

            # Normalize pelo range do objetivo
            f_min, f_max = F[idx[0], m], F[idx[-1], m]
            norm = (f_max - f_min) if f_max != f_min else 1.0

            # Calcule a distância de crowding para os pontos intermediários
            for i in range(1, L-1):
                crowding[idx[i]] += (F[idx[i+1], m] - F[idx[i-1], m]) / norm

        # Atribua os valores calculados aos indivíduos
        for i, ind in enumerate(front):
            ind.crowding = crowding[i]

    @staticmethod
    def tournament(a, b):
        if a.rank < b.rank:
            return a
        if b.rank < a.rank:
            return b
        return a if a.crowding > b.crowding else b

    def mating_pool(self, population, N):
        # Vetorização da seleção por torneio
        n_pop = len(population)
        # Gere pares de índices aleatórios para torneios
        idx1 = np.random.randint(0, n_pop, size=N)
        idx2 = np.random.randint(0, n_pop, size=N)

        pool = []
        for i, j in zip(idx1, idx2):
            winner = self.tournament(population[i], population[j])
            pool.append(winner)
        return pool

    @staticmethod
    def _sbx_vectorized(x1, x2, n_offspring, eta=15):
        """Versão vetorizada do Simulated Binary Crossover (SBX)"""
        n_var = len(x1)

        # Gere valores aleatórios de uma vez
        u = np.random.random((n_var, n_offspring//2))

        # Calcule beta para todos os genes de uma vez
        beta = np.where(u <= 0.5,
                        (2*u)**(1.0/(eta+1)),
                        (1.0/(2*(1-u)))**(1.0/(eta+1)))

        # Calcule os filhos
        children = np.zeros((n_offspring, n_var))
        for i in range(0, n_offspring, 2):
            j = i // 2
            children[i] = 0.5 * ((1+beta[:,j]) * x1 + (1-beta[:,j]) * x2)
            children[i+1] = 0.5 * ((1-beta[:,j]) * x1 + (1+beta[:,j]) * x2)

        return children

    def sbx(self, parent1, parent2, eta=15):
        """Versão wrapper da implementação vetorizada do SBX"""
        x1, x2 = np.array(parent1.x), np.array(parent2.x)
        children = self._sbx_vectorized(x1, x2, 2, eta)
        return Individual(children[0]), Individual(children[1])

    @staticmethod
    def _poly_mutation_vectorized(x, n_offspring, eta=20, p_m=None, xl=0, xu=1):
        """Versão vetorizada da Mutação Polinomial"""
        n_var = len(x)
        if p_m is None:
            p_m = 1.0 / n_var

        # Gere valores aleatórios de uma vez
        rand = np.random.random((n_var, n_offspring))
        mut_rand = np.random.random((n_var, n_offspring))

        # Determine quais genes serão mutados
        do_mutation = rand < p_m

        # Prepare xl e xu como arrays se necessário
        xl_arr = np.array([xl] * n_var) if not hasattr(xl, '__len__') else np.array(xl)
        xu_arr = np.array([xu] * n_var) if not hasattr(xu, '__len__') else np.array(xu)

        # Calcule delta
        delta = np.zeros((n_var, n_offspring))
        mask_less = mut_rand <= 0.5
        mask_greater = ~mask_less

        delta[mask_less] = (2 * mut_rand[mask_less]) ** (1.0 / (eta + 1)) - 1.0
        delta[mask_greater] = 1.0 - (2 * (1.0 - mut_rand[mask_greater])) ** (1.0 / (eta + 1))

        # Aplique a mutação vetorizada
        x_mutated = np.zeros((n_offspring, n_var))
        for i in range(n_offspring):
            mutated = x + delta[:, i] * (xu_arr - xl_arr)
            # Aplique limites
            mutated = np.maximum(np.minimum(mutated, xu_arr), xl_arr)
            # Aplique apenas nos genes selecionados para mutação
            mask = do_mutation[:, i]
            final_x = x.copy()
            final_x[mask] = mutated[mask]
            x_mutated[i] = final_x

        return x_mutated

    def poly_mutation(self, ind, eta=20, p_m=None, xl=None, xu=None):
        """Wrapper para a implementação vetorizada da mutação polinomial"""
        if xl is None:
            xl = self.xl
        if xu is None:
            xu = self.xu

        x = np.array(ind.x)
        mutated = self._poly_mutation_vectorized(x, 1, eta, p_m, xl, xu)
        ind.x = mutated[0]

    def make_offspring(self, pop, N, crossover_prob=0.9, eta_c=15, eta_m=20):
        # Otimização: geração vetorizada dos cromossomos dos pais
        n_var = len(pop[0].x)
        n_needed = N
        kids = []
        while len(kids) < n_needed:
            idxs = np.random.randint(0, len(pop), size=4)
            p1 = self.tournament(pop[idxs[0]], pop[idxs[1]])
            p2 = self.tournament(pop[idxs[2]], pop[idxs[3]])
            if np.random.random() < crossover_prob:
                c1, c2 = self.sbx(p1, p2, eta=eta_c)
            else:
                c1, c2 = deepcopy(p1), deepcopy(p2)
            self.poly_mutation(c1, eta=eta_m, xl=self.xl, xu=self.xu)
            self.poly_mutation(c2, eta=eta_m, xl=self.xl, xu=self.xu)
            kids.extend([c1, c2])
        return kids[:N]

    def environmental_selection(self, P, Q, N, M):
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

            # Use np.lexsort para ordenar pelos dois critérios (rank e crowding)
            # Ordenação deve ser por rank (crescente) e crowding (decrescente)
            last_front = fronts[i]
            crowding_values = np.array([ind.crowding for ind in last_front])
            # Como queremos ordenação decrescente por crowding, usamos o negativo
            sorted_indices = np.argsort(-crowding_values)

            # Adicione os melhores indivíduos da última frente
            needed = N - len(next_P)
            for j in sorted_indices[:needed]:
                next_P.append(last_front[j])

        return next_P

    def run(self, N=None, N_GEN=None, M=None, verbose=True):
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
        for g in range(N_GEN):
            Q = self.make_offspring(P, N)
            # Avalia apenas os filhos (vetorizado)
            XQ = np.array([ind.x for ind in Q])
            FQ = self.problem(XQ)
            for ind, f in zip(Q, FQ):
                ind.f = f
            P = self.environmental_selection(P, Q, N, M)
            if verbose and g % 50 == 0:
                best = min(P, key=lambda i: i.f[0])
                print(f"G {g}  |  best f:", best.f)
        self.population = P
        return P
