import random
import copy


class SelectionOperator:
    """
    Classe responsável pelas operações de seleção do algoritmo genético.
    """

    @staticmethod
    def tournament_selection(population, fitnesses, tournament_size=2):
        """
        Realiza a seleção por torneio.

        Args:
            population: Lista de indivíduos
            fitnesses: Lista de valores de fitness para cada indivíduo
            tournament_size: Tamanho do torneio

        Returns:
            Lista de indivíduos selecionados
        """
        selected = []
        pop_indices = list(range(len(population)))

        for _ in range(len(population)):
            # Seleciona indivíduos aleatórios para o torneio
            candidates = random.sample(pop_indices, tournament_size)

            # Encontra o indivíduo com o melhor fitness (menor makespan)
            winner_idx = min(candidates, key=lambda idx: fitnesses[idx])

            # Adiciona uma cópia para evitar modificar o original na população
            selected.append(copy.deepcopy(population[winner_idx]))

        return selected

    @staticmethod
    def roulette_wheel_selection(population, fitnesses):
        """
        Realiza a seleção por roleta. Para problemas de minimização, inverte o fitness.

        Args:
            population: Lista de indivíduos
            fitnesses: Lista de valores de fitness para cada indivíduo

        Returns:
            Lista de indivíduos selecionados
        """
        # Como o fitness é um makespan (menor é melhor), calculamos o inverso
        # Adiciona um pequeno epsilon para evitar divisão por zero
        epsilon = 1e-10
        inverse_fitnesses = [1.0 / (fit + epsilon) for fit in fitnesses]

        # Calcula a soma total dos fitness inversos
        total_fitness = sum(inverse_fitnesses)

        # Calcula as probabilidades de seleção (proporcional ao fitness inverso)
        probabilities = [fit / total_fitness for fit in inverse_fitnesses]

        # Realiza a seleção
        selected = []
        for _ in range(len(population)):
            # Seleciona um indivíduo com probabilidade proporcional ao fitness inverso
            selected_idx = random.choices(
                range(len(population)), weights=probabilities, k=1)[0]
            selected.append(copy.deepcopy(population[selected_idx]))

        return selected

    @staticmethod
    def rank_selection(population, fitnesses):
        """
        Realiza a seleção por ranking.

        Args:
            population: Lista de indivíduos
            fitnesses: Lista de valores de fitness para cada indivíduo

        Returns:
            Lista de indivíduos selecionados
        """
        # Ordena os índices da população pelo fitness (menor para maior)
        ranked_indices = sorted(range(len(fitnesses)),
                                key=lambda i: fitnesses[i])

        # Atribui um rank a cada indivíduo (o melhor tem o maior rank)
        ranks = [0] * len(population)
        for rank, idx in enumerate(ranked_indices):
            ranks[idx] = rank + 1  # Rank começa em 1

        # Calcula a soma dos ranks
        total_rank = sum(ranks)

        # Calcula as probabilidades de seleção (proporcional ao rank)
        probabilities = [rank / total_rank for rank in ranks]

        # Realiza a seleção
        selected = []
        for _ in range(len(population)):
            # Seleciona um indivíduo com probabilidade proporcional ao rank
            selected_idx = random.choices(
                range(len(population)), weights=probabilities, k=1)[0]
            selected.append(copy.deepcopy(population[selected_idx]))

        return selected

    @staticmethod
    def elitism_selection(population, fitnesses, elite_size, rest_selection_func):
        """
        Realiza a seleção com elitismo.

        Args:
            population: Lista de indivíduos
            fitnesses: Lista de valores de fitness para cada indivíduo
            elite_size: Número de indivíduos de elite a manter
            rest_selection_func: Função de seleção para o restante da população

        Returns:
            Lista de indivíduos selecionados
        """
        # Encontra os índices dos indivíduos com melhor fitness (menor makespan)
        elite_indices = sorted(range(len(fitnesses)),
                               key=lambda i: fitnesses[i])[:elite_size]

        # Seleciona os indivíduos de elite
        elite = [copy.deepcopy(population[idx]) for idx in elite_indices]

        # Prepara a população sem os elite para o restante da seleção
        remaining_population = [indiv for i, indiv in enumerate(
            population) if i not in elite_indices]
        remaining_fitnesses = [fit for i, fit in enumerate(
            fitnesses) if i not in elite_indices]

        # Seleciona o restante usando a função de seleção fornecida
        remaining_selected = rest_selection_func(
            remaining_population, remaining_fitnesses)

        # Seleciona apenas o número necessário para completar a população
        num_remaining_needed = len(population) - elite_size
        remaining_selected = remaining_selected[:num_remaining_needed]

        # Combina os elite com o restante selecionado
        return elite + remaining_selected
