import numpy as np
import random


def select_operator_ucb1(counts, rewards_sum, total_selections, exploration_factor=2.0):
    """
    Seleciona um operador usando o algoritmo UCB1.

    Args:
        counts: Lista com o número de vezes que cada operador foi selecionado
        rewards_sum: Lista com a soma das recompensas para cada operador
        total_selections: Número total de seleções feitas
        exploration_factor: Fator de exploração do algoritmo UCB1

    Returns:
        Índice do operador selecionado
    """
    num_operators = len(counts)

    # Fase de exploração inicial: Garante que cada operador seja testado pelo menos uma vez.
    not_tried_indices = [i for i, count in enumerate(counts) if count == 0]
    if not_tried_indices:
        return random.choice(not_tried_indices)

    # Calcula os scores UCB1 para cada operador
    ucb_scores = []
    # Evita log(0) ou divisão por zero
    log_total_selections = np.log(max(1, total_selections))

    for i in range(num_operators):
        count = counts[i]
        # Termo de Exploração (Exploitation Term) - Recompensa média
        average_reward = rewards_sum[i] / count

        # Termo de Exploração (Exploration Term)
        exploration_term = exploration_factor * np.sqrt(
            log_total_selections / count
        )

        ucb_scores.append(average_reward + exploration_term)

    # Retorna o índice do operador com o maior score UCB1
    return np.argmax(ucb_scores)


def update_operator_rewards(population, fitnesses, original_fitnesses,
                            crossover_counts, crossover_rewards_sum, total_crossover_selections,
                            mutation_counts, mutation_rewards_sum, total_mutation_selections):
    """
    Atualiza as contagens e recompensas UCB1 com base na melhoria de fitness.

    Args:
        population: Lista de indivíduos
        fitnesses: Lista de valores de fitness para cada indivíduo
        original_fitnesses: Lista de valores de fitness originais antes da aplicação dos operadores
        crossover_counts: Lista de contagens para cada operador de crossover
        crossover_rewards_sum: Lista de somas de recompensas para cada operador de crossover
        total_crossover_selections: Número total de seleções de operadores de crossover
        mutation_counts: Lista de contagens para cada operador de mutação
        mutation_rewards_sum: Lista de somas de recompensas para cada operador de mutação
        total_mutation_selections: Número total de seleções de operadores de mutação

    Returns:
        total_crossover_selections, total_mutation_selections: Valores atualizados
    """
    for i, indiv in enumerate(population):
        original_fitness = original_fitnesses[i]
        current_fitness = fitnesses[i]
        # Maior é melhor (redução do makespan)
        improvement = original_fitness - current_fitness

        # Se o indivíduo foi resultado de crossover
        if 'crossover_op_idx' in indiv:
            op_idx = indiv['crossover_op_idx']
            crossover_counts[op_idx] += 1
            # Recompensa pode ser a melhoria absoluta, relativa, ou binária (melhorou ou não)
            # Usando melhoria absoluta:
            reward = max(0, improvement)
            crossover_rewards_sum[op_idx] += reward
            total_crossover_selections += 1
            # Remove o índice para não contar de novo
            del indiv['crossover_op_idx']

        # Se o indivíduo foi resultado de mutação (pode ocorrer após crossover)
        if 'mutation_op_idx' in indiv:
            op_idx = indiv['mutation_op_idx']
            mutation_counts[op_idx] += 1
            # Usa a mesma lógica de recompensa
            reward = max(0, improvement)
            mutation_rewards_sum[op_idx] += reward
            total_mutation_selections += 1
            # Remove o índice
            del indiv['mutation_op_idx']

    return total_crossover_selections, total_mutation_selections
