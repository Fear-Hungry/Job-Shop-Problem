def population_diversity(population):
    # Calcula a diversidade média (distância de Hamming normalizada)
    size = len(population)
    if size < 2:
        return 0.0
    total_dist = 0
    count = 0
    for i in range(size):
        for j in range(i+1, size):
            dist = sum(1 for a, b in zip(
                population[i], population[j]) if a != b)
            total_dist += dist
            count += 1
    max_dist = len(population[0]) if population else 1
    return (total_dist / count) / max_dist if count > 0 else 0.0
