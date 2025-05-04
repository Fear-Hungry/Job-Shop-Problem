import random
import copy
from .genetic_solver import GeneticSolver
from typing import List, Dict, Optional


class PBTGeneticRunner:
    def __init__(self, jobs, num_jobs, num_machines, population_size: int = 8, block_generations: int = 10, total_blocks: int = 10):
        self.jobs = jobs
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.population_size = population_size  # Número de instâncias GA
        self.block_generations = block_generations  # Gerações por bloco
        self.total_blocks = total_blocks  # Quantos blocos executar
        self.gas: List[GeneticSolver] = []  # Instâncias de GeneticSolver
        # Lista de dicionários de hiperparâmetros
        self.hyperparams: List[Dict[str, float]] = []
        self.scores: List[float] = [float('inf')] * population_size
        self._init_population()

    def _init_population(self):
        for _ in range(self.population_size):
            mutation_rate = random.uniform(0.05, 0.5)
            crossover_rate = random.uniform(0.5, 0.95)
            elite_size = random.randint(1, 4)
            ga = GeneticSolver(
                self.jobs, self.num_jobs, self.num_machines,
                population_size=30,
                generations=self.block_generations,
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                elite_size=elite_size
            )
            self.gas.append(ga)
            self.hyperparams.append({
                'mutation_rate': mutation_rate,
                'crossover_rate': crossover_rate,
                'elite_size': elite_size
            })

    def _mutate_hyperparam(self, value: float, min_val: float, max_val: float, scale: float = 0.1, is_int: bool = False) -> float:
        perturbed = value + random.uniform(-scale, scale)
        perturbed = min(max(perturbed, min_val), max_val)
        return int(round(perturbed)) if is_int else perturbed

    def run(self) -> GeneticSolver:
        for block in range(self.total_blocks):
            print(f'\n=== Bloco {block+1}/{self.total_blocks} ===')
            # Executa cada GA por block_generations
            for i, ga in enumerate(self.gas):
                ga.generations = self.block_generations  # Garante número de gerações por bloco
                ga.crossover_rate = self.hyperparams[i]['crossover_rate']
                ga.mutation_rate = self.hyperparams[i]['mutation_rate']
                ga.elite_size = int(self.hyperparams[i]['elite_size'])
                schedule = ga.solve()
                makespan = schedule.get_makespan() if schedule else float('inf')
                self.scores[i] = makespan
                print(
                    f'GA {i}: makespan={makespan:.2f} | mut={ga.mutation_rate:.3f} cross={ga.crossover_rate:.3f} elite={ga.elite_size}')
            # Exploit/Explore: piores copiam melhores + mutação dos hiperparâmetros
            sorted_idx = sorted(range(self.population_size),
                                key=lambda i: self.scores[i])
            num_exploit = max(1, self.population_size // 4)
            for i in sorted_idx[-num_exploit:]:  # Piores
                j = random.choice(sorted_idx[:num_exploit])  # Um dos melhores
                # Copia população e hiperparâmetros
                self.gas[i].schedule = copy.deepcopy(self.gas[j].schedule)
                self.gas[i].jobs = copy.deepcopy(self.gas[j].jobs)
                # Mutação dos hiperparâmetros
                self.hyperparams[i]['mutation_rate'] = self._mutate_hyperparam(
                    self.hyperparams[j]['mutation_rate'], 0.05, 0.5, 0.05)
                self.hyperparams[i]['crossover_rate'] = self._mutate_hyperparam(
                    self.hyperparams[j]['crossover_rate'], 0.5, 0.95, 0.05)
                self.hyperparams[i]['elite_size'] = self._mutate_hyperparam(
                    self.hyperparams[j]['elite_size'], 1, 4, 1, is_int=True)
                print(
                    f'GA {i} explora GA {j} (herda hiperparâmetros e população)')
        # Após todos os blocos, retorna o melhor GA
        best_idx = self.scores.index(min(self.scores))
        print(
            f'\nMelhor GA: {best_idx} | makespan={self.scores[best_idx]:.2f}')
        return self.gas[best_idx]


# Exemplo de uso:
if __name__ == '__main__':
    # Supondo que você tenha jobs, num_jobs, num_machines definidos
    # Exemplo:
    # jobs = ...
    # num_jobs = ...
    # num_machines = ...
    # runner = PBTGeneticRunner(jobs, num_jobs, num_machines)
    # melhor_ga = runner.run()
    pass
