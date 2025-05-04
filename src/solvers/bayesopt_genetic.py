import random
try:
    from bayes_opt import BayesianOptimization
except ImportError:
    raise ImportError(
        'A biblioteca bayesian-optimization não está instalada. Instale com: pip install bayesian-optimization')
from .genetic_solver import GeneticSolver
from .pbt_genetic import PBTGeneticRunner


class BayesOptGeneticRunner:
    def __init__(self, jobs, num_jobs, num_machines, n_iter=20, init_points=5, population_size=8, block_generations=10, total_blocks=10):
        self.jobs = jobs
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.n_iter = n_iter
        self.init_points = init_points
        self.population_size = population_size
        self.block_generations = block_generations
        self.total_blocks = total_blocks
        self.best_params = None
        self.best_score = float('inf')
        self.history = []

    def _objective(self, mutation_rate, crossover_rate, elite_size):
        elite_size = int(round(elite_size))
        ga = GeneticSolver(
            self.jobs, self.num_jobs, self.num_machines,
            population_size=30,
            generations=self.block_generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elite_size=elite_size
        )
        schedule = ga.solve()
        makespan = schedule.get_makespan() if schedule else float('inf')
        # Armazena histórico
        self.history.append({
            'mutation_rate': mutation_rate,
            'crossover_rate': crossover_rate,
            'elite_size': elite_size,
            'makespan': makespan
        })
        # Minimização: retorna negativo para maximização do otimizador
        score = -makespan
        if makespan < self.best_score:
            self.best_score = makespan
            self.best_params = {
                'mutation_rate': mutation_rate,
                'crossover_rate': crossover_rate,
                'elite_size': elite_size
            }
        print(
            f"Avaliado: mut={mutation_rate:.3f}, cross={crossover_rate:.3f}, elite={elite_size} => makespan={makespan:.2f}")
        return score

    def run(self):
        pbounds = {
            'mutation_rate': (0.05, 0.5),
            'crossover_rate': (0.5, 0.95),
            'elite_size': (1, 4)
        }
        optimizer = BayesianOptimization(
            f=self._objective,
            pbounds=pbounds,
            random_state=42,
        )
        optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter
        )
        print(f"\nMelhores hiperparâmetros encontrados pela otimização bayesiana:")
        print(self.best_params)
        print(f"Melhor makespan: {self.best_score:.2f}")
        return self.best_params

    def run_with_pbt(self):
        # Roda a otimização bayesiana e inicializa o PBT com os melhores hiperparâmetros
        best_params = self.run()
        # Fallback para valores padrão se best_params for None
        if best_params is None:
            best_params = {'mutation_rate': 0.2,
                           'crossover_rate': 0.8, 'elite_size': 2}
        # Inicializa população do PBT com variações dos melhores hiperparâmetros
        jobs = self.jobs
        num_jobs = self.num_jobs
        num_machines = self.num_machines
        pbt_runner = PBTGeneticRunner(
            jobs, num_jobs, num_machines,
            population_size=self.population_size,
            block_generations=self.block_generations,
            total_blocks=self.total_blocks
        )
        # Sobrescreve hiperparâmetros iniciais do PBT
        for i in range(self.population_size):
            pbt_runner.hyperparams[i]['mutation_rate'] = min(
                max(random.gauss(best_params['mutation_rate'], 0.05), 0.05), 0.5)
            pbt_runner.hyperparams[i]['crossover_rate'] = min(
                max(random.gauss(best_params['crossover_rate'], 0.05), 0.5), 0.95)
            pbt_runner.hyperparams[i]['elite_size'] = int(
                round(min(max(random.gauss(best_params['elite_size'], 0.5), 1), 4)))
        print("\nPopulação inicial do PBT ajustada com base na otimização bayesiana!")
        melhor_ga = pbt_runner.run()
        return melhor_ga


# Exemplo de uso:
if __name__ == '__main__':
    # Supondo que você tenha jobs, num_jobs, num_machines definidos
    # Exemplo:
    # jobs = ...
    # num_jobs = ...
    # num_machines = ...
    # runner = BayesOptGeneticRunner(jobs, num_jobs, num_machines)
    # melhor_ga = runner.run_with_pbt()
    pass
