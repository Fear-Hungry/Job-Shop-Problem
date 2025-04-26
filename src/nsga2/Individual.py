import numpy as np

class Individual:
    def __init__(self, x):
        """Inicializa um indivíduo com Random-Key.
        Args:
            x: vetor de chaves aleatórias (valores entre 0 e 1)
        """
        self.x = np.array(x)           # vetor de chaves aleatórias
        self.f = [np.inf, np.inf, np.inf]  # [makespan, atrasos, balanceamento]
        self.rank = None               # frente de Pareto
        self.crowding = 0.0            # distância de aglomeração
        self.S = []                    # soluções dominadas por este
        self.n = 0                     # nº de soluções que o dominam
        self.schedule = None           # cronograma decodificado

    def decode(self, jobs, num_jobs, num_machines):
        """Decodifica as chaves aleatórias em um cronograma usando Giffler-Thompson.
        Args:
            jobs: lista de jobs e suas operações
            num_jobs: número total de jobs
            num_machines: número total de máquinas
        """
        # Ordena operações pelas chaves aleatórias
        n_ops = sum(len(job) for job in jobs)
        op_order = np.argsort(self.x[:n_ops])

        # Implementação simplificada de Giffler-Thompson
        machine_times = [0] * num_machines
        job_times = [0] * num_jobs
        job_counters = [0] * num_jobs
        schedule = []

        for op in op_order:
            job_id = op // num_machines
            op_idx = job_counters[job_id]
            machine, duration = jobs[job_id][op_idx]

            start = max(machine_times[machine], job_times[job_id])
            schedule.append((job_id, op_idx, machine, start, duration))

            machine_times[machine] = start + duration
            job_times[job_id] = start + duration
            job_counters[job_id] += 1

        self.schedule = schedule
        return schedule
