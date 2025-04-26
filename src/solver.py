class Solver:
    def __init__(self, jobs, num_jobs, num_machines):
        """
        Initializes a Solver instance for a job shop scheduling problem.

        Args:
            jobs (list[list[tuple[int, int]]]): A list of jobs, where each job is a list of
                (machine, duration) pairs.
            num_jobs (int): The number of jobs.
            num_machines (int): The number of machines.
        """
        self.jobs = jobs
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.schedule = []

    def solve(self):
        """
        Resolve o problema de Job Shop usando o NSGA-II e retorna o melhor cronograma encontrado.
        """
        from nsga2.NSGA2Perm import NSGA2Perm
        from nsga2.jobshop_eval import decode_permutation, multi_objective_evaluation
        from nsga2.Individual import Individual
        import numpy as np
        import random

        num_ops = sum(len(job) for job in self.jobs)
        pop_size = 100  # Tamanho sugerido no roteiro
        n_gen = 400     # Gerações sugeridas no roteiro

        # Cada operação é identificada por um índice único de 0 a num_ops-1
        op_ids = list(range(num_ops))

        # Função de avaliação para o NSGA-II
        def problem(X):
            return multi_objective_evaluation(X, self.jobs, self.num_jobs, self.num_machines)

        # Inicializar o NSGA-II para problemas de permutação
        nsga2 = NSGA2Perm(pop_size, n_gen, problem, None, None, op_ids)
        nsga2.IndividualClass = Individual
        nsga2.initialize_population()
        nsga2.evaluate_population()
        final_pop = nsga2.run(verbose=True)  # Ativando verbose para ver progresso

        # Seleciona o indivíduo com menor makespan (primeiro objetivo)
        best = min(final_pop, key=lambda ind: ind.f[0])
        best_perm = np.array(best.x).astype(int)
        schedule = decode_permutation(best_perm, self.jobs, self.num_jobs, self.num_machines)
        print(f"\nMelhores objetivos encontrados: Makespan={best.f[0]}, Tardiness={best.f[1]:.2f}, Load_SD={best.f[2]:.2f}")
        self.schedule = schedule
        return schedule

    def get_makespan(self, schedule):
        """
        Calculates the makespan of the given schedule.

        Args:
            schedule (list[tuple[int, int, int, int, int]]): A list of scheduled operations, where each operation is represented as a tuple:
                (job_id, operation_index, machine_id, start_time, duration).

        Returns:
            int: The maximum finish time of all operations in the schedule (i.e., the makespan).

        Raises:
            ValueError: If the schedule is empty.
        """
        if not schedule:
            raise ValueError("Schedule is empty")
        return max(start + duration for _, _, _, start, duration in schedule)

    def print_schedule(self, schedule):
        """
        Prints the schedule of operations in a human-readable format, along with the total makespan.

        Args:
            schedule (list[tuple[int, int, int, int, int]]): A list of scheduled operations, where
                each operation is represented as a tuple: (job_id, operation_index, machine_id,
                start_time, duration).
        """
        print("Schedule:")
        for job_id, op_idx, machine, start, duration in schedule:
            print(
                f"Job {job_id}, Op {op_idx} -> Machine {machine} | Start: {start}, Duration: {duration}"
            )
        print(f"\nTotal Makespan: {self.get_makespan(schedule)}")

    def is_valid_schedule(self, schedule):
        """
        Validates the given schedule for a job shop scheduling problem.

        This function checks for the following conditions to ensure the schedule's validity:
        1. No repeated operations for any job.
        2. Operations for each job are in the correct sequence and do not start before
        the previous operation ends.
        3. No overlapping operations on the same machine.

        Args:
            schedule (list[tuple[int, int, int, int, int]]): A list of scheduled operations,
                where each operation is represented as a tuple:
                (job_id, operation_index, machine_id, start_time, duration).

        Returns:
            bool: True if the schedule is valid, False otherwise. Prints detailed error
            messages if the schedule is invalid.
        """
        seen_operations = set()
        op_start_times = {}

        for job_id, op_idx, machine, start, duration in schedule:
            if (job_id, op_idx) in seen_operations:
                print(
                    f"Invalid schedule: Repeated operation for Job {job_id}, Operation {op_idx}."
                )
                return False
            seen_operations.add((job_id, op_idx))
            op_start_times[(job_id, op_idx)] = (start, start + duration)

        for job_id in range(self.num_jobs):
            for op_idx in range(1, len(self.jobs[job_id])):
                if (job_id, op_idx) in op_start_times and (
                    job_id,
                    op_idx - 1,
                ) in op_start_times:
                    prev_end = op_start_times[(job_id, op_idx - 1)][1]
                    curr_start = op_start_times[(job_id, op_idx)][0]
                    if curr_start < prev_end:
                        print(
                            f"Invalid schedule: Job {job_id} - Operation {op_idx} starts before previous ends."
                        )
                        return False

        machine_usage = {}
        for job_id, op_idx, machine, start, duration in schedule:
            if machine not in machine_usage:
                machine_usage[machine] = []
            machine_usage[machine].append((start, start + duration))

        for machine, intervals in machine_usage.items():
            intervals.sort()
            for i in range(1, len(intervals)):
                prev_end = intervals[i - 1][1]
                curr_start = intervals[i][0]
                if curr_start < prev_end:
                    print(
                        f"Invalid schedule: Overlapping operations on machine {machine}."
                    )
                    return False

        return True
