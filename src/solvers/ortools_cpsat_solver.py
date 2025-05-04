"""
Módulo que contém a implementação do solver OR-Tools CP-SAT para o Job Shop Scheduling.
"""
from solvers.base_solver import BaseSolver
from models.schedule import Schedule


class ORToolsCPSATSolver(BaseSolver):
    """
    Implementação de um solver para o problema Job Shop Scheduling usando Google OR-Tools CP-SAT.
    """

    def solve(self, time_limit: float = 30.0):
        """
        Resolve o problema Job Shop Scheduling usando Google OR-Tools CP-SAT.

        Args:
            time_limit (float): Limite máximo de tempo em segundos para o solver.

        Returns:
            Schedule: A agenda de operações resultante.
        """
        from ortools.sat.python import cp_model

        # Cria o modelo
        model = cp_model.CpModel()

        # Calcula o horizonte (duração total possível)
        horizon = sum(duration for job in self.jobs for _, duration in job)

        # Cria variáveis
        all_tasks = {}
        for job_id, job in enumerate(self.jobs):
            for op_idx, (machine_id, duration) in enumerate(job):
                suffix = f"J{job_id}_T{op_idx}"
                start = model.NewIntVar(0, horizon, f"start_{suffix}")
                end = model.NewIntVar(0, horizon, f"end_{suffix}")
                interval = model.NewIntervalVar(
                    start, duration, end, f"interval_{suffix}")
                all_tasks[(job_id, op_idx)] = (
                    start, end, interval, machine_id, duration)

        # Adiciona restrições de máquina (sem sobreposição nas máquinas)
        machine_to_intervals = {}
        for (job_id, op_idx), (_, _, interval, machine_id, _) in all_tasks.items():
            if machine_id not in machine_to_intervals:
                machine_to_intervals[machine_id] = []
            machine_to_intervals[machine_id].append(interval)

        for machine_id, intervals in machine_to_intervals.items():
            model.AddNoOverlap(intervals)

        # Adiciona restrições de precedência dentro de cada job
        for job_id, job in enumerate(self.jobs):
            for op_idx in range(len(job) - 1):
                # tempo final da operação atual
                prev_end = all_tasks[(job_id, op_idx)][1]
                # tempo inicial da próxima operação
                next_start = all_tasks[(job_id, op_idx + 1)][0]
                model.Add(next_start >= prev_end)

        # Define a variável makespan
        makespan = model.NewIntVar(0, horizon, "makespan")
        all_ends = [all_tasks[(job_id, len(job) - 1)][1]
                    for job_id, job in enumerate(self.jobs)]
        model.AddMaxEquality(makespan, all_ends)

        # Objetivo: minimizar makespan
        model.Minimize(makespan)

        # Resolve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(model)

        # Constrói agenda a partir da solução
        schedule = Schedule()
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for job_id, job in enumerate(self.jobs):
                for op_idx, (machine_id, duration) in enumerate(job):
                    start_time = solver.Value(all_tasks[(job_id, op_idx)][0])
                    schedule.add_operation(
                        job_id, op_idx, machine_id, start_time, duration)

            # Valida a agenda
            if not self.validator.is_valid(schedule):
                print("Warning: Generated schedule is invalid!")

            self.schedule = schedule
        else:
            print(
                f"No solution found. Solver status: {solver.StatusName(status)}")

        return schedule
