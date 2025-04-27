"""
Módulo que contém a classe ScheduleValidator para validar agendas de operações do Job Shop.
"""


class ScheduleValidator:
    """
    Classe responsável por validar uma agenda de operações.
    """
    def __init__(self, jobs, num_jobs, num_machines):
        """
        Inicializa o validador de agenda.

        Args:
            jobs (list[list[tuple[int, int]]]): Lista de jobs, onde cada job é uma lista de pares (máquina, duração).
            num_jobs (int): Número de jobs.
            num_machines (int): Número de máquinas.
        """
        self.jobs = jobs
        self.num_jobs = num_jobs
        self.num_machines = num_machines

    def is_valid(self, schedule):
        """
        Valida uma agenda para o problema Job Shop Scheduling.

        Verifica as seguintes condições para garantir a validade da agenda:
        1. Nenhuma operação repetida para qualquer job.
        2. As operações de cada job estão na sequência correta e não começam antes que a operação anterior termine.
        3. Sem sobreposição de operações na mesma máquina.

        Args:
            schedule (Schedule): A agenda a ser validada.

        Returns:
            bool: True se a agenda é válida, False caso contrário. Imprime mensagens de erro detalhadas.
        """
        operations = schedule.operations
        seen_operations = set()
        op_start_times = {}

        # Verifica operações repetidas
        for job_id, op_idx, machine, start, duration in operations:
            if (job_id, op_idx) in seen_operations:
                print(
                    f"Invalid schedule: Repeated operation for Job {job_id}, Operation {op_idx}."
                )
                return False
            seen_operations.add((job_id, op_idx))
            op_start_times[(job_id, op_idx)] = (start, start + duration)

        # Verifica sequência de operações para cada job
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

        # Verifica sobreposição de operações nas máquinas
        machine_usage = {}
        for job_id, op_idx, machine, start, duration in operations:
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