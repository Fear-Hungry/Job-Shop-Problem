def read_jobshop_instance(filepath):
    """
    Parses a Job Shop Scheduling Problem (JSSP) instance from a file.

    The file format should begin with a line containing two integers:
    the number of jobs and the number of machines. This is followed by
    one line per job, where each line contains pairs of integers
    representing (machine_id, duration) for each operation in the job.

    Returns:
        tuple: (jobs, num_jobs, num_machines)
            - jobs (list[list[tuple[int, int]]]): A list of jobs, each a list of (machine, duration) pairs
            - num_jobs (int): The number of jobs
            - num_machines (int): The number of machines

    Example:
        File content:
            3 3
            1 2 2 3 3 1
            2 1 3 2
            1 3 2 1

        Represents:
            - 3 jobs
            - 3 machines
            - First job has 3 operations: (1,2), (2,3), (3,1)
            - Second job has 2 operations: (2,1), (3,2)
            - Third job has 2 operations: (1,3), (2,1)
    """
    with open(filepath, "r") as file:
        lines = [
            line.strip() for line in file if line.strip() and not line.startswith("#")
        ]

    num_jobs, num_machines = map(int, lines[0].split())
    jobs = []

    for line in lines[1 : num_jobs + 1]:
        data = list(map(int, line.split()))
        job = []
        for i in range(0, len(data), 2):
            machine = data[i]
            duration = data[i + 1]
            job.append((machine, duration))
        jobs.append(job)

    return jobs, num_jobs, num_machines
