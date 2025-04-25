import sys
from collections import defaultdict

from src.common import read_jobshop_instance


def read_solution_order(path):
    """
    Reads a job shop scheduling solution from a file and returns it as a list of operation tuples.

    Each line should contain five integers: job_id, operation_index, machine_id, start_time, and duration.

    Args:
        path (str): The path to the solution file.

    Returns:
        list[tuple[int, int, int, int, int]]: List of tuples representing operations.
    """
    order = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"❌ Invalid line at {line_num}: {line}. Expected 5 values.")
                try:
                    order.append(tuple(map(int, parts)))
                except ValueError:
                    raise ValueError(f"❌ Line {line_num} contains non-integer values: {line}")
    return order


def check_duplicates(order):
    """
    Checks if a given schedule has any duplicate operations.

    Args:
        order (list[tuple[int, int, int, int, int]]): The schedule to check.

    Returns:
        bool: True if no duplicates are found, False otherwise.
    """
    seen = set()
    for operation in order:
        if operation in seen:
            print(f"❌ Duplicate operation: {operation}")
            return False
        seen.add(operation)
    return True


def check_all_operations_scheduled(jobs, order):
    """
    Checks if all required operations are scheduled and no extra operations are present.

    This function compares the required operations for each job against the operations
    provided in the order. It identifies any missing or extra operations and returns
    False if any discrepancies are found.

    Args:
        jobs (list[list[tuple[int, int]]]): A list of jobs, where each job is a list
            of operations represented as (machine, duration) tuples.
        order (list[tuple[int, int, int, int, int]]): A list of scheduled operations,
            with each operation represented as a tuple:
            (job_id, operation_index, machine_id, start_time, duration).

    Returns:
        bool: True if all required operations are scheduled and no extra operations
        are present; False otherwise. Prints details of missing or extra operations if any.
    """

    required_ops = set((j, o) for j in range(len(jobs)) for o in range(len(jobs[j])))
    given_ops = set((job, op) for job, op, _, _, _ in order)

    missing = required_ops - given_ops
    extra = given_ops - required_ops

    if missing:
        print(f"❌ Missing operations: {missing}")
        return False
    if extra:
        print(f"❌ Extra operations: {extra}")
        return False
    return True


def build_schedule(order):
    """
    Constructs a schedule from a list of ordered operations.

    This function validates the order of operations for each job and ensures that no operations are out of sequence.
    It calculates the ready times for each job and machine, updating them as operations are scheduled.
    The function returns a list representing the complete schedule.

    Args:
        order (list[tuple[int, int, int, int, int]]): A list of operations, where each operation is represented as a
            tuple: (job_id, operation_index, machine_id, start_time, duration).

    Returns:
        list[tuple[int, int, int, int, int]]: A list of scheduled operations, each represented as a tuple:
            (job_id, operation_index, machine_id, start_time, duration).

    Raises:
        ValueError: If an operation is found out of order for any job.
    """
    job_next_op = defaultdict(int)
    job_ready_time = defaultdict(int)
    machine_ready_time = defaultdict(int)
    schedule = []

    for job, op, machine, start_time, duration in order:
        if op != job_next_op[job]:
            raise ValueError(
                f"Operation out of order for job {job}: expected op {job_next_op[job]}, received {op}"
            )

        job_ready_time[job] = max(job_ready_time[job], start_time)
        machine_ready_time[machine] = max(machine_ready_time[machine], start_time)
        end_time = start_time + duration

        job_next_op[job] += 1
        machine_ready_time[machine] = end_time

        schedule.append((job, op, machine, start_time, duration))

    return schedule


def check_conflicts(schedule):
    """
    Checks if the given schedule has any conflicts between operations on the same machine.

    Args:
        schedule (list[tuple[int, int, int, int, int]]): A list of scheduled operations, each represented as a tuple:
            (job_id, operation_index, machine_id, start_time, duration).

    Returns:
        bool: True if the schedule has no conflicts, False otherwise.

    Prints a message for each conflict found, indicating the conflicting operations and the conflicting times.
    """
    machine_ops = defaultdict(list)
    for job, op, machine, start, duration in schedule:
        machine_ops[machine].append((start, start + duration, job, op))

    for machine, ops in machine_ops.items():
        ops.sort()
        for i in range(1, len(ops)):
            end_prev = ops[i - 1][1]
            start_curr = ops[i][0]
            if start_curr < end_prev:
                print(
                    f"❌ Conflict on machine {machine}: operation ({ops[i][2]},{ops[i][3]}) starts at {start_curr}, before the end of ({ops[i - 1][2]},{ops[i - 1][3]}) at {end_prev}."
                )
                return False
    return True


def main(instance_file, solution_file):
    """
    Validates a given job shop scheduling solution against the given instance.

    Prints a success message if the solution is valid, or an error message if it is not.

    Args:
        instance_file (str): The file path of the JSSP instance
        solution_file (str): The file path of the solution to validate
    """
    try:
        jobs, _, _ = read_jobshop_instance(instance_file)
        order = read_solution_order(solution_file)

        if not check_duplicates(order):
            raise ValueError("❌ Invalid solution: duplicate operations found.")

        if not check_all_operations_scheduled(jobs, order):
            raise ValueError("❌ Invalid solution: missing or extra operations.")

        schedule = build_schedule(order)
        
        if not check_conflicts(schedule):
            raise ValueError("❌ Invalid solution: machine conflicts detected.")

        print("✅ Valid solution (no machine conflicts).")
        makespan = max(start + duration for _, _, _, start, duration in schedule)
        print(f"⏱️ Makespan: {makespan}")

    except Exception as e:
        print(f"{e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python checker.py instance.txt solution.txt")
    else:
        main(sys.argv[1], sys.argv[2])
