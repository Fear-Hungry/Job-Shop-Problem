import os


def write_output(solution, instance_path):
    """
    Writes a job shop scheduling solution to a file in the "output" directory.

    The output filename is based on the instance file name, with the extension
    replaced by ".out.txt". The solution is expected to be a list of operations,
    where each operation is represented as a tuple:
    (job_id, operation_index, machine_id, start_time, duration).

    Args:
        solution (list[tuple[int, int, int, int, int]]):
            The solution to write, where each tuple represents:
            (job_id, op_index, machine, start_time, duration).
        instance_path (str):
            The file path of the instance, used to generate the output filename.

    Output Format:
        Each line of the output file will contain:
            job_id op_index machine start_time duration
        (separated by spaces)

    Example Output:
        0 0 1 0 3
        1 0 2 1 2
        0 1 0 3 4
    """
    instance_name = os.path.basename(instance_path).split(".")[0]

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{instance_name}.out.txt")

    with open(output_path, "w") as f:
        for job_id, op_idx, machine, start, duration in solution:
            f.write(f"{job_id} {op_idx} {machine} {start} {duration}\n")

    print(f"Output written to {output_path}")
