from typing import Tuple, List

# Operation tuple format: (job_id, op_id_within_job)
OpTuple = Tuple[int, int]
# Job data format: List[List[Tuple[machine_id, duration]]]
JobData = List[List[Tuple[int, int]]]

def has_path_in_job_graph(op1: OpTuple, op2: OpTuple, jobs: JobData) -> bool:
    """
    Checks if there is a job precedence path from op1 to op2.

    This is true if and only if op1 and op2 belong to the same job,
    and op1 occurs before op2 in that job's sequence.

    Args:
        op1: The first operation (job_id, op_id).
        op2: The second operation (job_id, op_id).
        jobs: The job data structure (unused in this simple check, but kept for potential future graph needs).

    Returns:
        True if op2 follows op1 in the same job, False otherwise.
    """
    job_id1, op_id1 = op1
    job_id2, op_id2 = op2

    # If operations are from different jobs, there's no direct job precedence path
    if job_id1 != job_id2:
        return False

    # If they are the same operation, there's no path from one to the other
    if op_id1 == op_id2:
        return False

    # If they are from the same job, a path exists from op1 to op2 iff op1 comes before op2
    return op_id1 < op_id2 