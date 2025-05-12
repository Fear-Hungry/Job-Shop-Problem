import collections

# --- Static Helper Function for Schedule Calculation ---
def _calculate_schedule_details_static(chromosome, jobs, num_jobs, num_machines):
    """
    Static helper function to calculate schedule details (operations list and makespan).
    Can be called from different processes without pickling the entire Solver object.

    Args:
        chromosome (list[tuple[int, int]]): Operation sequence.
        jobs (list[list[tuple[int, int]]]): Job definitions.
        num_jobs (int): Number of jobs.
        num_machines (int): Number of machines.

    Returns:
        tuple[list | None, float]: A tuple containing:
            - list: List of scheduled operations (job, op, machine, start, duration)
                    or None if the chromosome is invalid.
            - float: Calculated makespan, or float('inf') if invalid.
    """
    num_ops_total = sum(len(j) for j in jobs)
    scheduled_ops_list = []

    if not chromosome or len(chromosome) != num_ops_total:
        # print(f"Static Helper: Chromosome length mismatch or empty.")
        return None, float('inf')

    # Validation
    op_counts = {}
    for job_id, op_id in chromosome:
        op_counts[(job_id, op_id)] = op_counts.get((job_id, op_id), 0) + 1
    
    # Construct expected_ops based on the actual structure of jobs data
    expected_ops = set()
    for j_idx, job_list in enumerate(jobs):
        for op_idx_in_job in range(len(job_list)):
            expected_ops.add((j_idx, op_idx_in_job))
            
    actual_ops = set(op_counts.keys())
    if actual_ops != expected_ops or any(count != 1 for count in op_counts.values()):
        # print(f"Static Helper: Invalid chromosome structure.")
        return None, float('inf')

    # Simulation Setup
    machine_available = [0] * num_machines
    job_completion_time = [0] * num_jobs
    job_next_op_idx = [0] * num_jobs # Tracks which operation of a job is next *expected*
    op_priority = {op: i for i, op in enumerate(chromosome)}
    
    ready_operations = set()
    for j in range(num_jobs):
        if len(jobs[j]) > 0:
            ready_operations.add((j, 0)) # (job_id, op_id_within_job)

    num_scheduled = 0
    current_makespan = 0.0

    while num_scheduled < num_ops_total:
        best_op_to_schedule = None
        min_earliest_start_time = float('inf')
        best_priority_val = float('inf')

        if not ready_operations:
            # print("Static Helper Error: No ready operations but not all scheduled.")
            return None, float('inf') 

        for job_id, op_id in ready_operations:
            # op_id is the index within the specific job's list of operations
            machine_id, duration = jobs[job_id][op_id]
            
            earliest_start_time = max(job_completion_time[job_id], machine_available[machine_id])
            current_priority_for_op = op_priority[(job_id, op_id)]

            if earliest_start_time < min_earliest_start_time:
                min_earliest_start_time = earliest_start_time
                best_priority_val = current_priority_for_op
                best_op_to_schedule = (job_id, op_id)
            elif earliest_start_time == min_earliest_start_time and current_priority_for_op < best_priority_val:
                best_priority_val = current_priority_for_op
                best_op_to_schedule = (job_id, op_id)

        if best_op_to_schedule is None:
            # print("Static Helper Error: Could not select an operation.")
            return None, float('inf')

        j_sched, k_sched = best_op_to_schedule # k_sched is op_id_within_job
        m_sched, d_sched = jobs[j_sched][k_sched]
        start_time_sched = min_earliest_start_time 
        finish_time_sched = start_time_sched + d_sched

        scheduled_ops_list.append((j_sched, k_sched, m_sched, start_time_sched, d_sched))
        num_scheduled += 1

        machine_available[m_sched] = finish_time_sched
        job_completion_time[j_sched] = finish_time_sched
        current_makespan = max(current_makespan, finish_time_sched)
        job_next_op_idx[j_sched] += 1 # This job has completed one more operation

        ready_operations.remove((j_sched, k_sched))

        next_op_for_this_job = job_next_op_idx[j_sched]
        if next_op_for_this_job < len(jobs[j_sched]):
            ready_operations.add((j_sched, next_op_for_this_job))

    if num_scheduled != num_ops_total:
        # print(f"Static Helper Error: Scheduled op count mismatch.")
        return None, float('inf')
    
    # Ensure makespan is correct using max of machine available times too
    final_makespan_check = max(machine_available) if machine_available else 0
    # It could be that current_makespan (max of op finish times) is already correct
    # but using max(machine_available) is a robust way to get it.

    return scheduled_ops_list, final_makespan_check

def _calculate_makespan_static(chromosome, jobs, num_jobs, num_machines):
    """Static helper to calculate only the makespan using the detailed function."""
    _, makespan = _calculate_schedule_details_static(chromosome, jobs, num_jobs, num_machines)
    return makespan 