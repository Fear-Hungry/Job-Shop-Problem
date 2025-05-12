import collections
import heapq # Importar heapq

# --- Static Helper Function for Schedule Calculation ---
def _calculate_schedule_details_static(chromosome, jobs, num_jobs, num_machines):
    """
    Static helper function to calculate schedule details (operations list and makespan)
    using a min-heap for efficient selection of the next operation.

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

    # --- Simulation Setup using Heap --- 
    machine_available = [0] * num_machines
    job_completion_time = [0] * num_jobs
    job_next_op_idx = [0] * num_jobs # Tracks the *index* of the next operation for each job
    op_priority = {op: i for i, op in enumerate(chromosome)} # Priority based on chromosome order
    
    ready_heap = [] # Min-heap: (earliest_start_time, priority_value, job_id, op_id_in_job)
    
    # Initialize heap with the first operation of each job
    for j in range(num_jobs):
        if len(jobs[j]) > 0:
            op_id_in_job = 0
            job_id = j
            priority = op_priority[(job_id, op_id_in_job)]
            # Initial earliest start is 0, as job_completion and machine_available are 0
            heapq.heappush(ready_heap, (0, priority, job_id, op_id_in_job))

    num_scheduled = 0
    current_makespan = 0.0

    processed_in_this_step = set() # Avoid infinite loops if heap logic fails

    while num_scheduled < num_ops_total:
        if not ready_heap:
            # Should not happen if logic is correct and graph is valid
            # print("Error: Ready heap is empty but not all operations scheduled.")
            # Check if stuck due to some issue
            if num_scheduled < num_ops_total:
                 # If truly stuck, return invalid
                 # print(f"Stuck: Scheduled {num_scheduled}/{num_ops_total}")
                 return None, float('inf') 
            else: # All scheduled, break normally (though loop condition should handle this)
                 break

        # Pop the operation with the lowest start time / priority
        potential_est, priority_val, job_id, op_id = heapq.heappop(ready_heap)

        # --- Check if the popped operation is truly ready --- 
        machine_id, duration = jobs[job_id][op_id]
        actual_earliest_start = max(job_completion_time[job_id], machine_available[machine_id])

        if actual_earliest_start > potential_est:
            # The start time has shifted due to other operations completing.
            # Re-push with the updated actual earliest start time.
            heapq.heappush(ready_heap, (actual_earliest_start, priority_val, job_id, op_id))
            # Continue to the next iteration to pop the new best candidate
            continue 
            
        # --- Schedule the operation --- 
        start_time_sched = actual_earliest_start # Use the actual calculated start time
        finish_time_sched = start_time_sched + duration

        scheduled_ops_list.append((job_id, op_id, machine_id, start_time_sched, duration))
        num_scheduled += 1

        # Update state
        machine_available[machine_id] = finish_time_sched
        job_completion_time[job_id] = finish_time_sched
        current_makespan = max(current_makespan, finish_time_sched)
        job_next_op_idx[job_id] += 1 # Increment index for the *next* operation of this job

        # Add the next operation of this job to the heap if it exists
        next_op_idx_for_job = job_next_op_idx[job_id]
        if next_op_idx_for_job < len(jobs[job_id]):
            next_op_id = next_op_idx_for_job
            next_machine_id, _ = jobs[job_id][next_op_id]
            next_priority = op_priority[(job_id, next_op_id)]
            # Calculate potential earliest start for the next op
            next_potential_est = max(job_completion_time[job_id], machine_available[next_machine_id])
            heapq.heappush(ready_heap, (next_potential_est, next_priority, job_id, next_op_id))

    if num_scheduled != num_ops_total:
        # print(f"Static Helper Error: Scheduled op count mismatch at the end.")
        return None, float('inf')
    
    final_makespan_check = max(machine_available) if machine_available else 0

    return scheduled_ops_list, final_makespan_check

def _calculate_makespan_static(chromosome, jobs, num_jobs, num_machines):
    """Static helper to calculate only the makespan using the detailed function."""
    _, makespan = _calculate_schedule_details_static(chromosome, jobs, num_jobs, num_machines)
    return makespan 