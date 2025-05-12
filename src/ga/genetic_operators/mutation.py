from .base import MutationStrategy
import random
from ga.graph.disjunctive_graph import DisjunctiveGraph # Ensure DisjunctiveGraph is available

class StandardMutation(MutationStrategy):
    def mutate(self, chromosome, **kwargs):
        mutated = chromosome.copy()
        size = len(mutated)
        if size < 2:
            return mutated
        idx1, idx2 = random.sample(range(size), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        if hasattr(self, 'local_search_strategy') and self.local_search_strategy:
            return self.local_search_strategy.local_search(mutated)
        return mutated

class DisjunctiveMutation(MutationStrategy):
    def mutate(self, chromosome, machine_ops=None, graph_builder=None, dsu=None, **kwargs):
        if machine_ops is None or graph_builder is None:
            standard_mutation = StandardMutation()
            if hasattr(self, 'local_search_strategy'):
                standard_mutation.local_search_strategy = self.local_search_strategy
            return standard_mutation.mutate(chromosome)
        mutated = chromosome.copy()
        machines = list(machine_ops.keys())
        if not machines:
            return mutated
        machine_id = random.choice(machines)
        machine_operations = machine_ops[machine_id]
        if len(machine_operations) < 2:
            return mutated
        idx1, idx2 = random.sample(range(len(machine_operations)), 2)
        op1 = machine_operations[idx1]
        op2 = machine_operations[idx2]
        pos1 = mutated.index(op1)
        pos2 = mutated.index(op2)
        mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
        if dsu is not None:
            dsu.reset_partial([pos1, pos2])
            if not dsu.union(pos1, pos2):
                mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
                return mutated
        else:
            graph = graph_builder(mutated)
            if graph.has_cycle():
                mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
                return mutated
        if hasattr(self, 'local_search_strategy') and self.local_search_strategy:
            return self.local_search_strategy.local_search(mutated)
        return mutated

# --- Critical Path Based Mutations ---

class CriticalPathMutationBase(MutationStrategy):
    """Base class for mutations needing the critical path."""
    def _get_graph_and_critical_path(self, chromosome, graph_builder, op_durations, **kwargs):
        """Helper to build graph and get critical path."""
        if graph_builder is None:
            print("Error: graph_builder is required for critical path mutation.")
            return None, None, None
        try:
            # Assumes graph_builder needs only chromosome (and optionally use_dsu=False)
            graph = graph_builder(chromosome, use_dsu=False) # DSU might interfere with graph structure needed here
            critical_path_nodes = graph.get_critical_path(op_durations)
            if not critical_path_nodes:
                # print("Warning: Could not find critical path for mutation.")
                return graph, None, None # Return graph even if path is empty
            return graph, critical_path_nodes, graph.num_ops
        except Exception as e:
            print(f"Error building graph or getting critical path: {e}")
            return None, None, None

    def _get_op_durations_map(self, chromosome, jobs):
         """Creates a map from graph node index to duration."""
         op_to_idx = {op: idx for idx, op in enumerate(chromosome)}
         durations_map = {}
         for job_id, job_ops in enumerate(jobs):
             for op_id_in_job, (_, duration) in enumerate(job_ops):
                 op_tuple = (job_id, op_id_in_job)
                 if op_tuple in op_to_idx:
                     graph_idx = op_to_idx[op_tuple]
                     durations_map[graph_idx] = duration
                 # else: # Should not happen if chromosome is valid
                 #     print(f"Warning: Operation {op_tuple} not found in chromosome map.")
         return durations_map

class CriticalPathSwap(CriticalPathMutationBase):
    """Swaps two adjacent operations on the critical path."""
    def mutate(self, chromosome, graph_builder=None, jobs=None, **kwargs):
        if graph_builder is None or jobs is None:
            # Fallback to standard if needed info is missing
            # print("Fallback: Missing graph_builder or jobs for CriticalPathSwap")
            standard_mutation = StandardMutation()
            if hasattr(self, 'local_search_strategy'):
                 standard_mutation.local_search_strategy = self.local_search_strategy
            return standard_mutation.mutate(chromosome)

        op_durations = self._get_op_durations_map(chromosome, jobs)
        graph, critical_path_nodes, num_ops = self._get_graph_and_critical_path(chromosome, graph_builder, op_durations)

        if graph is None or critical_path_nodes is None or len(critical_path_nodes) < 2:
            # print("Fallback: CP too short or error in graph/CP calc.")
            standard_mutation = StandardMutation()
            if hasattr(self, 'local_search_strategy'):
                 standard_mutation.local_search_strategy = self.local_search_strategy
            return standard_mutation.mutate(chromosome) # Fallback

        # Find adjacent critical operations (u, v) where (u, v) is an edge in the graph
        candidates = []
        for i in range(len(critical_path_nodes) - 1):
            u_idx = critical_path_nodes[i]
            v_idx = critical_path_nodes[i+1]
            # Check if they are directly connected in the graph (critical edge)
            if v_idx in graph.adj[u_idx]:
                candidates.append((u_idx, v_idx))

        if not candidates:
            # print("Fallback: No adjacent critical operations found.")
            standard_mutation = StandardMutation()
            if hasattr(self, 'local_search_strategy'):
                 standard_mutation.local_search_strategy = self.local_search_strategy
            return standard_mutation.mutate(chromosome)

        # Select one adjacent pair to swap
        u_graph_idx, v_graph_idx = random.choice(candidates)

        # Get the corresponding (job, op) tuples from the chromosome
        # Need the original chromosome indexing for this
        idx_to_op = {idx: op for idx, op in enumerate(chromosome)}
        u_op = idx_to_op[u_graph_idx]
        v_op = idx_to_op[v_graph_idx]

        # Find their positions in the chromosome list
        try:
             u_chrom_idx = chromosome.index(u_op)
             v_chrom_idx = chromosome.index(v_op)
        except ValueError:
             print("Error finding critical ops in chromosome during swap.")
             return chromosome # Return original

        mutated = chromosome.copy()
        mutated[u_chrom_idx], mutated[v_chrom_idx] = mutated[v_chrom_idx], mutated[u_chrom_idx]

        # Check if the swap creates a cycle
        # Rebuild graph for the mutated chromosome to check cycle
        # Avoid DSU here as we need the explicit cycle check
        new_graph = graph_builder(mutated, use_dsu=False)
        if new_graph.has_cycle():
            # print("Swap created cycle, reverting.")
            return chromosome # Revert if cycle detected

        # Optional: Apply local search after successful mutation
        if hasattr(self, 'local_search_strategy') and self.local_search_strategy:
            # Pass necessary info if LS needs it, otherwise just the chromosome
            # ls_kwargs = {'jobs': jobs, 'num_machines': len(jobs[0]) if jobs else 0} # Example
            # The local search object should already have the context it needs.
            return self.local_search_strategy.local_search(mutated)

        return mutated

# TODO: Implement CriticalPathInsert and CriticalPath2Opt following similar patterns
# - CriticalPathInsert: Select critical node, find valid insertion points, insert, check cycle.
# - CriticalPath2Opt: Select segment of critical path, reverse it, check cycle.
