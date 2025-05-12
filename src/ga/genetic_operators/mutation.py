from .base import MutationStrategy
import random
from ga.graph.disjunctive_graph import DisjunctiveGraph
from ga.graph.graph_utils import has_path_in_job_graph

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
    def mutate(self, chromosome, machine_ops=None, graph_builder=None, dsu=None, jobs=None, **kwargs):
        if machine_ops is None or graph_builder is None or jobs is None:
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

        try:
            pos1 = mutated.index(op1)
            pos2 = mutated.index(op2)
        except ValueError:
            print(f"Warning: Could not find ops {op1} or {op2} from machine {machine_id} in chromosome during DisjunctiveMutation.")
            return mutated

        first_op_on_machine = op1 if idx1 < idx2 else op2
        second_op_on_machine = op2 if idx1 < idx2 else op1

        if has_path_in_job_graph(second_op_on_machine, first_op_on_machine, jobs):
            return mutated

        mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]

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
            standard_mutation = StandardMutation()
            if hasattr(self, 'local_search_strategy'):
                 standard_mutation.local_search_strategy = self.local_search_strategy
            return standard_mutation.mutate(chromosome)

        op_durations = self._get_op_durations_map(chromosome, jobs)
        graph, critical_path_nodes, num_ops = self._get_graph_and_critical_path(chromosome, graph_builder, op_durations)

        if graph is None or critical_path_nodes is None or len(critical_path_nodes) < 2:
            standard_mutation = StandardMutation()
            if hasattr(self, 'local_search_strategy'):
                 standard_mutation.local_search_strategy = self.local_search_strategy
            return standard_mutation.mutate(chromosome)

        candidates = []
        for i in range(len(critical_path_nodes) - 1):
            u_idx = critical_path_nodes[i]
            v_idx = critical_path_nodes[i+1]
            if v_idx in graph.adj.get(u_idx, set()):
                # Initialize flags for edge type for this iteration
                is_job_edge = False
                is_machine_edge = False

                u_op = graph.idx_to_op[u_idx]
                v_op = graph.idx_to_op[v_idx]

                if u_op[0] == v_op[0]: # Same job -> job precedence
                    is_job_edge = True
                else: # Different job -> must be machine precedence
                    is_machine_edge = True

                # We can only swap if it's a MACHINE precedence edge
                if is_machine_edge:
                    candidates.append((u_idx, v_idx))

        if not candidates:
            standard_mutation = StandardMutation()
            if hasattr(self, 'local_search_strategy'):
                 standard_mutation.local_search_strategy = self.local_search_strategy
            return standard_mutation.mutate(chromosome)

        u_graph_idx, v_graph_idx = random.choice(candidates)

        # Get the corresponding (job, op) tuples using graph's map
        # We now assume graph_builder adds this map to the graph object.
        if not hasattr(graph, 'idx_to_op'):
            # Fallback or error if the builder didn't add the map
            print("Error: graph object missing 'idx_to_op' attribute in CriticalPathSwap. Building manually.")
            # Build it manually as a fallback - this shouldn't happen now
            idx_to_op = {idx: op for idx, op in enumerate(chromosome)}
        else:
             idx_to_op = graph.idx_to_op

        try:
             u_op = idx_to_op[u_graph_idx] # The op that came first on the machine
             v_op = idx_to_op[v_graph_idx] # The op that came second on the machine
        except KeyError:
             print(f"Error: Graph index {u_graph_idx} or {v_graph_idx} not found in idx_to_op map.")
             return chromosome # Return original chromosome

        if has_path_in_job_graph(v_op, u_op, jobs):
            return chromosome

        try:
             u_chrom_idx = chromosome.index(u_op)
             v_chrom_idx = chromosome.index(v_op)
        except ValueError:
             print(f"Error finding critical ops {u_op} or {v_op} in chromosome during swap.")
             return chromosome

        mutated = chromosome.copy()
        mutated[u_chrom_idx], mutated[v_chrom_idx] = mutated[v_chrom_idx], mutated[u_chrom_idx]

        if hasattr(self, 'local_search_strategy') and self.local_search_strategy:
            return self.local_search_strategy.local_search(mutated)

        return mutated

# TODO: Implement CriticalPathInsert and CriticalPath2Opt following similar patterns
# - CriticalPathInsert: Select critical node, find valid insertion points, insert, check cycle.
# - CriticalPath2Opt: Select segment of critical path, reverse it, check cycle.
