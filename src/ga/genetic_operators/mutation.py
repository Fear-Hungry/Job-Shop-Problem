from .base import MutationStrategy
import random

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
