from .base import CrossoverStrategy
import random

class OrderCrossover(CrossoverStrategy):
    def __init__(self, local_search_strategy=None):
        super().__init__(local_search_strategy)

    def crossover(self, parent1, parent2):
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None]*size
        child[a:b] = parent1[a:b]
        fill = [gene for gene in parent2 if gene not in child[a:b]]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        if self.local_search_strategy:
            return self.local_search_strategy.local_search(child)
        return child

class PMXCrossover(CrossoverStrategy):
    def __init__(self, local_search_strategy=None):
        super().__init__(local_search_strategy)

    def crossover(self, parent1, parent2):
        size = len(parent1)
        if len(set(parent1)) != size or len(set(parent2)) != size:
            return parent1
        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        p1_val_to_idx = {val: i for i, val in enumerate(parent1)}
        p2_val_to_idx = {val: i for i, val in enumerate(parent2)}
        child[a:b] = parent1[a:b]
        child_segment_set = set(child[a:b])
        for i in range(a, b):
            p2_val = parent2[i]
            if p2_val not in child_segment_set:
                current_val_from_p1 = parent1[i]
                target_idx = p2_val_to_idx[current_val_from_p1]
                while a <= target_idx < b:
                    current_val_from_p1 = parent1[target_idx]
                    target_idx = p2_val_to_idx[current_val_from_p1]
                child[target_idx] = p2_val
        child_current_elements = set(filter(None, child))
        p2_elements_not_in_child = [elem for elem in parent2 if elem not in child_current_elements]
        idx_fill = 0
        for i in range(size):
            if child[i] is None:
                if idx_fill < len(p2_elements_not_in_child):
                    child[i] = p2_elements_not_in_child[idx_fill]
                    idx_fill += 1
        if len(set(filter(None, child))) != len(list(filter(None, child))) or None in child:
            return parent1
        if self.local_search_strategy:
            return self.local_search_strategy.local_search(child)
        return child

class CycleCrossover(CrossoverStrategy):
    def __init__(self, local_search_strategy=None):
        super().__init__(local_search_strategy)

    def crossover(self, parent1, parent2):
        size = len(parent1)
        child = [None] * size
        cycles = [0] * size
        cycle = 1
        idx = 0
        while None in child:
            if child[idx] is not None:
                idx = child.index(None)
            start = idx
            while True:
                child[idx] = parent1[idx] if cycle % 2 == 1 else parent2[idx]
                cycles[idx] = cycle
                idx = parent1.index(parent2[idx])
                if idx == start:
                    break
            cycle += 1
        if self.local_search_strategy:
            return self.local_search_strategy.local_search(child)
        return child

class PositionBasedCrossover(CrossoverStrategy):
    def __init__(self, local_search_strategy=None):
        super().__init__(local_search_strategy)

    def crossover(self, parent1, parent2):
        size = len(parent1)
        positions = sorted(random.sample(range(size), random.randint(1, size//2)))
        child = [None] * size
        for pos in positions:
            child[pos] = parent1[pos]
        fill = [gene for gene in parent2 if gene not in [child[p] for p in positions]]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        if self.local_search_strategy:
            return self.local_search_strategy.local_search(child)
        return child

class DisjunctiveCrossover(CrossoverStrategy):
    def __init__(self, local_search_strategy=None):
        super().__init__(local_search_strategy)

    def crossover(self, parent1, parent2, machine_ops_builder, graph_builder, use_dsu=False, dsu=None):
        child_machine_ops = {}
        all_machine_ids = machine_ops_builder(parent1).keys()
        if not all_machine_ids:
            all_machine_ids = range(len(machine_ops_builder(parent1)))

        for m in all_machine_ids:
            ops1 = machine_ops_builder(parent1).get(m, [])
            ops2 = machine_ops_builder(parent2).get(m, [])
            
            if not ops1 and not ops2:
                child_machine_ops[m] = []
                continue
            elif not ops1:
                child_machine_ops[m] = ops2[:]
                continue
            elif not ops2:
                child_machine_ops[m] = ops1[:]
                continue
                
            size = len(ops1)
            if len(ops2) != size:
                child_machine_ops[m] = ops1[:]
                continue

            a, b = sorted(random.sample(range(size), 2))
            child_ops_for_machine = [None]*size
            child_ops_for_machine[a:b] = ops1[a:b]
            
            fill_ops = [op for op in ops2 if op not in child_ops_for_machine[a:b]]
            
            idx = 0
            for i in range(size):
                if child_ops_for_machine[i] is None:
                    if idx < len(fill_ops):
                        child_ops_for_machine[i] = fill_ops[idx]
                        idx += 1
                    else:
                        pass
            
            child_machine_ops[m] = child_ops_for_machine
        
        new_chrom = []
        for m_id in sorted(child_machine_ops.keys()):
            new_chrom.extend(child_machine_ops[m_id])

        graph = graph_builder(new_chrom)
        
        if graph.has_cycle():
            return parent1

        if self.local_search_strategy:
            return self.local_search_strategy.local_search(new_chrom)
        
        return new_chrom
