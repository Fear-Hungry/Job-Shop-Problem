import random
from abc import ABC, abstractmethod
from typing import Optional, Union, Callable, Any
from .base import CrossoverStrategy

# Importar das classes base que ainda estão em genetic_operators.py (temporariamente)
# Assumindo que genetic_operators está um nível acima
from ..genetic_operators import LocalSearchStrategy
# Importar DSU e graph_builder se necessário para DisjunctiveCrossover
# from ..graph.dsu import DSU # Descomentar se DSU for usado diretamente aqui
# from ..graph import GraphBuilder # Ajustar o caminho conforme necessário


class OrderCrossover(CrossoverStrategy):
    """
    Implementação do operador de crossover OX (Order Crossover).
    """

    def crossover(self, parent1, parent2, **kwargs):
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
    """
    Implementação do operador de crossover PMX (Partially Mapped Crossover).
    """

    def crossover(self, parent1, parent2, **kwargs):
        size = len(parent1)
        # Garante que os pais sejam válidos antes de prosseguir
        if len(set(parent1)) != size or len(set(parent2)) != size:
            return parent1

        a, b = sorted(random.sample(range(size), 2))
        child = [None] * size
        # Dicionário para mapeamento reverso (valor -> índice) para performance
        p1_val_to_idx = {val: i for i, val in enumerate(parent1)}
        p2_val_to_idx = {val: i for i, val in enumerate(parent2)}

        # 1. Copia o segmento do primeiro pai
        child[a:b] = parent1[a:b]
        child_segment_set = set(child[a:b])

        # 2. Mapeamento PMX para elementos do segmento de parent2
        for i in range(a, b):
            p2_val = parent2[i]
            if p2_val not in child_segment_set:
                # Segue a cadeia de mapeamento até encontrar um slot vazio fora do segmento
                current_val_from_p1 = parent1[i]
                target_idx = p2_val_to_idx[current_val_from_p1]
                while a <= target_idx < b:
                    current_val_from_p1 = parent1[target_idx]
                    target_idx = p2_val_to_idx[current_val_from_p1]
                # Coloca o valor de p2 no slot encontrado
                child[target_idx] = p2_val

        # 3. Preenche os slots restantes (None) com elementos de parent2 que não foram usados
        child_current_elements = set(filter(None, child))
        p2_elements_not_in_child = [
            elem for elem in parent2 if elem not in child_current_elements]
        idx_fill = 0
        for i in range(size):
            if child[i] is None:
                if idx_fill < len(p2_elements_not_in_child):
                    child[i] = p2_elements_not_in_child[idx_fill]
                    idx_fill += 1

        # Verificação final
        if len(set(filter(None, child))) != len(list(filter(None, child))) or None in child:
            return parent1  # Retorna pai como fallback

        if self.local_search_strategy:
            return self.local_search_strategy.local_search(child)
        return child


class CycleCrossover(CrossoverStrategy):
    """
    Implementação do operador de crossover CX (Cycle Crossover).
    """

    def crossover(self, parent1, parent2, **kwargs):
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
    """
    Implementação do operador de crossover baseado em posição.
    """

    def crossover(self, parent1, parent2, **kwargs):
        size = len(parent1)
        positions = sorted(random.sample(
            range(size), random.randint(1, size//2)))
        child = [None] * size
        # Copia genes das posições escolhidas do primeiro pai
        for pos in positions:
            child[pos] = parent1[pos]
        # Preenche o restante na ordem do segundo pai
        fill = [gene for gene in parent2 if gene not in [child[p]
                                                         for p in positions]]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        if self.local_search_strategy:
            return self.local_search_strategy.local_search(child)
        return child


class DisjunctiveCrossover(CrossoverStrategy):
    """
    Implementação do operador de crossover disjuntivo específico para problemas job shop.
    """

    def crossover(self, parent1, parent2, machine_ops_builder, graph_builder, use_dsu=False, dsu=None, **kwargs):
        # Para cada máquina, faz OX entre as listas de operações
        child_machine_ops = {}
        for m in machine_ops_builder(tuple(parent1)).keys():
            ops1 = machine_ops_builder(tuple(parent1))[m]
            ops2 = machine_ops_builder(tuple(parent2))[m]
            size = len(ops1)
            a, b = sorted(random.sample(range(size), 2))
            child_ops = [None]*size
            child_ops[a:b] = ops1[a:b]
            fill = [op for op in ops2 if op not in child_ops[a:b]]
            idx = 0
            for i in range(size):
                if child_ops[i] is None:
                    child_ops[i] = fill[idx]
                    idx += 1
            child_machine_ops[m] = child_ops

        # Reconstrói cromossomo
        new_chrom = []
        for m in sorted(child_machine_ops.keys()):
            new_chrom.extend(child_machine_ops[m])

        # DSU parcimonioso: reset apenas das operações das máquinas afetadas
        if use_dsu and dsu is not None:
            for m, ops in child_machine_ops.items():
                indices_afetados = [i for i in range(len(ops))]
                dsu.reset_partial(indices_afetados)
                for i in range(len(ops) - 1):
                    dsu.union(i, i+1)
                for i in range(len(ops) - 1):
                    if dsu.connected(i, i+1):
                        return parent1
        elif use_dsu:
            graph = graph_builder(new_chrom, use_dsu=True)
            for m, ops in child_machine_ops.items():
                for i in range(len(ops) - 1):
                    u = new_chrom.index(ops[i])
                    v = new_chrom.index(ops[i+1])
                    if not graph.union(u, v):
                        return parent1

        # Opcionalmente, aplica busca local
        if self.local_search_strategy:
            return self.local_search_strategy.local_search(new_chrom)
        return new_chrom
