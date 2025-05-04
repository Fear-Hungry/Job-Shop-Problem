import random
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any  # Adicionado typing

# Importa classes base e dependências necessárias
from ..genetic_operators import MutationStrategy, LocalSearchStrategy
# from ..graph.dsu import DSU # Descomentar se DSU for usado diretamente
# from ..graph import GraphBuilder # Ajustar caminho se necessário


class StandardMutation(MutationStrategy):
    def __init__(self, local_search_strategy: Optional[LocalSearchStrategy] = None):
        self.local_search_strategy = local_search_strategy

    def mutate(self, chromosome, machine_ops=None, graph_builder=None, use_dsu=False, dsu=None):
        op = random.choice(['swap', 'inversion', 'scramble'])
        chrom = chromosome[:]
        if op == 'swap':
            if len(chrom) < 2:
                return chrom
            a, b = random.sample(range(len(chrom)), 2)
            chrom[a], chrom[b] = chrom[b], chrom[a]
        elif op == 'inversion':
            if len(chrom) < 2:
                return chrom
            a, b = sorted(random.sample(range(len(chrom)), 2))
            chrom[a:b] = list(reversed(chrom[a:b]))
        elif op == 'scramble':
            if len(chrom) < 3:
                return chrom
            a, b = sorted(random.sample(range(len(chrom)), 2))
            if b == a:
                return chrom
            sub = chrom[a:b]
            random.shuffle(sub)
            chrom[a:b] = sub
        if self.local_search_strategy:
            if hasattr(self.local_search_strategy, 'local_search'):
                return self.local_search_strategy.local_search(chrom)
        return chrom


class DisjunctiveMutation(MutationStrategy):
    def __init__(self, local_search_strategy: Optional[LocalSearchStrategy] = None):
        self.local_search_strategy = local_search_strategy

    def mutate(self, chromosome, machine_ops: Optional[dict] = None, graph_builder: Optional[Callable] = None, use_dsu=False, dsu=None):
        if machine_ops is None or graph_builder is None:
            # Fallback para mutação padrão se não houver info para disjuntiva
            # print("Aviso: DisjunctiveMutation sem machine_ops/graph_builder, usando Standard Mutation.")
            # Para isso funcionar, precisaríamos de uma instância de StandardMutation aqui
            # ou chamar StandardMutation.mutate como método estático (se fosse possível)
            # Opção mais simples: Retornar o cromossomo ou aplicar uma mutação simples aqui mesmo
            std_mut = StandardMutation()  # Cria instância temporária (sem busca local)
            return std_mut.mutate(chromosome)  # Aplica mutação padrão
            # return chromosome # Alternativa: não fazer nada

        eligible_machines = [
            m for m, ops in machine_ops.items() if len(ops) >= 2]
        if not eligible_machines:
            return chromosome

        machine = random.choice(eligible_machines)
        ops = machine_ops[machine][:]

        a, b = random.sample(range(len(ops)), 2)
        ops[a], ops[b] = ops[b], ops[a]

        new_machine_ops = machine_ops.copy()
        new_machine_ops[machine] = ops
        new_chrom = []
        for m in sorted(new_machine_ops.keys()):
            new_chrom.extend(new_machine_ops[m])

        try:
            graph = graph_builder(new_chrom)
            if graph.has_cycle():
                return chromosome
        except Exception as e:
            # print(f"Erro na validação de grafo em DisjunctiveMutation: {e}")
            return chromosome

        if self.local_search_strategy:
            if hasattr(self.local_search_strategy, 'local_search'):
                return self.local_search_strategy.local_search(new_chrom)

        return new_chrom
