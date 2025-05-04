import random
from .base import MutationStrategy
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any  # Adicionado typing

# Importa classes base e dependências necessárias
from ..genetic_operators import LocalSearchStrategy
# from ..graph.dsu import DSU # Descomentar se DSU for usado diretamente
# from ..graph import GraphBuilder # Ajustar caminho se necessário


class StandardMutation(MutationStrategy):
    """
    Implementação da estratégia de mutação padrão por troca (swap).
    """

    def mutate(self, chromosome, **kwargs):
        """
        Realiza a mutação por troca (swap) no cromossomo.

        Args:
            chromosome: Cromossomo a ser mutado
            **kwargs: Argumentos adicionais

        Returns:
            Cromossomo mutado
        """
        # Cria uma cópia do cromossomo para não modificar o original
        mutated = chromosome.copy()
        size = len(mutated)

        if size < 2:
            return mutated

        # Seleciona dois índices aleatórios para trocar
        idx1, idx2 = random.sample(range(size), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]

        # Opcionalmente, aplica busca local
        if hasattr(self, 'local_search_strategy') and self.local_search_strategy:
            return self.local_search_strategy.local_search(mutated)
        return mutated


class DisjunctiveMutation(MutationStrategy):
    """
    Implementação da estratégia de mutação disjuntiva específica para problemas job shop.
    """

    def mutate(self, chromosome, machine_ops=None, graph_builder=None, dsu=None, **kwargs):
        """
        Realiza a mutação disjuntiva no cromossomo.

        Args:
            chromosome: Cromossomo a ser mutado
            machine_ops: Mapeamento de máquinas para suas operações
            graph_builder: Função para construir o grafo disjuntivo
            dsu: Estrutura de dados DSU para verificação de ciclos
            **kwargs: Argumentos adicionais

        Returns:
            Cromossomo mutado
        """
        # Verifica se os argumentos necessários foram fornecidos
        if machine_ops is None or graph_builder is None:
            # Se não tiver os argumentos necessários, faz uma mutação padrão
            standard_mutation = StandardMutation()
            if hasattr(self, 'local_search_strategy'):
                standard_mutation.local_search_strategy = self.local_search_strategy
            return standard_mutation.mutate(chromosome)

        # Cria uma cópia do cromossomo para não modificar o original
        mutated = chromosome.copy()

        # Seleciona uma máquina aleatória
        machines = list(machine_ops.keys())
        if not machines:
            return mutated

        machine_id = random.choice(machines)
        machine_operations = machine_ops[machine_id]

        # Se a máquina tem menos de 2 operações, não há como trocar
        if len(machine_operations) < 2:
            return mutated

        # Seleciona duas posições aleatórias para trocar na sequência da máquina
        idx1, idx2 = random.sample(range(len(machine_operations)), 2)

        # Obtém as operações a serem trocadas
        op1 = machine_operations[idx1]
        op2 = machine_operations[idx2]

        # Encontra as posições dessas operações no cromossomo original
        pos1 = mutated.index(op1)
        pos2 = mutated.index(op2)

        # Realiza a troca
        mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]

        # Verifica se a mutação gerou um ciclo usando o DSU ou o graph_builder
        if dsu is not None:
            # Reinicia o DSU apenas para as posições afetadas
            dsu.reset_partial([pos1, pos2])
            # Adiciona as arestas do grafo disjuntivo
            # Se criar um ciclo, desfaz a mutação
            if not dsu.union(pos1, pos2):  # Se não puder unir, há um ciclo
                # Desfaz
                mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
                return mutated
        else:
            # Verifica usando o graph_builder
            graph = graph_builder(mutated)
            if graph.has_cycle():
                # Desfaz
                mutated[pos1], mutated[pos2] = mutated[pos2], mutated[pos1]
                return mutated

        # Opcionalmente, aplica busca local
        if hasattr(self, 'local_search_strategy') and self.local_search_strategy:
            return self.local_search_strategy.local_search(mutated)
        return mutated
