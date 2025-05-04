from abc import ABC, abstractmethod


class CrossoverStrategy(ABC):
    """
    Classe base abstrata para implementação de estratégias de crossover.
    """

    def __init__(self, local_search_strategy=None):
        """
        Inicializa a estratégia de crossover.

        Args:
            local_search_strategy: Estratégia de busca local opcional para refinar o resultado
        """
        self.local_search_strategy = local_search_strategy

    @abstractmethod
    def crossover(self, parent1, parent2, **kwargs):
        """
        Realiza o crossover entre dois pais para gerar um ou mais filhos.

        Args:
            parent1: Primeiro cromossomo pai
            parent2: Segundo cromossomo pai
            **kwargs: Argumentos adicionais específicos da implementação

        Returns:
            Um ou mais filhos gerados pelo crossover
        """
        pass
