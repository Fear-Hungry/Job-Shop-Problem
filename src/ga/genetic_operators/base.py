from abc import ABC, abstractmethod

class CrossoverStrategy(ABC):
    """
    Classe base abstrata para implementação de estratégias de crossover.
    """
    def __init__(self, local_search_strategy=None):
        self.local_search_strategy = local_search_strategy

    @abstractmethod
    def crossover(self, parent1, parent2, **kwargs):
        """
        Realiza o crossover entre dois pais para gerar um ou mais filhos.
        """
        pass

class MutationStrategy(ABC):
    """
    Classe base abstrata para implementação de estratégias de mutação.
    """
    def __init__(self, local_search_strategy=None):
        self.local_search_strategy = local_search_strategy

    @abstractmethod
    def mutate(self, chromosome, **kwargs):
        """
        Realiza a mutação em um cromossomo.
        """
        pass

class LocalSearchStrategy(ABC):
    @abstractmethod
    def local_search(self, chromosome, **kwargs):
        """
        Aplica busca local a um cromossomo para tentar melhorá-lo.
        """
        pass
