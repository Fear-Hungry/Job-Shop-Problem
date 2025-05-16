from abc import ABC, abstractmethod

# Estratégia de Crossover Base


class CrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1, parent2, **kwargs):
        """Realiza o cruzamento entre dois pais para gerar um ou mais filhos."""
        pass

# Estratégia de Mutação Base


class MutationStrategy(ABC):
    @abstractmethod
    def mutate(self, chromosome, **kwargs):
        """Realiza a mutação em um cromossomo."""
        pass

# Estratégia de Busca Local Base


class LocalSearchStrategy(ABC):
    @abstractmethod
    def local_search(self, chromosome, **kwargs):
        """Aplica uma estratégia de busca local para melhorar um cromossomo."""
        pass
