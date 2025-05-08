from abc import ABC, abstractmethod


class LocalSearchStrategy(ABC):
    """
    Classe base abstrata para implementação de estratégias de busca local.
    """

    def __init__(self, fitness_func=None):
        """
        Inicializa a estratégia de busca local.

        Args:
            fitness_func: Função para calcular o fitness de um cromossomo
        """
        self.fitness_func = fitness_func

    @abstractmethod
    def local_search(self, chromosome, **kwargs):
        """
        Aplica uma estratégia de busca local para melhorar um cromossomo.

        Args:
            chromosome: Cromossomo a ser melhorado
            **kwargs: Argumentos adicionais específicos da implementação

        Returns:
            Cromossomo melhorado
        """
        pass
