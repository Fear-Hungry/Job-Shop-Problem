from abc import ABC, abstractmethod


class MutationStrategy(ABC):
    """
    Classe base abstrata para implementação de estratégias de mutação.
    """

    def __init__(self, local_search_strategy=None):
        """
        Inicializa a estratégia de mutação.

        Args:
            local_search_strategy: Estratégia de busca local opcional para refinar o resultado
        """
        self.local_search_strategy = local_search_strategy

    @abstractmethod
    def mutate(self, chromosome, **kwargs):
        """
        Realiza a mutação em um cromossomo.

        Args:
            chromosome: Cromossomo a ser mutado
            **kwargs: Argumentos adicionais específicos da implementação

        Returns:
            Cromossomo mutado
        """
        pass
