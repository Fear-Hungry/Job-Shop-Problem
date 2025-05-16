import random
from abc import ABC, abstractmethod
from typing import List, Tuple
import logging
from .simple_operators import SwapOperator, InversionOperator, ScrambleOperator, TwoOptOperator, ThreeOptOperator
from .block_operators import BlockMoveOperator, BlockSwapOperator

logger = logging.getLogger(__name__)

class BaseNeighborhoodOperator(ABC):
    """Classe base abstrata para operadores de vizinhança."""
    def __init__(self, rng: random.Random):
        self.rng = rng

    @abstractmethod
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Aplica o operador de vizinhança ao cromossomo.

        Args:
            chrom: O cromossomo (lista de tuplas (job_id, op_id)).

        Returns:
            Um novo cromossomo (lista) resultante da aplicação do operador.
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
