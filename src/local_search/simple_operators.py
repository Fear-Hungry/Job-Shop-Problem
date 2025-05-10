import random
from typing import List, Tuple
from .base import LocalSearchStrategy
from abc import ABC, abstractmethod

class BaseNeighborhoodOperator(ABC):
    """
    Classe base abstrata para operadores de vizinhança simples.
    """
    def __init__(self, rng: random.Random):
        self.rng = rng

    @abstractmethod
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Aplica o operador de vizinhança ao cromossomo.
        Args:
            chrom: O cromossomo (lista de tuplas (job_id, op_id)).
        Returns:
            Um novo cromossomo (lista) resultante da aplicação do operador.
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__

class SwapOperator(BaseNeighborhoodOperator):
    """
    Operador que troca dois elementos aleatórios do cromossomo.
    Útil para explorar vizinhanças simples e diversificar a busca.
    """
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        size = len(chrom)
        if size < 2:
            return chrom[:]
        new_chrom = chrom[:]
        a, b = self.rng.sample(range(size), 2)
        new_chrom[a], new_chrom[b] = new_chrom[b], new_chrom[a]
        return new_chrom

class InversionOperator(BaseNeighborhoodOperator):
    """
    Operador que inverte um segmento aleatório do cromossomo.
    Útil para explorar vizinhanças de inversão, podendo escapar de ótimos locais simples.
    """
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        size = len(chrom)
        if size < 2:
            return chrom[:]
        new_chrom = chrom[:]
        a, b = sorted(self.rng.sample(range(size), 2))
        if a == b:
            return new_chrom
        new_chrom[a:b+1] = list(reversed(new_chrom[a:b+1]))
        return new_chrom

class ScrambleOperator(BaseNeighborhoodOperator):
    """
    Operador que embaralha um segmento aleatório do cromossomo.
    Útil para diversificação local sem alterar toda a estrutura global.
    """
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        size = len(chrom)
        if size < 2:
            return chrom[:]
        new_chrom = chrom[:]
        a, b = sorted(self.rng.sample(range(size), 2))
        if b == a:
            return new_chrom
        sub_segment = new_chrom[a:b+1]
        if len(sub_segment) < 2:
            return new_chrom
        self.rng.shuffle(sub_segment)
        new_chrom[a:b+1] = sub_segment
        return new_chrom

class TwoOptOperator(BaseNeighborhoodOperator):
    """
    Operador 2-opt: inverte um segmento, útil para eliminar cruzamentos e melhorar sequências.
    Muito usado em problemas de roteamento e sequenciamento.
    """
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        size = len(chrom)
        if size < 2:
            return chrom[:]
        i, k = sorted(self.rng.sample(range(size), 2))
        if i == k:
            return chrom[:]
        new_chrom = chrom[:]
        segment_to_reverse = new_chrom[i:k+1]
        segment_to_reverse.reverse()
        new_chrom[i:k+1] = segment_to_reverse
        return new_chrom

class ThreeOptOperator(BaseNeighborhoodOperator):
    """
    Operador 3-opt: realiza trocas e inversões entre três segmentos distintos.
    Permite saltos maiores no espaço de busca, útil para escapar de ótimos locais mais complexos.
    """
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        size = len(chrom)
        if size < 3:
            return chrom[:]
        try:
            a, b, c = sorted(self.rng.sample(range(size + 1), 3))
        except ValueError:
            return chrom[:]
        s1 = chrom[:a]
        s2 = chrom[a:b]
        s3 = chrom[b:c]
        s4 = chrom[c:]
        if not s2 and not s3:
            return chrom[:]
        s2_reversed = list(reversed(s2))
        s3_reversed = list(reversed(s3))
        possible_moves = []
        if s2: possible_moves.append(s1 + s2_reversed + s3 + s4)
        if s3: possible_moves.append(s1 + s2 + s3_reversed + s4)
        if s2 and s3: possible_moves.append(s1 + s2_reversed + s3_reversed + s4)
        if s2 and s3: possible_moves.append(s1 + s3 + s2 + s4)
        if s2 and s3: possible_moves.append(s1 + s3_reversed + s2 + s4)
        if s2 and s3: possible_moves.append(s1 + s3 + s2_reversed + s4)
        if s2 and s3: possible_moves.append(s1 + s3_reversed + s2_reversed + s4)
        if not possible_moves:
            return chrom[:]
        return self.rng.choice(possible_moves) 