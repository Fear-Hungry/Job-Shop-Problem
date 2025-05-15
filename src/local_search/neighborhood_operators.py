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


class SwapOperator(BaseNeighborhoodOperator):
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        size = len(chrom)
        if size < 2:
            return chrom[:]
        new_chrom = chrom[:]
        a, b = self.rng.sample(range(size), 2)
        new_chrom[a], new_chrom[b] = new_chrom[b], new_chrom[a]
        return new_chrom


class InversionOperator(BaseNeighborhoodOperator):
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        size = len(chrom)
        if size < 2:
            return chrom[:]
        new_chrom = chrom[:]
        a, b = sorted(self.rng.sample(range(size), 2))
        # Certifique-se de que a fatia não seja vazia se a == b após sorted (não deveria acontecer com sample(..., 2))
        if a == b: # Caso extremo, embora sample(size, 2) deva evitar isso para size >=2
             return new_chrom
        new_chrom[a:b+1] = list(reversed(new_chrom[a:b+1])) # b+1 para incluir o elemento b
        return new_chrom


class ScrambleOperator(BaseNeighborhoodOperator):
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        size = len(chrom)
        if size < 2: # Pode embaralhar um segmento de tamanho 1 (não faz nada) ou 0.
            return chrom[:]
        new_chrom = chrom[:]
        # Para Scramble, precisamos de pelo menos dois pontos para definir um segmento que pode ser embaralhado.
        # Se size < 2, sample falhará. Se size == 2, a e b serão 0 e 1.
        if size < 2:
             return new_chrom
        a, b = sorted(self.rng.sample(range(size), 2))
        if b == a: # Segmento de tamanho 0 ou 1, não há o que embaralhar
            return new_chrom

        # O segmento é chrom[a:b+1]
        sub_segment = new_chrom[a:b+1]
        if len(sub_segment) < 2: # Não há o que embaralhar
            return new_chrom

        self.rng.shuffle(sub_segment)
        new_chrom[a:b+1] = sub_segment
        return new_chrom


class TwoOptOperator(BaseNeighborhoodOperator):
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        size = len(chrom)
        if size < 2:
            return chrom[:]
        # Para 2-opt, precisamos de pelo menos 2 pontos para definir um segmento a ser invertido
        # Os pontos i e k são os *inícios* das arestas a serem quebradas.
        # Arestas (chrom[i-1], chrom[i]) e (chrom[k-1], chrom[k])
        # No contexto de sequência, selecionamos dois índices i, k e invertemos chrom[i:k+1]
        i, k = sorted(self.rng.sample(range(size), 2))
        if i == k: # Não há segmento para inverter
            return chrom[:]

        new_chrom = chrom[:]
        segment_to_reverse = new_chrom[i:k+1]
        segment_to_reverse.reverse()
        new_chrom[i:k+1] = segment_to_reverse
        return new_chrom


class ThreeOptOperator(BaseNeighborhoodOperator):
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        size = len(chrom)
        if size < 3:
            return chrom[:]
        

        try:
            a, b, c = sorted(self.rng.sample(range(size +1), 3)) # Pontos de corte de 0 a size
        except ValueError: # Not enough items to sample (size < 3)
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
        # Caso 1: S1 S2 S3 S4 (original, não incluído normalmente)
        # Movimentos 2-opt (1 segmento invertido)
        if s2: possible_moves.append(s1 + s2_reversed + s3 + s4) # Inverte S2
        if s3: possible_moves.append(s1 + s2 + s3_reversed + s4) # Inverte S3
        if s2 and s3: possible_moves.append(s1 + s2_reversed + s3_reversed + s4) # Inverte S2 e S3 (ainda um tipo de 2-opt "duplo")

        # Movimentos 3-opt (reordenamento de 3 segmentos)
        # A B C -> A C B (S1 S2 S3 S4 -> S1 S3 S2 S4)
        if s2 and s3: possible_moves.append(s1 + s3 + s2 + s4)
        # A B C -> A C' B (S1 S3_reversed S2 S4)
        if s2 and s3: possible_moves.append(s1 + s3_reversed + s2 + s4)
        # A B C -> A C B' (S1 S3 S2_reversed S4)
        if s2 and s3: possible_moves.append(s1 + s3 + s2_reversed + s4)
        # A B C -> A C' B' (S1 S3_reversed S2_reversed S4)
        if s2 and s3: possible_moves.append(s1 + s3_reversed + s2_reversed + s4)



        if not possible_moves: # Se nenhum movimento foi gerado (ex: s2 ou s3 vazios)
            return chrom[:]

        return self.rng.choice(possible_moves)


class BlockMoveOperator(BaseNeighborhoodOperator):
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        size = len(chrom)
        if size < 2: # Precisa de pelo menos 1 elemento para mover, e 1 lugar para mover.
                     # Se size=1, block_len=1, remaining_len=0, k=0. Insere em si mesmo. OK.
                     # Se size=0, não faz nada.
            return chrom[:]

        # Seleciona o bloco [a, b)
        # Para que o bloco tenha pelo menos tamanho 1, b > a.
        if size == 0: return chrom[:]
        # a pode ser de 0 a size-1. b pode ser de a+1 a size.
        a = self.rng.randint(0, size -1) # Pelo menos um elemento para o bloco começar.
        b = self.rng.randint(a + 1, size) # Bloco de pelo menos tamanho 1.

        block = chrom[a:b]

        remaining = chrom[:a] + chrom[b:]

        # Seleciona a posição de inserção k (0 a len(remaining))
        # Se remaining é vazio (bloco era o cromossomo inteiro), len(remaining) = 0, k = 0.
        k = self.rng.randint(0, len(remaining))

        new_chrom = remaining[:k] + block + remaining[k:]
        return new_chrom


class BlockSwapOperator(BaseNeighborhoodOperator):
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        size = len(chrom)
        if size < 2: # Precisa de pelo menos dois elementos para poder trocar
            return chrom[:]
            
        # Bloco 1: [a,b)
        idx1_b1, idx2_b1 = sorted(self.rng.sample(range(size + 1), 2))
        a, b = idx1_b1, idx2_b1
        if a == b: return chrom[:] # Bloco de tamanho 0
        block1 = chrom[a:b]
        
        # Bloco 2: [c,d) não sobreposto
        possible_c_starts = []
        # Tentar encontrar c,d antes de a
        if a > 0:
            for c_candidate_len in range(1, a + 1):
                for c_candidate_start in range(a - c_candidate_len +1):
                    possible_c_starts.append((c_candidate_start, c_candidate_start + c_candidate_len))
        # Tentar encontrar c,d depois de b
        if b < size:
            for c_candidate_len in range(1, (size - b) + 1):
                for c_candidate_start in range(b, (size - c_candidate_len) +1):
                     possible_c_starts.append((c_candidate_start, c_candidate_start + c_candidate_len))

        if not possible_c_starts:
            return chrom[:] # Não foi possível encontrar um segundo bloco não sobreposto

        c, d = self.rng.choice(possible_c_starts)
        block2 = chrom[c:d]
        
        # Garantir que os blocos não se sobrepõem na lógica de preenchimento
        # Se Bloco 2 (c,d) vem antes de Bloco 1 (a,b)
        if d <= a:
            # Partes: chrom[:c], block2, chrom[d:a], block1, chrom[b:]
            # Trocado: chrom[:c], block1, chrom[d:a], block2, chrom[b:]
            final_chrom = chrom[:c] + block1 + chrom[d:a] + block2 + chrom[b:]
        # Se Bloco 1 (a,b) vem antes de Bloco 2 (c,d)
        elif b <= c:
            # Partes: chrom[:a], block1, chrom[b:c], block2, chrom[d:]
            # Trocado: chrom[:a], block2, chrom[b:c], block1, chrom[d:]
            final_chrom = chrom[:a] + block2 + chrom[b:c] + block1 + chrom[d:]
        else:
            # Sobreposição, o que não deveria acontecer pela seleção de c,d
            logger.error("BlockSwap: Sobreposição inesperada de blocos. Retornando original.")
            return chrom[:]

        return final_chrom
