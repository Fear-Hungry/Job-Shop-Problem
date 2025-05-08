import random
from abc import ABC, abstractmethod
from typing import List, Tuple
import logging

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
        if size < 3: # 3-opt requer pelo menos 3 itens para escolher 3 pontos de corte diferentes.
                     # Se for menor que 3, ou mesmo 4 dependendo da interpretação de "segmento", não faz sentido.
                     # Para ter 3 segmentos não vazios entre 4 pontos (incluindo início e fim), precisa de chrom >= 3.
            return chrom[:]

        # Seleciona 3 pontos de corte distintos i, j, k. Os segmentos são S1=[0,i-1], S2=[i,j-1], S3=[j,k-1], S4=[k,size-1]
        # Ou, se i,j,k são os índices dos elementos *após* os quais quebramos:
        # Quebra após chrom[i], chrom[j], chrom[k]
        # S1=chrom[0...i], S2=chrom[i+1...j], S3=chrom[j+1...k], S4=chrom[k+1...N-1]

        # A implementação original selecionava a,b,c como índices e os segmentos eram chrom[:a], chrom[a:b], chrom[b:c], chrom[c:]
        # Isso significa que a,b,c são pontos de início de segmentos (exclusive para o final do anterior).
        # Para ter s2 e s3 não vazios, precisamos b > a e c > b.
        if size < 3: # Garante que sample(range(size), 3) funcione
            return chrom[:]

        try:
            a, b, c = sorted(self.rng.sample(range(size +1), 3)) # Pontos de corte de 0 a size
        except ValueError: # Not enough items to sample (size < 3)
            return chrom[:]

        s1 = chrom[:a]
        s2 = chrom[a:b]
        s3 = chrom[b:c]
        s4 = chrom[c:]

        # A lógica original não garantia que s2 ou s3 fossem não-vazios, mas tinha um `if not s2 or not s3: return chrom`
        # Se a,b,c podem ser iguais, ou adjacentes, os segmentos podem ser vazios.
        # Ex: a=0, b=0, c=1 => s1=[], s2=[], s3=chrom[0:1], s4=chrom[1:]
        # Para 3-opt significativo, s2 e s3 (os segmentos do meio que são manipulados) devem existir.
        # A lógica original é mais simples: se s2 ou s3 estiverem vazios, retorna original.
        if not s2 and not s3: # Se ambos são vazios, não há muito o que fazer. Se um é vazio, ainda pode ser como 2-opt.
                              # A lógica original era `if not s2 or not s3: return chrom`
            return chrom[:] # Simplificado: se os segmentos principais de troca são vazios.

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

        # Casos onde S4 é movido, etc., são mais complexos e geralmente cobertos por 4-opt ou combinações.
        # A lista original de 7 movimentos era:
        # s1 + s2_reversed + s3 + s4,
        # s1 + s2 + s3_reversed + s4,
        # s1 + s2_reversed + s3_reversed + s4,
        # s1 + s3 + s2 + s4,
        # s1 + s3 + s2_reversed + s4,
        # s1 + s3_reversed + s2 + s4,
        # s1 + s3_reversed + s2_reversed + s4
        # Todos estes dependem de s2 e s3 existirem e serem manipuláveis.

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
        if size < 2: # Precisa de pelo menos dois elementos para poder trocar (mesmo que blocos de tamanho 1)
            return chrom[:]

        # 1. Selecionar o primeiro bloco [a, b)
        # Para garantir bloco de tamanho >= 1: a de 0 a size-1, b de a+1 a size
        a = self.rng.randint(0, size - 1)
        b = self.rng.randint(a + 1, size)
        block1 = chrom[a:b]
        block1_len = len(block1)

        # Criar lista de índices não pertencentes ao bloco 1
        remaining_indices = [i for i in range(size) if not (a <= i < b)]
        if not remaining_indices or len(remaining_indices) < 1:
            # Não há elementos suficientes fora do bloco 1 para formar um bloco 2 de tamanho >=1
            return chrom[:]

        # 2. Selecionar o segundo bloco [c, d) a partir dos índices restantes
        # c_start_index_in_remaining: índice em remaining_indices para o início do bloco 2
        # d_end_index_in_remaining: índice em remaining_indices para o fim do bloco 2

        # Tamanho do bloco 2: de 1 até len(remaining_indices)
        block2_len = self.rng.randint(1, len(remaining_indices))

        # Ponto de início do bloco 2 nos remaining_indices
        c_start_idx_in_rem = self.rng.randint(0, len(remaining_indices) - block2_len)
        block2_indices_in_rem = remaining_indices[c_start_idx_in_rem : c_start_idx_in_rem + block2_len]

        block2 = [chrom[i] for i in block2_indices_in_rem]

        # Agora precisamos reconstruir o cromossomo. A ordem relativa dos elementos
        # que não estão em block1 nem em block2 deve ser preservada.

        # Esta lógica é mais complexa que a original que assumia blocos contíguos e não sobrepostos.
        # A lógica original era para blocos contíguos e não sobrepostos no cromossomo original.
        # Vou reimplementar a lógica original que era mais simples e provavelmente a intenção.

        # Reset para lógica original (mais simples):
        size = len(chrom)
        if size < 2: return chrom[:]

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

        new_chrom = [None] * size
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
