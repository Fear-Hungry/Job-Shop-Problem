import random
from typing import List, Tuple
from .simple_operators import BaseNeighborhoodOperator

class BlockMoveOperator(BaseNeighborhoodOperator):
    """
    Move um bloco contínuo de operações para outra posição do cromossomo.
    Útil para grandes perturbações locais, permitindo saltos maiores no espaço de busca.
    """
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        size = len(chrom)
        if size < 2:
            return chrom[:]
        a = self.rng.randint(0, size - 1)
        b = self.rng.randint(a + 1, size)
        block = chrom[a:b]
        remaining = chrom[:a] + chrom[b:]
        k = self.rng.randint(0, len(remaining))
        new_chrom = remaining[:k] + block + remaining[k:]
        return new_chrom

class BlockSwapOperator(BaseNeighborhoodOperator):
    """
    Troca dois blocos não sobrepostos do cromossomo.
    Permite grandes mudanças estruturais, útil para escapar de ótimos locais profundos.
    """
    def apply(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        size = len(chrom)
        if size < 2:
            return chrom[:]
        idx1_b1, idx2_b1 = sorted(self.rng.sample(range(size + 1), 2))
        a, b = idx1_b1, idx2_b1
        if a == b:
            return chrom[:]
        block1 = chrom[a:b]
        possible_c_starts = []
        if a > 0:
            for c_candidate_len in range(1, a + 1):
                for c_candidate_start in range(a - c_candidate_len + 1):
                    possible_c_starts.append((c_candidate_start, c_candidate_start + c_candidate_len))
        if b < size:
            for c_candidate_len in range(1, (size - b) + 1):
                for c_candidate_start in range(b, (size - c_candidate_len) + 1):
                    possible_c_starts.append((c_candidate_start, c_candidate_start + c_candidate_len))
        if not possible_c_starts:
            return chrom[:]
        c, d = self.rng.choice(possible_c_starts)
        block2 = chrom[c:d]
        if d <= a:
            final_chrom = chrom[:c] + block1 + chrom[d:a] + block2 + chrom[b:]
        elif b <= c:
            final_chrom = chrom[:a] + block2 + chrom[b:c] + block1 + chrom[d:]
        else:
            return chrom[:]
        return final_chrom 