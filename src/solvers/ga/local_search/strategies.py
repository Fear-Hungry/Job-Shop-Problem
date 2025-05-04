import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any

# Importa classe base
from ..genetic_operators import LocalSearchStrategy


class VNDLocalSearch(LocalSearchStrategy):
    def __init__(self, fitness_func: Callable[[list], float], use_advanced_neighborhoods=False, max_tries_per_neighborhood=10):
        self.fitness_func = fitness_func
        self.use_advanced_neighborhoods = use_advanced_neighborhoods
        self.max_tries_per_neighborhood = max_tries_per_neighborhood

    def _2opt(self, chrom):
        size = len(chrom)
        if size < 2:
            return chrom
        a, b = sorted(random.sample(range(size), 2))
        return chrom[:a] + list(reversed(chrom[a:b])) + chrom[b:]

    def _3opt(self, chrom):
        size = len(chrom)
        if size < 3:
            return chrom
        a, b, c = sorted(random.sample(range(size), 3))
        opt1 = chrom[:a] + list(reversed(chrom[a:b])) + chrom[b:c] + chrom[c:]
        opt2 = chrom[:a] + chrom[a:b] + list(reversed(chrom[b:c])) + chrom[c:]
        opt3 = chrom[:a] + list(reversed(chrom[a:b])) + \
            list(reversed(chrom[b:c])) + chrom[c:]
        opt4 = chrom[:a] + chrom[b:c] + chrom[a:b] + chrom[c:]
        opt6 = chrom[:a] + list(reversed(chrom[b:c])) + chrom[a:b] + chrom[c:]
        opt7 = chrom[:a] + chrom[b:c] + list(reversed(chrom[a:b])) + chrom[c:]
        opts = [opt1, opt2, opt3, opt4, opt6, opt7]
        return random.choice(opts)

    def local_search(self, chromosome, use_advanced: Optional[bool] = None):
        start_vnd_time = time.time()
        vnd_iterations = 0
        if use_advanced is None:
            use_advanced = self.use_advanced_neighborhoods
        neighborhoods = ['swap', 'inversion', 'scramble']
        if use_advanced:
            neighborhoods += ['2opt', '3opt']

        best_chrom = chromosome[:]
        try:
            best_fit = self.fitness_func(best_chrom)
        except Exception as e:
            # print(f"Erro ao calcular fitness inicial em VND: {e}")
            return chromosome

        improved = True
        while improved:
            vnd_iterations += 1
            start_iter_time = time.time()
            improved = False
            neighbors_evaluated_total_iter = 0

            for nh in neighborhoods:
                neighbors_evaluated_in_nh = 0
                improvement_found_in_nh = False
                for _ in range(self.max_tries_per_neighborhood):
                    neighbors_evaluated_in_nh += 1
                    neighbors_evaluated_total_iter += 1
                    candidate = best_chrom[:]

                    try:
                        if nh == 'swap':
                            if len(candidate) < 2:
                                continue
                            a, b = random.sample(range(len(candidate)), 2)
                            candidate[a], candidate[b] = candidate[b], candidate[a]
                        elif nh == 'inversion':
                            if len(candidate) < 2:
                                continue
                            a, b = sorted(random.sample(
                                range(len(candidate)), 2))
                            candidate[a:b] = list(reversed(candidate[a:b]))
                        elif nh == 'scramble':
                            if len(candidate) < 3:
                                continue
                            a, b = sorted(random.sample(
                                range(len(candidate)), 2))
                            if b == a:
                                continue
                            sub = candidate[a:b]
                            random.shuffle(sub)
                            candidate[a:b] = sub
                        elif nh == '2opt':
                            candidate = self._2opt(candidate)
                        elif nh == '3opt':
                            candidate = self._3opt(candidate)
                        else:
                            continue

                        if candidate == best_chrom:
                            continue

                        fit = self.fitness_func(candidate)
                        if fit < best_fit:
                            # print(f"        Melhoria encontrada! {nh}: {best_fit:.2f} -> {fit:.2f}")
                            best_chrom = candidate
                            best_fit = fit
                            improvement_found_in_nh = True
                            break
                    except IndexError as e:
                        # print(f"Erro de índice em VND ({nh}): {e}. Cromossomo: {candidate}")
                        continue
                    except Exception as e:
                        # print(f"Erro ao aplicar {nh} ou calcular fitness: {e}")
                        continue

                if improvement_found_in_nh:
                    improved = True
                    break

            end_iter_time = time.time()
            # print(f"      [VND Iter {vnd_iterations}] Total Neighbors: {neighbors_evaluated_total_iter} | Time: {end_iter_time - start_iter_time:.4f}s | Improved in Iter: {improved} | Current Best Fit: {best_fit:.2f}")

        end_vnd_time = time.time()
        # print(f"    [VND Final] Tempo Total: {end_vnd_time - start_vnd_time:.4f}s | Iterações VND: {vnd_iterations} | Melhor Fitness: {best_fit:.2f}")
        return best_chrom
