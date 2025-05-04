import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict, List, Tuple
import logging
from enum import Enum
import concurrent.futures
import os
import operator  # Importar o módulo operator
import copy  # Adicionado para cópia profunda
import math

# Importa classe base
from ..genetic_operators import LocalSearchStrategy
from solvers.ortools_cpsat_solver import ORToolsCPSATSolver
# Importações locais devem vir depois das absolutas
# from solvers.ga.local_search.neighborhood_utils import get_critical_path, get_operations_on_critical_path

logger = logging.getLogger(__name__)

# Define Enum para os tipos de vizinhança


class NeighborhoodType(Enum):
    """Enumeração para os tipos de operadores de busca de vizinhança."""
    SWAP = 'swap'
    INVERSION = 'inversion'
    SCRAMBLE = 'scramble'
    TWO_OPT = '2opt'
    THREE_OPT = '3opt'
    LNS_SHAKE = 'lns_shake'  # Adiciona um tipo para o shake LNS
    BLOCK_MOVE = 'block_move'  # Novo: Mover bloco
    BLOCK_SWAP = 'block_swap'  # Novo: Trocar blocos
    CRITICAL_INSERT = 'critical_insert'  # Novo: Inserção na rota crítica
    # Novo: Troca de blocos na rota crítica
    CRITICAL_BLOCK_SWAP = 'critical_block_swap'
    CRITICAL_2OPT = 'critical_2opt'  # Novo: 2-opt na rota crítica


class VNDLocalSearch(LocalSearchStrategy):
    """Implementa a estratégia de busca local Variable Neighborhood Descent (VND)
    com ordenação adaptativa de vizinhanças e shaking LNS periódico.

    Esta estratégia explora uma sequência adaptativamente ordenada de estruturas de vizinhança.
    Se uma melhoria for encontrada em uma vizinhança, a busca reinicia a partir da
    primeira vizinhança com a solução melhorada. A ordem das vizinhanças é ajustada
    dinamicamente com base em sua taxa de sucesso recente.
    Se a busca estagnar por um número definido de iterações, um Large Neighborhood Search (LNS)
    "shaking" é aplicado para diversificar a busca, seguido por mais refino VND.
    Suporta avaliação paralela de vizinhos dentro de cada estrutura usando
    concurrent.futures.ProcessPoolExecutor.

    Atributos:
        fitness_func (Callable[[list], float]): Função para avaliar o fitness do cromossomo.
        use_advanced_neighborhoods (bool): Se deve incluir 2-opt e 3-opt.
        max_tries_per_neighborhood (int): Máximo de vizinhos a avaliar por estrutura (exceto LNS).
        rng (random.Random): Instância do gerador de números aleatórios para reprodutibilidade.
        max_workers (int): Número de workers paralelos para avaliação de fitness.
        lns_shake_frequency (int): Número de iterações VND sem melhoria para acionar o LNS shake.
                                     Se 0 ou None, o LNS shake é desativado.
        lns_shake_intensity (float): Proporção (0.0 a 1.0) do cromossomo a ser embaralhada no LNS shake.
        operator_map (dict): Mapeia Enums NeighborhoodType para seus métodos correspondentes.
        neighborhood_stats (dict): Armazena estatísticas de tentativas e sucessos por vizinhança.
        all_neighborhoods (list[NeighborhoodType]): Lista de todos os tipos de vizinhança VND disponíveis.
    """

    def __init__(self, fitness_func: Callable[[list], float],
                 jobs: list,  # Adicionado
                 num_machines: int,  # Adicionado
                 use_advanced_neighborhoods=False,
                 max_tries_per_neighborhood=10,
                 random_seed: Optional[int] = None,
                 max_workers: Optional[int] = None,
                 lns_shake_frequency: Optional[int] = 5,
                 lns_shake_intensity: float = 0.2,
                 lns_solver_time_limit: float = 0.1,  # Tempo limite para o CP-SAT no LNS
                 initial_shake_type: Optional[NeighborhoodType] = NeighborhoodType.SCRAMBLE,
                 initial_lns_shake_intensity: float = 0.1,
                 use_block_operators: bool = True,
                 # Novo: Habilita operadores de rota crítica
                 use_critical_path_operators: bool = True,
                 # Parâmetros para o Orquestrador UCB1
                 use_orchestrator: bool = True,
                 ucb1_exploration_factor: float = 1.0,
                 orchestrator_initial_attempts: int = 1,
                 orchestrator_initial_reward: float = 0.0,
                 # Max de tentativas por iteração do orquestrador (se não 1)
                 orchestrator_tries_per_pick: int = 1
                 ):
        """Inicializa a estratégia VNDLocalSearch com opções avançadas.

        Inclui LNS shake periódico, shake inicial opcional, operadores de bloco opcionais,
        operadores de rota crítica opcionais e orquestração de vizinhanças UCB1 opcional.

        Args:
            # ... (outros args)
            use_orchestrator: Se True, usa NeighborhoodOrchestrator (UCB1) em vez de VND tradicional.
            ucb1_exploration_factor: Parâmetro 'c' para UCB1 (maior = mais exploração).
            orchestrator_initial_attempts: Tentativas iniciais para cada operador no UCB1.
            orchestrator_initial_reward: Recompensa inicial para cada operador no UCB1.
            orchestrator_tries_per_pick: Quantas vezes tentar o operador escolhido pelo UCB1 a cada iteração.
        """
        self.fitness_func = fitness_func
        self.jobs_data = jobs  # Armazena dados dos jobs
        self.num_machines = num_machines  # Armazena número de máquinas
        self.use_advanced_neighborhoods = use_advanced_neighborhoods
        # Mantido para referência, mas não usado com orquestrador
        self.max_tries_per_neighborhood = max_tries_per_neighborhood
        self.rng = random.Random(random_seed)
        self.max_workers = max_workers if max_workers is not None else 4

        if lns_shake_frequency is not None and lns_shake_frequency > 0:
            self.lns_shake_frequency = lns_shake_frequency
            if not 0.0 < lns_shake_intensity <= 1.0:
                raise ValueError("lns_shake_intensity deve estar entre (0, 1]")
            self.lns_shake_intensity = lns_shake_intensity
            self.perform_lns_shake = True
            logger.info(
                f"LNS Shake habilitado: Frequência={lns_shake_frequency}, Intensidade={lns_shake_intensity:.2f}")
        else:
            self.lns_shake_frequency = 0
            self.lns_shake_intensity = 0.0
            self.perform_lns_shake = False
            logger.info("LNS Shake desabilitado.")

        # Configuração do Shake Inicial
        self.initial_shake_type = initial_shake_type
        self.initial_lns_shake_intensity = initial_lns_shake_intensity
        if self.initial_shake_type:
            if self.initial_shake_type not in [NeighborhoodType.SWAP, NeighborhoodType.INVERSION, NeighborhoodType.SCRAMBLE, NeighborhoodType.LNS_SHAKE]:
                raise ValueError(
                    f"Tipo de shake inicial inválido: {self.initial_shake_type}. Use SWAP, INVERSION, SCRAMBLE ou LNS_SHAKE.")
            if self.initial_shake_type == NeighborhoodType.LNS_SHAKE and not (0.0 < initial_lns_shake_intensity <= 1.0):
                raise ValueError(
                    "initial_lns_shake_intensity deve estar entre (0, 1] se initial_shake_type for LNS_SHAKE.")
            logger.info(f"Shake Inicial habilitado com tipo: {self.initial_shake_type.name}" +
                        (f", Intensidade LNS: {self.initial_lns_shake_intensity:.2f}" if self.initial_shake_type == NeighborhoodType.LNS_SHAKE else ""))
        else:
            logger.info("Shake Inicial desabilitado.")

        # Configuração Operadores de Bloco
        self.use_block_operators = use_block_operators
        if self.use_block_operators:
            logger.info(
                "Operadores de Bloco (BLOCK_MOVE, BLOCK_SWAP) habilitados.")
        else:
            logger.info("Operadores de Bloco desabilitados.")

        # Configuração Operadores de Rota Crítica
        self.use_critical_path_operators = use_critical_path_operators
        if self.use_critical_path_operators:
            logger.info(
                "Operadores de Rota Crítica (CRITICAL_INSERT, CRITICAL_BLOCK_SWAP, CRITICAL_2OPT) habilitados.")
        else:
            logger.info("Operadores de Rota Crítica desabilitados.")

        # Configuração do Orquestrador
        self.use_orchestrator = use_orchestrator
        self.orchestrator = None
        if self.use_orchestrator:
            self.ucb1_exploration_factor = ucb1_exploration_factor
            self.orchestrator_initial_attempts = orchestrator_initial_attempts
            self.orchestrator_initial_reward = orchestrator_initial_reward
            self.orchestrator_tries_per_pick = max(
                1, orchestrator_tries_per_pick)  # Garante >= 1

            # Cria a instância do orquestrador com as vizinhanças VND
            self.orchestrator = NeighborhoodOrchestrator(
                neighborhoods=self.all_neighborhoods,  # Passa a lista de vizinhanças ativas
                c=self.ucb1_exploration_factor,
                initial_attempts=self.orchestrator_initial_attempts,
                initial_reward=self.orchestrator_initial_reward
            )
            self.orchestrator.set_rng(self.rng)  # Passa a instância do RNG
            logger.info(
                f"Orquestrador UCB1 Habilitado (c={self.ucb1_exploration_factor}, tries_per_pick={self.orchestrator_tries_per_pick}).")
        else:
            # Se não usar orquestrador, precisa da lógica antiga de stats (ou uma simplificada)
            # Por enquanto, vamos focar no orquestrador. Se False, o comportamento atual será o VND padrão.
            logger.info("Usando VND padrão (ordenação por taxa de sucesso).")

        # Mapeamento de tipo de vizinhança para método de operação
        self.operator_map = {
            NeighborhoodType.SWAP: self._apply_swap,
            NeighborhoodType.INVERSION: self._apply_inversion,
            NeighborhoodType.SCRAMBLE: self._apply_scramble,
            NeighborhoodType.TWO_OPT: self._2opt,
            NeighborhoodType.THREE_OPT: self._3opt,
            NeighborhoodType.LNS_SHAKE: self._apply_lns_shake,  # Adiciona o método LNS
            NeighborhoodType.BLOCK_MOVE: self._apply_block_move,
            NeighborhoodType.BLOCK_SWAP: self._apply_block_swap,
            # Mapeia novos operadores (implementações placeholder abaixo)
            NeighborhoodType.CRITICAL_INSERT: self._apply_critical_insert,
            NeighborhoodType.CRITICAL_BLOCK_SWAP: self._apply_critical_block_swap,
            NeighborhoodType.CRITICAL_2OPT: self._apply_critical_2opt
        }

        # Define a lista base de vizinhanças VND
        base_neighborhoods = [NeighborhoodType.SWAP,
                              NeighborhoodType.INVERSION, NeighborhoodType.SCRAMBLE]
        advanced_neighborhoods = [NeighborhoodType.TWO_OPT,
                                  NeighborhoodType.THREE_OPT]
        block_neighborhoods = [NeighborhoodType.BLOCK_MOVE,
                               NeighborhoodType.BLOCK_SWAP]
        critical_path_neighborhoods = [NeighborhoodType.CRITICAL_INSERT,
                                       NeighborhoodType.CRITICAL_BLOCK_SWAP,
                                       NeighborhoodType.CRITICAL_2OPT]

        self.all_neighborhoods = base_neighborhoods
        if self.use_advanced_neighborhoods:
            self.all_neighborhoods.extend(advanced_neighborhoods)
        if self.use_block_operators:
            self.all_neighborhoods.extend(block_neighborhoods)
        if self.use_critical_path_operators:
            self.all_neighborhoods.extend(critical_path_neighborhoods)

        # Armazena tempo limite do CP-SAT
        self.lns_solver_time_limit = lns_solver_time_limit

    # --- Métodos de Operação de Vizinhança ---

    def _apply_swap(self, chrom: list) -> list:
        """Aplica o operador de vizinhança swap (troca).

        Seleciona aleatoriamente dois índices distintos e troca os elementos
        nestas posições.

        Args:
            chrom: O cromossomo (lista) a ser modificado.

        Returns:
            Uma nova lista de cromossomo com os elementos trocados, ou a lista
            original se o tamanho do cromossomo for menor que 2.
        """
        size = len(chrom)
        if size < 2:
            return chrom
        new_chrom = chrom[:]
        a, b = self.rng.sample(range(size), 2)
        new_chrom[a], new_chrom[b] = new_chrom[b], new_chrom[a]
        return new_chrom

    def _apply_inversion(self, chrom: list) -> list:
        """Aplica o operador de vizinhança inversion (inversão).

        Seleciona aleatoriamente dois índices distintos e inverte a sublista
        entre eles.

        Args:
            chrom: O cromossomo (lista) a ser modificado.

        Returns:
            Uma nova lista de cromossomo com a sublista invertida, ou a lista
            original se o tamanho do cromossomo for menor que 2.
        """
        size = len(chrom)
        if size < 2:
            return chrom
        new_chrom = chrom[:]
        a, b = sorted(self.rng.sample(range(size), 2))
        new_chrom[a:b] = list(reversed(new_chrom[a:b]))
        return new_chrom

    def _apply_scramble(self, chrom: list) -> list:
        """Aplica o operador de vizinhança scramble (embaralhamento).

        Seleciona aleatoriamente dois índices distintos e embaralha os elementos
        na sublista entre eles.

        Args:
            chrom: O cromossomo (lista) a ser modificado.

        Returns:
            Uma nova lista de cromossomo com a sublista embaralhada, ou a lista
            original se o tamanho do cromossomo for menor que 3 ou a sublista
            selecionada for muito pequena.
        """
        size = len(chrom)
        if size < 3:
            return chrom
        new_chrom = chrom[:]
        a, b = sorted(self.rng.sample(range(size), 2))
        if b == a + 1 or b == a:
            return chrom
        sub = new_chrom[a:b]
        self.rng.shuffle(sub)
        new_chrom[a:b] = sub
        return new_chrom

    def _2opt(self, chrom: list) -> list:
        """Aplica o operador de vizinhança 2-opt.

        Seleciona dois índices a, b e inverte o segmento entre a e b.
        Equivalente a quebrar duas arestas e reconectá-las de forma diferente.

        Args:
            chrom: O cromossomo (lista) representando um tour/sequência.

        Returns:
            Uma nova lista de cromossomo representando a troca 2-opt, ou a lista
            original se o tamanho do cromossomo for menor que 2.
        """
        size = len(chrom)
        if size < 2:
            return chrom
        a, b = sorted(self.rng.sample(range(size), 2))
        return chrom[:a] + list(reversed(chrom[a:b])) + chrom[b:]

    def _3opt(self, chrom: list) -> list:
        """Aplica uma variante aleatória do operador de vizinhança 3-opt.

        Seleciona três índices distintos a, b, c (ordenados) que definem quatro
        segmentos: S1 = chrom[:a], S2 = chrom[a:b], S3 = chrom[b:c], S4 = chrom[c:].
        Gera 7 possíveis rearranjos desses segmentos (alguns com S2 ou S3 invertidos)
        e retorna aleatoriamente um deles. Isso corresponde a quebrar 3 "arestas"
        (antes de a, b, c) e reconectá-las de uma das 7 maneiras não triviais.

        Args:
            chrom: O cromossomo (lista) representando uma sequência.

        Returns:
            Uma nova lista de cromossomo representando um movimento 3-opt aleatório,
            ou a lista original se o tamanho do cromossomo for menor que 3.
        """
        size = len(chrom)
        if size < 3:
            return chrom
        # Seleciona 3 pontos de corte distintos e ordena-os.
        a, b, c = sorted(self.rng.sample(range(size), 3))

        # Segmentos definidos pelos pontos de corte a, b, c
        s1 = chrom[:a]
        s2 = chrom[a:b]  # Segmento entre a (incl.) e b (excl.)
        s3 = chrom[b:c]  # Segmento entre b (incl.) e c (excl.)
        s4 = chrom[c:]  # Segmento de c (incl.) até o fim

        # Evita trabalho se algum segmento intermediário estiver vazio (improvável com sample, mas seguro)
        if not s2 or not s3:
            return chrom  # Não é possível fazer um 3-opt significativo

        # Calcula segmentos invertidos uma vez para reuso
        s2_reversed = list(reversed(s2))
        s3_reversed = list(reversed(s3))

        # Gera as 7 combinações não-identidade possíveis de 3-opt
        # (Baseado nas formas de reconectar após 3 quebras)
        possible_moves = [
            # Movimentos que invertem um segmento (equivalente a 2-opt)
            s1 + s2_reversed + s3 + s4,  # Inverte S2
            s1 + s2 + s3_reversed + s4,  # Inverte S3
            # Movimentos 3-opt propriamente ditos
            s1 + s2_reversed + s3_reversed + s4,  # Inverte S2 e S3
            # Troca S2 e S3 (move S2 para depois de S3)
            s1 + s3 + s2 + s4,
            s1 + s3 + s2_reversed + s4,  # Troca S2 e S3, inverte S2
            s1 + s3_reversed + s2 + s4,  # Troca S2 e S3, inverte S3
            s1 + s3_reversed + s2_reversed + s4  # Troca S2 e S3, inverte ambos
        ]

        # Escolhe aleatoriamente um dos 7 movimentos possíveis (não-identidade)
        return self.rng.choice(possible_moves)

    def _apply_lns_shake(self, chrom: list) -> list:
        """Aplica o LNS Shake: seleciona um subconjunto, tenta reotimizá-lo com CP-SAT,
           e o reinsere na ordem otimizada. Se CP-SAT falhar, embaralha aleatoriamente.

        Args:
            chrom: O cromossomo (lista) a ser modificado.

        Returns:
            Uma nova lista de cromossomo com uma subseção embaralhada, ou a lista
            original se a intensidade for muito baixa ou o cromossomo muito pequeno.
        """
        size = len(chrom)
        if not self.perform_lns_shake or size < 2:
            return chrom

        # Garante pelo menos 2
        num_to_shake = max(2, int(size * self.lns_shake_intensity))
        if num_to_shake >= size:
            logger.warning(
                "Intensidade do LNS shake muito alta, embaralhando tudo.")
            new_chrom = chrom[:]
            self.rng.shuffle(new_chrom)
            return new_chrom

        # 1. Seleciona índices e operações a serem "destruídas e reconstruídas"
        indices_to_shake = sorted(self.rng.sample(range(size), num_to_shake))
        ops_to_shake = [chrom[i] for i in indices_to_shake]
        # Guarda mapeamento inverso para reconstrução
        op_to_original_index = {op: idx for op,
                                idx in zip(ops_to_shake, indices_to_shake)}

        logger.debug(
            f"    Aplicando LNS Shake em {num_to_shake} operações nos índices: {indices_to_shake}")

        # 2. Cria o mini-problema JSSP para o CP-SAT
        # Trata cada operação selecionada como um job de uma única operação
        mini_jobs = []
        op_map_mini_to_global = {}
        for i, (job_id, op_id) in enumerate(ops_to_shake):
            machine_id, duration = self.jobs_data[job_id][op_id]
            mini_jobs.append([(machine_id, duration)])
            # Mapeia (mini_job_id, 0) para (job_id, op_id) original
            op_map_mini_to_global[(i, 0)] = (job_id, op_id)

        # 3. Tenta resolver o mini-problema com CP-SAT
        cpsat_solution_order = None
        try:
            mini_solver = ORToolsCPSATSolver(
                mini_jobs, len(mini_jobs), self.num_machines)
            # Usa um tempo limite muito curto! Remove verbose=False e converte para int
            # Nota: float < 1.0 será convertido para 0, pode precisar de ajuste
            mini_schedule = mini_solver.solve(
                time_limit=self.lns_solver_time_limit)

            if mini_schedule and mini_schedule.operations:
                # Ordena as operações do mini schedule pelo start_time
                mini_schedule.operations.sort(
                    key=lambda x: x[3])  # x[3] é start_time
                # Extrai a ordem otimizada das operações globais
                cpsat_solution_order = [op_map_mini_to_global[(mini_job_id, mini_op_id)]
                                        for mini_job_id, mini_op_id, _, _, _ in mini_schedule.operations]
                logger.debug(
                    f"    LNS Shake: CP-SAT encontrou ordem otimizada para subproblema.")
            else:
                logger.warning(
                    "    LNS Shake: CP-SAT não encontrou solução para o subproblema.")

        except Exception as e:
            logger.error(
                f"    Erro ao executar CP-SAT para LNS Shake: {e}", exc_info=True)
            # Deixa cpsat_solution_order como None para acionar o fallback

        # 4. Reconstrói o cromossomo
        new_chrom = chrom[:]  # Começa com cópia do original
        if cpsat_solution_order:
            # Reinsere as operações na ordem otimizada pelo CP-SAT nos índices originais
            for i, original_index in enumerate(indices_to_shake):
                new_chrom[original_index] = cpsat_solution_order[i]
        else:
            # Fallback: Embaralhamento aleatório das operações selecionadas nos índices originais
            logger.warning(
                "    LNS Shake: Fallback para embaralhamento aleatório.")
            values_to_shuffle = [new_chrom[i] for i in indices_to_shake]
            self.rng.shuffle(values_to_shuffle)
            for i, original_index in enumerate(indices_to_shake):
                new_chrom[original_index] = values_to_shuffle[i]

        return new_chrom

    def _apply_block_move(self, chrom: list) -> list:
        """Aplica o operador de vizinhança block move.

        Seleciona um bloco contíguo aleatório e o insere em uma posição aleatória diferente.

        Args:
            chrom: O cromossomo (lista) a ser modificado.

        Returns:
            Uma nova lista de cromossomo com o bloco movido, ou a lista original
            se o tamanho do cromossomo for menor que 3.
        """
        size = len(chrom)
        if size < 3:
            return chrom

        # Seleciona o bloco [a, b)
        a = self.rng.randint(0, size - 1)
        b = self.rng.randint(a + 1, size)
        block = chrom[a:b]
        block_len = len(block)

        # Remove o bloco
        remaining = chrom[:a] + chrom[b:]

        # Seleciona a posição de inserção k (0 a len(remaining))
        k = self.rng.randint(0, len(remaining))

        # Insere o bloco
        new_chrom = remaining[:k] + block + remaining[k:]
        return new_chrom

    def _apply_block_swap(self, chrom: list) -> list:
        """Aplica o operador de vizinhança block swap.

        Seleciona dois blocos contíguos não sobrepostos aleatórios e troca suas posições.

        Args:
            chrom: O cromossomo (lista) a ser modificado.

        Returns:
            Uma nova lista de cromossomo com os blocos trocados, ou a lista original
            se o tamanho do cromossomo for menor que 4 ou não for possível selecionar
            blocos não sobrepostos.
        """
        size = len(chrom)
        # Precisa de pelo menos 2 elementos para trocar blocos (de tamanho 1)
        if size < 2:
            return chrom

        # 1. Selecionar o primeiro bloco [a, b)
        # Escolhe dois índices distintos de 0 a size para definir o bloco
        idx1, idx2 = self.rng.sample(range(size + 1), 2)
        a, b = min(idx1, idx2), max(idx1, idx2)
        block1_len = b - a

        if block1_len == 0:  # Não deve acontecer com sample, mas por segurança
            return chrom
        if block1_len == size:
            # Bloco 1 ocupa tudo, não há espaço para Bloco 2
            return chrom

        # 2. Calcular tamanho restante e escolher tamanho do Bloco 2
        remaining_len = size - block1_len
        if remaining_len < 1:  # Não há espaço para Bloco 2 de tamanho >= 1
            return chrom
        block2_len = self.rng.randint(1, remaining_len)

        # 3. Identificar posições iniciais válidas para o Bloco 2
        valid_c_positions = []
        # Posições antes do Bloco 1: [0, a)
        max_c_before = a - block2_len
        if max_c_before >= 0:
            valid_c_positions.extend(range(max_c_before + 1))

        # Posições depois do Bloco 1: [b, size)
        min_c_after = b
        max_c_after = size - block2_len
        if min_c_after <= max_c_after:
            valid_c_positions.extend(range(min_c_after, max_c_after + 1))

        # 4. Escolher posição inicial c para o Bloco 2
        if not valid_c_positions:
            # Não foi possível encontrar posição para Bloco 2 com o tamanho escolhido
            # Poderia tentar outros tamanhos, mas retornamos o original por simplicidade
            logger.debug(
                f"Block Swap: Não foi possível encontrar posição para bloco 2 de tamanho {block2_len} não sobreposto a [{a},{b})")
            return chrom

        c = self.rng.choice(valid_c_positions)
        d = c + block2_len

        # 5. Extrair blocos e reconstruir
        block1 = chrom[a:b]
        block2 = chrom[c:d]

        if d <= a:  # Bloco 2 vem antes do Bloco 1
            # Segmentos: [0, c), block2=[c,d), [d, a), block1=[a,b), [b, size)
            new_chrom = chrom[:c] + block1 + chrom[d:a] + block2 + chrom[b:]
        else:  # b <= c, Bloco 1 vem antes do Bloco 2
            # Segmentos: [0, a), block1=[a,b), [b, c), block2=[c,d), [d, size)
            new_chrom = chrom[:a] + block2 + chrom[b:c] + block1 + chrom[d:]

        return new_chrom

    # --- Métodos dos Operadores de Rota Crítica (Placeholders) ---

    def _calculate_schedule_and_critical_path(self, chrom: list) -> tuple[dict[tuple[int, int], float], list[tuple[int, int]], float]:
        """Calcula os tempos de término das operações, o makespan e a rota crítica.

        Args:
            chrom: A sequência de operações (lista de tuplas (job_id, op_id)).

        Returns:
            Uma tupla contendo:
            - completion_times: Dicionário mapeando (job_id, op_id) para seu tempo de término.
            - critical_path: Lista de operações (job_id, op_id) na rota crítica.
            - makespan: O tempo de término da última operação.
            Retorna ({}, [], 0.0) em caso de erro ou se o cromossomo estiver vazio.
        """
        if not chrom:
            return {}, [], 0.0

        num_total_ops = len(chrom)
        completion_times = {}  # (job_id, op_id) -> end_time
        machine_release_times = {m: 0.0 for m in range(self.num_machines)}
        job_release_times = {j: 0.0 for j in range(len(self.jobs_data))}
        # Mapeia (job_id, op_id) para a máquina e duração
        op_details = {(j, i): self.jobs_data[j][i] for j, job in enumerate(
            self.jobs_data) for i in range(len(job))}
        # Mapeia (job_id, op_id) para seu predecessor no job
        job_predecessors = {}
        for j, job in enumerate(self.jobs_data):
            for i in range(len(job)):
                if i > 0:
                    job_predecessors[(j, i)] = (j, i - 1)

        # Constrói a ordem das operações por máquina a partir do cromossomo
        machine_sequences: Dict[int, List[Tuple[int, int]]] = {
            m: [] for m in range(self.num_machines)}
        machine_predecessors = {}
        for op_tuple in chrom:
            machine_id, _ = op_details[op_tuple]
            if machine_sequences[machine_id]:  # Se já tem operação na máquina
                last_op_on_machine = machine_sequences[machine_id][-1]
                machine_predecessors[op_tuple] = last_op_on_machine
            machine_sequences[machine_id].append(op_tuple)

        # Calcula tempos de término iterativamente (similar à ordenação topológica)
        # Precisamos processar na ordem correta, garantindo que predecessores sejam calculados antes
        # Uma forma é iterar até que todos os tempos sejam calculados (ou detectar ciclo/erro)
        ops_to_process = set(chrom)
        processed_ops = set()
        iterations = 0
        max_iterations = num_total_ops * 2  # Heurística para evitar loop infinito

        while ops_to_process and iterations < max_iterations:
            processed_in_iter = set()
            for op_job_id, op_op_id in list(ops_to_process):
                op_tuple = (op_job_id, op_op_id)
                machine_id, duration = op_details[op_tuple]

                # Verifica se predecessores já foram processados
                job_pred = job_predecessors.get(op_tuple)
                machine_pred = machine_predecessors.get(op_tuple)

                can_process = True
                job_pred_end_time = 0.0
                machine_pred_end_time = 0.0

                if job_pred and job_pred not in processed_ops:
                    can_process = False
                elif job_pred:
                    job_pred_end_time = completion_times[job_pred]

                if machine_pred and machine_pred not in processed_ops:
                    # Se o predecessor de máquina ainda não foi processado, espera
                    can_process = False
                elif machine_pred:
                    # O tempo de término do predecessor na máquina é o tempo de liberação da máquina para esta op
                    machine_pred_end_time = completion_times[machine_pred]

                if can_process:
                    # Tempo de início é o max(fim_pred_job, fim_pred_maquina)
                    start_time = max(job_pred_end_time, machine_pred_end_time)
                    end_time = start_time + duration
                    completion_times[op_tuple] = end_time

                    # Atualiza tempos de liberação (não estritamente necessário aqui, mas útil)
                    job_release_times[op_job_id] = end_time
                    # Atualiza liberação da máquina
                    machine_release_times[machine_id] = end_time

                    processed_ops.add(op_tuple)
                    processed_in_iter.add(op_tuple)

            if not processed_in_iter:
                # Se nenhuma operação pôde ser processada, há um problema (ciclo?)
                logger.error(
                    f"Não foi possível processar todas as operações. Possível ciclo ou erro. Restantes: {ops_to_process}")
                # Retorna vazio para indicar falha
                return {}, [], 0.0

            ops_to_process -= processed_in_iter
            iterations += 1

        if ops_to_process:
            logger.error(
                f"Cálculo do cronograma excedeu max_iterations. Restantes: {ops_to_process}")
            return {}, [], 0.0

        # Calcula Makespan
        makespan = max(completion_times.values()) if completion_times else 0.0

        # Encontra a Rota Crítica (Backtracking do Makespan)
        critical_path = []
        if not completion_times:
            return completion_times, critical_path, makespan

        # Encontra a(s) última(s) operação(ões) que definem o makespan
        last_ops = [op for op, ct in completion_times.items() if ct ==
                    makespan]
        if not last_ops:
            return completion_times, [], makespan  # Deveria ter ao menos uma

        # Começa o backtracking a partir de uma das últimas operações
        # Escolhe uma arbitrariamente se houver mais de uma (pode haver múltiplos caminhos críticos)
        current_op = last_ops[0]

        while current_op is not None:
            critical_path.append(current_op)
            job_pred = job_predecessors.get(current_op)
            machine_pred = machine_predecessors.get(current_op)
            # end_time - duration
            current_op_start_time = completion_times[current_op] - \
                op_details[current_op][1]

            prev_op = None
            # Verifica qual predecessor (job ou máquina) define o tempo de início da op atual
            job_pred_end_time = completion_times.get(job_pred, -1.0)
            machine_pred_end_time = completion_times.get(machine_pred, -1.0)

            # Se o fim do predecessor do job é igual ao início da op atual, ele está no caminho crítico
            # Tolerância float
            if job_pred is not None and abs(job_pred_end_time - current_op_start_time) < 1e-6:
                prev_op = job_pred
            # Se o fim do predecessor da máquina é igual ao início da op atual, ele está no caminho crítico
            elif machine_pred is not None and abs(machine_pred_end_time - current_op_start_time) < 1e-6:
                prev_op = machine_pred
            # Se nenhum predecessor define o tempo de início, chegamos ao começo (ou erro)
            # Pode acontecer se a primeira operação do job/máquina for a crítica
            elif job_pred is None and machine_pred is None and current_op_start_time == 0.0:
                prev_op = None  # Chegou ao início
            # Primeira op do job, mas não da máquina
            elif job_pred is None and abs(machine_pred_end_time - current_op_start_time) < 1e-6:
                prev_op = machine_pred
            else:
                # logger.warning(f"Backtracking da rota crítica parou inesperadamente em {current_op}. Start: {current_op_start_time}, JobPredEnd: {job_pred_end_time}, MachPredEnd: {machine_pred_end_time}")
                prev_op = None  # Interrompe

            current_op = prev_op

        critical_path.reverse()  # Reverte para ter a ordem correta (início -> fim)
        return completion_times, critical_path, makespan

    def _apply_critical_insert(self, chrom: list) -> list:
        """Aplica o operador Critical Path Insertion Move.

        Seleciona uma operação na rota crítica e tenta movê-la para outra
        posição na sequência da *mesma máquina* dentro do cromossomo,
        respeitando as restrições de precedência do job.
        """
        completion_times, critical_path, _ = self._calculate_schedule_and_critical_path(
            chrom)
        if not critical_path or len(critical_path) < 1:
            return chrom  # Retorna original se não há rota crítica

        # Escolhe uma operação crítica aleatória para mover
        op_to_move = self.rng.choice(critical_path)
        job_id, op_id = op_to_move
        machine_id, duration = self.jobs_data[job_id][op_id]

        # Encontra a posição atual da operação no cromossomo
        try:
            current_index = chrom.index(op_to_move)
        except ValueError:
            logger.error(
                f"Operação crítica {op_to_move} não encontrada no cromossomo? Impossível.")
            return chrom  # Erro interno

        # Identifica as operações predecessoras e sucessoras no JOB
        job_pred = None
        job_succ = None
        if op_id > 0:
            job_pred = (job_id, op_id - 1)
        if op_id < len(self.jobs_data[job_id]) - 1:
            job_succ = (job_id, op_id + 1)

        # Encontra os índices no cromossomo dos predecessores/sucessores do job (se existirem)
        job_pred_index = -1
        if job_pred:
            try:
                job_pred_index = chrom.index(job_pred)
            except ValueError:
                pass  # Pode não estar no cromossomo ainda (se erro no cálculo)

        job_succ_index = len(chrom)
        if job_succ:
            try:
                job_succ_index = chrom.index(job_succ)
            except ValueError:
                pass

        # Encontra todas as posições *no cromossomo* das operações da mesma máquina
        machine_op_indices = [i for i, op in enumerate(
            chrom) if self.jobs_data[op[0]][op[1]][0] == machine_id]

        valid_insertion_points = []
        for target_index in machine_op_indices:
            # O ponto de inserção é *antes* do target_index
            insert_pos = target_index
            # Se a op no target_index é a própria op_to_move, a inserção seria na posição dela
            if chrom[target_index] == op_to_move:
                continue  # Não pode inserir na própria posição atual

            # Verifica restrição de precedência:
            # A nova posição (insert_pos) deve ser > índice do pred do job
            # A nova posição (insert_pos) deve ser < índice do succ do job
            # Ajuste: se estamos inserindo *antes* de target_index, a posição
            # efetiva da op_to_move será 'insert_pos'. Precisamos garantir que
            # o índice do pred seja < insert_pos e o índice do succ seja > insert_pos.
            # Contudo, a lógica de índices muda se movermos o elemento.

            # Abordagem mais simples: construir o cromossomo candidato e verificar
            temp_chrom = chrom[:current_index] + chrom[current_index+1:]
            # Calcula o índice de inserção no cromossomo temporário
            temp_insert_index = -1
            if insert_pos > current_index:
                temp_insert_index = insert_pos - 1  # Ajuste porque removemos antes
            else:
                temp_insert_index = insert_pos

            # Insere no ponto candidato
            candidate_chrom = temp_chrom[:temp_insert_index] + \
                [op_to_move] + temp_chrom[temp_insert_index:]

            # Verifica se as precedências do job são mantidas no candidato
            try:
                new_op_index = candidate_chrom.index(op_to_move)
                new_pred_index = -1
                if job_pred:
                    new_pred_index = candidate_chrom.index(job_pred)
                new_succ_index = len(candidate_chrom)
                if job_succ:
                    new_succ_index = candidate_chrom.index(job_succ)

                if new_pred_index < new_op_index < new_succ_index:
                    # Armazena o índice original de destino
                    valid_insertion_points.append(insert_pos)

            except ValueError:
                # Se pred/succ não estiver no cromossomo candidato, algo está errado
                logger.warning(
                    f"Pred/Succ de {op_to_move} não encontrado no cromossomo candidato ao testar inserção em {insert_pos}")
                continue

        # Adiciona a possibilidade de inserir no final da sequência da máquina
        last_machine_op_index = machine_op_indices[-1]
        # Inserir *depois* da última operação da máquina
        insert_pos_after_last = last_machine_op_index + 1
        # Verifica precedências para inserção no final da máquina
        temp_chrom_end = chrom[:current_index] + chrom[current_index+1:]
        candidate_chrom_end = temp_chrom_end[:insert_pos_after_last] + [
            op_to_move] + temp_chrom_end[insert_pos_after_last:]
        try:
            new_op_index_end = candidate_chrom_end.index(op_to_move)
            new_pred_index_end = -1
            if job_pred:
                new_pred_index_end = candidate_chrom_end.index(job_pred)
            # Sucessor não importa para inserção no fim
            if new_pred_index_end < new_op_index_end:
                valid_insertion_points.append(insert_pos_after_last)
        except ValueError:
            logger.warning(
                f"Pred de {op_to_move} não encontrado ao testar inserção no fim da máquina")

        if not valid_insertion_points:
            logger.debug(
                f"Nenhuma posição de inserção válida encontrada para {op_to_move} na rota crítica.")
            return chrom  # Nenhuma mudança possível

        # Escolhe um ponto de inserção válido aleatoriamente
        chosen_insertion_point = self.rng.choice(valid_insertion_points)

        # Constrói o cromossomo final
        # Remove o elemento original
        final_temp_chrom = chrom[:current_index] + chrom[current_index+1:]
        # Ajusta o índice de inserção com base na posição original removida
        if chosen_insertion_point > current_index:
            final_insert_index = chosen_insertion_point - 1
        else:
            final_insert_index = chosen_insertion_point

        new_chrom = final_temp_chrom[:final_insert_index] + \
            [op_to_move] + final_temp_chrom[final_insert_index:]

        logger.debug(
            f"    Critical Insert: Movid {op_to_move} de {current_index} para {final_insert_index} (relativo ao temp).")
        return new_chrom

    def _apply_critical_block_swap(self, chrom: list) -> list:
        """Aplica o operador Critical Block Swap (versão simplificada).

        Seleciona duas operações *adjacentes na mesma máquina* que estão
        *ambas na rota crítica* e troca suas posições no cromossomo.
        """
        completion_times, critical_path, _ = self._calculate_schedule_and_critical_path(
            chrom)
        if not critical_path or len(critical_path) < 2:
            return chrom

        critical_path_set = set(critical_path)
        possible_swaps = []

        # Encontra pares de operações adjacentes na *mesma máquina* que estão na rota crítica
        op_details = {(j, i): self.jobs_data[j][i] for j, job in enumerate(
            self.jobs_data) for i in range(len(job))}
        machine_sequences: Dict[int, List[Tuple[int, int]]] = {
            m: [] for m in range(self.num_machines)}
        op_indices_in_chrom = {op: idx for idx, op in enumerate(chrom)}

        for op_tuple in chrom:
            machine_id, _ = op_details[op_tuple]
            machine_sequences[machine_id].append(op_tuple)

        for machine_id, sequence in machine_sequences.items():
            for i in range(len(sequence) - 1):
                op1 = sequence[i]
                op2 = sequence[i+1]
                # Verifica se ambas estão na rota crítica
                if op1 in critical_path_set and op2 in critical_path_set:
                    # Verifica se a troca respeitaria precedências de JOB
                    # Trocar op1 e op2 significa colocar op2 antes de op1 na máquina
                    op1_job, op1_id = op1
                    op2_job, op2_id = op2

                    # op2 não pode ser sucessor de op1 no mesmo job
                    is_op2_succ_of_op1 = (
                        op1_job == op2_job and op2_id == op1_id + 1)
                    # op1 não pode ser sucessor de op2 no mesmo job
                    is_op1_succ_of_op2 = (
                        op1_job == op2_job and op1_id == op2_id + 1)

                    if not is_op2_succ_of_op1 and not is_op1_succ_of_op2:
                        idx1 = op_indices_in_chrom.get(op1)
                        idx2 = op_indices_in_chrom.get(op2)
                        if idx1 is not None and idx2 is not None:
                            # Armazena índices no cromossomo
                            possible_swaps.append(tuple(sorted((idx1, idx2))))

        if not possible_swaps:
            logger.debug(
                "Nenhuma troca válida de blocos críticos adjacentes encontrada.")
            return chrom

        # Escolhe um par aleatório para trocar
        idx1, idx2 = self.rng.choice(possible_swaps)

        new_chrom = chrom[:]
        new_chrom[idx1], new_chrom[idx2] = new_chrom[idx2], new_chrom[idx1]
        logger.debug(
            f"    Critical Block Swap: Trocou {chrom[idx1]} (idx {idx1}) com {chrom[idx2]} (idx {idx2})")
        return new_chrom

    def _apply_critical_2opt(self, chrom: list) -> list:
        """Aplica o operador Critical Path 2-opt (versão por bloco no cromossomo).

        Encontra um bloco *contíguo no cromossomo* onde *todas* as operações
        pertencem à rota crítica e aplica uma inversão (2-opt) nesse bloco.
        """
        completion_times, critical_path, _ = self._calculate_schedule_and_critical_path(
            chrom)
        # Precisa de ao menos 2 ops na rota
        if not critical_path or len(critical_path) < 2:
            return chrom

        critical_path_set = set(critical_path)
        # Lista de tuplas (start_index, end_index) de blocos críticos no cromossomo
        critical_blocks = []

        start = -1
        for i, op in enumerate(chrom):
            if op in critical_path_set:
                if start == -1:
                    start = i  # Início de um bloco crítico
            else:
                if start != -1:
                    # Fim de um bloco crítico (pelo menos 1 op)
                    # Para 2-opt, precisamos de blocos de tamanho >= 2
                    if i - start >= 2:
                        # Bloco é [start, i)
                        critical_blocks.append((start, i))
                    start = -1  # Reseta
        # Verifica se o último bloco termina no final do cromossomo
        if start != -1 and len(chrom) - start >= 2:
            critical_blocks.append((start, len(chrom)))

        if not critical_blocks:
            logger.debug(
                "Nenhum bloco contíguo de operações críticas (tam >= 2) encontrado no cromossomo.")
            return chrom

        # Escolhe um bloco crítico aleatório
        # a = start_idx, b = end_idx (exclusive)
        a, b = self.rng.choice(critical_blocks)

        # Aplica 2-opt (inversão) nesse bloco
        new_chrom = chrom[:a] + list(reversed(chrom[a:b])) + chrom[b:]

        # Validação Opcional: Verificar se a inversão violou precedências de job
        # (Pode acontecer se o bloco invertido continha ops do mesmo job)
        # Por simplicidade, vamos assumir que a inversão é válida ou que
        # a avaliação de fitness subsequente penalizará soluções inválidas.
        # Uma validação robusta aqui seria mais complexa.

        logger.debug(
            f"    Critical 2-opt: Inverteu o bloco crítico [{a}:{b}] no cromossomo.")
        return new_chrom

    # --- Método Principal de Busca Local ---

    def local_search(self, chromosome, use_advanced: Optional[bool] = None):
        """Executa a busca Variable Neighborhood Descent (VND) adaptativa com LNS shake periódico.

        Explora iterativamente as vizinhanças VND disponíveis. A ordem das vizinhanças
        é reavaliada a cada iteração com base em sua taxa de sucesso histórica.
        Para cada vizinhança, gera `max_tries_per_neighborhood` candidatos
        e avalia seus fitness em paralelo usando `ProcessPoolExecutor`.
        Se uma solução melhor for encontrada em qualquer vizinhança, a busca reinicia
        a partir da primeira vizinhança (na ordem atualizada) com a nova melhor solução.
        Se a busca estagnar (sem melhorias por `lns_shake_frequency` iterações),
        um LNS shake é aplicado para diversificar, e o VND continua.
        O processo para quando uma passagem completa por todas as vizinhanças VND não
        resulta em melhoria e o LNS shake (se aplicável) também não levou a uma melhoria
        na iteração subsequente.

        Args:
            chromosome: O cromossomo inicial (lista) a ser melhorado.
            use_advanced: Sobrescreve a configuração `use_advanced_neighborhoods` para
                          esta chamada específica, se fornecido. (Nota: a lista de vizinhanças
                          disponíveis é definida na inicialização).

        Returns:
            O melhor cromossomo encontrado após o término do processo VND.
        """
        start_vnd_time = time.time()
        vnd_iterations = 0
        non_improving_iterations = 0  # Contador para LNS Shake
        # Determina quais vizinhanças VND usar
        current_neighborhoods_list = self.all_neighborhoods[:]
        if self.use_orchestrator and self.orchestrator is None:
            logger.error(
                "Orquestrador habilitado mas não inicializado. Abortando busca local.")
            return chromosome
        # Se não usar orquestrador, inicializa stats para VND padrão
        if not self.use_orchestrator:
            self.neighborhood_stats = {nh_type: {'attempts': 0, 'successes': 0, 'success_rate': 0.0}
                                       for nh_type in current_neighborhoods_list}

        best_chrom = chromosome[:]
        initial_fitness_calculated = False
        evaluated_solutions_cache = set()

        # 0. Aplicar Shake Inicial (Opcional)
        if self.initial_shake_type:
            shake_operator = self.operator_map.get(self.initial_shake_type)
            if shake_operator:
                logger.debug(
                    f"    Aplicando Shake Inicial do tipo: {self.initial_shake_type.name}")
                # Guarda intensidade original do LNS periódico se necessário
                original_lns_intensity = self.lns_shake_intensity
                if self.initial_shake_type == NeighborhoodType.LNS_SHAKE:
                    # Usa intensidade específica para o shake inicial
                    self.lns_shake_intensity = self.initial_lns_shake_intensity

                shaken_chrom = shake_operator(best_chrom)

                # Restaura intensidade original do LNS periódico se foi alterada
                if self.initial_shake_type == NeighborhoodType.LNS_SHAKE:
                    self.lns_shake_intensity = original_lns_intensity

                if shaken_chrom != best_chrom:
                    best_chrom = shaken_chrom
                    # Adiciona ao cache SE for diferente
                    evaluated_solutions_cache.add(tuple(best_chrom))
                    logger.debug(
                        "    Cromossomo modificado pelo Shake Inicial.")
                else:
                    logger.debug("    Shake Inicial não alterou o cromossomo.")
            else:
                logger.error(
                    f"Operador para Shake Inicial ({self.initial_shake_type.name}) não encontrado!")
        else:  # Adiciona original ao cache se não houve shake
            evaluated_solutions_cache.add(tuple(best_chrom))

        try:
            best_fit = self.fitness_func(best_chrom)
            initial_fitness_calculated = True
            logger.debug(
                f"    [VND Start] Initial Fitness (após shake, se houver): {best_fit:.2f}")
        except Exception as e:
            logger.error(
                f"Erro ao calcular fitness inicial em VND: {e}", exc_info=True)
            return chromosome

        if not initial_fitness_calculated:
            # Este bloco não deve ser alcançado devido ao return acima, mas por segurança:
            logger.error(
                "Não foi possível calcular o fitness inicial. Abortando VND.")
            return chromosome

        keep_searching = True

        # Cria o executor uma vez fora do loop principal
        # Garante fechamento mesmo com exceções
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            while keep_searching:
                vnd_iterations += 1
                start_iter_time = time.time()
                improvement_in_iteration = False  # Flag para melhoria nesta iteração específica
                neighbors_evaluated_total_iter = 0
                best_improvement_in_iter = 0.0  # Para rastrear a melhor recompensa na iteração

                # --- Lógica de Seleção: Orquestrador UCB1 ou VND Padrão ---
                if self.use_orchestrator:
                    # Garante que o orquestrador existe (Linter fix)
                    if self.orchestrator is None:
                        logger.error(
                            "Orquestrador é None mesmo com use_orchestrator=True. Abortando.")
                        return best_chrom  # Ou raise Exception

                    # 1. Orquestrador seleciona a vizinhança
                    nh_type = self.orchestrator.pick()
                    operator = self.operator_map.get(nh_type)

                    if not operator:
                        logger.error(
                            f"Operador para vizinhança {nh_type.name} selecionada pelo orquestrador não encontrado!")
                        # O que fazer aqui? Pular a iteração? Tentar pegar outro?
                        # Por segurança, vamos pular esta iteração.
                        non_improving_iterations += 1  # Conta como não melhoria
                        continue

                    logger.debug(
                        f"      [Orchestrator Iter {vnd_iterations}] Picked: {nh_type.name} (Tries: {self.orchestrator_tries_per_pick})")

                    improvement_found_in_pick = False
                    total_reward_for_pick = 0.0
                    candidates_to_evaluate = []

                    # 2. Gera N candidatos com o operador escolhido
                    for _ in range(self.orchestrator_tries_per_pick):
                        candidate_chrom = operator(best_chrom)
                        # Avalia apenas se for novo e diferente
                        if candidate_chrom != best_chrom:
                            candidate_tuple = tuple(candidate_chrom)
                            if candidate_tuple not in evaluated_solutions_cache:
                                candidates_to_evaluate.append(candidate_chrom)
                                evaluated_solutions_cache.add(
                                    candidate_tuple)  # Marca como visto

                    neighbors_evaluated_total_iter += len(
                        candidates_to_evaluate)

                    if not candidates_to_evaluate:
                        logger.debug(
                            f"        Nenhum candidato novo/único gerado por {nh_type.name}.")
                        # Garante que o orquestrador existe (Linter fix)
                        if self.orchestrator is None:
                            logger.error(
                                "Orquestrador é None ao tentar atualizar com falha. Abortando.")
                            return best_chrom
                        # Atualiza o orquestrador com recompensa 0
                        self.orchestrator.update(nh_type, 0.0)
                        # Incrementa contador de não melhoria e vai para próxima iteração
                        improvement_in_iteration = False  # Garante que está False
                    else:
                        # 3. Avalia os candidatos gerados (em paralelo)
                        futures_map = {executor.submit(
                            self.fitness_func, cand): cand for cand in candidates_to_evaluate}
                        processed_futures = 0
                        best_candidate_in_batch = None
                        best_fit_in_batch = best_fit  # Começa com o fitness atual

                        try:
                            for future in concurrent.futures.as_completed(futures_map):
                                processed_futures += 1
                                candidate = futures_map[future]
                                try:
                                    fit = future.result()
                                    if fit < best_fit_in_batch:  # Encontrou uma solução melhor que a *melhor atual*
                                        improvement_found_in_pick = True
                                        improvement_in_iteration = True  # Marca melhoria geral na iteração
                                        # Calcula recompensa relativa à *melhor solução antes desta avaliação*
                                        reward = max(0.0, best_fit - fit)
                                        best_improvement_in_iter = max(
                                            best_improvement_in_iter, reward)  # Guarda a maior melhoria

                                        # Atualiza melhor global
                                        previous_fit = best_fit
                                        # Cópia defensiva
                                        best_chrom = candidate[:]
                                        best_fit = fit
                                        best_fit_in_batch = fit  # Atualiza melhor dentro do batch tbm

                                        logger.debug(
                                            f"        Melhoria encontrada! {nh_type.name}: {previous_fit:.2f} -> {best_fit:.2f}")
                                        # Não saímos do loop as_completed, avaliamos todos do batch

                                except concurrent.futures.CancelledError:
                                    logger.debug(
                                        "        Tarefa cancelada encontrada.")
                                except Exception as exc:
                                    logger.error(
                                        f'      Erro ao avaliar candidato para {nh_type.name} [Fitness Func]: {exc}', exc_info=True)

                            # 4. Atualiza o orquestrador com a *melhor* recompensa obtida neste pick
                            # (Poderia ser a média, mas usar a melhor incentiva picos de melhoria)
                            self.orchestrator.update(
                                nh_type, best_improvement_in_iter)

                            if not improvement_found_in_pick:
                                logger.debug(
                                    f"        Nenhuma melhoria encontrada por {nh_type.name} nesta rodada.")

                        except Exception as e:
                            logger.error(
                                f"Erro durante a submissão/processamento de futuros para {nh_type.name} [Executor]: {e}", exc_info=True)
                            # Atualiza com recompensa 0 em caso de erro na avaliação
                            self.orchestrator.update(nh_type, 0.0)
                            improvement_in_iteration = False  # Garante que está False

                else:
                    # --- Lógica VND Padrão (mantida se use_orchestrator=False) ---
                    # 1. Calcular taxas de sucesso e reordenar vizinhanças VND
                    for nh_type_std in current_neighborhoods_list:
                        stats = self.neighborhood_stats[nh_type_std]
                        if stats['attempts'] > 0:
                            stats['success_rate'] = stats['successes'] / \
                                stats['attempts']
                        else:
                            stats['success_rate'] = 0.0
                    ordered_neighborhoods = sorted(
                        current_neighborhoods_list, key=lambda nh: self.neighborhood_stats[nh]['success_rate'], reverse=True)
                    logger.debug(
                        f"      [VND Std Iter {vnd_iterations}] Neighborhood Order: {[nh.name for nh in ordered_neighborhoods]}")

                    # 2. Iterar sobre as vizinhanças VND ordenadas
                    for nh_type_std in ordered_neighborhoods:
                        # ... (Lógica original do VND padrão para gerar N vizinhos, avaliar em paralelo, first improvement) ...
                        # Esta parte precisa ser mantida/restaurada se quisermos alternar entre os modos.
                        # Por ora, vamos focar no modo orquestrador. Se use_orchestrator for False,
                        # o código atual não fará a busca local padrão.
                        logger.warning(
                            "Lógica VND padrão não totalmente implementada/restaurada neste branch.")
                        # Simula nenhuma melhoria no modo padrão não implementado
                        improvement_in_iteration = False
                        break  # Sai do loop de vizinhanças padrão

                # --- Fim do loop de vizinhanças ---
                end_iter_time = time.time()
                logger.debug(
                    f"      [{'Orch' if self.use_orchestrator else 'VND Std'} Iter {vnd_iterations}] Neighbors Evaluated: {neighbors_evaluated_total_iter} | Time: {end_iter_time - start_iter_time:.4f}s | Improved: {improvement_in_iteration} | Best Fit: {best_fit:.2f}")

                # 3. Verificar estagnação e aplicar LNS Shake se necessário
                if not improvement_in_iteration:
                    non_improving_iterations += 1
                    logger.debug(
                        f"      Iteração {vnd_iterations} sem melhoria. Contador Estagnação: {non_improving_iterations}/{self.lns_shake_frequency if self.perform_lns_shake else 'N/A'}")

                    # LNS Shake
                    if self.perform_lns_shake and non_improving_iterations >= self.lns_shake_frequency:
                        logger.info(
                            f"    [LNS Shake Triggered] Aplicando shake após {non_improving_iterations} iterações sem melhoria.")
                        # Usa o executor global para avaliar o pós-shake? Não, fitness é chamado diretamente.
                        shaken_chrom = self._apply_lns_shake(best_chrom)

                        if shaken_chrom != best_chrom:
                            shaken_tuple = tuple(shaken_chrom)
                            # Avalia apenas se for novo
                            if shaken_tuple not in evaluated_solutions_cache:
                                try:
                                    # Recalcula o fitness após o shake
                                    shaken_fit = self.fitness_func(
                                        shaken_chrom)
                                    # Adiciona ao cache após avaliação bem-sucedida
                                    evaluated_solutions_cache.add(shaken_tuple)
                                    logger.info(
                                        f"    [LNS Shake Applied] Novo fitness após shake: {shaken_fit:.2f}")

                                    # Verifica se o shake melhorou a solução atual
                                    if shaken_fit < best_fit:
                                        logger.info(
                                            "    LNS Shake resultou em melhoria direta!")
                                        # Cópia defensiva
                                        best_chrom = shaken_chrom[:]
                                        best_fit = shaken_fit
                                        improvement_in_iteration = True  # Considera como melhoria na iteração
                                        non_improving_iterations = 0  # Reseta estagnação
                                    else:
                                        logger.info(
                                            "    LNS Shake não melhorou o fitness atual, mas diversificou.")
                                        # Mantém o cromossomo pós-shake para a próxima iteração VND
                                        # Cópia defensiva
                                        best_chrom = shaken_chrom[:]
                                        # Atualiza o fitness mesmo que pior (para exploração)
                                        best_fit = shaken_fit
                                        # Não reseta non_improving_iterations aqui se não melhorou

                                except Exception as e:
                                    logger.error(
                                        f"Erro ao calcular fitness após LNS shake: {e}", exc_info=True)
                                    # Decide se quer parar ou continuar. Manter best_chrom antigo?
                                    # Por segurança, não altera best_chrom se avaliação falhar
                                    logger.warning(
                                        "    Mantendo cromossomo anterior ao LNS shake devido a erro na avaliação.")
                            else:
                                logger.info(
                                    "    Cromossomo pós-shake já estava no cache, LNS não gerou novidade.")
                                # Não faz nada, continua com o best_chrom atual
                                # Não reseta non_improving_iterations

                        else:  # shaken_chrom == best_chrom
                            logger.warning(
                                "LNS Shake não alterou o cromossomo.")
                            # Não reseta non_improving_iterations

                        # Se houve melhoria (direta do shake), o contador já foi resetado.
                        # Se não houve melhoria (ou shake não alterou/foi cacheado), o contador continua.
                        # Se o shake foi aplicado (mesmo sem melhoria), resetamos o gatilho do shake para
                        # dar chance ao VND de refinar a nova solução (ou a antiga se shake falhou)
                        # antes de tentar outro shake imediatamente.
                        if self.perform_lns_shake:  # Só reseta se o shake foi tentado
                            non_improving_iterations = 0  # Reseta mesmo sem melhoria para forçar VND pós-shake
                            logger.debug(
                                "    Contador de estagnação LNS resetado após tentativa de shake.")

                    # Não houve melhoria E não é hora do shake (ou shake desativado)
                    else:
                        # Paramos a busca global
                        keep_searching = False
                        logger.debug(
                            f"    VND estagnado ({non_improving_iterations} iterações s/ melhoria) e sem LNS shake pendente. Finalizando busca local.")
                # Houve melhoria na iteração (vinda de uma vizinhança VND)
                else:
                    non_improving_iterations = 0  # Reseta contador de estagnação
                    # A adaptação de tentativas já foi feita ao sair do loop de vizinhanças

                # Condição de parada adicional (ex: limite de tempo/iterações total) pode ser adicionada aqui

            # --- Fim do loop while keep_searching ---
        # --- Fim do 'with executor' --- O executor é fechado aqui

        end_vnd_time = time.time()
        total_evaluated = len(evaluated_solutions_cache)
        logger.debug(
            f"    [VND Final] Tempo Total: {end_vnd_time - start_vnd_time:.4f}s | Iterações VND: {vnd_iterations} | Soluções Únicas Avaliadas: {total_evaluated} | Melhor Fitness Final: {best_fit:.2f}")
        return best_chrom


# --- Classe para Orquestração de Vizinhanças (MAB/AOS) ---

class NeighborhoodOrchestrator:
    """Gerencia a seleção de operadores de vizinhança usando UCB1.

    Trata cada tipo de vizinhança como um "braço" de um Multi-Armed Bandit.
    Seleciona o próximo operador a ser aplicado com base em uma combinação
    de desempenho histórico (exploração) e incerteza (exploração).
    """

    def __init__(self, neighborhoods: List[NeighborhoodType], c: float = 1.0, initial_attempts: int = 1, initial_reward: float = 0.0):
        """Inicializa o orquestrador.

        Args:
            neighborhoods: Lista dos tipos de vizinhança (braços) a serem gerenciados.
            c: Parâmetro de exploração para UCB1. Valores maiores incentivam mais exploração.
            initial_attempts: Contagem inicial de tentativas para cada braço (evita div/0).
            initial_reward: Recompensa inicial para cada braço.
        """
        if not neighborhoods:
            raise ValueError("A lista de vizinhanças não pode ser vazia.")
        if initial_attempts <= 0:
            raise ValueError("initial_attempts deve ser positivo.")

        self.neighborhoods = list(neighborhoods)
        # stats[n] = [sum_of_rewards, attempts]
        self.stats = {n: [float(initial_reward), int(initial_attempts)]
                      for n in self.neighborhoods}
        # Soma total de tentativas em todos os braços
        self.total_trials = sum(s[1] for s in self.stats.values())
        self.c = c
        logger.info(
            f"NeighborhoodOrchestrator inicializado com {len(self.neighborhoods)} vizinhanças e c={self.c}")

    def pick(self) -> NeighborhoodType:
        """Seleciona a próxima vizinhança a ser explorada usando UCB1.

        UCB1 Score = average_reward + c * sqrt(ln(total_trials) / attempts_for_arm)

        Returns:
            O NeighborhoodType com a maior pontuação UCB1.
        """
        best_neighborhood = None
        max_ucb_score = -float('inf')

        # Garante que total_trials seja pelo menos 1 para math.log
        log_total_trials = math.log(max(1, self.total_trials))

        for n in self.neighborhoods:
            sum_rewards, attempts = self.stats[n]
            if attempts == 0:  # Deve ser prevenido por initial_attempts > 0
                ucb_score = float('inf')  # Prioriza braços não tentados
            else:
                average_reward = sum_rewards / attempts
                exploration_bonus = self.c * \
                    math.sqrt(log_total_trials / attempts)
                ucb_score = average_reward + exploration_bonus

            # logger.debug(f"  UCB1 Score for {n.name}: {ucb_score:.4f} (AvgRew={average_reward:.4f}, ExpBonus={exploration_bonus:.4f})")

            if ucb_score > max_ucb_score:
                max_ucb_score = ucb_score
                best_neighborhood = n
            # Desempate aleatório simples se pontuações forem iguais (ou muito próximas)
            elif abs(ucb_score - max_ucb_score) < 1e-9 and self.rng.choice([True, False]):
                best_neighborhood = n

        if best_neighborhood is None:
            # Fallback: se algo deu errado, escolhe aleatoriamente
            logger.warning(
                "UCB1 não conseguiu selecionar um braço, escolhendo aleatoriamente.")
            best_neighborhood = self.rng.choice(self.neighborhoods)

        # logger.debug(f"    Orchestrator picked: {best_neighborhood.name}")
        return best_neighborhood

    def update(self, neighborhood: NeighborhoodType, reward: float):
        """Atualiza as estatísticas de um braço após uma tentativa.

        Args:
            neighborhood: O tipo de vizinhança que foi aplicado.
            reward: A recompensa obtida (ex: melhoria no fitness).
        """
        if neighborhood not in self.stats:
            logger.error(
                f"Tentativa de atualizar vizinhança desconhecida: {neighborhood}")
            return

        self.stats[neighborhood][0] += reward
        self.stats[neighborhood][1] += 1
        self.total_trials += 1
        # logger.debug(f"    Orchestrator updated {neighborhood.name}: Reward={reward:.4f}, New SumRew={self.stats[neighborhood][0]:.4f}, New Att={self.stats[neighborhood][1]}")

    # Adiciona referência ao rng da classe VND para desempate
    # Isso requer passar o rng do VND para o construtor ou definir depois
    def set_rng(self, rng_instance: random.Random):
        self.rng = rng_instance
