import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
import logging
from enum import Enum
import concurrent.futures
import os
import operator  # Importar o módulo operator
import copy  # Adicionado para cópia profunda

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
                 enable_adaptive_tries: bool = True,
                 min_tries: int = 3,
                 max_tries_dynamic_factor: float = 3.0,
                 tries_increase_factor: float = 1.5,
                 tries_decrease_factor: float = 1.2,
                 stagnation_threshold_increase: int = 2,
                 improvement_streak_decrease: int = 3):
        """Inicializa a estratégia VNDLocalSearch com opções avançadas.

        Inclui LNS shake periódico, shake inicial opcional, operadores de bloco opcionais,
        enable_adaptive_tries: Se True, ajusta dinamicamente o número de tentativas por vizinhança.
        min_tries: Limite inferior para o número de tentativas por vizinhança (se adaptativo).
        max_tries_dynamic_factor: Multiplicador sobre `max_tries_per_neighborhood` para definir o limite superior dinâmico.
        tries_increase_factor: Fator pelo qual aumentar as tentativas durante estagnação.
        tries_decrease_factor: Fator pelo qual diminuir as tentativas após melhorias consecutivas.
        stagnation_threshold_increase: Número de iterações sem melhoria para começar a aumentar as tentativas.
        improvement_streak_decrease: Número de iterações com melhoria consecutivas para diminuir as tentativas.
        """
        self.fitness_func = fitness_func
        self.jobs_data = jobs  # Armazena dados dos jobs
        self.num_machines = num_machines  # Armazena número de máquinas
        self.use_advanced_neighborhoods = use_advanced_neighborhoods
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

        # --- Configuração Adaptação de Tentativas ---
        self.enable_adaptive_tries = enable_adaptive_tries
        if self.enable_adaptive_tries:
            if not (min_tries >= 1 and tries_increase_factor > 1.0 and tries_decrease_factor > 1.0 and
                    stagnation_threshold_increase >= 1 and improvement_streak_decrease >= 1 and
                    max_tries_dynamic_factor >= 1.0):
                raise ValueError(
                    "Parâmetros inválidos para adaptação de tentativas.")
            self.base_max_tries = max_tries_per_neighborhood
            self.min_tries = min_tries
            # Calcula max dinâmico, garantindo que seja >= base e >= min_tries
            self.max_tries_dynamic = max(self.base_max_tries, self.min_tries, int(
                self.base_max_tries * max_tries_dynamic_factor))
            self.tries_increase_factor = tries_increase_factor
            self.tries_decrease_factor = tries_decrease_factor
            self.stagnation_threshold_increase = stagnation_threshold_increase
            self.improvement_streak_decrease = improvement_streak_decrease
            logger.info(
                f"Adaptação de Tentativas Habilitada: Base={self.base_max_tries}, Min={self.min_tries}, MaxDyn={self.max_tries_dynamic}, StagnThr={stagnation_threshold_increase}, ImprStrk={improvement_streak_decrease}")
        else:
            logger.info("Adaptação de Tentativas Desabilitada.")

        logger.info(
            f"VNDLocalSearch inicializado com max_workers={self.max_workers}")

        # Mapeamento de tipo de vizinhança para método de operação
        self.operator_map = {
            NeighborhoodType.SWAP: self._apply_swap,
            NeighborhoodType.INVERSION: self._apply_inversion,
            NeighborhoodType.SCRAMBLE: self._apply_scramble,
            NeighborhoodType.TWO_OPT: self._2opt,
            NeighborhoodType.THREE_OPT: self._3opt,
            NeighborhoodType.LNS_SHAKE: self._apply_lns_shake,  # Adiciona o método LNS
            NeighborhoodType.BLOCK_MOVE: self._apply_block_move,
            NeighborhoodType.BLOCK_SWAP: self._apply_block_swap
        }

        # Define a lista base de vizinhanças VND
        base_neighborhoods = [NeighborhoodType.SWAP,
                              NeighborhoodType.INVERSION, NeighborhoodType.SCRAMBLE]
        advanced_neighborhoods = [NeighborhoodType.TWO_OPT,
                                  NeighborhoodType.THREE_OPT]
        block_neighborhoods = [NeighborhoodType.BLOCK_MOVE,
                               NeighborhoodType.BLOCK_SWAP]

        self.all_neighborhoods = base_neighborhoods
        if self.use_advanced_neighborhoods:
            self.all_neighborhoods.extend(advanced_neighborhoods)
        if self.use_block_operators:
            self.all_neighborhoods.extend(block_neighborhoods)

        # Inicializa estatísticas para cada vizinhança VND (LNS não entra na ordenação)
        self.neighborhood_stats = {
            nh_type: {
                'attempts': 0,
                'successes': 0,
                'success_rate': 0.0,
                'current_max_tries': self.base_max_tries if self.enable_adaptive_tries else self.max_tries_per_neighborhood,
                'consecutive_failures': 0  # Novo: contador de falhas consecutivas por vizinhança
            }
            for nh_type in self.all_neighborhoods
        }

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

                # 1. Calcular taxas de sucesso e reordenar vizinhanças VND
                for nh_type in current_neighborhoods_list:
                    stats = self.neighborhood_stats[nh_type]
                    if stats['attempts'] > 0:
                        stats['success_rate'] = stats['successes'] / \
                            stats['attempts']
                    else:
                        stats['success_rate'] = 0.0

                ordered_neighborhoods = sorted(
                    current_neighborhoods_list,
                    key=lambda nh: self.neighborhood_stats[nh]['success_rate'],
                    reverse=True
                )
                logger.debug(
                    f"      [VND Iter {vnd_iterations}] Neighborhood Order: {[nh.name for nh in ordered_neighborhoods]}")

                # 2. Iterar sobre as vizinhanças VND ordenadas
                for nh_type in ordered_neighborhoods:
                    improvement_found_in_nh = False
                    # Pega stats da vizinhança atual
                    stats_nh = self.neighborhood_stats[nh_type]
                    stats_nh['attempts'] += 1

                    candidates = []
                    operator = self.operator_map.get(nh_type)
                    if not operator:
                        logger.warning(  # Warning em vez de error para não parar tudo
                            f"Operador não encontrado para o tipo de vizinhança: {nh_type}, pulando.")
                        continue

                    # Usa o número de tentativas específico da vizinhança (adaptativo ou fixo)
                    num_tries_this_round = stats_nh['current_max_tries']

                    logger.debug(
                        f"        Explorando {nh_type.name} (Max Tries: {num_tries_this_round})")

                    generated_count = 0
                    unique_candidates_added = 0
                    for _ in range(num_tries_this_round):
                        candidate = operator(best_chrom)
                        generated_count += 1
                        # Verifica se é diferente do atual e se já não foi avaliado
                        if candidate != best_chrom:
                            candidate_tuple = tuple(candidate)
                            if candidate_tuple not in evaluated_solutions_cache:
                                # Adiciona para avaliação
                                candidates.append(candidate)
                                # Marca como avaliado (ou a ser avaliado) para evitar duplicatas na avaliação
                                evaluated_solutions_cache.add(candidate_tuple)
                                unique_candidates_added += 1
                            # else: logger.debug(f"    Candidato {candidate_tuple} já no cache.") # Log opcional
                        # else: logger.debug("    Candidato gerado igual ao atual.")

                    # Contabiliza apenas os que serão avaliados
                    neighbors_evaluated_total_iter += unique_candidates_added

                    if not candidates:
                        logger.debug(
                            f"        Nenhum candidato único e novo gerado para {nh_type.value} após {generated_count} tentativas.")
                        # --- Adaptação de Tentativas (Sem Candidatos Únicos) ---
                        # Considera isso uma falha para a adaptação
                        if self.enable_adaptive_tries:
                            stats_nh['consecutive_failures'] += 1
                            if stats_nh['consecutive_failures'] >= self.stagnation_threshold_increase:
                                # Diminui tentativas se estagnado (mesma lógica de falha na avaliação)
                                new_tries = max(self.min_tries, int(
                                    stats_nh['current_max_tries'] / self.tries_decrease_factor))
                                if new_tries < stats_nh['current_max_tries']:
                                    stats_nh['current_max_tries'] = new_tries
                                    logger.debug(
                                        f"        [Adaptive Tries - {nh_type.name}] Diminuindo para {new_tries} após {stats_nh['consecutive_failures']} falhas (sem candidatos).")
                                # stats_nh['consecutive_failures'] = 0 # Resetar aqui ou após avaliação? Resetar aqui.
                        continue  # Pula para a próxima vizinhança

                    logger.debug(
                        f"        Avaliando {len(candidates)} candidatos únicos para {nh_type.name}...")

                    # --- Usa o executor existente ---
                    try:
                        # Não cria mais um executor aqui, usa o do 'with' externo
                        futures_map = {executor.submit(
                            self.fitness_func, cand): cand for cand in candidates}
                        processed_futures = 0

                        for future in concurrent.futures.as_completed(futures_map):
                            processed_futures += 1
                            candidate = futures_map[future]
                            try:
                                fit = future.result()
                                # logger.debug(f"          Candidato avaliado (Fit: {fit:.2f}) vs Best (Fit: {best_fit:.2f})") # Log detalhado
                                if fit < best_fit:
                                    improvement_found_in_nh = True
                                    improvement_in_iteration = True  # Marca que houve melhoria na iteração
                                    previous_fit = best_fit
                                    # Cria cópia defensiva
                                    best_chrom = candidate[:]
                                    best_fit = fit
                                    stats_nh['successes'] += 1
                                    logger.debug(
                                        f"        Melhoria encontrada! {nh_type.name}: {previous_fit:.2f} -> {best_fit:.2f} (First Improvement)")

                                    # Tenta cancelar futuros restantes
                                    cancelled_count = 0
                                    pending_futures = 0
                                    for f_key, f_val in futures_map.items():  # Iterar sobre o dict original
                                        if not f_key.done():
                                            pending_futures += 1
                                            if f_key.cancel():
                                                cancelled_count += 1
                                            # else: logger.debug("      Não foi possível cancelar futuro.") # Futuro já poderia estar rodando/completo
                                    if pending_futures > 0:
                                        logger.debug(
                                            f"        Canceladas {cancelled_count}/{pending_futures} tarefas pendentes.")

                                    break  # Sai do loop as_completed

                            except concurrent.futures.CancelledError:
                                logger.debug(
                                    "        Tarefa cancelada encontrada no loop as_completed (esperado após melhoria).")
                            except Exception as exc:
                                logger.error(
                                    f'      Erro ao avaliar candidato para {nh_type.value} [Fitness Func]: {exc}', exc_info=True)
                                # Não para o VND inteiro, apenas ignora este candidato problemático

                        # Fim do loop as_completed

                        # Se o loop as_completed terminou E *nenhuma* melhoria foi encontrada nesta vizinhança
                        if not improvement_found_in_nh:
                            logger.debug(
                                f"        Nenhuma melhoria encontrada para {nh_type.name} após avaliar {processed_futures} candidatos.")
                            # --- Adaptação de Tentativas (Sem Melhoria na Avaliação) ---
                            if self.enable_adaptive_tries:
                                stats_nh['consecutive_failures'] += 1
                                # logger.debug(f"          Falhas consecutivas {nh_type.name}: {stats_nh['consecutive_failures']}") # Debug
                                if stats_nh['consecutive_failures'] >= self.stagnation_threshold_increase:
                                    new_tries = max(self.min_tries, int(
                                        stats_nh['current_max_tries'] / self.tries_decrease_factor))
                                    # Só diminui se for realmente menor
                                    if new_tries < stats_nh['current_max_tries']:
                                        stats_nh['current_max_tries'] = new_tries
                                        logger.debug(
                                            f"        [Adaptive Tries - {nh_type.name}] Diminuindo para {new_tries} após {stats_nh['consecutive_failures']} falhas (sem melhoria).")
                                    # Resetar contador após diminuir? Não, espera melhorar.
                                    # else: logger.debug(f"          Tentativas já no mínimo ({self.min_tries}) ou fator não reduziu.") # Debug
                        # else: # Houve melhoria, a adaptação (resetar falhas) é feita após sair do loop de vizinhanças

                    except Exception as e:
                        logger.error(
                            f"Erro durante a submissão/processamento de futuros para {nh_type.value} [Executor]: {e}", exc_info=True)
                        # Decide se quer continuar para a próxima vizinhança ou parar
                        continue  # Exemplo: Pula para a próxima vizinhança em caso de erro no executor

                    # Se uma melhoria foi encontrada (improvement_found_in_nh é True), sai do loop de vizinhanças
                    if improvement_found_in_nh:
                        # --- Adaptação de Tentativas (Com Melhoria) ---
                        if self.enable_adaptive_tries:
                            # Reseta falhas da vizinhança que teve sucesso
                            if stats_nh['consecutive_failures'] > 0:
                                logger.debug(
                                    f"        [Adaptive Tries - {nh_type.name}] Resetando {stats_nh['consecutive_failures']} falhas consecutivas após sucesso.")
                                stats_nh['consecutive_failures'] = 0

                             # Opcional: Aumentar tentativas após sucesso (comentado por enquanto)
                             # increase_trigger_rate = 0.8 # Ex: Aumentar se taxa de sucesso recente for alta
                             # recent_attempts = max(1, stats_nh['attempts'] % 10) # Últimos 10 por ex. (simplificado)
                             # recent_successes = max(0, stats_nh['successes'] % 5) # (simplificado)
                             # if recent_successes / recent_attempts >= increase_trigger_rate:
                             #      new_tries = min(self.max_tries_dynamic, int(stats_nh['current_max_tries'] * self.tries_increase_factor))
                             #      if new_tries > stats_nh['current_max_tries']:
                             #           stats_nh['current_max_tries'] = new_tries
                             #           logger.debug(f"        [Adaptive Tries - {nh_type.name}] Aumentando para {new_tries} devido a alta taxa de sucesso.")

                        break  # Reinicia o VND com a nova melhor solução a partir da vizinhança mais promissora
                    # --- Fim da Refatoração ---

                # --- Fim do loop de vizinhanças ---
                end_iter_time = time.time()
                logger.debug(
                    f"      [VND Iter {vnd_iterations}] Neighbors Evaluated This Iter: {neighbors_evaluated_total_iter} | Time: {end_iter_time - start_iter_time:.4f}s | Improved in Iter: {improvement_in_iteration} | Current Best Fit: {best_fit:.2f}")

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

                # Condição de parada adicional (ex: limite de tempo/iterações) pode ser adicionada aqui

            # --- Fim do loop while keep_searching ---
        # --- Fim do 'with executor' --- O executor é fechado aqui

        end_vnd_time = time.time()
        total_evaluated = len(evaluated_solutions_cache)
        logger.debug(
            f"    [VND Final] Tempo Total: {end_vnd_time - start_vnd_time:.4f}s | Iterações VND: {vnd_iterations} | Soluções Únicas Avaliadas: {total_evaluated} | Melhor Fitness Final: {best_fit:.2f}")
        return best_chrom
