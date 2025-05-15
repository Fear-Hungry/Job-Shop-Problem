import random
import time
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict, List, Tuple
import logging
from enum import Enum
import concurrent.futures
import os
import operator
import copy
import math
import itertools
import threading

from .base import LocalSearchStrategy
from solvers.ortools_cpsat_solver import ORToolsCPSATSolver
from .operator_utils import calculate_schedule_and_critical_path
# Importa os novos operadores
from .neighborhood_operators import (
    BaseNeighborhoodOperator, SwapOperator, InversionOperator, ScrambleOperator,
    TwoOptOperator, ThreeOptOperator, BlockMoveOperator, BlockSwapOperator
)

logger = logging.getLogger(__name__)

class NeighborhoodType(Enum):
    SWAP = 'swap'
    INVERSION = 'inversion'
    SCRAMBLE = 'scramble'
    TWO_OPT = '2opt'
    THREE_OPT = '3opt'
    LNS_SHAKE = 'lns_shake'
    BLOCK_MOVE = 'block_move'
    BLOCK_SWAP = 'block_swap'
    CRITICAL_INSERT = 'critical_insert'
    CRITICAL_BLOCK_SWAP = 'critical_block_swap'
    CRITICAL_2OPT = 'critical_2opt'
    CRITICAL_LNS = 'critical_lns'

# Placeholder para a classe que o GA usará para gerenciar soluções globais entre threads
class GlobalSolutionPoolPlaceholder:
    def __init__(self):
        self.solutions: Dict[Any, Tuple[list, float]] = {} # thread_id -> (chrom, fitness)
        self.lock = threading.Lock() # Para segurança em ambiente multithread
        logger.info("GlobalSolutionPoolPlaceholder inicializado.")

    def report(self, thread_id: Any, chrom: list, fitness: float):
        with self.lock:
            self.solutions[thread_id] = (list(chrom), fitness)
            # logger.debug(f"Thread {thread_id} reportou fitness {fitness:.2f}")

    def get_best_excluding(self, thread_id: Any) -> Tuple[Optional[list], Optional[float]]:
        with self.lock:
            best_chrom, best_fitness = None, float('inf')
            if not self.solutions:
                return None, None
            for tid, (chrom, fitness) in self.solutions.items():
                if tid != thread_id and fitness < best_fitness:
                    best_fitness = fitness
                    best_chrom = list(chrom)
            # logger.debug(f"Thread {thread_id} obteve melhor peer: {best_fitness if best_fitness != float('inf') else 'N/A'}")
            return best_chrom, (best_fitness if best_fitness != float('inf') else None)

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
                 jobs: list,
                 num_machines: int,
                 use_advanced_neighborhoods=False,
                 max_tries_per_neighborhood=10,
                 random_seed: Optional[int] = None,
                 max_workers: Optional[int] = None,
                 lns_shake_frequency: Optional[int] = 5,
                 lns_shake_intensity: float = 0.2,
                 lns_solver_time_limit: float = 0.1,
                 initial_shake_type: Optional[NeighborhoodType] = NeighborhoodType.SCRAMBLE,
                 initial_lns_shake_intensity: float = 0.1,
                 use_block_operators: bool = True,
                 use_critical_path_operators: bool = True,
                 use_orchestrator: bool = True,
                 ucb1_exploration_factor: float = 1.0,
                 orchestrator_initial_attempts: int = 1,
                 orchestrator_initial_reward: float = 0.0,
                 orchestrator_tries_per_pick: int = 1,
                 # Parâmetros para Reactive VND
                 thread_id: Any = 0, # ID da "thread" de LS
                 shared_solution_pool: Optional[GlobalSolutionPoolPlaceholder] = None,
                 share_frequency: int = 10, # Frequência para compartilhar soluções
                 reactive_update_N: int = 50, # Frequência para aprendizado reativo de operadores
                 reactive_learning_eta: float = 0.1, # Taxa de aprendizado para gradient bandit
                 # Parâmetros adicionais para shake avançado
                 max_shake_intensity: float = 0.8,
                 enable_multi_phase_shake: bool = True,
                 critical_ops_ratio: float = 0.7
                 ):
        """Inicializa a estratégia VNDLocalSearch com opções avançadas.

        Inclui LNS shake periódico, shake inicial opcional, operadores de bloco opcionais,
        operadores de rota crítica opcionais e orquestração de vizinhanças UCB1 opcional.

        Args:
            # ... (outros args)
            max_shake_intensity: Intensidade máxima permitida para o shake
            enable_multi_phase_shake: Se True, usa shake multi-fase adaptativo
            critical_ops_ratio: Proporção de operações críticas no shake
        """
        self.fitness_func = fitness_func
        self.jobs_data = jobs
        self.num_machines = num_machines
        self.use_advanced_neighborhoods = use_advanced_neighborhoods
        self.max_tries_per_neighborhood = max_tries_per_neighborhood
        self.rng = random.Random(random_seed)
        self.max_workers = max_workers if max_workers is not None else 4

        # Pré-cálculo de op_details e job_predecessors
        self.op_details = {(j, i): self.jobs_data[j][i] for j, job_ops_list in enumerate(self.jobs_data) for i in range(len(job_ops_list))}
        self.job_predecessors = {}
        for j, job_ops_list in enumerate(self.jobs_data):
            for i in range(len(job_ops_list)):
                if i > 0:
                    self.job_predecessors[(j, i)] = (j, i - 1)

        # Configurações para Reactive VND
        self.thread_id = thread_id
        self.shared_solution_pool = shared_solution_pool
        self.share_frequency = share_frequency
        self.reactive_update_N = reactive_update_N
        self.reactive_learning_eta = reactive_learning_eta

        # Configurações para shake avançado
        self.max_shake_intensity = max_shake_intensity
        self.enable_multi_phase_shake = enable_multi_phase_shake
        self.critical_ops_ratio = critical_ops_ratio
        self.consecutive_shakes_without_improvement = 0
        self.shake_strategy_stats = {
            "critical_path": {"attempts": 0, "successes": 0},
            "mixed_strategy": {"attempts": 0, "successes": 0},
            "strong_diversification": {"attempts": 0, "successes": 0},
            "cp_sat": {"attempts": 0, "successes": 0},
            "alternative": {"attempts": 0, "successes": 0}
        }
        self.alternative_strategies = ["block_inversion", "guided_shuffle", "machine_based_reordering"]
        self.last_shake_makespan = float('inf')
        self.last_shake_type = None

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

        self.use_block_operators = use_block_operators
        self.use_critical_path_operators = use_critical_path_operators
        self.lns_solver_time_limit = lns_solver_time_limit

        # --- Instanciar Operadores ---
        self.op_swap = SwapOperator(self.rng)
        self.op_inversion = InversionOperator(self.rng)
        self.op_scramble = ScrambleOperator(self.rng)
        self.op_2opt = TwoOptOperator(self.rng)
        self.op_3opt = ThreeOptOperator(self.rng)
        self.op_block_move = BlockMoveOperator(self.rng)
        self.op_block_swap = BlockSwapOperator(self.rng)
        # Nota: Operadores LNS e de Rota Crítica ainda são métodos por enquanto

        # --- Configurar lista de vizinhanças e operator_map ---
        base_neighborhoods = [NeighborhoodType.SWAP,
                              NeighborhoodType.INVERSION, NeighborhoodType.SCRAMBLE]
        advanced_neighborhoods = [NeighborhoodType.TWO_OPT, NeighborhoodType.THREE_OPT]
        block_neighborhoods = [NeighborhoodType.BLOCK_MOVE, NeighborhoodType.BLOCK_SWAP]
        critical_path_neighborhoods = [NeighborhoodType.CRITICAL_INSERT,
                                       NeighborhoodType.CRITICAL_BLOCK_SWAP,
                                       NeighborhoodType.CRITICAL_2OPT,
                                       NeighborhoodType.CRITICAL_LNS]
        lns_neighborhood = [NeighborhoodType.LNS_SHAKE]

        self.all_neighborhoods = base_neighborhoods
        self.operator_map: Dict[NeighborhoodType, Callable] = {
            NeighborhoodType.SWAP: self.op_swap.apply,
            NeighborhoodType.INVERSION: self.op_inversion.apply,
            NeighborhoodType.SCRAMBLE: self.op_scramble.apply,
            # Ainda são métodos:
            NeighborhoodType.LNS_SHAKE: self._apply_lns_shake,
            NeighborhoodType.CRITICAL_INSERT: self._apply_critical_insert,
            NeighborhoodType.CRITICAL_BLOCK_SWAP: self._apply_critical_block_swap,
            NeighborhoodType.CRITICAL_2OPT: self._apply_critical_2opt,
            NeighborhoodType.CRITICAL_LNS: self._apply_critical_lns,
        }
        if self.use_advanced_neighborhoods:
            self.all_neighborhoods.extend(advanced_neighborhoods)
            self.operator_map[NeighborhoodType.TWO_OPT] = self.op_2opt.apply
            self.operator_map[NeighborhoodType.THREE_OPT] = self.op_3opt.apply
        if self.use_block_operators:
            self.all_neighborhoods.extend(block_neighborhoods)
            self.operator_map[NeighborhoodType.BLOCK_MOVE] = self.op_block_move.apply
            self.operator_map[NeighborhoodType.BLOCK_SWAP] = self.op_block_swap.apply
        if self.use_critical_path_operators:
            self.all_neighborhoods.extend(critical_path_neighborhoods)
            # Os métodos _apply_critical_* já estão mapeados acima

        # Adicionar LNS ao mapa se estiver habilitado (o método _apply_lns_shake já está no mapa)
        # A lista all_neighborhoods é usada principalmente pelo orquestrador agora.

        # --- Inicialização para Aprendizado Reativo de Operadores ---
        self.operator_probabilities: Dict[NeighborhoodType, float] = {
            nt: 1.0 / len(self.operator_map) for nt in self.operator_map.keys()
        }
        self.reactive_attempts_count = 0 # Contador para o gatilho de atualização reativa
        # --- Configuração do Orquestrador ---
        self.use_orchestrator = use_orchestrator
        self.orchestrator = None
        if self.use_orchestrator:
            self.ucb1_exploration_factor = ucb1_exploration_factor
            self.orchestrator_initial_attempts = orchestrator_initial_attempts
            self.orchestrator_initial_reward = orchestrator_initial_reward
            self.orchestrator_tries_per_pick = max(1, orchestrator_tries_per_pick)
            # Usa a lista final de vizinhanças ativas
            active_neighborhoods = list(self.operator_map.keys())
            self.orchestrator = NeighborhoodOrchestrator(
                neighborhoods=active_neighborhoods,
                c=self.ucb1_exploration_factor,
                initial_attempts=self.orchestrator_initial_attempts,
                initial_reward=self.orchestrator_initial_reward,
                operator_probabilities_ref=self.operator_probabilities # Passa a referência
            )
            self.orchestrator.set_rng(self.rng)
            logger.info(
                f"Orquestrador UCB1 Habilitado com {len(active_neighborhoods)} vizinhanças (c={self.ucb1_exploration_factor}, tries_per_pick={self.orchestrator_tries_per_pick}).")
        else:
            logger.info("Usando VND padrão (ordenação por taxa de sucesso).")

        # --- Configuração do Hyperparameter UCB para LNS Meta-Adaptação ---
        if self.perform_lns_shake:
            # Define grid de braços (intensity, frequency)
            intensities = [self.lns_shake_intensity * f for f in (0.5, 1.0, 1.5)]
            freqs = [self.lns_shake_frequency, max(1, self.lns_shake_frequency * 2)]
            arms = list(itertools.product(intensities, freqs))
            self.hyper_orch = HyperparameterOrchestrator(
                arms=arms,
                c=self.ucb1_exploration_factor,
                initial_attempts=orchestrator_initial_attempts,
                initial_reward=0.0
            )
            self.hyper_orch.set_rng(self.rng)
            logger.info(f"Hyperparameter UCB habilitado com {len(arms)} braços (intensity, freq)")

    
    def _apply_lns_shake(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Aplica o operador LNS Shake: embaralha uma fração do cromossomo usando um solver exato ou aleatoriedade.
        Útil para diversificação global quando a busca local estagna.
        """
        size = len(chrom)
        if not self.perform_lns_shake or size < 2:
            return chrom
            
        # Se o shake multi-fase está habilitado, utiliza abordagem por fases
        if self.enable_multi_phase_shake:
            return self._apply_multi_phase_shake(chrom)
        
        # Caso contrário, usa o método de shake padrão aprimorado
        # Usar intensidade progressiva
        intensity = self._get_progressive_shake_intensity()
        num_to_shake = max(2, int(size * intensity))
        
        if num_to_shake >= size:
            logger.warning("Intensidade do LNS shake muito alta, embaralhando tudo.")
            new_chrom = chrom[:]
            self.rng.shuffle(new_chrom)
            return new_chrom
            
        # Considerar caminho crítico na seleção
        completion_times, critical_path, _ = calculate_schedule_and_critical_path(
            chrom, self.num_machines, self.op_details, self.job_predecessors
        )
            
        indices_to_shake = []
        
        if critical_path:
            # Priorizar operações do caminho crítico
            critical_ops_set = set(critical_path)
            critical_indices = [i for i, op in enumerate(chrom) if op in critical_ops_set]
            non_critical_indices = [i for i, op in enumerate(chrom) if op not in critical_ops_set]
            
            num_critical = min(len(critical_indices), int(num_to_shake * self.critical_ops_ratio))
            num_non_critical = min(len(non_critical_indices), num_to_shake - num_critical)
            
            if num_critical > 0:
                indices_to_shake.extend(self.rng.sample(critical_indices, num_critical))
            if num_non_critical > 0:
                indices_to_shake.extend(self.rng.sample(non_critical_indices, num_non_critical))
        else:
            # Fallback para seleção aleatória
            indices_to_shake = self.rng.sample(range(size), num_to_shake)
            
        indices_to_shake = sorted(indices_to_shake)
        
        # logger.debug(f"    Aplicando LNS Shake em {num_to_shake} operações nos índices: {indices_to_shake}")
        
        # Tentar resolver com CP-SAT
        new_chrom = self._apply_cp_sat_to_indices(indices_to_shake, chrom)
        
        if new_chrom != chrom:
            # logger.debug(f"    LNS Shake: CP-SAT encontrou ordem otimizada.")
            return new_chrom
        
        # Se CP-SAT falhar, usar estratégias alternativas
        logger.warning("    LNS Shake: Fallback para estratégias alternativas.")
        return self._apply_alternative_shake(indices_to_shake, chrom)

    def _apply_critical_insert(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Move uma operação da rota crítica para outra posição válida na mesma máquina.
        Permite explorar vizinhanças baseadas na estrutura do caminho crítico.
        """
        completion_times, critical_path, _ = calculate_schedule_and_critical_path(chrom, self.num_machines, self.op_details, self.job_predecessors)
        if not critical_path or len(critical_path) < 1:
            return chrom
        op_to_move = self.rng.choice(critical_path)
        job_id, op_id = op_to_move
        try:
            current_index = chrom.index(op_to_move)
        except ValueError:
            logger.error(f"Operação crítica {op_to_move} não encontrada no cromossomo? Impossível.")
            return chrom
        job_pred = (job_id, op_id - 1) if op_id > 0 else None
        job_succ = (job_id, op_id + 1) if op_id < len(self.jobs_data[job_id]) - 1 else None
        op_to_move_machine_id, _ = self.jobs_data[op_to_move[0]][op_to_move[1]]
        machine_op_indices = [i for i, op in enumerate(chrom) if self.jobs_data[op[0]][op[1]][0] == op_to_move_machine_id]
        valid_insertion_points = []
        for target_index in machine_op_indices:
            if chrom[target_index] == op_to_move:
                continue
            temp_chrom = chrom[:current_index] + chrom[current_index+1:]
            temp_insert_index = target_index - 1 if target_index > current_index else target_index
            candidate_chrom = temp_chrom[:temp_insert_index] + [op_to_move] + temp_chrom[temp_insert_index:]
            try:
                new_op_index = candidate_chrom.index(op_to_move)
                new_pred_index = candidate_chrom.index(job_pred) if job_pred else -1
                new_succ_index = candidate_chrom.index(job_succ) if job_succ else len(candidate_chrom)
                if new_pred_index < new_op_index < new_succ_index:
                    valid_insertion_points.append(target_index)
            except ValueError:
                continue
        if not valid_insertion_points:
            # logger.debug(f"Nenhuma posição de inserção válida encontrada para {op_to_move}.")
            return chrom
        chosen_insertion_point_in_original_chrom = self.rng.choice(valid_insertion_points)
        final_temp_chrom = chrom[:current_index] + chrom[current_index+1:]
        final_insert_index_in_temp_chrom = chosen_insertion_point_in_original_chrom - 1 if chosen_insertion_point_in_original_chrom > current_index else chosen_insertion_point_in_original_chrom
        new_chrom = final_temp_chrom[:final_insert_index_in_temp_chrom] + [op_to_move] + final_temp_chrom[final_insert_index_in_temp_chrom:]
        # logger.debug(f"    Critical Insert: Movi {op_to_move} para índice {chosen_insertion_point_in_original_chrom}.")
        return new_chrom

    def _apply_critical_block_swap(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Troca blocos de operações dentro do caminho crítico, respeitando restrições de precedência.
        Útil para grandes saltos estruturais guiados pelo caminho crítico.
        """
        completion_times, critical_path, _ = calculate_schedule_and_critical_path(chrom, self.num_machines, self.op_details, self.job_predecessors)
        if not critical_path or len(critical_path) < 2:
            return chrom
        critical_path_set = set(critical_path)
        possible_swaps = []
        op_details_local = {(j, i): self.jobs_data[j][i] for j, job_ops in enumerate(self.jobs_data) for i in range(len(job_ops))}
        machine_sequences: Dict[int, List[Tuple[int, int]]] = {m: [] for m in range(self.num_machines)}
        op_indices_in_chrom = {op: idx for idx, op in enumerate(chrom)}
        for op_tuple in chrom:
            if op_tuple not in op_details_local: continue # Sanity check
            machine_id, _ = op_details_local[op_tuple]
            machine_sequences[machine_id].append(op_tuple)
        for machine_id, sequence in machine_sequences.items():
            for i in range(len(sequence) - 1):
                op1 = sequence[i]
                op2 = sequence[i+1]
                if op1 in critical_path_set and op2 in critical_path_set:
                    op1_job, op1_id = op1
                    op2_job, op2_id = op2
                    is_op2_succ_of_op1 = (op1_job == op2_job and op2_id == op1_id + 1)
                    is_op1_succ_of_op2 = (op1_job == op2_job and op1_id == op2_id + 1)
                    if not is_op2_succ_of_op1 and not is_op1_succ_of_op2:
                        idx1 = op_indices_in_chrom.get(op1)
                        idx2 = op_indices_in_chrom.get(op2)
                        if idx1 is not None and idx2 is not None:
                            possible_swaps.append(tuple(sorted((idx1, idx2))))
        if not possible_swaps:
            # logger.debug("Nenhuma troca válida de blocos críticos adjacentes encontrada.")
            return chrom
        idx1, idx2 = self.rng.choice(possible_swaps)
        new_chrom = chrom[:]
        new_chrom[idx1], new_chrom[idx2] = new_chrom[idx2], new_chrom[idx1]
        # logger.debug(f"    Critical Block Swap: Trocou {chrom[idx1]} com {chrom[idx2]}")
        return new_chrom

    def _apply_critical_2opt(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Aplica 2-opt restrito ao caminho crítico, invertendo segmentos críticos.
        """
        completion_times, critical_path, _ = calculate_schedule_and_critical_path(chrom, self.num_machines, self.op_details, self.job_predecessors)
        if not critical_path or len(critical_path) < 2:
            return chrom
        critical_path_set = set(critical_path)
        critical_blocks = []
        start = -1
        for i, op in enumerate(chrom):
            if op in critical_path_set:
                if start == -1:
                    start = i
            else:
                if start != -1:
                    if i - start >= 2:
                        critical_blocks.append((start, i))
                    start = -1
        if start != -1 and len(chrom) - start >= 2:
            critical_blocks.append((start, len(chrom)))
        if not critical_blocks:
            # logger.debug("Nenhum bloco contíguo de operações críticas (tam >= 2) encontrado.")
            return chrom
        a, b = self.rng.choice(critical_blocks)
        new_chrom = chrom[:a] + list(reversed(chrom[a:b])) + chrom[b:]
        # logger.debug(f"    Critical 2-opt: Inverteu o bloco crítico [{a}:{b}].")
        return new_chrom

    def _apply_critical_lns(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Aplica LNS restrito ao caminho crítico, embaralhando apenas operações críticas.
        """
        # Calcula rota crítica e makespan
        completion_times, critical_path, makespan = calculate_schedule_and_critical_path(
            chrom, self.num_machines, self.op_details, self.job_predecessors
        )
        if not critical_path or len(critical_path) < 2:
            return chrom
        # Cria janela temporal dinâmica baseada em intensidade LNS
        window_size = max(1, int(makespan * self.lns_shake_intensity))
        threshold = makespan - window_size
        window_ops = [op for op in critical_path if completion_times.get(op, 0.0) >= threshold]
        block_candidates = window_ops if len(window_ops) >= 2 else critical_path
        # Define tamanho de bloco entre 2 e min(5, len(candidates))
        max_block = min(5, len(block_candidates))
        block_size = max(2, min(max_block, int(len(block_candidates) * self.lns_shake_intensity)))
        # Seleciona posição aleatória para bloco
        start_idx = self.rng.randint(0, len(block_candidates) - block_size)
        block_ops = block_candidates[start_idx:start_idx + block_size]
        # Mapeia mini-jobs para o solver
        mini_jobs = []
        op_map = {}
        for i, (job_id, op_id) in enumerate(block_ops):
            machine_id, duration = self.jobs_data[job_id][op_id]
            mini_jobs.append([(machine_id, duration)])
            op_map[i] = (job_id, op_id)
        # Resolve subproblema
        try:
            mini_solver = ORToolsCPSATSolver(
                mini_jobs, len(mini_jobs), self.num_machines)
            mini_schedule = mini_solver.solve(time_limit=self.lns_solver_time_limit)
        except Exception:
            mini_schedule = None
        new_chrom = chrom[:]
        # Se solução válida, reordena bloco
        if mini_schedule and hasattr(mini_schedule, 'operations') and mini_schedule.operations:
            # Ordena por tempo de início
            ops_sorted = sorted(mini_schedule.operations, key=lambda x: x[3])
            # Extrai nova ordem de operações globais
            new_order = [op_map[job_idx] for job_idx, _, _, _, _ in ops_sorted]
            # Posições originais no cromossomo
            indices = [new_chrom.index(op) for op in block_ops]
            # Reinsere na nova ordem
            for pos, op in zip(indices, new_order):
                new_chrom[pos] = op
        else:
            # Fallback: embaralha aleatoriamente o bloco
            indices = [chrom.index(op) for op in block_ops]
            vals = [new_chrom[i] for i in indices]
            self.rng.shuffle(vals)
            for pos, op in zip(indices, vals):
                new_chrom[pos] = op
        return new_chrom

    # --- Novo método para reparar sequências ---
    def _repair_sequence(self, chrom: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Repara a sequência do cromossomo para garantir validade após operadores destrutivos.
        """
        new_chrom = list(chrom)
        for job_id in range(len(self.jobs_data)):
            ops_j = [op for op in new_chrom if op[0] == job_id]
            if not ops_j:
                continue
            sorted_ops = sorted(ops_j, key=lambda x: x[1])
            result = []
            idx = 0
            for op in new_chrom:
                if op[0] == job_id:
                    result.append(sorted_ops[idx])
                    idx += 1
                else:
                    result.append(op)
            new_chrom = result
        return new_chrom

    def local_search(self, chromosome: List[Tuple[int, int]], use_advanced: Optional[bool] = None) -> List[Tuple[int, int]]:
        """
        Executa a busca local sobre o cromossomo fornecido, retornando a melhor solução encontrada.
        """
        start_vnd_time = time.time()
        vnd_iterations = 0
        non_improving_iterations = 0

        # Inicializa estatísticas de vizinhanças para VND e Orquestrador
        all_nh = list(self.operator_map.keys())
        self.neighborhood_stats = {nh: {'attempts': 0, 'successes': 0, 'success_rate': 0.0} for nh in all_nh}

        if self.use_orchestrator and self.orchestrator is None:
            logger.error("Orquestrador habilitado mas não inicializado.")
            return chromosome
        if not self.use_orchestrator: # Garante que neighborhood_stats seja inicializado para VND padrão
            active_neighborhoods_std = list(self.operator_map.keys()) # Usa todos os operadores mapeados
            self.neighborhood_stats = {nh_type: {'attempts': 0, 'successes': 0, 'success_rate': 0.0}
                                       for nh_type in active_neighborhoods_std}

        best_chrom = chromosome[:]
        initial_fitness_calculated = False
        # MODIFICADO: Cache de fitness (cromossomo_tupla -> valor_fitness)
        evaluated_solutions_cache: Dict[Tuple[Any,...], float] = {}

        if self.initial_shake_type:
            shake_operator_method = self.operator_map.get(self.initial_shake_type)
            if shake_operator_method:
                original_lns_intensity = self.lns_shake_intensity
                if self.initial_shake_type == NeighborhoodType.LNS_SHAKE:
                    self.lns_shake_intensity = self.initial_lns_shake_intensity
                
                raw_shaken = shake_operator_method(best_chrom)
                shaken_chrom = self._repair_sequence(raw_shaken) # Reparar antes de avaliar
                
                if self.initial_shake_type == NeighborhoodType.LNS_SHAKE:
                    self.lns_shake_intensity = original_lns_intensity # Restaurar intensidade
                
                if shaken_chrom != best_chrom:
                    best_chrom = shaken_chrom
            else:
                logger.error(f"Operador para Shake Inicial ({self.initial_shake_type.name}) não encontrado!")
        
        # Calcula/obtém fitness inicial para best_chrom (que pode ter sido alterado pelo shake inicial)
        try:
            current_best_chrom_tuple = tuple(best_chrom)
            if current_best_chrom_tuple in evaluated_solutions_cache:
                best_fit = evaluated_solutions_cache[current_best_chrom_tuple]
            else:
                best_fit = self.fitness_func(best_chrom)
                evaluated_solutions_cache[current_best_chrom_tuple] = best_fit
            initial_fitness_calculated = True
        except Exception as e:
            logger.error(f"Erro ao calcular fitness inicial em VND: {e}", exc_info=True)
            return chromosome 
        
        if not initial_fitness_calculated: 
            logger.error("Não foi possível calcular o fitness inicial.")
            return chromosome

        keep_searching = True
        while keep_searching:
            vnd_iterations += 1
            start_iter_time = time.time()
            improvement_in_iteration = False
            
            actual_fitness_calculations_this_iter = 0 
            neighbors_generated_this_iter = 0 # Total de vizinhos gerados (antes de verificar se são iguais ao best_chrom)


            reward_for_reactive_update = 0.0
            chosen_operator_for_reactive_update = None 
            shake_just_happened = False 

            if self.use_orchestrator:
                if self.orchestrator is None: 
                    logger.error("Orquestrador é None. Abortando.")
                    return best_chrom

                nh_type = self.orchestrator.pick()
                operator_method = self.operator_map.get(nh_type)
                chosen_operator_for_reactive_update = nh_type
                if not operator_method:
                    logger.error(f"Operador para {nh_type.name} não encontrado! Pulando.")
                    non_improving_iterations += 1 
                    continue

                improvement_found_in_pick = False
                
                # Processar self.orchestrator_tries_per_pick tentativas
                for try_num in range(self.orchestrator_tries_per_pick):
                    raw_candidate = operator_method(best_chrom)
                    candidate_chrom = self._repair_sequence(raw_candidate)
                    neighbors_generated_this_iter +=1

                    if candidate_chrom == best_chrom:
                        continue

                    candidate_tuple = tuple(candidate_chrom)
                    fit = -1.0
                    
                    if candidate_tuple in evaluated_solutions_cache: 
                        fit = evaluated_solutions_cache[candidate_tuple]
                    else:
                        try:
                            fit = self.fitness_func(candidate_chrom)
                            actual_fitness_calculations_this_iter += 1
                            evaluated_solutions_cache[candidate_tuple] = fit
                        except Exception as e:
                            logger.error(f"Erro ao calcular fitness para candidato (Orch) de {nh_type.name}: {e}", exc_info=True)
                            continue 

                    if fit < best_fit:
                        previous_fit = best_fit
                        best_chrom = candidate_chrom[:]
                        best_fit = fit
                        improvement_in_iteration = True
                        reward_for_reactive_update = max(reward_for_reactive_update, previous_fit - best_fit)
                        improvement_found_in_pick = True
                        self.neighborhood_stats[nh_type]['successes'] += 1
                        break 
                    
                self.neighborhood_stats[nh_type]['attempts'] += self.orchestrator_tries_per_pick
                self.orchestrator.update(nh_type, reward_for_reactive_update)
                # No log de if not improvement_found_in_pick

            else: # VND Padrão
                if not hasattr(self, 'neighborhood_stats') or not self.neighborhood_stats: # Checagem robusta
                     logger.error("VND Padrão: neighborhood_stats não inicializado corretamente.")
                     return best_chrom 

                # Atualiza taxas de sucesso para ordenação
                active_neighborhoods_std = list(self.neighborhood_stats.keys())
                for nh_key in active_neighborhoods_std:
                    stats = self.neighborhood_stats[nh_key]
                    if stats['attempts'] > 0:
                        stats['success_rate'] = stats['successes'] / stats['attempts']
                    else:
                        stats['success_rate'] = 0.0
                
                ordered_neighborhoods = sorted(
                    active_neighborhoods_std, key=lambda nh_k: self.neighborhood_stats[nh_k]['success_rate'], reverse=True)

                for nh_type_std in ordered_neighborhoods:
                    operator_method = self.operator_map.get(nh_type_std)
                    if not operator_method:
                        logger.error(f"Operador VND padrão para {nh_type_std.name} não encontrado.")
                        continue

                    improvement_found_in_nh = False
                    attempts_in_nh = 0
                    
                    while attempts_in_nh < self.max_tries_per_neighborhood:
                        raw_candidate = operator_method(best_chrom)
                        candidate_chrom = self._repair_sequence(raw_candidate)
                        attempts_in_nh += 1
                        neighbors_generated_this_iter +=1

                        if candidate_chrom == best_chrom:
                            continue

                        candidate_tuple = tuple(candidate_chrom)
                        fit = -1.0

                        if candidate_tuple in evaluated_solutions_cache: 
                            fit = evaluated_solutions_cache[candidate_tuple]
                        else:
                            try:
                                fit = self.fitness_func(candidate_chrom)
                                actual_fitness_calculations_this_iter += 1
                                evaluated_solutions_cache[candidate_tuple] = fit
                            except Exception as e:
                                logger.error(f"Erro ao calcular fitness para candidato (VND Std) de {nh_type_std.name}: {e}", exc_info=True)
                                continue 

                        if fit < best_fit:
                            previous_fit = best_fit
                            best_chrom = candidate_chrom[:]
                            best_fit = fit
                            improvement_in_iteration = True
                            improvement_found_in_nh = True
                            self.neighborhood_stats[nh_type_std]['successes'] += 1
                            break 
                    
                    self.neighborhood_stats[nh_type_std]['attempts'] += attempts_in_nh
                    if improvement_found_in_nh:
                        break 
            
            end_iter_time = time.time()
            
            # --- Aprendizado Reativo de Operadores (Gradient Bandit) ---
            if chosen_operator_for_reactive_update and self.use_orchestrator : # Só se orquestrador usou um operador
                self.reactive_attempts_count += 1
                if self.reactive_attempts_count % self.reactive_update_N == 0 and self.orchestrator:
                    total_sum_rewards = sum(stat[0] for stat in self.orchestrator.stats.values())
                    total_num_trials = self.orchestrator.total_trials
                    baseline_reward = total_sum_rewards / total_num_trials if total_num_trials > 0 else 0.0

                    prob_chosen_op = self.operator_probabilities[chosen_operator_for_reactive_update]
                    delta_p_chosen = self.reactive_learning_eta * (reward_for_reactive_update - baseline_reward) * (1 - prob_chosen_op)
                    self.operator_probabilities[chosen_operator_for_reactive_update] += delta_p_chosen

                    for op_type_reactive in self.operator_probabilities:
                        if op_type_reactive != chosen_operator_for_reactive_update:
                            prob_other_op = self.operator_probabilities[op_type_reactive]
                            # Ajuste para contribuição negativa de outros operadores
                            delta_p_other = self.reactive_learning_eta * (reward_for_reactive_update - baseline_reward) * (-prob_other_op)
                            self.operator_probabilities[op_type_reactive] += delta_p_other # Nota: delta_p_other pode ser negativo

                    current_sum_probs = 0.0
                    for op_type_reactive in self.operator_probabilities:
                        self.operator_probabilities[op_type_reactive] = max(0.001, self.operator_probabilities[op_type_reactive]) 
                        current_sum_probs += self.operator_probabilities[op_type_reactive]
                    if current_sum_probs > 0 : # Evitar divisão por zero
                        for op_type_reactive in self.operator_probabilities:
                            self.operator_probabilities[op_type_reactive] /= current_sum_probs

            if not improvement_in_iteration:
                non_improving_iterations += 1

                if self.perform_lns_shake and non_improving_iterations >= self.lns_shake_frequency:
                    logger.info(f"    [LNS Shake Triggered] Após {non_improving_iterations} iterações sem melhoria.")
                    
                    if hasattr(self, 'hyper_orch'):
                        arm = self.hyper_orch.pick()
                        new_int, new_freq = arm
                        # prev_int, prev_freq = self.lns_shake_intensity, self.lns_shake_frequency # Guardar para recompensa
                        self.lns_shake_intensity, self.lns_shake_frequency = new_int, new_freq
                    
                    prev_best_fit_for_shake_reward = best_fit # Fitness antes do shake
                    
                    raw_shaken = self._apply_lns_shake(best_chrom)
                    shaken_chrom = self._repair_sequence(raw_shaken)
                    
                    self.consecutive_shakes_without_improvement += 1
                    
                    if shaken_chrom != best_chrom:
                        shaken_tuple = tuple(shaken_chrom)
                        shaken_fit = -1.0
                        
                        if shaken_tuple in evaluated_solutions_cache:
                            shaken_fit = evaluated_solutions_cache[shaken_tuple]
                        else:
                            try:
                                shaken_fit = self.fitness_func(shaken_chrom)
                                actual_fitness_calculations_this_iter +=1 
                                evaluated_solutions_cache[shaken_tuple] = shaken_fit
                            except Exception as e:
                                logger.error(f"Erro ao calcular fitness pós LNS shake: {e}", exc_info=True)
                                shaken_fit = float('inf') 
                        
                        if shaken_fit != float('inf'): 
                            if self.last_shake_type and self.last_shake_type in self.shake_strategy_stats: # Garante que last_shake_type é válido
                                if shaken_fit < self.last_shake_makespan: # Usa o makespan do shake anterior da mesma ESTRATÉGIA
                                     self.shake_strategy_stats[self.last_shake_type]["successes"] += 1
                            
                            if hasattr(self, 'hyper_orch') and 'arm' in locals(): # Verifica se 'arm' foi definido
                                reward_shake = max(0.0, prev_best_fit_for_shake_reward - shaken_fit)
                                self.hyper_orch.update(arm, reward_shake)
                                    
                            if shaken_fit < best_fit:
                                logger.info("    LNS Shake resultou em melhoria direta!")
                                best_chrom = shaken_chrom[:]
                                best_fit = shaken_fit
                                improvement_in_iteration = True 
                                self.consecutive_shakes_without_improvement = 0
                            else: # Mesmo se não melhorar globalmente, adota para diversificação
                                logger.info("    LNS Shake diversificou, mas não melhorou globalmente. Adotando.")
                                best_chrom = shaken_chrom[:] # Adota mesmo assim
                                best_fit = shaken_fit       # Atualiza best_fit para o fitness do shaken
                            
                            self.last_shake_makespan = shaken_fit # Atualiza o makespan do último shake bem sucedido
                    
                    non_improving_iterations = 0 
                    shake_just_happened = True 

                if self.shared_solution_pool and vnd_iterations % self.share_frequency == 0:
                    self.shared_solution_pool.report(self.thread_id, best_chrom, best_fit)
                    peer_chrom, peer_fitness = self.shared_solution_pool.get_best_excluding(self.thread_id)
                    if peer_chrom and peer_fitness is not None and peer_fitness < best_fit:
                        logger.info(f"Thread {self.thread_id} adotou solução de peer com fitness {peer_fitness:.2f} (anterior {best_fit:.2f}).")
                        best_chrom = peer_chrom[:]
                        best_fit = peer_fitness
                        evaluated_solutions_cache[tuple(best_chrom)] = best_fit 
                        non_improving_iterations = 0 
                        improvement_in_iteration = True 
                elif not self.perform_lns_shake or non_improving_iterations < self.lns_shake_frequency:
                    if not shake_just_happened and not improvement_in_iteration: # Condição de parada mais clara
                        keep_searching = False
            else: # Houve melhoria na iteração
                non_improving_iterations = 0
                self.consecutive_shakes_without_improvement = 0 # Reset se houve melhoria por qualquer meio

        end_vnd_time = time.time()
        total_unique_solutions_in_cache = len(evaluated_solutions_cache) 
        # logger.info(
        #      f"    [VND Final] Tempo: {end_vnd_time - start_vnd_time:.4f}s | Iters: {vnd_iterations} | Fitness Calcs: ??? | Soluções no Cache: {total_unique_solutions_in_cache} | Melhor Fitness: {best_fit:.2f}")
        return best_chrom

    def _get_progressive_shake_intensity(self):
        """Retorna intensidade crescente baseada no número de shakes consecutivos sem melhoria.
        
        A intensidade começa no valor base e aumenta gradualmente até o valor máximo
        conforme ocorrem mais shakes consecutivos sem melhoria.
        
        Returns:
            float: Intensidade de shake ajustada entre base_intensity e max_intensity
        """
        base_intensity = self.lns_shake_intensity
        
        # Limita a intensidade máxima para evitar embaralhar o cromossomo inteiro
        max_intensity = min(self.max_shake_intensity, base_intensity * 3)
        
        # Escala baseada no número de shakes consecutivos sem melhoria
        # A cada 5 shakes sem melhoria, chega à intensidade máxima
        scale_factor = min(1.0, self.consecutive_shakes_without_improvement / 5)
        intensity = base_intensity + (max_intensity - base_intensity) * scale_factor
        
        # logger.debug(f"    Intensidade progressiva de shake: {intensity:.3f} " + 
        #             f"(base: {base_intensity:.3f}, shakes consecutivos: {self.consecutive_shakes_without_improvement})")
        
        return intensity

    def _find_contiguous_blocks(self, indices):
        """Encontra blocos contíguos de índices em uma lista.
        
        Args:
            indices: Lista de índices ordenados
            
        Returns:
            list: Lista de tuplas (início, fim) de blocos contíguos
        """
        if not indices:
            return []
            
        blocks = []
        start = indices[0]
        last = start
        
        for idx in indices[1:]:
            if idx == last + 1:
                last = idx
            else:
                blocks.append((start, last))
                start = idx
                last = idx
                
        blocks.append((start, last))
        return blocks
        
    def _group_by_job(self, operations):
        """Agrupa operações pelo job a que pertencem.
        
        Args:
            operations: Lista de operações como tuplas (job_id, op_id)
            
        Returns:
            dict: Mapeamento de job_id para lista de operações daquele job
        """
        job_groups = {}
        for op in operations:
            job_id = op[0]
            if job_id not in job_groups:
                job_groups[job_id] = []
            job_groups[job_id].append(op)
        return job_groups
    
    def _group_by_machine(self, operations):
        """Agrupa operações pela máquina em que são processadas.
        
        Args:
            operations: Lista de operações como tuplas (job_id, op_id)
            
        Returns:
            dict: Mapeamento de machine_id para lista de operações naquela máquina
        """
        machine_groups = {}
        for op in operations:
            job_id, op_id = op
            machine_id, _ = self.jobs_data[job_id][op_id]
            if machine_id not in machine_groups:
                machine_groups[machine_id] = []
            machine_groups[machine_id].append(op)
        return machine_groups
    
    def _apply_alternative_shake(self, indices_to_shake, chrom, strategy=None):
        """Aplica estratégias alternativas de shake quando CP-SAT falha.
        
        Args:
            indices_to_shake: Índices das operações a serem reorganizadas
            chrom: Cromossomo atual
            strategy: Estratégia específica a usar, ou None para escolha aleatória
            
        Returns:
            list: Novo cromossomo após aplicação da estratégia alternativa
        """
        # Seleciona uma estratégia aleatoriamente ou usa a indicada
        if strategy is None:
            strategy = self.rng.choice(self.alternative_strategies)
        
        # logger.debug(f"    Aplicando estratégia alternativa de shake: {strategy}")
        self.shake_strategy_stats["alternative"]["attempts"] += 1
        self.last_shake_type = "alternative"
        
        new_chrom = chrom[:]
        ops_to_shake = [chrom[i] for i in indices_to_shake]
        
        if strategy == "block_inversion":
            # Inverte blocos contíguos dentro dos índices selecionados
            blocks = self._find_contiguous_blocks(indices_to_shake)
            for start, end in blocks:
                if end - start >= 1:  # Inverte blocos com pelo menos 2 elementos
                    new_chrom[start:end+1] = list(reversed(new_chrom[start:end+1]))
                    
        elif strategy == "guided_shuffle":
            # Embaralha, mas mantém operações do mesmo job próximas
            job_groups = self._group_by_job(ops_to_shake)
            
            # Embaralha a ordem dos jobs e cada job internamente
            job_ids = list(job_groups.keys())
            self.rng.shuffle(job_ids)
            
            new_ops = []
            for job_id in job_ids:
                job_ops = job_groups[job_id]
                self.rng.shuffle(job_ops)
                new_ops.extend(job_ops)
                
            # Substitui as operações originais pelas reorganizadas
            for i, idx in enumerate(indices_to_shake):
                new_chrom[idx] = new_ops[i]
                
        elif strategy == "machine_based_reordering":
            # Reordena por máquinas para tentar reduzir ociosidade
            machine_groups = self._group_by_machine(ops_to_shake)
            
            # Coleta operações agrupadas por máquina em ordem aleatória
            machine_ids = list(machine_groups.keys())
            self.rng.shuffle(machine_ids)
            
            new_ops = []
            for machine_id in machine_ids:
                new_ops.extend(machine_groups[machine_id])
                
            # Substitui as operações originais pelas reorganizadas
            for i, idx in enumerate(indices_to_shake):
                if i < len(new_ops):
                    new_chrom[idx] = new_ops[i]
        
        return new_chrom

    def _apply_critical_path_shake(self, chrom):
        """Aplica shake focado no caminho crítico.
        
        Esta é a Fase 1 do shake multi-fase, com intensidade mais leve
        focada principalmente no caminho crítico.
        
        Args:
            chrom: Cromossomo a ser modificado
            
        Returns:
            list: Novo cromossomo após shake
        """
        size = len(chrom)
        if size < 2:
            return chrom
            
        self.shake_strategy_stats["critical_path"]["attempts"] += 1
        self.last_shake_type = "critical_path"
        
        # Calcula o caminho crítico atual
        completion_times, critical_path, _ = calculate_schedule_and_critical_path(
            chrom, self.num_machines, self.op_details, self.job_predecessors
        )
            
        if not critical_path:
            # Fallback para shake padrão
            logger.warning("Caminho crítico vazio no _apply_critical_path_shake, usando shuffle padrão")
            num_to_shake = max(2, int(size * self._get_progressive_shake_intensity() * 0.5))
            indices_to_shake = sorted(self.rng.sample(range(size), num_to_shake))
            return self._apply_alternative_shake(indices_to_shake, chrom, "guided_shuffle")
        
        # Utiliza uma intensidade mais leve para esta fase
        intensity = min(0.8, self._get_progressive_shake_intensity() * 0.7)
        
        # Seleção priorizada de operações críticas
        critical_ops_set = set(critical_path)
        critical_indices = [i for i, op in enumerate(chrom) if op in critical_ops_set]
        non_critical_indices = [i for i, op in enumerate(chrom) if op not in critical_ops_set]
        
        num_to_shake = max(2, int(len(critical_path) * intensity))
        num_critical = min(len(critical_indices), int(num_to_shake * 0.8))
        num_non_critical = min(len(non_critical_indices), num_to_shake - num_critical)
        
        # logger.debug(f"    Critical Path Shake: Escolhendo {num_critical} ops críticas e {num_non_critical} não-críticas")
        
        indices_to_shake = []
        if num_critical > 0:
            indices_to_shake.extend(self.rng.sample(critical_indices, num_critical))
        if num_non_critical > 0:
            indices_to_shake.extend(self.rng.sample(non_critical_indices, num_non_critical))
            
        indices_to_shake = sorted(indices_to_shake)
        
        # Tenta usar CP-SAT para este conjunto menor primeiro
        new_chrom = self._apply_cp_sat_to_indices(indices_to_shake, chrom)
        
        if new_chrom != chrom:
            self.shake_strategy_stats["cp_sat"]["successes"] += 1
            return new_chrom
        
        # Se CP-SAT falhar, usa inversão de blocos (menos disruptivo)
        return self._apply_alternative_shake(indices_to_shake, chrom, "block_inversion")
    
    def _apply_mixed_strategy_shake(self, chrom):
        """Aplica shake com estratégia mista.
        
        Esta é a Fase 2 do shake multi-fase, equilibrando intensidade e direcionamento.
        
        Args:
            chrom: Cromossomo a ser modificado
            
        Returns:
            list: Novo cromossomo após shake
        """
        size = len(chrom)
        if size < 2:
            return chrom
            
        self.shake_strategy_stats["mixed_strategy"]["attempts"] += 1
        self.last_shake_type = "mixed_strategy"
        
        # Intensidade moderada para esta fase
        intensity = self._get_progressive_shake_intensity()
        num_to_shake = max(3, int(size * intensity))
        
        # Obtém o caminho crítico, mas não foca tanto nele quanto a Fase 1
        completion_times, critical_path, _ = calculate_schedule_and_critical_path(
            chrom, self.num_machines, self.op_details, self.job_predecessors
        )
            
        if critical_path:
            critical_ops_set = set(critical_path)
            critical_indices = [i for i, op in enumerate(chrom) if op in critical_ops_set]
            non_critical_indices = [i for i, op in enumerate(chrom) if op not in critical_ops_set]
            
            # Equilíbrio entre operações críticas e não críticas
            num_critical = min(len(critical_indices), int(num_to_shake * 0.6))
            num_non_critical = min(len(non_critical_indices), num_to_shake - num_critical)
            
            indices_to_shake = []
            if num_critical > 0:
                indices_to_shake.extend(self.rng.sample(critical_indices, num_critical))
            if num_non_critical > 0:
                indices_to_shake.extend(self.rng.sample(non_critical_indices, num_non_critical))
        else:
            # Fallback se não houver caminho crítico
            indices_to_shake = sorted(self.rng.sample(range(size), num_to_shake))
            
        indices_to_shake = sorted(indices_to_shake)
        
        # Tenta CP-SAT primeiro
        new_chrom = self._apply_cp_sat_to_indices(indices_to_shake, chrom)
        
        if new_chrom != chrom:
            self.shake_strategy_stats["cp_sat"]["successes"] += 1
            return new_chrom
        
        # Se falhar, escolhe aleatoriamente entre as estratégias alternativas
        return self._apply_alternative_shake(indices_to_shake, chrom)
    
    def _apply_strong_diversification_shake(self, chrom):
        """Aplica shake com forte diversificação.
        
        Esta é a Fase 3 do shake multi-fase, com alta intensidade
        para escapar de regiões de forte atração.
        
        Args:
            chrom: Cromossomo a ser modificado
            
        Returns:
            list: Novo cromossomo após shake
        """
        size = len(chrom)
        if size < 2:
            return chrom
            
        self.shake_strategy_stats["strong_diversification"]["attempts"] += 1
        self.last_shake_type = "strong_diversification"
        
        # Alta intensidade para esta fase
        intensity = min(0.9, self._get_progressive_shake_intensity() * 1.3)
        num_to_shake = max(5, int(size * intensity))
        
        # Nesta fase, misturamos caminho crítico e seleção aleatória com viés para
        # máquinas gargalo (máquinas com maior tempo total de ocupação)
        # Identificar máquinas gargalo
        machine_loads = {m: 0.0 for m in range(self.num_machines)}
        
        for job_id, ops in enumerate(self.jobs_data):
            for op_id, (machine_id, duration) in enumerate(ops):
                machine_loads[machine_id] += duration
                
        # Ordenar máquinas por carga (decrescente)
        bottleneck_machines = sorted(machine_loads.keys(), key=lambda m: machine_loads[m], reverse=True)
        
        # Obter operações de máquinas gargalo (top 30% das máquinas)
        num_bottleneck = max(1, int(self.num_machines * 0.3))
        bottleneck_set = set(bottleneck_machines[:num_bottleneck])
        
        bottleneck_indices = []
        for i, (job_id, op_id) in enumerate(chrom):
            machine_id, _ = self.jobs_data[job_id][op_id]
            if machine_id in bottleneck_set:
                bottleneck_indices.append(i)
                
        # Selecionar operações para shake
        if len(bottleneck_indices) >= num_to_shake * 0.5:
            # Temos operações de gargalo suficientes
            to_take = min(len(bottleneck_indices), int(num_to_shake * 0.7))
            indices_to_shake = self.rng.sample(bottleneck_indices, to_take)
            
            # Complementar com operações aleatórias
            remaining = num_to_shake - to_take
            other_indices = [i for i in range(size) if i not in indices_to_shake]
            if remaining > 0 and other_indices:
                to_add = min(len(other_indices), remaining)
                indices_to_shake.extend(self.rng.sample(other_indices, to_add))
        else:
            # Fallback para seleção aleatória
            indices_to_shake = self.rng.sample(range(size), num_to_shake)
            
        indices_to_shake = sorted(indices_to_shake)
        
        # Tenta CP-SAT primeiro
        new_chrom = self._apply_cp_sat_to_indices(indices_to_shake, chrom)
        
        if new_chrom != chrom:
            self.shake_strategy_stats["cp_sat"]["successes"] += 1
            return new_chrom
        
        # Se falhar, alterna entre estratégias mais diversificadoras
        strategy = self.rng.choice(["guided_shuffle", "machine_based_reordering"])
        return self._apply_alternative_shake(indices_to_shake, chrom, strategy)
        
    def _apply_cp_sat_to_indices(self, indices_to_shake, chrom):
        """Utiliza o resolvedor CP-SAT para otimizar um subconjunto de operações.
        
        Args:
            indices_to_shake: Índices das operações a serem otimizadas
            chrom: Cromossomo atual
            
        Returns:
            list: Novo cromossomo após otimização CP-SAT ou o original se falhar
        """
        self.shake_strategy_stats["cp_sat"]["attempts"] += 1
        self.last_shake_type = "cp_sat"
        
        ops_to_shake = [chrom[i] for i in indices_to_shake]
        op_to_original_index = {op: idx for op, idx in zip(ops_to_shake, indices_to_shake)}
        
        mini_jobs = []
        op_map_mini_to_global = {}
        
        for i, (job_id, op_id) in enumerate(ops_to_shake):
            if job_id >= len(self.jobs_data) or op_id >= len(self.jobs_data[job_id]):
                logger.error(f"CP-SAT: Índice inválido ({job_id}, {op_id}) acessado em jobs_data.")
                return chrom
                
            machine_id, duration = self.jobs_data[job_id][op_id]
            mini_jobs.append([(machine_id, duration)])
            op_map_mini_to_global[(i, 0)] = (job_id, op_id)
            
        cpsat_solution_order = None
        try:
            mini_solver = ORToolsCPSATSolver(
                mini_jobs, len(mini_jobs), self.num_machines)
            mini_schedule = mini_solver.solve(
                time_limit=self.lns_solver_time_limit)
                
            if mini_schedule and mini_schedule.operations:
                mini_schedule.operations.sort(key=lambda x: x[3])  # Ordenar por tempo de início
                cpsat_solution_order = [op_map_mini_to_global[(mini_job_id, mini_op_id)]
                                       for mini_job_id, mini_op_id, _, _, _ in mini_schedule.operations]
                # logger.debug(f"    CP-SAT encontrou ordem otimizada para {len(indices_to_shake)} operações.")
            else:
                logger.warning("    CP-SAT não encontrou solução para o subproblema.")
                return chrom
        except Exception as e:
            logger.error(f"    Erro ao executar CP-SAT: {e}", exc_info=True)
            return chrom
            
        new_chrom = chrom[:]
        if cpsat_solution_order:
            for i, original_index in enumerate(indices_to_shake):
                new_chrom[original_index] = cpsat_solution_order[i]
                
        return new_chrom
    
    def _apply_multi_phase_shake(self, chrom):
        """Aplica shake multi-fase adaptativo.
        
        Escolhe a fase apropriada com base no histórico de estagnação.
        
        Args:
            chrom: Cromossomo a ser modificado
            
        Returns:
            list: Novo cromossomo após shake
        """
        # Determina a fase baseada no número de shakes consecutivos sem melhoria
        if self.consecutive_shakes_without_improvement < 2:
            # Fase 1: Shake leve focado no caminho crítico
            logger.info(f"    [Shake Fase 1] Shake leve focado no caminho crítico.")
            return self._apply_critical_path_shake(chrom)
        elif self.consecutive_shakes_without_improvement < 4:
            # Fase 2: Shake moderado com estratégia mista
            logger.info(f"    [Shake Fase 2] Shake moderado com estratégia mista.")
            return self._apply_mixed_strategy_shake(chrom)
        else:
            # Fase 3: Shake intenso com forte diversificação
            logger.info(f"    [Shake Fase 3] Shake intenso com forte diversificação.")
            return self._apply_strong_diversification_shake(chrom)

class NeighborhoodOrchestrator:
    def __init__(self, neighborhoods: List[NeighborhoodType], c: float = 1.0, initial_attempts: int = 1, initial_reward: float = 0.0, operator_probabilities_ref: Optional[Dict[NeighborhoodType, float]] = None):
        if not neighborhoods:
            raise ValueError("A lista de vizinhanças não pode ser vazia.")
        if initial_attempts <= 0:
            raise ValueError("initial_attempts deve ser positivo.")
        self.neighborhoods = list(neighborhoods)
        self.stats = {n: [float(initial_reward), int(initial_attempts)] for n in self.neighborhoods}
        self.total_trials = sum(s[1] for s in self.stats.values())
        self.c = c
        self.rng = random.Random()
        self.operator_probabilities_ref = operator_probabilities_ref # Referência para pesos reativos
        logger.info(
            f"NeighborhoodOrchestrator inicializado com {len(self.neighborhoods)} vizinhanças, c={self.c}")

    def pick(self) -> NeighborhoodType:
        best_neighborhood = None
        max_ucb_score = -float('inf')
        log_total_trials = math.log(max(1, self.total_trials))
        for n in self.neighborhoods:
            sum_rewards, attempts = self.stats[n]
            if attempts == 0:
                ucb_score = float('inf')
            else:
                average_reward = sum_rewards / attempts
                exploration_bonus = self.c * math.sqrt(log_total_trials / attempts)
                ucb_value = average_reward + exploration_bonus
                # Pondera pelo peso reativo, se disponível
                prob_weight = self.operator_probabilities_ref.get(n, 1.0) if self.operator_probabilities_ref else 1.0
                ucb_score = ucb_value * prob_weight
            if ucb_score > max_ucb_score:
                max_ucb_score = ucb_score
                best_neighborhood = n
            elif abs(ucb_score - max_ucb_score) < 1e-9 and self.rng.choice([True, False]):
                best_neighborhood = n
        if best_neighborhood is None:
            logger.warning("UCB1 não selecionou, escolhendo aleatoriamente.")
            best_neighborhood = self.rng.choice(self.neighborhoods)
        return best_neighborhood

    def update(self, neighborhood: NeighborhoodType, reward: float):
        if neighborhood not in self.stats:
            logger.error(f"Tentativa de atualizar vizinhança desconhecida: {neighborhood}")
            return
        self.stats[neighborhood][0] += reward
        self.stats[neighborhood][1] += 1
        self.total_trials += 1

    def set_rng(self, rng_instance: random.Random):
        self.rng = rng_instance

class HyperparameterOrchestrator:
    """Orquestrador UCB1 para escolher pares de hiperparâmetros (intensidade, frequência)"""
    def __init__(self, arms: list, c: float = 1.0, initial_attempts: int = 1, initial_reward: float = 0.0):
        if not arms:
            raise ValueError("Lista de hiperparâmetros não pode ser vazia.")
        self.arms = list(arms)
        self.stats = {a: [float(initial_reward), int(initial_attempts)] for a in self.arms}
        self.total_trials = sum(v[1] for v in self.stats.values())
        self.c = c
        self.rng = random.Random()

    def pick(self):
        best = None
        max_score = -float('inf')
        log_total = math.log(max(1, self.total_trials))
        for a in self.arms:
            sum_r, attempts = self.stats[a]
            if attempts == 0:
                score = float('inf')
            else:
                score = (sum_r/attempts) + self.c * math.sqrt(log_total/attempts)
            if score > max_score:
                max_score = score
                best = a
        return best if best is not None else self.rng.choice(self.arms)

    def update(self, arm, reward: float):
        if arm not in self.stats:
            return
        self.stats[arm][0] += reward
        self.stats[arm][1] += 1
        self.total_trials += 1

    def set_rng(self, rng_instance: random.Random):
        self.rng = rng_instance
