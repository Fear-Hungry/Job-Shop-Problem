# Conteúdo inicial para jssp_drl_env.py 

import gymnasium as gym # Mudança para Gymnasium
from gymnasium import spaces # Mudança para Gymnasium
import numpy as np
import random
import logging # Adicionado para logging
import copy # Adicionado para cópias profundas de cromossomos

# Supondo que seus módulos GA e Local Search possam ser importados assim:
# Estes imports precisarão ser ajustados conforme a estrutura final do seu projeto
# e como você expõe as funcionalidades necessárias.
from src.ga.solver import GeneticAlgorithmSolver # Ou os componentes que você precisa
from src.ga.fitness import FitnessCalculator # Exemplo, se você tiver uma classe fitness
from src.ga.population import Population # Exemplo
from src.ga.initialization import RandomInitialization # Exemplo
from src.ga.selection import TournamentSelection # Exemplo
from src.ga.genetic_operators.crossover import PMXCrossover # Exemplo
from src.ga.genetic_operators.mutation import SwapMutation # Exemplo

from src.local_search.strategies import VNDLocalSearch
from src.local_search.neighborhood_operators import SwapOperator # e outros se for usar granularmente

logger = logging.getLogger(__name__) # Adicionado


class JSSPDrlHyperHeuristicEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 4} # Atualizado para Gymnasium

    def __init__(self, jobs_data, num_machines,
                 population_size=50,
                 ga_generations_per_drl_step=1, # Quantas gerações GA por ação DRL "GA_STEP"
                 max_drl_steps_per_episode=100,
                 # Parâmetros para operadores GA
                 tournament_size_ga: int = 3,
                 crossover_rate_ga: float = 0.8,
                 mutation_rate_ga: float = 0.1,
                 elitism_size_ga: int = 1,
                 # Parâmetros para VND (apenas os mais importantes, outros podem ser default no VND)
                 vnd_max_tries_per_neighborhood: int = 10,
                 vnd_lns_shake_frequency: int = 0, # Desabilitar LNS no VND por padrão para simplificar
                 vnd_lns_shake_intensity: float = 0.2
                 ):
        super(JSSPDrlHyperHeuristicEnv, self).__init__()

        self.jobs_data = jobs_data
        self.num_machines = num_machines
        self.population_size = population_size
        self.ga_generations_per_drl_step = ga_generations_per_drl_step
        self.max_drl_steps_per_episode = max_drl_steps_per_episode
        self.current_drl_step = 0

        # Parâmetros GA
        self.tournament_size_ga = tournament_size_ga
        self.crossover_rate_ga = crossover_rate_ga
        self.mutation_rate_ga = mutation_rate_ga
        self.elitism_size_ga = max(0, elitism_size_ga) # Garantir não negativo

        # --- Definição do Espaço de Ação ---
        # Ação 0: Executar N gerações do Algoritmo Genético completo (Sel+Cross+Mut+Eval)
        # Ação 1: Aplicar VND ao melhor indivíduo da população
        # Ação 2: Aplicar apenas Mutação em X% da população (pode ser útil para diversificar)
        # Ação 3: Aplicar apenas Crossover para gerar uma nova população (intensificar boas features)
        # Ação 4: Aplicar VND em P% dos melhores indivíduos (mais intensivo)
        self.n_actions = 5
        self.action_space = spaces.Discrete(self.n_actions)

        # --- Definição do Espaço de Observação ---
        # 1. Melhor fitness atual (normalizado)
        # 2. Fitness médio da população (normalizado)
        # 3. Diversidade da população (desvio padrão do fitness, normalizado)
        # 4. Número de passos DRL desde a última melhoria global (normalizado)
        # 5. Proporção do orçamento de passos DRL restantes
        # (Adicionar mais features pode ser útil, ex: taxa de sucesso de operadores recentes)
        self.n_obs_features = 5
        self.observation_space = spaces.Box(low=0.0, high=1.0, # Normalizado entre 0 e 1
                                            shape=(self.n_obs_features,), dtype=np.float32)

        # --- Componentes do GA ---
        self.fitness_calculator = FitnessCalculator(jobs_data, num_machines)
        # A função de fitness que o GA e o VND usarão
        self.calculate_fitness_func = self.fitness_calculator.calculate_makespan

        self.population_initializer = RandomInitialization(
            jobs_data=self.jobs_data,
            num_machines=self.num_machines,
            evaluate_population=False # O ambiente DRL controlará a avaliação inicial
        )
        self.selection_op = TournamentSelection(tournament_size=self.tournament_size_ga)
        self.crossover_op = PMXCrossover(crossover_rate=1.0) # Taxa interna do operador é 1.0, controlaremos externamente
        self.mutation_op = SwapMutation(mutation_rate=1.0)   # Taxa interna do operador é 1.0, controlaremos externamente

        # --- Componentes de Busca Local ---
        self.vnd_search = VNDLocalSearch(
            fitness_func=self.calculate_fitness_func,
            jobs=self.jobs_data,
            num_machines=self.num_machines,
            max_tries_per_neighborhood=vnd_max_tries_per_neighborhood,
            lns_shake_frequency=vnd_lns_shake_frequency,
            lns_shake_intensity=vnd_lns_shake_intensity,
            use_orchestrator=False, # Para simplificar inicialmente, podemos habilitar depois
            # Outras configs do VND podem ser adicionadas ou usar defaults
        )

        # Estado interno do ambiente DRL
        self.best_fitness_overall = float('inf')
        self.steps_since_last_improvement = 0
        self.current_population_chromosomes: list[list[tuple[int, int]]] = []
        self.current_population_fitnesses: list[float] = []

        # Para normalização do fitness (estimativas iniciais, podem ser adaptativas)
        # Estes são exemplos, você precisará ajustá-los ou usar uma normalização mais robusta.
        self.min_expected_fitness = 50 # Exemplo
        self.max_expected_fitness = 5000 # Exemplo

        logger.info(f"JSSPDrlHyperHeuristicEnv inicializado com {self.n_actions} ações e {self.n_obs_features} features de observação.")

    def _normalize_fitness(self, fitness_value):
        if self.max_expected_fitness == self.min_expected_fitness:
            return 0.5 # Evita divisão por zero, retorna valor neutro
        # Inverte para que menor (melhor) fitness seja mais próximo de 1.0
        norm_val = (self.max_expected_fitness - fitness_value) / (self.max_expected_fitness - self.min_expected_fitness)
        return np.clip(norm_val, 0.0, 1.0)

    def _get_obs(self):
        if not self.current_population_fitnesses:
            return np.zeros(self.n_obs_features, dtype=np.float32)

        current_best_fitness = min(self.current_population_fitnesses)
        avg_fitness = np.mean(self.current_population_fitnesses)
        diversity = np.std(self.current_population_fitnesses) if len(self.current_population_fitnesses) > 1 else 0.0

        # Normalização
        norm_best_fitness = self._normalize_fitness(current_best_fitness)
        norm_avg_fitness = self._normalize_fitness(avg_fitness)
        
        # Normalizar diversidade (ex: 0 a 200 de std dev -> 0 a 1)
        # Max_expected_diversity pode ser estimado ou aprendido
        max_expected_diversity = (self.max_expected_fitness - self.min_expected_fitness) / 4 
        norm_diversity = np.clip(diversity / max_expected_diversity if max_expected_diversity > 0 else 0.0, 0.0, 1.0)
        
        norm_steps_since_imp = np.clip(self.steps_since_last_improvement / (self.max_drl_steps_per_episode / 2.0), 0.0, 1.0) # Estagnação atinge 1.0 na metade do episódio
        norm_progress = self.current_drl_step / self.max_drl_steps_per_episode

        obs = np.array([
            norm_best_fitness,
            norm_avg_fitness,
            norm_diversity,
            norm_steps_since_imp,
            norm_progress
        ], dtype=np.float32)
        return obs

    def _calculate_reward(self, old_overall_best_fitness, new_overall_best_fitness):
        # Recompensa baseada na melhoria do MELHOR fitness GERAL encontrado até agora.
        # Isso incentiva o agente a encontrar novas soluções de ponta.
        reward = 0.0
        improvement_overall = old_overall_best_fitness - new_overall_best_fitness

        if improvement_overall > 0:
            # Recompensa proporcional à magnitude da melhoria.
            # A escala pode ser ajustada. Se fitnesses variam muito, normalizar a melhoria pode ser bom.
            reward += improvement_overall * 0.1 # Ex: melhoria de 10 pontos -> recompensa de 1
            logger.debug(f"Recompensa positiva por melhoria global: {reward:.2f}")
        elif improvement_overall < 0: # Piorou o melhor global (raro se sempre mantemos o melhor)
            reward -= abs(improvement_overall) * 0.05 # Pequena penalidade
            logger.debug(f"Recompensa negativa por piora global: {reward:.2f}")
        else: # Sem mudança no melhor global
            # Poderíamos dar uma pequena recompensa negativa por passo para incentivar a eficiência,
            # ou recompensar a melhoria do fitness médio da população, ou manutenção da diversidade.
            # Por agora, sem recompensa/penalidade por estagnação no melhor global.
            pass
        
        # Adicionar uma pequena penalidade por passo para incentivar a conclusão mais rápida?
        # reward -= 0.01 
        
        return reward

    def reset(self):
        self.current_drl_step = 0
        self.steps_since_last_improvement = 0
        self.seed_val = random.randint(0, 1_000_000) if self.seed_val is None else self.seed_val # Para consistência se seed não for passado no reset
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        
        # (Re)inicializar operadores com o seed, se eles o utilizarem para reprodutibilidade interna
        # Nossos operadores atuais não parecem ter um método set_seed, então a semente global é usada.
        # Se tivessem, seria algo como: self.selection_op.seed(self.seed_val)

        # Inicializar a população do GA
        initial_pop_tuples = self.population_initializer.generate_population(self.population_size)
        self.current_population_chromosomes = [list(item[0]) for item in initial_pop_tuples] # item[0] é o cromossomo
        
        # Avaliar a população inicial
        self.current_population_fitnesses = []
        for chrom in self.current_population_chromosomes:
            try:
                fitness = self.calculate_fitness_func(chrom)
                self.current_population_fitnesses.append(fitness)
            except Exception as e:
                logger.error(f"Erro ao calcular fitness para cromossomo no reset: {chrom}. Erro: {e}", exc_info=True)
                # Lidar com erro: atribuir fitness muito alto ou remover indivíduo?
                self.current_population_fitnesses.append(float('inf'))


        if self.current_population_fitnesses:
             self.best_fitness_overall = min(self.current_population_fitnesses)
        else:
            # Isso não deveria acontecer se a inicialização funcionar
            logger.error("População vazia ou toda com erro após inicialização no reset!")
            self.best_fitness_overall = float('inf')
            # Forçar um estado de erro ou retornar observação padrão de falha
            return np.zeros(self.n_obs_features, dtype=np.float32)


        logger.info(f"Ambiente DRL resetado. Melhor fitness inicial: {self.best_fitness_overall:.2f}")
        obs, info = self._get_obs_and_info() # Modificado para retornar info também
        return obs, info # Gym padrão para reset a partir da v0.26

    def _execute_ga_cycle(self, current_chromosomes: list, current_fitnesses: list, custom_mutation_op=None, custom_crossover_rate=None, custom_mutation_rate=None):
        """
        Executa um ciclo de GA: elitismo, seleção, crossover, mutação, avaliação.
        Controla taxas de crossover e mutação externamente.
        Usa deepcopy para mutação, pois SwapMutation modifica no local.
        """
        if not current_chromosomes or not current_fitnesses or len(current_chromosomes) != len(current_fitnesses):
            logger.error("Dados de população inválidos para _execute_ga_cycle.")
            # Retornar cópias para evitar modificar o estado anterior em caso de erro aqui
            return copy.deepcopy(current_chromosomes), list(current_fitnesses)

        next_pop_chromosomes = []
        population_with_fitness = list(zip(current_chromosomes, current_fitnesses))

        # 1. Elitismo
        if self.elitism_size_ga > 0 and len(population_with_fitness) >= self.elitism_size_ga:
            # Ordenar por fitness (menor é melhor)
            elites_with_fitness = sorted(population_with_fitness, key=lambda x: x[1])[:self.elitism_size_ga]
            next_pop_chromosomes.extend([copy.deepcopy(chrom) for chrom, fit in elites_with_fitness])

        # 2. Geração de Filhos (Seleção, Crossover, Mutação)
        num_offspring_to_generate = self.population_size - len(next_pop_chromosomes)
        
        # Usar taxas customizadas se fornecidas, senão usar as do ambiente
        active_crossover_rate = custom_crossover_rate if custom_crossover_rate is not None else self.crossover_rate_ga
        active_mutation_rate = custom_mutation_rate if custom_mutation_rate is not None else self.mutation_rate_ga
        mutation_operator_to_use = custom_mutation_op if custom_mutation_op else self.mutation_op

        offspring_generated_count = 0
        max_selection_attempts_per_parent_pair = 10 # Para evitar loops infinitos se a população for muito homogênea

        while offspring_generated_count < num_offspring_to_generate:
            # Seleciona pais
            # TournamentSelection.select espera [(cromossomo, fitness), ...] e retorna um cromossomo.
            parent1_chrom = self.selection_op.select(population_with_fitness) # Retorna o cromossomo
            
            selection_attempts = 0
            parent2_chrom = self.selection_op.select(population_with_fitness)
            while parent1_chrom == parent2_chrom and selection_attempts < max_selection_attempts_per_parent_pair:
                 parent2_chrom = self.selection_op.select(population_with_fitness)
                 selection_attempts += 1
            
            if parent1_chrom == parent2_chrom and selection_attempts >= max_selection_attempts_per_parent_pair:
                logger.warning("Não foi possível selecionar dois pais diferentes. Usando o mesmo pai para gerar via mutação ou cópia.")
                # Estratégia: Se não conseguir pais diferentes, pegue um e mute-o ou apenas copie.
                child1_cand = copy.deepcopy(parent1_chrom)
                # Tenta gerar o segundo filho de forma similar ou pular.
                # Por simplicidade aqui, vamos focar em gerar um filho e preencher o resto depois se necessário.
            else:
                # Crossover
                if random.random() < active_crossover_rate:
                    # PMXCrossover.crossover recebe dois cromossomos e retorna (filho1, filho2)
                    # Os operadores foram instanciados com taxa 1.0, então eles sempre tentam cruzar.
                    child1_cand_raw, child2_cand_raw = self.crossover_op.crossover(copy.deepcopy(parent1_chrom), copy.deepcopy(parent2_chrom))
                else:
                    # Sem crossover, filhos são cópias dos pais
                    child1_cand_raw = copy.deepcopy(parent1_chrom)
                    child2_cand_raw = copy.deepcopy(parent2_chrom)

            # Processar o primeiro candidato a filho
            # Mutação
            if random.random() < active_mutation_rate:
                # SwapMutation.mutate modifica o cromossomo no local e o retorna.
                # Precisamos passar uma cópia para não afetar o original (child1_cand_raw)
                # se ele for usado novamente (ex: se child2_cand_raw for o mesmo objeto).
                child1_mutated = mutation_operator_to_use.mutate(copy.deepcopy(child1_cand_raw))
                next_pop_chromosomes.append(child1_mutated)
            else:
                next_pop_chromosomes.append(copy.deepcopy(child1_cand_raw)) # Adiciona sem mutação
            offspring_generated_count += 1

            if offspring_generated_count < num_offspring_to_generate:
                # Processar o segundo candidato a filho
                if random.random() < active_mutation_rate:
                    child2_mutated = mutation_operator_to_use.mutate(copy.deepcopy(child2_cand_raw))
                    next_pop_chromosomes.append(child2_mutated)
                else:
                    next_pop_chromosomes.append(copy.deepcopy(child2_cand_raw))
                offspring_generated_count += 1
        
        # Avaliar a nova população (apenas os filhos gerados, pois os elites já têm fitness)
        newly_generated_chromosomes = next_pop_chromosomes[self.elitism_size_ga:]
        newly_generated_fitnesses = []
        for chrom in newly_generated_chromosomes:
            try:
                fitness = self.calculate_fitness_func(list(chrom)) # Garantir que é lista
                newly_generated_fitnesses.append(fitness)
            except Exception as e:
                logger.warning(f"Erro ao calcular fitness para filho: {chrom}. Erro: {e}. Fitness Inf.")
                newly_generated_fitnesses.append(float('inf'))

        # Montar a população final e fitnesses
        final_pop_chromosomes = next_pop_chromosomes[:self.elitism_size_ga] + newly_generated_chromosomes
        final_pop_fitnesses = [fit for chrom, fit in elites_with_fitness][:self.elitism_size_ga] + newly_generated_fitnesses 
        
        # Garantir o tamanho da população (caso num_offspring_to_generate não seja par)
        # Se next_pop_chromosomes ficou maior que population_size devido ao elitismo + filhos, truncar.
        # Se ficou menor, precisaria de uma estratégia de preenchimento (ex: repetir os melhores filhos ou re-selecionar).
        # A lógica acima tenta gerar exatamente `num_offspring_to_generate`.
        
        if len(final_pop_chromosomes) > self.population_size:
            final_pop_chromosomes = final_pop_chromosomes[:self.population_size]
            final_pop_fitnesses = final_pop_fitnesses[:self.population_size]
        elif len(final_pop_chromosomes) < self.population_size:
            logger.warning(f"Tamanho da população final ({len(final_pop_chromosomes)}) menor que o esperado ({self.population_size}). Preenchendo com cópias dos melhores da geração anterior...")
            num_to_fill = self.population_size - len(final_pop_chromosomes)
            # Pega os melhores da população ANTERIOR (que entrou na função) para preencher
            # Isso é uma forma simples, poderia ser mais sofisticado.
            if population_with_fitness:
                 fill_candidates = sorted(population_with_fitness, key=lambda x: x[1])
                 for i in range(num_to_fill):
                     if not fill_candidates: break # Sem mais candidatos
                     chrom_to_add, fit_to_add = fill_candidates[i % len(fill_candidates)] # Pega circularmente dos melhores
                     final_pop_chromosomes.append(copy.deepcopy(chrom_to_add))
                     final_pop_fitnesses.append(fit_to_add) # Fitness já conhecido
            else: # Se a população de entrada estava vazia, não há como preencher.
                logger.error("Não foi possível preencher a população pois a população de entrada estava vazia.")

        return final_pop_chromosomes, final_pop_fitnesses

    def _get_obs_and_info(self): # Helper para combinar lógica de observação e info
        obs = self._get_obs()
        info = {
            'current_best_fitness_in_pop': min(self.current_population_fitnesses) if self.current_population_fitnesses else float('inf'),
            'overall_best_fitness': self.best_fitness_overall,
            'steps_since_last_improvement': self.steps_since_last_improvement,
            'drl_step': self.current_drl_step
        }
        return obs, info

    def step(self, action: int):
        self.current_drl_step += 1
        old_overall_best_fitness = self.best_fitness_overall

        if not self.current_population_chromosomes or not self.current_population_fitnesses:
            logger.error("ERRO: Tentando dar step sem população inicializada ou avaliada.")
            obs, info = self._get_obs_and_info()
            return obs, 0, True, True, info # Adicionado truncated=True para Gym v0.26+

        current_info = {'action_taken': action, 'status': 'success'} # Info para o step atual

        # --- Lógica da Ação ---
        if action == 0: # Executar N gerações do Algoritmo Genético
            logger.debug(f"DRL Action: Executar GA (gerações: {self.ga_generations_per_drl_step})")
            try:
                temp_chroms = copy.deepcopy(self.current_population_chromosomes)
                temp_fitnesses = list(self.current_population_fitnesses) # list() cria cópia superficial
                for _ in range(self.ga_generations_per_drl_step):
                    temp_chroms, temp_fitnesses = self._execute_ga_cycle(temp_chroms, temp_fitnesses)
                self.current_population_chromosomes = temp_chroms
                self.current_population_fitnesses = temp_fitnesses
                current_info['ga_generations_executed'] = self.ga_generations_per_drl_step
            except Exception as e:
                logger.error(f"Erro durante a Ação GA: {e}", exc_info=True)
                current_info['status'] = 'error_ga_step'


        elif action == 1: # Aplicar VND ao melhor indivíduo
            logger.debug("DRL Action: Aplicar VND ao melhor")
            try:
                if not self.current_population_chromosomes:
                    logger.warning("Sem cromossomos para aplicar VND.")
                    current_info['status'] = 'no_chrom_for_vnd'
                else:
                    best_idx = np.argmin(self.current_population_fitnesses)
                    best_chrom_before_vnd = copy.deepcopy(self.current_population_chromosomes[best_idx])
                    fitness_before_vnd = self.current_population_fitnesses[best_idx]

                    chrom_after_vnd = self.vnd_search.local_search(best_chrom_before_vnd) # VND retorna o cromossomo
                    fitness_after_vnd = self.calculate_fitness_func(list(chrom_after_vnd))
                    
                    self.current_population_chromosomes[best_idx] = chrom_after_vnd
                    self.current_population_fitnesses[best_idx] = fitness_after_vnd
                    logger.debug(f"VND no melhor: {fitness_before_vnd:.2f} -> {fitness_after_vnd:.2f}")
                    current_info['vnd_improvement'] = fitness_before_vnd - fitness_after_vnd
            except Exception as e:
                logger.error(f"Erro durante a Ação VND no melhor: {e}", exc_info=True)
                current_info['status'] = 'error_vnd_best'

        elif action == 2: # Aplicar apenas Mutação em X% da população
            mutation_target_percentage = 0.2 # Ex: 20%
            num_to_mutate = int(self.population_size * mutation_target_percentage)
            logger.debug(f"DRL Action: Aplicar Mutação em {num_to_mutate} indivíduos")
            try:
                if not self.current_population_chromosomes: # Checagem adicional
                     logger.warning("Ação 2: População vazia, não é possível aplicar mutação em lote.")
                     current_info['status'] = 'empty_pop_for_batch_mutation'
                else:
                    indices_to_mutate = random.sample(range(len(self.current_population_chromosomes)), min(num_to_mutate, len(self.current_population_chromosomes)))
                    mutated_count_actual = 0
                    for idx in indices_to_mutate:
                        # A taxa de mutação interna do self.mutation_op é 1.0, então ele sempre tenta mutar.
                        # O controle de taxa (se queremos mutar ESTE indivíduo ou não) deve ser externo aqui se necessário.
                        # Por simplicidade, aqui estamos mutando todos os selecionados (num_to_mutate).
                        original_chrom = self.current_population_chromosomes[idx]
                        chrom_to_mutate = copy.deepcopy(original_chrom) # Mutate modifica no local
                        mutated_chrom = self.mutation_op.mutate(chrom_to_mutate) 
                        self.current_population_chromosomes[idx] = mutated_chrom
                        self.current_population_fitnesses[idx] = self.calculate_fitness_func(list(mutated_chrom))
                        mutated_count_actual +=1
                    current_info['mutated_count'] = mutated_count_actual
            except Exception as e:
                logger.error(f"Erro durante a Ação de Mutação em Lote: {e}", exc_info=True)
                current_info['status'] = 'error_batch_mutation'

        elif action == 3: # Aplicar Crossover para gerar uma nova população (com mutação do ambiente)
            logger.debug("DRL Action: Foco em Crossover (com mutação padrão do ambiente)")
            try:
                temp_chroms = copy.deepcopy(self.current_population_chromosomes)
                temp_fitnesses = list(self.current_population_fitnesses)
                # Usará self.crossover_rate_ga e self.mutation_rate_ga definidos no ambiente
                self.current_population_chromosomes, self.current_population_fitnesses = \
                    self._execute_ga_cycle(temp_chroms, temp_fitnesses, \
                                         custom_crossover_rate=self.crossover_rate_ga, # Explicito, mas já seria o padrão
                                         custom_mutation_rate=self.mutation_rate_ga)   # Explicito
                current_info['crossover_focus_executed'] = True
            except Exception as e:
                logger.error(f"Erro durante a Ação de Foco em Crossover: {e}", exc_info=True)
                current_info['status'] = 'error_crossover_focus'
        
        elif action == 4: # Aplicar VND em P% dos melhores indivíduos
            vnd_target_percentage = 0.1 # Ex: 10% dos melhores
            num_to_vnd = max(1, int(self.population_size * vnd_target_percentage))
            logger.debug(f"DRL Action: Aplicar VND em {num_to_vnd} melhores indivíduos")
            try:
                if not self.current_population_chromosomes:
                    logger.warning("Sem cromossomos para aplicar VND em lote.")
                    current_info['status'] = 'no_chrom_for_batch_vnd'
                else:
                    # Pega os índices dos N melhores
                    best_indices_for_vnd = np.argsort(self.current_population_fitnesses)[:num_to_vnd]
                    improvements_vnd_batch = []
                    for idx in best_indices_for_vnd:
                        chrom_before_vnd = copy.deepcopy(self.current_population_chromosomes[idx])
                        fit_before_vnd = self.current_population_fitnesses[idx]
                        
                        chrom_after_vnd = self.vnd_search.local_search(chrom_before_vnd)
                        fit_after_vnd = self.calculate_fitness_func(list(chrom_after_vnd))
                        
                        self.current_population_chromosomes[idx] = chrom_after_vnd
                        self.current_population_fitnesses[idx] = fit_after_vnd
                        improvements_vnd_batch.append(fit_before_vnd - fit_after_vnd)
                    logger.debug(f"VND em lote: {num_to_vnd} indivíduos. Melhorias médias: {np.mean(improvements_vnd_batch) if improvements_vnd_batch else 0:.2f}")
                    current_info['vnd_batch_count'] = num_to_vnd
                    current_info['vnd_batch_avg_improvement'] = np.mean(improvements_vnd_batch) if improvements_vnd_batch else 0
            except Exception as e:
                logger.error(f"Erro durante a Ação VND em Lote: {e}", exc_info=True)
                current_info['status'] = 'error_vnd_batch'


        # Atualizar fitness e informações de melhoria
        current_pop_best_fitness = min(f for f in self.current_population_fitnesses if f is not None and f != float('inf')) if any(f is not None and f != float('inf') for f in self.current_population_fitnesses) else float('inf')
        
        if current_pop_best_fitness < self.best_fitness_overall:
            logger.info(f"Nova melhor solução global encontrada! Fitness: {current_pop_best_fitness:.2f} (anterior: {self.best_fitness_overall:.2f})")
            self.best_fitness_overall = current_pop_best_fitness
            self.steps_since_last_improvement = 0
        else:
            self.steps_since_last_improvement += 1

        reward = self._calculate_reward(old_overall_best_fitness, self.best_fitness_overall)

        # Condição de término do episódio DRL
        done = False
        truncated = False # Para Gym v0.26+

        if self.current_drl_step >= self.max_drl_steps_per_episode:
            logger.info("Máximo de DRL steps atingido.")
            done = True # Ou truncated = True se for um limite de tempo artificial
            truncated = True 

        if self.steps_since_last_improvement >= self.max_drl_steps_per_episode * 0.75: 
             logger.info("Estagnação detectada.")
             done = True
             truncated = True # Estagnação também pode ser vista como um truncamento
        
        if np.isnan(self.best_fitness_overall) or (np.isinf(self.best_fitness_overall) and current_pop_best_fitness == float('inf')):
            logger.critical(f"Fitness global inválido ({self.best_fitness_overall}) e população sem solução válida. Forçando término.")
            done = True
            reward = -float(self.max_drl_steps_per_episode) 

        obs, step_info = self._get_obs_and_info() # Pega obs e info atualizados
        current_info.update(step_info) # Combina com as infos da ação específica

        return obs, reward, done, truncated, current_info # Gym v0.26+ espera (obs, rew, terminated, truncated, info)

    def render(self, mode='human', close=False):
        if mode == 'human':
            print(f"DRL Step: {self.current_drl_step}/{self.max_drl_steps_per_episode}")
            pop_min_fit = min(f for f in self.current_population_fitnesses if f is not None and f != float('inf')) if any(f is not None and f != float('inf') for f in self.current_population_fitnesses) else float('nan')
            valid_fitnesses = [f for f in self.current_population_fitnesses if f is not None and f != float('inf')]
            pop_avg_fit = np.mean(valid_fitnesses) if valid_fitnesses else float('nan')
            pop_std_fit = np.std(valid_fitnesses) if len(valid_fitnesses) > 1 else float('nan')
            print(f"  Pop Fitness (Min/Avg/Std): {pop_min_fit:.2f} / {pop_avg_fit:.2f} / {pop_std_fit:.2f}")
            print(f"  Overall Best Fitness: {self.best_fitness_overall:.2f}")
            print(f"  Steps Since Last Global Improvement: {self.steps_since_last_improvement}")
            
    def close(self):
        pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG) # Para ver logs detalhados do ambiente
    # Exemplo de como usar o ambiente (para teste)
    sample_jobs_data = [
        [(0, 3), (1, 2), (2, 2)], 
        [(0, 2), (2, 1), (1, 4)], 
        [(1, 4), (2, 3)]          
    ]
    num_machines = 3

    env = JSSPDrlHyperHeuristicEnv(jobs_data=sample_jobs_data, num_machines=num_machines, population_size=10, max_drl_steps_per_episode=20)
    
    # Teste com Stable Baselines3 (opcional, mas bom para verificar compatibilidade)
    try:
        from stable_baselines3.common.env_checker import check_env
        check_env(env, warn=True) # Avisa sobre potenciais problemas mas não quebra
        logger.info("Verificação do ambiente com Stable Baselines3 check_env passou (ou passou com avisos).")
    except ImportError:
        logger.warning("Stable Baselines3 não instalado. Pulando check_env.")
    except Exception as e:
        logger.error(f"Erro durante check_env: {e}", exc_info=True)


    obs, info = env.reset()
    logger.info(f"Initial Observation: {obs}, Info: {info}")

    for i in range(30): # Mais steps para testar
        action = env.action_space.sample() 
        logger.info(f"--- Step {i+1}, Action: {action} ---")
        obs, reward, done, truncated, info = env.step(action) # Atualizado para Gym v0.26+
        env.render()
        logger.info(f"Observation: {obs}")
        logger.info(f"Reward: {reward:.4f}")
        logger.info(f"Done: {done}, Truncated: {truncated}")
        logger.info(f"Info: {info}")
        if done or truncated:
            logger.info("Episode finished.")
            obs, info = env.reset() # Atualizado para Gym v0.26+
            logger.info(f"Environment reset. New Initial Observation: {obs}, Info: {info}")
            # break # Descomente para parar após o primeiro 'done' ou 'truncated'
    env.close() 