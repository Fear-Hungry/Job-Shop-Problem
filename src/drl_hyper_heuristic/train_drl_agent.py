import sys # Adicionado para modificar o sys.path
import os # Adicionado para manipulação de caminhos

# Adicionar a raiz do projeto ao sys.path
# Isso permite que o script seja executado de diferentes locais e ainda encontre o pacote 'src'
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Script para treinar o agente DRL para o JSSP Hyper-Heuristic

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor # Para monitorar o ambiente

from src.drl_hyper_heuristic.jssp_drl_env import JSSPDrlHyperHeuristicEnv # Importa nosso ambiente
import logging

# Configurar logging para o treinamento
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Iniciando o script de treinamento do agente DRL...")

    # --- Dados do Problema JSSP (Exemplo) ---
    # Você pode carregar instâncias de arquivos ou definir outras aqui
    sample_jobs_data = [
        [(0, 3), (1, 2), (2, 2)],  # Job 0
        [(0, 2), (2, 1), (1, 4)],  # Job 1
        [(1, 4), (2, 3)]           # Job 2
    ]
    num_machines = 3
    problem_instance_name = "sample_3x3"

    # --- Diretórios para salvar modelos e logs ---
    # Usar os.path.join para melhor portabilidade entre OS
    base_output_dir = os.path.join(PROJECT_ROOT, "outputs") # Salvar outputs na raiz do projeto
    log_dir = os.path.join(base_output_dir, "drl_logs", problem_instance_name)
    model_save_dir = os.path.join(base_output_dir, "drl_models", problem_instance_name)
    model_save_path = os.path.join(model_save_dir, "jssp_ppo_agent")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # --- Configuração do Ambiente ---
    logger.info(f"Configurando o ambiente JSSP para a instância: {problem_instance_name}")
    env = JSSPDrlHyperHeuristicEnv(
        jobs_data=sample_jobs_data,
        num_machines=num_machines,
        population_size=30,            
        ga_generations_per_drl_step=1,
        max_drl_steps_per_episode=100, 
        tournament_size_ga=3,
        crossover_rate_ga=0.8,
        mutation_rate_ga=0.2,        
        elitism_size_ga=1,
        vnd_max_tries_per_neighborhood=5,
        seed=42                        
    )
    env = Monitor(env, log_dir)

    # --- Configuração do Agente PPO ---
    # Hiperparâmetros do PPO (podem ser ajustados/otimizados depois)
    # Para problemas mais complexos, 'MlpPolicy' (Multi-Layer Perceptron) é comum.
    # verbose=1 para logs de treinamento do SB3.
    # tensorboard_log para visualização do treinamento.
    # learning_rate, n_steps, batch_size, etc., são hiperparâmetros chave.
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=3e-4,       
        n_steps=256,              
        batch_size=64,            
        n_epochs=10,              
        gamma=0.99,               
        gae_lambda=0.95,          
        clip_range=0.2,           
        ent_coef=0.0,             
        vf_coef=0.5,              
        max_grad_norm=0.5,        
        seed=42                   
    )

    # --- Callbacks (Opcional) ---
    # Salvar o modelo periodicamente
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=log_dir,
        name_prefix="ppo_jssp_checkpoint"
    )

    # --- Treinamento ---
    total_timesteps_to_train = 50000 
    logger.info(f"Iniciando treinamento do agente PPO por {total_timesteps_to_train} timesteps...")
    try:
        model.learn(
            total_timesteps=total_timesteps_to_train,
            callback=checkpoint_callback,
            progress_bar=True 
        )
        logger.info("Treinamento concluído.")
        model.save(model_save_path)
        logger.info(f"Modelo final salvo em: {model_save_path}.zip")
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}", exc_info=True)
        return

    # --- Teste Rápido do Modelo Treinado (Opcional) ---
    logger.info("Testando o modelo treinado...")
    try:
        del model 
        loaded_model = PPO.load(f"{model_save_path}.zip", env=env) # Adicionar .zip ao carregar

        obs, info = env.reset()
        for i in range(100): 
            action, _states = loaded_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if i % 20 == 0:
                 logger.info(f"Teste - Passo: {i+1}, Ação: {action}, Recompensa: {reward:.3f}, Melhor Fitness Ep: {info.get('overall_best_fitness', float('inf')):.2f}")
            if terminated or truncated:
                logger.info("Episódio de teste finalizado.")
                obs, info = env.reset()
        logger.info("Teste rápido concluído.")
    except Exception as e:
        logger.error(f"Erro durante o teste do modelo carregado: {e}", exc_info=True)

if __name__ == '__main__':
    main() 