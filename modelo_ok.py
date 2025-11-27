import os
import sys
import gymnasium as gym


# --- 1. Configuración de PATHs de SUMO ---
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)
from stable_baselines3 import PPO
import sumo_rl
import supersuit as ss # Necesario para adaptar PettingZoo a StableBaselines

# --- 2. Configuración de Archivos y Variables ---
NET_FILE = "sumo_files/RotondaFinal.net.xml"
ROUTE_FILE = "sumo_files/RotondaFinal.rou.xml"
MODEL_NAME = "ppo_rotonda_model"
MODEL_FILE = f"{MODEL_NAME}.zip"

# Cantidad de pasos para entrenar (si no existe el modelo). 
# 100,000 es un buen comienzo. Para pruebas rápidas usa 5,000.
TRAIN_TIMESTEPS = 3

# ==========================================
#      LÓGICA PRINCIPAL: SI / SINO
# ==========================================

if os.path.exists(MODEL_FILE):
    # ---------------------------------------
    # BLOQUE 1: EL MODELO EXISTE -> VISUALIZAR
    # ---------------------------------------
    print(f"✅ Modelo '{MODEL_FILE}' encontrado.")
    print(">>> Iniciando modo VISUALIZACIÓN (GUI)...")

    # Creamos entorno con GUI activada
    env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=True,  # <--- VERLO
        num_seconds=3600
    )
    # Parche para el error de render_mode
    env.unwrapped.render_mode = "human"

    # Wrappers (Deben ser idénticos al entrenamiento)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')

    # Cargar y Ejecutar
    model = PPO.load(MODEL_NAME)
    
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if any(dones): 
            done = True
            
    env.close()
    print("Simulación finalizada.")

else:
    # ---------------------------------------
    # BLOQUE 2: EL MODELO NO EXISTE -> ENTRENAR
    # ---------------------------------------
    print(f"❌ Modelo '{MODEL_FILE}' NO encontrado.")
    print(">>> Iniciando modo ENTRENAMIENTO (Sin GUI)...")

    # Creamos entorno SIN GUI (más rápido para aprender)
    env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=False, # <--- VELOCIDAD
        num_seconds=3600,
        out_csv_name="results/resultados_entrenamiento"
    )
    # Parche para el error de render_mode
    env.unwrapped.render_mode = "rgb_array"

    # Wrappers
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')

    # Definir el modelo PPO
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.001,
        gamma=0.99
    )

    print(f"Entrenando por {TRAIN_TIMESTEPS} pasos...")
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    
    # Guardar
    model.save(MODEL_NAME)
    print(f"💾 Entrenamiento finalizado. Modelo guardado como '{MODEL_FILE}'.")
    print("Ejecuta este script de nuevo para ver la simulación.")
    
    env.close()