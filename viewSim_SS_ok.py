import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
import sumo_rl
import supersuit as ss

# --- Configuración de PATHs de SUMO ---
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

# --- Configuración de Archivos y Variables ---
NET_FILE = "sumo_files/RotondaAmerica.net.xml"
ROUTE_FILE = "sumo_files/DemandaAmerica.rou.xml"
MODEL_NAME = "ppo_rotonda_model_mitad"
MODEL_FILE = f"{MODEL_NAME}.zip"

def main():
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Error: No se encuentra el archivo '{MODEL_FILE}'. Ejecuta primero el script de entrenamiento.")
        return

    print(f"✅ Modelo '{MODEL_FILE}' encontrado.")
    print(">>> Iniciando modo VISUALIZACIÓN (GUI)...")

    # Creamos entorno con GUI activada
    env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=True,  # Importante: True para ver la simulación
        num_seconds=3600
    )
    
    # Parche para el error de render_mode
    env.unwrapped.render_mode = "human"

    # Wrappers (Deben ser IDÉNTICOS al entrenamiento)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')

    # Cargar el modelo
    model = PPO.load(MODEL_NAME)
    
    obs = env.reset()
    done = False
    
    while not done:
        # deterministic=True es mejor para evaluación/visualización
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        if any(dones): 
            done = True
            
    env.close()
    print("Simulación finalizada.")

if __name__ == "__main__":
    main()