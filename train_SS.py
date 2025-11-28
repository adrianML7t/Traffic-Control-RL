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
MODEL_NAME = "ppo_rotonda_model_mitad_10k"
TRAIN_TIMESTEPS = 10000

def main():
    print(">>> Iniciando modo ENTRENAMIENTO (Sin GUI)...")

    # Creamos entorno SIN GUI (más rápido para aprender)
    env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=False, # Importante: False para velocidad
        num_seconds=3600,
        out_csv_name="resultsMitadDemanda7/resultados_entrenamiento"
    )
    
    # Parche para el error de render_mode
    env.unwrapped.render_mode = "rgb_array"

    # Wrappers (Deben coincidir con los de visualización)
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
    print(f"💾 Entrenamiento finalizado. Modelo guardado como '{MODEL_NAME}.zip'.")
    
    env.close()

if __name__ == "__main__":
    main()