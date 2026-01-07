import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
import sumo_rl
import supersuit as ss
import numpy as np

# --- Configuración de PATHs de SUMO ---
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

# --- Configuración de Archivos y Variables ---

NET_FILE = "final-sumo-files/RotondaFinal8semaforo.net.xml"
ROUTE_FILE = "final-sumo-files/DemandaReal_Dia.rou.xml"
MODEL_NAME = "ModeloFinalNuevo"
TRAIN_TIMESTEPS = 5000000

def reward_fn_propuesta(ts):
    # --- Colas en entradas ---
    total_queue = ts.get_total_queued()
    lanes_queue = ts.get_lanes_queue()
    max_entry_queue = max(lanes_queue)

    # --- Predeterminada ---
    diff_wait = ts._diff_waiting_time_reward()

    # --- Presión ---
    pressure = ts.get_pressure()

    # --- Balance de accesos ---
    imbalance = np.std(lanes_queue)         # alta si una entrada domina

    # --- Normalizaciones ---
    total_queue_n = total_queue / 100.0
    pressure_n = pressure / 30.0
    reward = (
        -0.35 * total_queue_n        # congestión global
        + 0.25 * diff_wait            # mejora real
        + 0.2 * pressure_n            # rotación / salida
        - 0.15 * imbalance            # castiga dominancia
    )

    return reward

def main():
    print(">>> Iniciando modo ENTRENAMIENTO (Sin GUI)...")

    # Creamos entorno SIN GUI (visual) para aprender
    env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=False,
        num_seconds=4000,
        out_csv_name="resultsFinal/resultados",
        
        #Restricciones
        delta_time=5,
        min_green=5,
        max_green = 100,
        yellow_time = 4,
        enforce_max_green = True,

        #Funcion de recompensa + info
        reward_fn = reward_fn_propuesta,
        add_per_agent_info = True,
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
        gamma=0.99,
        ent_coef = 0.05,
        tensorboard_log="./tensorboard_logs/" #Para usar tensorboard
    )

    print(f"Entrenando por {TRAIN_TIMESTEPS} pasos...")
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    
    # Guardar
    model.save(MODEL_NAME)
    print(f"Entrenamiento finalizado. Modelo guardado como '{MODEL_NAME}.zip'.")
    
    env.close()

if __name__ == "__main__":
    main()