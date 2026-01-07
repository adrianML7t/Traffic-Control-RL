import os
import sys

# --- Configuración de PATHs de SUMO ---
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

import gymnasium as gym
from stable_baselines3 import PPO
import sumo_rl
import supersuit as ss
import numpy as np

# --- Configuración de Archivos y Variables ---
NET_FILE = "sumo_files/RotondaFinal3.net.xml"
ROUTE_FILE = "sumo_files/DemandaReal_MenosFlujo_Dia.rou.xml"
MODEL_NAME = "resultsColas_exp_doc"

def reward_fn_propuesta(ts):
    # --- Colas en entradas ---
    total_queue = ts.get_total_queued()
    lanes_queue = ts.get_lanes_queue()      # [0,1] por entrada
    max_entry_queue = max(lanes_queue)

    # --- Dinámica temporal ---
    diff_wait = ts._diff_waiting_time_reward()

    # --- Pressure (flujo neto) ---
    pressure = ts.get_pressure()

    # --- Balance de accesos (evita monopolio) ---
    imbalance = np.std(lanes_queue)         # alta si una entrada domina

    # --- Normalizaciones ---
    total_queue_n = total_queue / 100.0
    pressure_n = pressure / 30.0

    # --- Recompensa ---
    reward = (
        -0.35 * total_queue_n        # congestión global
        + 0.25 * diff_wait            # mejora real
        + 0.2 * pressure_n            # rotación / salida
        - 0.15 * imbalance            # castiga dominancia
    )

    return reward

def main():
    print(f">>> Cargando modelo: {MODEL_NAME} y abriendo SUMO-GUI...")

    # 1. Crear el entorno CON GUI
    # Nota: Es vital que los parámetros físicos (min_green, max_green, fixed_ts)
    # sean IDÉNTICOS al entrenamiento para que el modelo sepa qué hacer.
    env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=True, # Importante: False para velocidad
        num_seconds=4000,
        out_csv_name="resultsFinal/resultados",
        
        #Restricciones
        delta_time=5,
        min_green=5,
        max_green = 60,
        yellow_time = 4,
        enforce_max_green = True,
        reward_fn = reward_fn_propuesta,
        add_per_agent_info = True,
    )

    # 2. Wrappers (SuperSuit)
    # Tienen que estar EN EL MISMO ORDEN que en el entrenamiento.

    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')

    # 3. Cargar el modelo entrenado
    # Verificamos que el archivo existe primero
    model_path = f"{MODEL_NAME}.zip"
    if not os.path.exists(model_path):
        print(f"ERROR: No se encuentra el archivo '{model_path}'")
        return

    # Cargamos el modelo
    model = PPO.load(model_path)

    # 4. Bucle de Simulación
    print(">>> Iniciando simulación. Presiona 'Start' (Play) en la ventana de SUMO.")
    
    obs = env.reset()
    done = False
    try:
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            
            # Ejecutar paso en el entorno
            obs, rewards, dones, info = env.step(action)
            if isinstance(dones, (list, tuple)) or hasattr(dones, '__iter__'):
                done = all(dones) 
            else:
                done = dones
                
    except KeyboardInterrupt:
        print("\n>>> Simulación interrumpida por el usuario.")
    except Exception as e:
        print(f"\n>>> Error durante la simulación: {e}")
    finally:
        print(">>> Cerrando entorno...")
        env.close()

if __name__ == "__main__":
    main()