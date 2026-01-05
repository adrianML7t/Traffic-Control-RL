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
TRAIN_TIMESTEPS = 1000000


def reward_fn_mixta(ts):
    return 0.7*ts._queue_reward() + 0.3*ts._diff_waiting_time_reward()

def reward_fn_roundabout(ts):
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



def reward_fn_documentada(ts):
    # 1. Obtenemos la ocupación de los carriles (Lista de 0.0 a 1.0)
    # Fuente: TrafficSignal.get_lanes_queue()
    queues = ts.get_lanes_queue()
    
    # 2. Obtenemos el tiempo de espera (Lista de segundos)
    # Fuente: TrafficSignal.get_accumulated_waiting_time_per_lane()
    waiting_time = sum(ts.get_accumulated_waiting_time_per_lane())
    
    # CÁLCULO DE LA RECOMPENSA
    # -----------------------
    # A. Penalización por bloqueo total (Suma de todos los carriles)
    # Multiplicamos por 50 porque queues son valores pequeños (0-1).
    penalty_total_queue = -50.0 * sum(queues)
    
    # B. Penalización por "Cuello de Botella" (El peor carril)
    # Si la lista está vacía, ponemos 0. Si no, cogemos el máximo.
    # Multiplicamos por 100 para que el agente tenga PÁNICO si un solo carril se llena (tu entrada derecha).
    worst_lane = max(queues) if queues else 0.0
    penalty_worst_lane = -100.0 * worst_lane
    
    # C. Penalización por Tiempo de Espera
    # Factor pequeño (0.05) porque el tiempo de espera puede ser miles de segundos.
    penalty_waiting = -0.05 * waiting_time
    
    # Suma total
    reward = penalty_total_queue + penalty_worst_lane + penalty_waiting
    
    # Normalización final para estabilidad numérica (Value Loss controlado)
    return reward / 100.0


def reward_fn_aggresive(ts):
    queues = ts.get_lanes_queue() 
    # Penalizamos: 
    # 1. Suma de ocupación de todos los carriles (*50 para dar peso a porcentajes 0-1)
    # 2. El carril más lleno (*100 para priorizar desbloqueo agresivo del peor carril)
    # 3. Tiempo de espera (peso bajo para no opacar la gestión de colas)
    return (-50.0 * sum(queues) - 100.0 * max(queues) - 0.05 * sum(ts.get_accumulated_waiting_time_per_lane())) / 100.0
   
def reward_fn_sin_norm(ts):
    queue = ts.get_total_queued()
    waiting_time = sum(ts.get_accumulated_waiting_time_per_lane())
    reward = -1.0 * queue - 0.01 * waiting_time
    return reward / 100.0

def reward_fn2(ts):
    queue = ts.get_total_queued()         # vehículos en cola
    waiting = sum(ts.get_accumulated_waiting_time_per_lane())  # tiempo total de espera
    speed = ts.get_average_speed()         # velocidad media (m/s)

    reward = -0.6 * queue - 0.4 * waiting + 0.2 * speed

    reward_normalizada = reward / 100.0 
    # Opcional: Clipping para asegurar que no explote
    reward_normalizada = max(min(reward_normalizada, 1.0), -1.0)
    return reward_normalizada


def main():
    print(">>> Iniciando modo ENTRENAMIENTO (Sin GUI)...")

    # Creamos entorno SIN GUI (más rápido para aprender)
    env = sumo_rl.parallel_env(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=False, # Importante: False para velocidad
        num_seconds=4000,
        out_csv_name="resultsFinal/resultados",
        
        #########
        delta_time=5,
        min_green=5,
        max_green = 100,
        yellow_time = 4,
        enforce_max_green = True,
        
        #delta_time=5, #Para que tome decisiones frecuentes
        reward_fn = reward_fn_roundabout,
        #fixed_ts = True, # Respeta las fases del conf SUMO inicial
        #########
        # reward_fn = ["queue", "diff-waiting-time", "pressure"]
        # reward_weights = [0.5, 0.3, 0.2]

        add_per_agent_info = True,
       # additional_sumo_cmd="--tripinfo-output results_test/tripinfo.xml --emissions-output results_test/emissions.xml"
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
        gamma=0.99,
        ent_coef = 0.05,
        tensorboard_log="./tensorboard_logs/"
    )

    print(f"Entrenando por {TRAIN_TIMESTEPS} pasos...")
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    
    # Guardar
    model.save(MODEL_NAME)
    print(f"💾 Entrenamiento finalizado. Modelo guardado como '{MODEL_NAME}.zip'.")
    
    env.close()

if __name__ == "__main__":
    main()
