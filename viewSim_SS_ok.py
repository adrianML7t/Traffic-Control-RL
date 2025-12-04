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
NET_FILE = "Traficoreal/RotondaFinal.net.xml"
ROUTE_FILE = "Traficoreal/DemandaReal.rou.xml"
MODEL_NAME = "modelo-entrenado"
MODEL_FILE = f"{MODEL_NAME}.zip"

def reward_fc(ts):
    # wait_list = ts.get_accumulated_waiting_time_per_lane()
    # penal_tiempo_espera = -1.5 * sum(wait_list)
    # penal_presion = -2.0 * ts.get_pressure()
    penal_colas = -1.0 * ts.get_total_queued()
    reward = penal_colas
    return reward

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
        use_gui=True, # Importante: False para velocidad
        num_seconds=1000,
        #out_csv_name="resultsMitadDemanda7/resultados_entrenamiento",
        
        #########
        min_green=60,   # 10 segundos es razonable (200 era excesivo)
        max_green = 90,
        enforce_max_green = True,
        
        delta_time=10, #Para que tome decisiones frecuentes
        reward_fn = reward_fc,
        fixed_ts = True, #respeta lsa Fases del conf SUMO inicial
        #########
        
        add_per_agent_info = True
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