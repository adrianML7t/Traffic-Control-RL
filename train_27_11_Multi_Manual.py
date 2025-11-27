import os
import sys

#Arreglar el path de SUMO
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"
tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
sys.path.append(tools)

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import pettingzoo as pz
import sumo_rl as sum

#Ficheros
NET_FILE = "sumo_files/RotondaFinal.net.xml"
ROUTE_FILE = "sumo_files/RotondaFinal.rou.xml"
TLS_IDS = ["E", "N", "S", "W"]

#Entorno pettingzoo 
# (https://lucasalegre.github.io/sumo-rl/api/pettingzoo/)

#Diferencia entre sum.env y sum.parallel_env?? parece que los dos son multiagente
#AEC VS paralelo ? https://pettingzoo.farama.org/api/aec/#about-aec

env = sum.parallel_env(
    net_file = NET_FILE,
    route_file = ROUTE_FILE,
    use_gui = True,
    num_seconds = 3600,
)

#s_env.ts_ids = TLS_IDS

observations = env.reset()

#???
while env.agents:
    # Genera una acción aleatoria para CADA agente activo.
    # Esta es la parte donde se insertaría una política (modelo RL) si existiera.
    actions = {}
    for agent in env.agents:
        # La acción aleatoria se toma del espacio de acción del agente.
        actions[agent] = env.action_space(agent).sample()

    # Ejecuta un paso de simulación en el entorno con todas las acciones.
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Opcional: Imprimir las recompensas del paso
    # print(f"Paso - Recompensas: {rewards}")

# --- 4. CIERRE DEL ENTORNO ---
print("Simulación finalizada.")
env.close()
