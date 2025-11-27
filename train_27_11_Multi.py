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

env = sum.parallel_env(
    net_file = NET_FILE,
    route_file = ROUTE_FILE,
    use_gui = True,
    num_seconds = 3600
)



