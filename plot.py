import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import os
import seaborn as sns

# --- CONFIGURACIÓN ---
# La ruta base que pusiste en 'out_csv_name'
FOLDER_PREFIX = "resultsMitadDemanda7/xabi-resultados" 
METRIC_COL = "system_total_waiting_time" # La columna que quieres evaluar
# Otras opciones: "mean_waiting_time", "system_total_stopped", "reward"

def get_episode_data():
    data = []
    # Buscamos todos los archivos que coincidan con el patrón
    # SUMO-RL suele nombrar: nombre_conn0_ep1.csv, nombre_conn0_ep2.csv...
    pattern = FOLDER_PREFIX + "*_conn*.csv"
    files = glob.glob(pattern)
    
    print(f"Leyendo {len(files)} archivos de episodios... (Esto puede tardar un poco)")

    for f in files:
        # Extraemos el número del episodio usando Expresiones Regulares (Regex)
        # Busca "ep" seguido de digitos
        match = re.search(r"ep(\d+)", f)
        if match:
            ep_num = int(match.group(1))
            
            try:
                df = pd.read_csv(f)
                # Calculamos la MEDIA de espera de todo ese episodio (30 mins)
                # Si el tráfico fue fluido, este número será bajo.
                val = df[METRIC_COL].mean()
                
                data.append({"Episode": ep_num, "Average Waiting Time": val})
            except Exception as e:
                print(f"Error leyendo {f}: {e}")

    return pd.DataFrame(data)

def main():
    df = get_episode_data()
    
    if df.empty:
        print("ERROR: No se encontraron datos. Verifica la ruta 'FOLDER_PREFIX'.")
        return

    # Ordenamos por número de episodio para ver la evolución temporal
    df = df.sort_values("Episode")

    # --- PLOT ---
    sns.set(style="darkgrid")
    plt.figure(figsize=(10, 6))
    
    # Graficamos los datos crudos (puntos transparentes)
    sns.scatterplot(data=df, x="Episode", y="Average Waiting Time", alpha=0.3, color="gray", label="Episodio individual")
    
    # Graficamos una linea suavizada (Media Móvil) para ver la tendencia clara
    # window=20 significa que promedia los ultimos 20 episodios
    df["Trend"] = df["Average Waiting Time"].rolling(window=20).mean()
    sns.lineplot(data=df, x="Episode", y="Trend", color="blue", linewidth=2.5, label="Tendencia (Media móvil 20)")

    plt.title(f"Evolución del Aprendizaje (Total: {len(df)} episodios)")
    plt.xlabel("Número de Episodio")
    plt.ylabel("Tiempo de Espera Promedio (s)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()