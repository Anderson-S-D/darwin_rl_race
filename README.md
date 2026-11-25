# DARwIn-OP RL Race: Competencia de Locomoción Bípeda en Webots

Simulación 3D en **Webots** donde tres instancias del robot humanoide **DARwIn-OP (`RobotisOp2`)** compiten en tiempo real en pistas **sin obstáculos** y **con obstáculos**, controladas por tres algoritmos de Aprendizaje por Refuerzo (RL): **DQN**, **A2C** y **PPO**.

---

## 1. Arquitectura de Software

El proyecto aplica principios **SOLID** y el **Patrón Strategy** para desacoplar la física del simulador de la lógica matemática de cada agente:

* **Contrato Polimórfico (`BaseAgent`)**: Todos los algoritmos heredan de una clase abstracta común (`select_action`, `store_transition`, `update`, `save_weights`, `load_weights`), permitiendo intercambiar modelos en entrenamiento o ejecutarlos simultáneamente en carrera sin modificar el código del entorno.
* **Abstracción de Locomoción (`DarwinGaitController`)**: Dado que el DARwIn-OP posee 20 grados de libertad, se implementa un generador de marcha bípeda sinusoidal (**CPG** - *Central Pattern Generator*) en Python puro con balanceo lateral, compensación de tobillos y braceo. Sobre esta capa, los agentes operan en un espacio de **5 macro-acciones discretas**.
* **Entorno Estándar (`DarwinRaceEnv`)**: Envuelve la API `Supervisor` de Webots bajo la interfaz estándar de **Gymnasium**, gestionando reinicios automáticos ante caídas, cálculo de recompensas y normalización de sensores.

### Estructura del Repositorio

```text
darwin_rl_race/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py                # Clase abstracta (Patrón Strategy) y persistencia .pth
│   ├── dqn_agent.py                 # Deep Q-Network (Off-policy)
│   ├── a2c_agent.py                 # Advantage Actor-Critic (On-policy)
│   └── ppo_agent.py                 # Proximal Policy Optimization (On-policy + Clipping)
├── controllers/
│   ├── rl_trainer_controller/
│   │   ├── rl_trainer_controller.py # Orquestador de entrenamiento y checkpoints
│   │   ├── darwin_env.py            # Entorno Gymnasium (estados, acciones, recompensa)
│   │   └── gait_controller.py       # Cinemática de marcha bípeda CPG (20 motores)
│   └── race_controller/
│       └── race_controller.py       # Controlador de carrera simultánea (modo evaluación)
├── worlds/
│   ├── pista_entrenamiento.wbt      # Arena individual para entrenamiento y ajuste
│   ├── pista_sin_obstaculos.wbt     # Pista de competencia abierta (3 corredores)
│   └── pista_con_obstaculos.wbt     # Pista de competencia con bloques físicos (3 corredores)
├── weights/                         # Pesos entrenados exportados (.pth)
├── metrics/                         # Registros históricos por episodio (.csv)
├── .gitignore
└── README.md
```

---

## 2. Formulación del Problema de Aprendizaje por Refuerzo

### Espacio de Observación (8 variables normalizadas en `[-1, 1]`)
1. `dist_goal`: Distancia longitudinal restante hacia la meta (`goal_x = 10.0 m`).
2. `lateral_offset`: Desviación lateral respecto al eje central del carril (`Y`).
3. `acc_x`: Aceleración lineal frontal (detecta inclinación o frenado brusco).
4. `acc_z`: Aceleración vertical (estabilidad del centro de masa).
5. `gyro_y`: Velocidad angular de cabeceo (*pitch*, anticipa caídas hacia adelante o atrás).
6. `ds_left`: Lectura normalizada del sensor de distancia frontal-izquierdo.
7. `ds_right`: Lectura normalizada del sensor de distancia frontal-derecho.
8. `is_standing`: Indicador binario de verticalidad (`+1.0` si altura del torso $Z > 0.22\text{ m}$, `-1.0` en caso contrario).

### Espacio de Acciones Discretas (5 macro-acciones)
| ID | Acción | Amplitud Avance ($X$) | Amplitud Giro ($A$) | Comportamiento Físico |
| :--- | :--- | :---: | :---: | :--- |
| **`0`** | Avance rápido | `1.00` | `0.00` | Zancada máxima en línea recta. |
| **`1`** | Avance prudente | `0.55` | `0.00` | Paso medio estable para aproximación. |
| **`2`** | Giro izquierda | `0.45` | `+0.60` | Avance reducido con rotación en cadera hacia `+Y`. |
| **`3`** | Giro derecha | `0.45` | `-0.60` | Avance reducido con rotación en cadera hacia `-Y`. |
| **`4`** | Estabilización | `0.15` | `0.00` | Paso corto de recuperación de equilibrio. |

### Función de Recompensa
$$R_t = 15.0 \cdot (x_t - x_{t-1}) - 0.04 + R_{\text{obstáculo}} + R_{\text{terminal}}$$
* **Progreso longitudinal**: $+15.0$ por cada metro avanzado hacia $+X$.
* **Penalización temporal**: $-0.04$ por paso para incentivar velocidad.
* **Proximidad a obstáculos**: $-1.5$ si `min(ds_left, ds_right) < 0.25`.
* **Eventos terminales**: $-20.0$ por caída ($Z < 0.20\text{ m}$) y $+50.0$ al cruzar la meta ($X \ge 10.0\text{ m}$).

---

## 3. Algoritmos Implementados

| Algoritmo | Archivo | Paradigma | Mecanismo Clave |
| :--- | :--- | :--- | :--- |
| **DQN** | `dqn_agent.py` | *Off-policy* (Valor) | *Experience Replay* (`20,000` transiciones), *Target Network* sincronizada cada 200 pasos, política $\epsilon$-greedy y pérdida Huber (`SmoothL1Loss`). |
| **A2C** | `a2c_agent.py` | *On-policy* (Actor-Crítico) | *Backbone* compartido con doble salida ($\pi_\theta(a\Vert{}s)$ y $V_\phi(s)$), actualizaciones cada `32` pasos, normalización de ventaja y regularización por entropía. |
| **PPO** | `ppo_agent.py` | *On-policy* (Región de confianza) | Redes Actor y Crítico separadas (`Tanh`), estimación de ventaja generalizada (**GAE**, $\lambda=0.95$) y función objetivo recortada (*Clipping*, $\epsilon=0.2$) sobre trayectorias de `256` pasos. |

---

## 4. Requisitos e Instalación (Windows y macOS)

### Requisitos Previos
* **Webots R2025a** (o superior).
* **Python 3.10 – 3.12**.
* Librerías: `torch`, `numpy`, `gymnasium`.

### Configuración en Windows
Abre **PowerShell** o **CMD** en la raíz del proyecto (`darwin_rl_race`):

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install numpy gymnasium
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```

**Vincular el entorno en Webots (Windows):**
1. Abre Webots y ve a **`Tools > Preferences > General`**.
2. En el campo **`Python command`**, ingresa la ruta absoluta al ejecutable del entorno virtual:
   ```text
   C:\Users\<TU_USUARIO>\Documents\darwin_rl_race\venv\Scripts\python.exe
   ```

---

### Configuración en macOS
Abre **Terminal** (`zsh` / `bash`) en la raíz del proyecto (`darwin_rl_race`):

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch numpy gymnasium
```

**Vincular el entorno en Webots (macOS):**
1. Abre Webots y ve a **`Webots > Settings...`** (o **`Preferences`**) **`> General`**.
2. En el campo **`Python command`**, ingresa la ruta absoluta al binario del entorno virtual:
   ```text
   /Users/<TU_USUARIO>/Documents/darwin_rl_race/venv/bin/python
   ```

---

## 5. Guía de Ejecución

### Fase A: Entrenamiento y Ajuste por Algoritmo
1. Abre `worlds/pista_entrenamiento.wbt` en Webots. Verifica que el nodo `RobotisOp2` tenga `supervisor TRUE`, `controller "rl_trainer_controller"` y altura inicial `translation 0 0 0.30`.
2. Abre `controllers/rl_trainer_controller/rl_trainer_controller.py` y configura el algoritmo deseado:
   ```python
   SELECTED_ALGORITHM = "ppo"   # Opciones: "dqn", "a2c", "ppo"
   EXTRA_EPISODES = 250
   RESUME_CHECKPOINT = None     # Ej. "weights/ppo_ep250.pth" para reanudar con obstáculos
   ```
3. Guarda el script (`Ctrl + S` en Windows / `Cmd + S` en macOS) y presiona **Play** en Webots (puedes activar el modo **Fast** en la barra superior para acelerar el entrenamiento).
4. Al finalizar, los pesos se almacenarán en `weights/<algoritmo>_ep<N>.pth` y el historial en `metrics/<algoritmo>_training.csv`.

### Fase B: Carrera Simultánea en Tiempo Real (3 Robots)
1. Abre `worlds/pista_sin_obstaculos.wbt` o `worlds/pista_con_obstaculos.wbt`.
2. Verifica que existan **3 instancias de `RobotisOp2`** ubicadas en `X = 0.0, Z = 0.30` (separadas en el eje `Y`, por ejemplo `Y = -1.0, 0.0, 1.0`), todas con `supervisor TRUE`, `controller "race_controller"` y los siguientes nombres exactos en el campo **`name`**:
   * `darwin_ppo` $\rightarrow$ carga `weights/ppo_ep350.pth` (o `ppo_ep250.pth`)
   * `darwin_dqn` $\rightarrow$ carga `weights/dqn_ep350.pth` (o `dqn_ep250.pth`)
   * `darwin_a2c` $\rightarrow$ carga `weights/a2c_ep350.pth` (o `a2c_ep250.pth`)
3. Presiona **Play** en modo **Real-time**. Cada robot ejecutará su política aprendida en modo determinista (`eval_mode=True`) y reportará en consola su tiempo exacto al cruzar la meta (`[META]`).

---

## 6. Registro de Métricas

Durante el entrenamiento se genera un archivo CSV por algoritmo en `metrics/` con las columnas:

```csv
episode,reward,distance_m,steps,fallen,avg_loss
```

Estos archivos permiten graficar la curva de aprendizaje (recompensa acumulada por episodio), la distancia máxima alcanzada y la reducción en la tasa de caídas para comparar el desempeño entre los tres modelos.
