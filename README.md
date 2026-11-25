# DARwIn-OP RL Race: Competencia de Locomoción Bípeda en Webots

Simulación 3D en **Webots** donde múltiples instancias del robot humanoide **DARwIn-OP (`RobotisOp2`)** compiten en tiempo real en pistas **sin obstáculos** y **con obstáculos**, controladas por algoritmos de Aprendizaje por Refuerzo (RL): **DQN**, **A2C**, **PPO** y un agente propio (**MyAgent**).

---

## 1. Arquitectura del Proyecto

El proyecto aplica principios **SOLID** y el **Patrón Strategy** para desacoplar la física de Webots de la lógica de aprendizaje de cada agente:

* **Contrato Polimórfico (`agents/base_agent.py`)**: Todos los algoritmos heredan de `BaseAgent` (`select_action`, `store_transition`, `update`, `save_weights`, `load_weights`), permitiendo entrenarlos de forma independiente y cargar sus pesos `.pth` simultáneamente en la carrera final.
* **Abstracción de Locomoción (`gait_controller.py`)**: Dado que el DARwIn-OP posee 20 grados de libertad, se implementa un generador de marcha bípeda sinusoidal (**CPG** - *Central Pattern Generator*) en Python puro con balanceo lateral, compensación de tobillos y braceo. Sobre esta capa, los agentes deciden entre **5 macro-acciones discretas**.
* **Entorno Estándar (`darwin_env.py`)**: Envuelve la API `Supervisor` de Webots bajo la interfaz de **Gymnasium**, gestionando reinicios automáticos ante caídas, normalización de sensores y cálculo de recompensas.

### Estructura del Repositorio

```text
darwin_rl_race/
├── agents/
│   ├── base_agent.py                    # Clase abstracta (Patrón Strategy) y persistencia .pth
│   ├── dqn_agent.py                     # Deep Q-Network (Off-policy)
│   ├── a2c_agent.py                     # Advantage Actor-Critic (On-policy)
│   ├── ppo_agent.py                     # Proximal Policy Optimization (On-policy + Clipping)
├── controllers/
│   ├── dqn/
│   │   ├── dqn.py                       # Controlador de entrenamiento para DQN
│   │   ├── darwin_env.py                # Entorno Gymnasium del robot
│   │   └── gait_controller.py           # Cinemática de marcha bípeda CPG
│   ├── a2c/
│   │   ├── a2c.py                       # Controlador de entrenamiento para A2C
│   │   ├── darwin_env.py
│   │   └── gait_controller.py
│   ├── ppo/
│   │   ├── ppo.py                       # Controlador de entrenamiento para PPO
│   │   ├── darwin_env.py
│   │   └── gait_controller.py
│   └── race_supervisor_controller/
│       ├── race_manager.py              # Gestor de carrera simultánea en tiempo real
│       ├── darwin_env.py
│       └── gait_controller.py
├── worlds/
│   ├── pista_entrenamiento.wbt          # Escenario individual de entrenamiento
│   ├── pista_sin_obstaculos.wbt         # Arena de competencia abierta
│   └── pista_con_obstaculos.wbt         # Arena de competencia con obstáculos físicos
├── weights/                             # Pesos entrenados exportados (.pth)
├── metrics/                             # Historial de métricas por episodio (.csv)
├── .gitignore
└── README.md
```

---

## 2. Formulación del Entorno RL

### Espacio de Observación (8 variables normalizadas en `[-1, 1]`)
1. `dist_goal`: Distancia longitudinal restante hacia la meta (`goal_x = 10.0 m`).
2. `lateral_offset`: Desviación lateral respecto al centro del carril (`Y`).
3. `acc_x`: Aceleración lineal frontal.
4. `acc_z`: Aceleración vertical del torso.
5. `gyro_y`: Velocidad angular de cabeceo (*pitch*).
6. `ds_left`: Lectura normalizada del sensor de distancia frontal-izquierdo.
7. `ds_right`: Lectura normalizada del sensor de distancia frontal-derecho.
8. `is_standing`: Indicador binario de verticalidad (`+1.0` si $Z > 0.22\text{ m}$, `-1.0` en caso contrario).

### Espacio de Acciones Discretas (5 macro-acciones)
| ID | Acción | Amplitud Avance ($X$) | Amplitud Giro ($A$) | Comportamiento Físico |
| :--- | :--- | :---: | :---: | :--- |
| **`0`** | Avance rápido | `1.00` | `0.00` | Zancada máxima en línea recta. |
| **`1`** | Avance prudente | `0.55` | `0.00` | Paso medio estable. |
| **`2`** | Giro izquierda | `0.45` | `+0.60` | Avance con rotación en cadera hacia `+Y`. |
| **`3`** | Giro derecha | `0.45` | `-0.60` | Avance con rotación en cadera hacia `-Y`. |
| **`4`** | Estabilización | `0.15` | `0.00` | Paso corto de recuperación de equilibrio. |

### Función de Recompensa
$$R_t = 15.0 \cdot (x_t - x_{t-1}) - 0.04 + R_{\text{obstáculo}} + R_{\text{terminal}}$$
* **Progreso longitudinal**: $+15.0$ por metro avanzado hacia $+X$.
* **Penalización por tiempo**: $-0.04$ por paso para incentivar velocidad.
* **Proximidad a obstáculos**: $-1.5$ si `min(ds_left, ds_right) < 0.25`.
* **Eventos terminales**: $-20.0$ por caída ($Z < 0.20\text{ m}$) y $+50.0$ al cruzar la meta ($X \ge 10.0\text{ m}$).

---

## 3. Algoritmos Implementados

| Algoritmo | Clase / Archivo | Paradigma | Mecanismo Principal |
| :--- | :--- | :--- | :--- |
| **DQN** | `agents/dqn_agent.py` | *Off-policy* (Valor) | *Experience Replay* (`20,000` transiciones), *Target Network* sincronizada cada 200 pasos, política $\epsilon$-greedy y pérdida Huber (`SmoothL1Loss`). |
| **A2C** | `agents/a2c_agent.py` | *On-policy* (Actor-Crítico) | *Backbone* compartido con salidas Actor $\pi_\theta(a\Vert{}s)$ y Crítico $V_\phi(s)$, actualizaciones cada `32` pasos, ventaja normalizada y regularización por entropía. |
| **PPO** | `agents/ppo_agent.py` | *On-policy* (Región de confianza) | Redes Actor y Crítico independientes (`Tanh`), estimación de ventaja generalizada (**GAE**, $\lambda=0.95$) y función objetivo recortada (*Clipping*, $\epsilon=0.2$) en bloques de `256` pasos. |

---

## 4. Instalación y Configuración (Windows y macOS)

### Requisitos Previos
* **Webots R2025a** (o superior).
* **Python 3.10 – 3.12**.
* Dependencias: `torch`, `numpy`, `gymnasium`.

### Configuración en Windows
Desde **PowerShell** o **CMD** en la raíz `darwin_rl_race`:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install numpy gymnasium
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```

**Configurar Python en Webots (Windows):**
1. Abre **`Tools > Preferences > General`**.
2. En **`Python command`**, especifica la ruta al ejecutable del entorno virtual:
   ```text
   C:\Users\<TU_USUARIO>\Documents\darwin_rl_race\venv\Scripts\python.exe
   ```

---

### Configuración en macOS
Desde **Terminal** (`zsh` / `bash`) en la raíz `darwin_rl_race`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch numpy gymnasium
```

**Configurar Python en Webots (macOS):**
1. Abre **`Webots > Settings...`** (o **`Preferences`**) **`> General`**.
2. En **`Python command`**, especifica la ruta al binario del entorno virtual:
   ```text
   /Users/<TU_USUARIO>/Documents/darwin_rl_race/venv/bin/python
   ```

---

## 5. Ejecución

### Entrenamiento Individual por Algoritmo
1. Abre `worlds/pista_entrenamiento.wbt` en Webots.
2. En el nodo `RobotisOp2`, verifica que `supervisor` esté en `TRUE`, `translation` en `0 0 0.30` y asigna en `controller` el algoritmo a entrenar:
   * `"dqn"` (ejecuta `controllers/dqn/dqn.py`)
   * `"a2c"` (ejecuta `controllers/a2c/a2c.py`)
   * `"ppo"` (ejecuta `controllers/ppo/ppo.py`)
3. Guarda el mundo (`Ctrl + S` / `Cmd + S`) y presiona **Play** (puedes usar modo **Fast** para acelerar los episodios). Los pesos se exportarán a `weights/` y los registros a `metrics/`.

### Carrera Simultánea en Tiempo Real
1. Abre `worlds/pista_sin_obstaculos.wbt` o `worlds/pista_con_obstaculos.wbt`.
2. Configura las instancias de `RobotisOp2` en la línea de salida (`X = 0.0, Z = 0.30`, separadas en `Y`), asignando a cada robot su nombre identificador (`darwin_dqn`, `darwin_a2c`, `darwin_ppo`, `darwin_custom`) y el controlador de carrera (`race_supervisor_controller`).
3. Presiona **Play** en modo **Real-time** para visualizar la competencia y el tiempo de llegada en consola.
