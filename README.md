# TFG_Adrian_Fernandez_Casas
Recopilación de los códigos utilizados para el TFG Control en impedancia para la actuación rápida de robots con visión basada en eventos

## 📂 Estructura del Proyecto

A continuación se detalla la organización de los módulos principales del software:

```text
.
├── 📂 rl_package             # Módulos de Aprendizaje por Refuerzo (SAC)
│   ├── simplificado.py       # Entrenamiento con espacio de acción unificado
│   └── record_move.py        # Registro de trayectorias y demostraciones
│
├── 📂 robot_real             # Implementación en hardware real (C++/ViSP)
│   ├── visp_aruco.cpp        # Seguimiento de marcadores ArUco
│   └── visp_pelota.cpp       # Detección y seguimiento de esferas
│
├── 📂 scripts_simulacion     # Algoritmos de control en entorno virtual
│   ├── control_adaptativo.py # Estrategias de adaptación dinámica
│   ├── control_impedancia.py # Control de interacción física segura
│   └── visual_servoing.py    # Servovisual (IBVS) basado en eventos
│
├── CMakeLists.txt            # Configuración de compilación para los nodos C++
└── launcher_isaac.sh         # Script de arranque para el simulador NVIDIA Isaac Gym
