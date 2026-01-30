# TFG_Adrian_Fernandez_Casas
Recopilación de los códigos utilizados para el TFG Control en impedancia para la actuación rápida de robots con visión basada en eventos

## 📂 Estructura del Proyecto

A continuación se detalla la organización de los módulos principales del software:

```text
.
├── 📂 rl_package                     # Paquete de aprendizaje por refuerzo
│   ├── 📂 rl_package                 # Carpeta de los módulos
│   │   ├── entrenamiento.py           # Implementación final del entrenamiento
│   │   ├── perfil_trapezoidal.py      # Entrenamiento para generar ganancias siguiendo un perfil trapezoidal
│   │   ├── one_joint.py               # Entrenamiento inicial, moviendo una única articulación
│   │   ├── auxiliar.py                # Librería auxiliar para modificar SAC y generar nuevas poses
│   │   └── record_move.py             # Registro de trayectorias y demostraciones
│   └── setup.py                       # Definición de los módulos necesaria para ROS 2
│
├── 📂 robot_real                                    # Implementación con el robot real
│   ├── saveVideo.cpp                                 # Mueve al robot a una serie de trayectorias cartesianas, guardadas en txt, guardando tanto el robot_state (a 60Hz) como el video
│   ├── saveVideo_joint.cpp                           # Mueve al robot a una serie de trayectorias articulares y guarda el video
│   ├── go_home.cpp                                   # Mueve a home mediante movimiento articular
│   ├── calibracion_joint.cpp                         # Realiza la autocalibración de la cámara mediante movimiento articular, generando los parámetros intrínsecos y extrínsecos
│   ├── control_cartesiano.cpp                        # Movimiento cartesiano, mueve el end effector a una pose seleccionada, con una determinada velocidad lineal y angular
│   ├── control_impedancia.cpp                        # Utiliza la base de control_cartesiano para controlar el robot en impedancia
│   ├── moveTrajectory.cpp                            # Lee un .txt con poses y realiza una trayectoria cartesiana
│   ├── moveJointTrajectory.cpp                       # Lee un .txt con configuraciones articulares y realiza una trayectoria articular
│   ├── visual_servoing_apriltag.cpp                  # Realiza control visual con ViSP para seguir un AprilTag
│   ├── visual_servoing_pelota_rgb.cpp                # Realiza control visual con ViSP para seguir la pelota mediante la detección por RGB
│   ├── visual_servoing_pelota_yolo.cpp               # Realiza control visual con ViSP para seguir la pelota mediante la detección con YOLOv4
│   ├── visual_servoing_pelota_eventos_cliente.cpp    # Realiza control visual con ViSP para seguir la pelota con la información de la cámara de eventos
│   └── visual_servoing_pelota_eventos_server.py      # Detecta la pelota utilizando la cámara de eventos y se lo pasa al cliente
│
├── 📂 scripts_simulacion      # Scripts para la simulación en Isaac Sim
│   ├── control_adaptativo.py  # Control en impedancia con aceleración constante
│   ├── control_impedancia.py  # Control en impedancia para RL
│   └── visual_servoing.py     # Control visual IBVS para detectar la pelota
│
├── CMakeLists.txt            # Configuración de compilación para los nodos C++
└── launcher_isaac.sh         # Script de arranque para el simulador NVIDIA Isaac Sim
