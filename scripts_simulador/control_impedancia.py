#!/usr/bin/env python3
"""
Franka Panda Joint Impedance Controller with Isaac Sim ROS2 Bridge
Fixed version for Isaac Sim 5.1.0

Usage:
    python impedancia_standalone_fixed.py
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": True,
    "physics_hz": 5000,      # Frecuencia física
})

# ===== AÑADE ESTO JUSTO AQUÍ =====
import carb

# Desactivar rate limiting (limita a 30 Hz por defecto)
settings = carb.settings.get_settings()
settings.set("/app/runloops/main/rateLimitEnabled", False)
settings.set("/app/window/drawModeRepaint", False)
settings.set("/app/runloops/main/maxFPS", 0)  # 0 = sin límite

print("[INIT] Rate limiting desactivado - frecuencia libre")

import numpy as np
import sympy as sp
import omni.kit.app
from isaacsim.core.api import World
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka import Franka
import carb
from roboticstoolbox import models
import time
from isaacsim.core.utils.extensions import enable_extension
import omni.graph.core as og
from omni.isaac.core.articulations import ArticulationView

# Habilitar la extensión ROS2 Bridge
enable_extension("isaacsim.ros2.bridge")

# Esperar a que se cargue
simulation_app.update()
simulation_app.update()

# Importar rclpy después de habilitar la extensión
import rclpy
from std_msgs.msg import Float32MultiArray
from rosgraph_msgs.msg import Clock
from rclpy.node import Node

#importamos librerias para liberar memoria
import gc
import torch
import threading
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo
import matplotlib.pyplot as plt

class FrankaJointImpedanceController:
    """Controlador de impedancia para articulaciones del Franka Panda"""
    
    def __init__(self, robot, robot_view):
        self.robot = robot
        self.robot_view = robot_view  # Guardamos la vista
        self.stiffness = np.array([600., 600., 600., 600., 250., 600., 200.])
        self.damping = 2 * np.sqrt(self.stiffness * 2.5)
        # self.stiffness = np.array([12_000., 12_000., 12_000., 12_000., 12_000., 12_000., 12_000])
        # self.damping = np.array([1000., 1000., 1000., 1000., 1000., 1000., 1000.])
        self.target_positions = None
        self.modelo = models.DH.Panda()
        self.dq_past = np.zeros(7)

        # Flags para actualizar en el hilo principal
        self.pending_k_update = None
        self.pending_d_update = None
        self.pending_target_update = None

        # Inicializamos listas para cada joint
        self.torques_plot = [ [] for _ in range(7) ]

    def set_target(self, joint_positions):
        """Establece las posiciones objetivo de las articulaciones"""
        self.target_positions = np.array(joint_positions, dtype=float)

    def update_stiffness(self, new_k):
        """Actualiza la rigidez del controlador"""
        self.pending_k_update = np.array(new_k, dtype=float)
        # Actualizar damping asociado
        # self.pending_d_update = np.array(2 * np.sqrt(new_k * 2.5), dtype=float)

    def update_damping(self, new_d):
        """Actualiza el amortiguamiento del controlador"""
        self.pending_d_update = np.array(new_d, dtype=float)

    def update_target(self, new_target):
        """Actualiza el objetivo del controlador"""
        self.pending_target_update = np.array(new_target, dtype=float)

    def apply_pending_updates(self):
        """Aplica actualizaciones pendientes de parámetros"""
        if self.pending_k_update is not None:
            if len(self.pending_k_update) == 7:
                self.stiffness = self.pending_k_update
            self.pending_k_update = None

        if self.pending_d_update is not None:
            if len(self.pending_d_update) == 7:
                self.damping = self.pending_d_update
            self.pending_d_update = None

        if self.pending_target_update is not None:
            if len(self.pending_target_update) == 7:
                self.target_positions = self.pending_target_update
            self.pending_target_update = None

    def compute_impedance_torques(self):
        if self.target_positions is None:
            return np.zeros(7)

        # 1. Obtener datos dinámicos NATIVOS de Isaac Sim
        # get_mass_matrices devuelve un tensor de (num_robots, num_dofs, num_dofs)
        # Como solo hay 1 robot, tomamos el índice [0]
        
        # IMPORTANTE: ArticulationView devuelve datos de TODOS los joints (9 para Panda).
        # Debemos recortar las matrices para usar solo los primeros 7 (brazo).
        
        try:
            # Obtener matrices completas (9x9)
            M_full = self.robot_view.get_mass_matrices(clone=False)[0]
            C_full = self.robot_view.get_coriolis_and_centrifugal_forces(clone=False)[0]
            G_full = self.robot_view.get_generalized_gravity_forces(clone=False)[0]
            
            # Recortar a 7x7 y 7x1
            M = M_full[:7, :7]  # Matriz de inercia
            # M_diag = np.diag(M)
            # sp.pprint(sp.Matrix(np.round(M,2)))
            # print(np.diag(M))
            C = C_full[:7]      # Vector Coriolis
            G = G_full[:7]      # Vector Gravedad
            
            # Obtener estado actual (usando el robot o la view, ambos funcionan)
            q = self.robot_view.get_joint_positions(clone=False)[0][:7]
            dq = self.robot_view.get_joint_velocities(clone=False)[0][:7]
            
            # Calcular error
            position_error = self.target_positions - q
            
            # Ley de Control: M * (Kp * error - Kd * vel) + Coriolis + Gravedad
            # Nota: En la ecuación nativa, C y G ya son fuerzas/torques, no matrices que se multiplican.
            # Isaac devuelve C como el vector de fuerzas C(q,dq)*dq
            torques = M @ (self.stiffness * position_error - self.damping * dq) + C + G

            return torques
        except Exception as e:
            print(f"Error calculando torques: {e}")
            return np.zeros(7)


class ROS2Bridge(Node):
    """Nodo ROS2 para comunicación con el controlador"""
    
    def __init__(self, controller_callback, world):
        super().__init__('franka_impedance_node')
        self.controller_callback = controller_callback
        
        # Crear subscribers
        self.k_sub = self.create_subscription(
            Float32MultiArray, '/k_gains', self.k_callback, 10)
        self.d_sub = self.create_subscription(
            Float32MultiArray, '/d_gains', self.d_callback, 10)
        self.target_sub = self.create_subscription(
            Float32MultiArray, '/q_goals', self.target_callback, 10)
        self.reset_sub = self.create_subscription(
            Float32MultiArray, '/reset_sim', self.reset_callback, 10)
        
        # Crear publishers
        self.pos_pub = self.create_publisher(Float32MultiArray, '/joint_pos', 10)
        self.vel_pub = self.create_publisher(Float32MultiArray, '/joint_vel', 10)
        self.jerk_pub = self.create_publisher(Float32MultiArray, '/joint_jerks', 10)
        self.reset_done_pub = self.create_publisher(Float32MultiArray, '/sim_reset_done', 10)
        self.reset_req_pub = self.create_publisher(Float32MultiArray, '/sim_reset_request', 10)
        self.torques_pub = self.create_publisher(Float32MultiArray, '/torques', 10)
        self.clock_publisher = self.create_publisher(Clock, '/clock', 10)
        self.world = world
        
        print("[ROS2] Nodo ROS2 creado exitosamente")

    def k_callback(self, msg):
        """Callback para recibir ganancias K"""
        if len(msg.data) == 7:
            self.controller_callback('k', np.array(msg.data))
            # print(f"[ROS] Nuevo K recibido: {np.round(msg.data, 1)}")
            

    def d_callback(self, msg):
        """Callback para recibir ganancias D"""
        if len(msg.data) == 7:
            self.controller_callback('d', np.array(msg.data))
            # print(f"[ROS] Nuevo D recibido: {np.round(msg.data, 1)}")

    def target_callback(self, msg):
        """Callback para recibir posiciones objetivo"""
        if len(msg.data) == 7:
            self.controller_callback('target', np.array(msg.data))
            print(f"[ROS] Nuevo target recibido: {np.round(msg.data, 3)}")

    def reset_callback(self, msg):
        """Callback para recibir solicitud de reset"""
        print("[ROS] Reinicio de simulación solicitado por RL...")
        self.controller_callback('reset', None)

    def publish_pos(self, data):
        """Publica error de posición"""
        msg = Float32MultiArray()
        msg.data = data.tolist()
        self.pos_pub.publish(msg)
        # print(f"[ROS] Error publicado: {np.round(data, 3)}")

    def publish_velocity(self, data):
        """Publica velocidades"""
        msg = Float32MultiArray()
        msg.data = data.tolist()
        self.vel_pub.publish(msg)

    def publish_jerk(self, data):
        """Publica jerks"""
        msg = Float32MultiArray()
        msg.data = data.tolist()
        self.jerk_pub.publish(msg)

    def publish_reset_done(self):
        """Publica confirmación de reset"""
        msg = Float32MultiArray()
        msg.data = [1.0]
        self.reset_done_pub.publish(msg)

    def publish_reset_request(self):
        """Publica solicitud de reset"""
        msg = Float32MultiArray()
        msg.data = [1.0]
        self.reset_req_pub.publish(msg)

    def publish_torques(self, data):
        """Publica solicitud de reset"""
        msg = Float32MultiArray()
        msg.data = data.tolist()
        self.torques_pub.publish(msg)

    def publish_clock(self):
        while simulation_app.is_running():
            # Obtener tiempo de simulación
            sim_time = self.world._current_time
            
            # Crear mensaje Clock
            clock_msg = Clock()
            clock_msg.clock.sec = int(sim_time)
            clock_msg.clock.nanosec = int((sim_time - int(sim_time)) * 1e9)
            
            # Publicar
            self.clock_publisher.publish(clock_msg)
            time.sleep(0.0005)  # Publicar a 1000 Hz



class FrankaImpedanceSimulation:
    """Clase principal de simulación con control de impedancia usando ROS2"""
    
    # Límites articulares del Franka Panda (rad)
    JOINT_LIMITS_LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
    JOINT_LIMITS_UPPER = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
    
    # Posición home del robot
    HOME_POSITION = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    
    def __init__(self):
        self.world = None
        self.franka = None
        self.controller = None
        self.frame = 0
        self.subscription = None
        self.initialized = False
        self.setup_done = False
        self.wait_frames = 0

        # ROS2
        self.ros2_node = None
        self.ros2_initialized = False

        # Variables para cálculo de derivadas
        self.last_time = None
        self.last_dq = np.zeros(7)
        self.last_ddq = np.zeros(7)
        
        # Control de reset
        self.reset_requested = False
        self.resetting = False
        self.post_reset_frames = 0
        self.initial_joint_positions = None
        self.reset_count = 0
        self.consecutive_resets = 0
        self.last_reset_time = 0
        self.episodes = 0
        self.last_k = np.array([600., 600., 600., 600., 250., 600., 200.])
        self.last_process_time = [0,0]

    def setup(self):
        """Configura el mundo y el robot"""
        try:
            # Obtener el World singleton
            self.world = World.instance()
            if self.world is None:
                self.world = World(stage_units_in_meters=1.0)
                self.world.scene.add_default_ground_plane()
            
            # self.world.set_physics_step_size(0.001)  # 1 ms
            # self.world.set_min_simulation_frame_rate(1000)  # 1000 Hz

            self.world.set_simulation_dt(physics_dt=0.001, rendering_dt=0.001)  # 1 ms
            # get_current_stage = self.world.get_stage()
            # get_current_stage.SetTimeCodesPerSecond(0.002)

            # Agregar el robot usando el método correcto
            if not self.world.scene.object_exists("franka"):
                self.franka = self.world.scene.add(
                    Franka(prim_path="/World/Franka", name="franka", position=np.zeros(3))
                )
            else:
                self.franka = self.world.scene.get_object("franka")
            
            self.setup_done = True
            print("[SETUP] Mundo y robot configurados correctamente")

            #iniciamos la simulación
            self.world.play()
        except Exception as e:
            import traceback
            print(f"[ERROR] Error en setup: {e}")
            print(traceback.format_exc())
            self.setup_done = False

    def initialize(self):
        """Inicializa el controlador y ROS2"""
        if not self.setup_done:
            return False
            
        if self.franka is None or not self.franka.handles_initialized:
            self.franka.initialize()
            return False

        positions = self.franka.get_joint_positions()[:7]
        self.initial_joint_positions = positions.copy()

        if self.world.scene.object_exists("franka_view"):
            # Si ya existe (de un intento anterior), la recuperamos
            self.franka_view = self.world.scene.get_object("franka_view")
        else:
            # Si no existe, la creamos y añadimos
            self.franka_view = ArticulationView(
                prim_paths_expr="/World/Franka", 
                name="franka_view"
            )
            self.world.scene.add(self.franka_view)

        # self.world.set_simulation_dt(0.001)  # 1 ms
        if not self.franka_view._is_initialized:
            self.franka_view.initialize()
        
        self.controller = FrankaJointImpedanceController(self.franka, self.franka_view)
        self.controller.set_target(positions)

        # Inicializar ROS2
        if not self.ros2_initialized:
            try:
                rclpy.init()
                self.ros2_node = ROS2Bridge(self.ros2_callback, self.world)
                self.ros2_initialized = True

                # lanzamos el spin en un hilo aparte
                self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.ros2_node,), daemon=True)
                self.spin_thread.start()

                # Ejecutar en thread separado
                self.thread_clk = threading.Thread(target=self.ros2_node.publish_clock)
                self.thread_clk.start()

                print("[ROS2] ROS2 inicializado correctamente")
            except Exception as e:
                print(f"[ROS2] Error inicializando ROS2: {e}")
                return False

        self.initialized = True
        print(f"[INIT] Controlador inicializado")
        print(f"  Stiffness: {self.controller.stiffness}")
        print(f"  Damping: {self.controller.damping}")

        # Publicar confirmación de reset
        if self.ros2_node is not None:
            self.ros2_node.publish_reset_done()
            print("[INIT] Confirmación de init publicada")

        return True

    def ros2_callback(self, msg_type, data):
        """Callback unificado para mensajes ROS2"""
        if msg_type == 'k':
            self.controller.update_stiffness(data)
        elif msg_type == 'd':
            self.controller.update_damping(data)
        elif msg_type == 'target':
            self.controller.update_target(data)
            self.episodes += 1
        elif msg_type == 'reset':
            self.request_reset()

    def request_reset(self):
        """Solicita un reset que se ejecutará en el próximo frame"""
        current_time = self.world._current_time
        
        # Prevenir resets demasiado frecuentes
        if current_time - self.last_reset_time < 0.5:
            print("[RESET] Reset ignorado: demasiado reciente (< 0.5s)")
            return
        
        self.reset_requested = True
        self.last_reset_time = current_time
        self.consecutive_resets += 1
        
        if self.consecutive_resets > 5:
            print(f"[RESET] ADVERTENCIA: {self.consecutive_resets} resets consecutivos")
        
        print("[RESET] Reset solicitado, se ejecutará en el próximo frame")
        
        # Notificar al RL sobre el reset solicitado
        self.ros2_node.publish_reset_request()

    def execute_reset(self):
        """Ejecuta el reset de forma segura"""
        if self.resetting:
            return
            
        self.resetting = True
        print("[RESET] Ejecutando reset del mundo...")
        
        try:
            # Resetear robot a posición home
            if self.franka is not None:
                zero_velocities = np.zeros(9)
                full_positions = np.zeros(9)
                full_positions[:7] = self.HOME_POSITION
                
                # Aplicar posición y velocidad cero
                self.franka.set_joint_positions(full_positions)
                self.franka.set_joint_velocities(zero_velocities)
                
                # Aplicar torques cero
                self.franka.apply_action(ArticulationAction(
                    joint_positions=full_positions,
                    joint_velocities=zero_velocities,
                    joint_efforts=np.zeros(9)
                ))
            
            # Reinicializar el controlador con posiciones home
            self.controller.set_target(self.HOME_POSITION)
            
            # Resetear variables de tiempo y derivadas
            self.last_time = None
            self.last_dq = np.zeros(7)
            self.last_ddq = np.zeros(7)
            self.frame = 0
            
            # Publicar confirmación de reset
            if self.ros2_node is not None:
                self.ros2_node.publish_reset_done()
                print("[RESET] Confirmación de reset publicada")
            
            self.post_reset_frames = 0
            self.resetting = False
            self.reset_requested = False
            self.reset_count += 1
            
            print(f"[RESET] Reset #{self.reset_count} completado exitosamente")
            
        except Exception as e:
            import traceback
            print(f"[RESET] Error durante reset: {e}")
            print(f"[RESET] Traceback: {traceback.format_exc()}")
            
            self.resetting = False
            self.reset_requested = False

    def check_for_collision_or_instability(self, q, q_error):
        """Detecta condiciones que requieren reset"""
        # Detectar NaNs
        if np.isnan(q).any() or np.isnan(q_error).any():
            print("[COLLISION] NaN detectado en posiciones o error")
            return True
        
        # Detectar límites articulares excedidos
        margin = 0.2  # Margen de seguridad en radianes
        if np.any(q < self.JOINT_LIMITS_LOWER - margin) or \
           np.any(q > self.JOINT_LIMITS_UPPER + margin):
            print("[COLLISION] Límites articulares excedidos")
            return True
        
        return False

    def on_update(self, event):
        """Callback principal del loop de simulación"""
        if not self.setup_done:
            self.setup()
            return
        
        if not self.world.is_playing():
            return
        
        if not self.initialized:
            self.wait_frames += 1
            if self.wait_frames < 10:
                return
            if not self.initialize():
                return

        tmp_proc = time.time()

        if self.episodes >= 100:
            # print("Cerramos programa para liberar memoria")
            # self.ros2_node.publish_reset_request() #le avisamos al RL que reiniciamos
            # rclpy.spin_once(self.ros2_node, timeout_sec=0.0)
            self.episodes = 0
            gc.collect()
            torch.cuda.empty_cache()
            # self.stop()
            # simulation_app.close()
            # return

        # Procesar mensajes ROS2
        # if self.ros2_node is not None:
        #     rclpy.spin_once(self.ros2_node, timeout_sec=0.0)

        # Ejecutar reset si fue solicitado
        if self.reset_requested and not self.resetting:
            self.execute_reset()
            return
        
        # Esperar frames adicionales después del reset
        if self.post_reset_frames < 10:
            self.post_reset_frames += 1
            if self.post_reset_frames == 10:
                self.consecutive_resets = max(0, self.consecutive_resets - 1)
            return

        # Obtener estado actual
        try:
            q = self.franka.get_joint_positions()[:7]
            dq = self.franka.get_joint_velocities()[:7]
            q_error = self.controller.target_positions - q

            controller = self.franka.get_articulation_controller()
            controller.set_gains(np.array([0.0]*9), np.array([0.0]*9))

        except Exception as e:
            print(f"Error obteniendo estado del robot: {e}")
            self.request_reset()
            return

        # Verificar condiciones de colisión o inestabilidad
        if self.check_for_collision_or_instability(q, q_error):
            self.request_reset()
            return

        # Calcular jerk (derivada de la aceleración)
        current_time = self.world._current_time
        jerk = None
        if self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                ddq = (dq - self.last_dq) / dt
                jerk = (ddq - self.last_ddq) / dt
                self.last_ddq = ddq
        self.last_dq = dq
        self.last_time = current_time

        # Aplicar actualizaciones pendientes del controlador, por si llega nueva K
        # self.controller.apply_pending_updates()

        # print(f"Current K: {np.round(self.controller.stiffness, 1)}")
        # print(f"Last K:    {np.round(self.last_k, 1)}")
        # Publicar estado cuando nos llegue el nuevo K
        if self.ros2_node is not None:
            self.ros2_node.publish_pos(q)
            self.ros2_node.publish_velocity(dq)
            if jerk is not None:
                self.ros2_node.publish_jerk(jerk)
            
            # self.last_k = self.controller.stiffness.copy() #actualizamos last_k

        # Aplicar cambios pendientes
        self.controller.apply_pending_updates()

        # Calcular y aplicar torques
        torques = self.controller.compute_impedance_torques()
        full_torques = np.zeros(9)
        full_torques[:7] = torques
        if self.ros2_node is not None:
            self.ros2_node.publish_torques(torques)
        
        try:
            self.franka.apply_action(ArticulationAction(joint_efforts=full_torques))
        except Exception as e:
            print(f"Error aplicando torques: {e}")
            self.request_reset()
            return

        tmp_sim = self.world._current_time
        tmp_cpu = time.time()

        # Log periódico
        if self.frame % 100 == 0:
            error_norm = np.linalg.norm(q_error)
            np.set_printoptions(suppress=True)   # desactiva notación científica
            print(f"\n[Frame {self.frame}]")
            print(f"  Error: {np.round(q_error, 3)} rad (norm: {error_norm:.3f})")
            print(f"  Vel: {np.round(dq, 3)} rad/s")
            if jerk is not None:
                print(f"  Jerk: {np.round(jerk, 1)} rad/s³")
            print(f"  K: {np.round(self.controller.stiffness, 1)}")
            print(f"  D: {np.round(self.controller.damping, 1)}")
            print(f"  Torques: {np.round(torques, 2)} Nm")
            print(f"\n  Sim Delta Time: {(tmp_sim - self.last_process_time[0])*1000:.2f} ms")
            print(f"  CPU Delta Time: {(tmp_cpu - self.last_process_time[1])*1000:.2f} ms")
            print(f"  Process Time (CPU): {(tmp_cpu - tmp_proc)*1000:.2f} ms")

        self.last_process_time = [tmp_sim, tmp_cpu]

        self.frame += 1

    def start(self):
        """Inicia la simulación"""
        update_stream = omni.kit.app.get_app().get_update_event_stream()
        self.subscription = update_stream.create_subscription_to_pop(
            self.on_update, 
            name="franka_joint_impedance_control"
        )
        print("\n" + "="*60)
        print("FRANKA IMPEDANCE CONTROL SIMULATION")
        print("="*60)
        print("Simulación iniciada. Presiona PLAY en Isaac Sim.")
        print("\nTópicos ROS2 disponibles:")
        print("  - /k_gains (input): Ganancias de rigidez [Float32MultiArray]")
        print("  - /d_gains (input): Ganancias de amortiguamiento [Float32MultiArray]")
        print("  - /q_goals (input): Posiciones objetivo [Float32MultiArray]")
        print("  - /reset_sim (input): Solicitud de reset [Float32MultiArray]")
        print("  - /joint_error (output): Error de posición [Float32MultiArray]")
        print("  - /joint_vel (output): Velocidades articulares [Float32MultiArray]")
        print("  - /joint_jerks (output): Jerks articulares [Float32MultiArray]")
        print("  - /sim_reset_done (output): Confirmación de reset [Float32MultiArray]")
        print("\nPara probar la conexión, ejecuta en otra terminal:")
        print("  ros2 topic list")
        print("  ros2 topic echo /joint_error")
        print("="*60 + "\n")

    def stop(self):
        """Detiene la simulación y limpia recursos"""
        print("\n[STOP] Deteniendo simulación...")
        
        if self.subscription:
            self.subscription.unsubscribe()
            self.subscription = None
        
        if self.ros2_node is not None:
            self.ros2_node.destroy_node()
            
        if self.ros2_initialized:
            rclpy.shutdown()
            if self.spin_thread.is_alive():
                self.spin_thread.join(timeout=1)
            
            if self.thread_clk.is_alive():
                self.thread_clk.join(timeout=1)
        
        print("[STOP] Simulación detenida correctamente")


# ==================== Main Entry Point ====================

def main():
    """Función principal"""
    try:
        # Crear e iniciar simulación
        sim = FrankaImpedanceSimulation()
        sim.start()
        
        # Loop principal
        while simulation_app.is_running():
            simulation_app.update()
        
        # Cleanup al salir
        sim.stop()
        simulation_app.close()
        
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupción por teclado detectada")
        # Cleanup al salir
        sim.stop()
        simulation_app.close()
    except Exception as e:
        print(f"[MAIN] Error crítico: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()


if __name__ == "__main__":
    main()
