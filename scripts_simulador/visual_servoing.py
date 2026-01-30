#!/usr/bin/env python3
"""
Franka Panda Visual Servoing - Trayectoria 4 Puntos
Uso: Ejecutar y pulsar la tecla 'Q' (en la ventana de Isaac Sim) para guardar gráfica y salir.
"""

from isaacsim import SimulationApp
# Iniciamos simulador
simulation_app = SimulationApp({
    "headless": False,
    "physics_hz": 60,
})

import numpy as np
import omni.kit.app
from isaacsim.core.api import World
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.franka import Franka
import time
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_rot_matrix
from pxr import Gf, UsdGeom, Usd
from omni.isaac.sensor import Camera
from omni.isaac.core.objects import FixedSphere, DynamicSphere
from isaacsim.core.api.materials import PhysicsMaterial
import cv2
import carb
import carb.input
import omni.appwindow
import os

# Configuración de Matplotlib (Backend no interactivo)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

carb.settings.get_settings().set_bool("/rtx/shadows/enabled", False)

# Esperar carga inicial
simulation_app.update()

class FrankaImpedanceSimulation:
    
    # Límites y Home
    JOINT_LIMITS_LOWER = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
    JOINT_LIMITS_UPPER = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
    HOME_POSITION = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
    
    def __init__(self):
        self.world = None
        self.franka = None
        self.frame = 0
        self.initialized = False
        self.setup_done = False
        self.wait_frames = 0
        self.last_dq = np.zeros(7)
        self.last_time = None
        self.post_reset_frames = 0
        self.target_positions = self.HOME_POSITION.copy()
        self.first_pose = True
        
        self.last_error = None # Para control derivativo
        # Historial para gráfica
        self.features_history = [] 
        
        # Control de parada
        self.stop_requested = False
        self.input_sub = None # Suscripción al teclado
        self.is_closing = False

    def setup(self):
        try:
            self.world = World.instance()
            if self.world is None:
                self.world = World(stage_units_in_meters=1.0)
                self.world.scene.add_default_ground_plane()

            self.world.set_simulation_dt(physics_dt=1/60.0, rendering_dt=1/60.0)

            if not self.world.scene.object_exists("franka"):
                self.franka = self.world.scene.add(
                    Franka(prim_path="/World/Franka", name="franka", position=np.zeros(3))
                )
                 
                # Cámara
                assets_root = get_assets_root_path()
                d455_path = assets_root + "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
                cam_base_path = "/World/Franka/panda_hand/rsd455"
                add_reference_to_stage(usd_path=d455_path, prim_path=cam_base_path)
                cam_xform = XFormPrim(prim_path=cam_base_path)
                cam_xform.set_local_pose(translation=np.array([0.075, 0.0, 0.05]),
                                        orientation=euler_angles_to_quat(np.array([-180, -90, 0]), degrees=True))
                
                def find_rgb_camera(base_path):
                    stage = get_current_stage()
                    for prim in Usd.PrimRange(stage.GetPrimAtPath(base_path)):
                        if prim.IsA(UsdGeom.Camera): return prim.GetPath().pathString
                    return None
                
                real_cam_path = find_rgb_camera(cam_base_path)
                self.camera = Camera(prim_path=real_cam_path, resolution=(1280, 720))
                self.camera.initialize()

                # Cubo Objetivo
                mat_path = "/World/Physics_Materials/ball_mat"
                ball_mat = PhysicsMaterial(prim_path=mat_path, dynamic_friction=0.2, static_friction=0.2, restitution=0.8)
                # self.target_ball = self.world.scene.add(FixedSphere(
                #     prim_path="/World/StaticSphere",
                #     name="static_ball",
                #     position=np.array([0.35, -0.2, 0.06]), # Posición diferente para no chocar
                #     radius=0.06,
                #     color=np.array([1, 0, 0]), # También roja
                #     physics_material=ball_mat
                # ))
                # Bola Dinámica (para interacción física)
                self.target_ball = self.world.scene.add(DynamicSphere(
                    prim_path="/World/Sphere",
                    name="dynamic_ball",
                    position=np.array([0.35, 0.2, 0.06]),
                    radius=0.06,
                    color=np.array([1, 0, 0]),
                    mass=1.0,
                    physics_material=ball_mat))

            else:
                self.franka = self.world.scene.get_object("franka")
            
            self.setup_done = True
            
            # CONFIGURAR TECLADO AQUÍ
            self.setup_keyboard()
            
            self.world.play()
            
        except Exception as e:
            print(f"[ERROR] Setup: {e}")

    def setup_keyboard(self):
        """Se suscribe a los eventos del teclado de Isaac Sim"""
        app_window = omni.appwindow.get_default_app_window()
        keyboard = app_window.get_keyboard()
        # Suscribirse al evento
        self.input_sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(
            keyboard, self.on_keyboard_event
        )
        print("[INPUT] Teclado configurado. Pulsa 'Q' para guardar y salir.")

    def on_keyboard_event(self, event, *args):
        """Callback que se ejecuta al pulsar una tecla"""
        # Chequear si es un evento de pulsar tecla (KEY_PRESS) y si es la Q
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.Q:
                print("\n[USER] ¡Tecla Q detectada! Iniciando guardado...")
                self.stop_requested = True
        return True

    def initialize(self):
        if not self.setup_done: return False
        if self.franka is None or not self.franka.handles_initialized:
            self.franka.initialize()
            return False

        if self.world.scene.object_exists("franka_view"):
            self.franka_view = self.world.scene.get_object("franka_view")
        else:
            self.franka_view = ArticulationView(prim_paths_expr="/World/Franka", name="franka_view")
            self.world.scene.add(self.franka_view)

        if not self.franka_view._is_initialized:
            self.franka_view.initialize()

        self.initialized = True
        return True

    def order_points(self, pts):
        """Ordena: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # TL
        rect[2] = pts[np.argmax(s)] # BR
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # TR
        rect[3] = pts[np.argmax(diff)] # BL
        return rect

    def save_trajectory_graph(self):
        """Genera la gráfica de trayectoria (Verde) vs Meta (Rojo)"""
        if not self.features_history:
            print("[PLOT] No hay datos suficientes para graficar.")
            return

        print(f"[PLOT] Generando gráfica con {len(self.features_history)} frames...")
        
        # 1. Datos históricos
        history = np.array(self.features_history)
        
        # 2. Calcular Metas en Píxeles
        Z_des = 0.2
        val_norm = (0.06 / Z_des) / 2.0
        
        # Puntos meta normalizados (Orden: TL, TR, BR, BL)
        targets_norm = np.array([
            [-val_norm, -val_norm], 
            [ val_norm, -val_norm], 
            [ val_norm,  val_norm], 
            [-val_norm,  val_norm]
        ])
        
        # Desnormalizar
        K = self.camera.get_intrinsics_matrix()
        fx = K[0, 0]; fy = K[1, 1]
        px = K[0, 2]; py = K[1, 2]
        
        targets_px = np.zeros_like(targets_norm)
        targets_px[:, 0] = targets_norm[:, 0] * fx + px
        targets_px[:, 1] = targets_norm[:, 1] * fy + py

        # 3. Plotear
        plt.figure(figsize=(10, 8))
        
        # Trayectorias (Verde)
        labels = ["TL", "TR", "BR", "BL"]
        for i in range(4):
            plt.plot(history[:, i, 0], history[:, i, 1], color='green', linewidth=2, alpha=0.6)
            # Punto de inicio
            plt.scatter(history[0, i, 0], history[0, i, 1], color='green', s=20, marker='o')

        # Metas (Rojo)
        plt.scatter(targets_px[:, 0], targets_px[:, 1], color='red', marker='x', s=150, linewidths=3, label='Meta (Z=0.15m)')
        
        plt.title(f"Trayectoria Visual Servoing (4 Puntos)")
        plt.xlabel("U (Píxeles)")
        plt.ylabel("V (Píxeles)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.5)
        
        # Invertir eje Y (Coordenadas de imagen)
        plt.gca().invert_yaxis()
        plt.axis('equal') # Mantener proporción de aspecto cuadrada
        
        # Guardar
        idx = 1
        while True:
            filename = f"./graficas/trayectoria_visual{idx}.png"
            if not os.path.exists(filename):
                break
            idx += 1
            
        plt.savefig(filename)
        plt.close()
        print(f"[PLOT] Gráfica guardada en: {os.path.abspath(filename)}")

    def on_update(self, event):
        # 0. CHEQUEO DE PARADA (Tecla Q)
        if self.stop_requested:
            # Si ya estamos cerrando, no hacemos nada (evita guardar 2 veces)
            if self.is_closing:
                return
            
            # Marcamos que hemos empezado el proceso de cierre
            self.is_closing = True
            
            print("[STOP] Deteniendo simulación...")
            self.save_trajectory_graph()
            simulation_app.close()
            return

        if not self.setup_done:
            self.setup()
            return
        if not self.world.is_playing():
            return
        if not self.initialized:
            self.wait_frames += 1
            if self.wait_frames < 10: return
            if not self.initialize(): return

        if self.post_reset_frames < 10:
            self.post_reset_frames += 1
            return

        # --- Obtener Estado Robot ---
        dt = self.world.get_physics_dt()
        q = self.franka.get_joint_positions()[:7]
        q_error = self.target_positions - q

        if self.first_pose:
            self.franka.apply_action(ArticulationAction(joint_positions=np.append(self.target_positions, [0.0, 0.0])))
            if np.linalg.norm(q_error) < 0.02:
                self.first_pose = False
                print("[RUN] Inicio Visual Servoing...")
            return
            
        # --- Lógica de Control ---
        J_hand = self.franka_view.get_jacobians()[0][8, :, :7]
        
        if not hasattr(self, "camera_prim"): self.camera_prim = XFormPrim(self.camera.prim_path)
        if not hasattr(self, "ee_prim"): self.ee_prim = XFormPrim("/World/Franka/panda_hand")
        
        p_cam, q_cam = self.camera_prim.get_world_pose()
        R_cam = quat_to_rot_matrix(q_cam)
        p_hand, _ = self.ee_prim.get_world_pose()

        # Visión
        frame = self.camera.get_rgba()
        img_rgb = frame[:, :, :3].astype(np.uint8)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        lower1 = np.array([0, 70, 50]); upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 70, 50]); upper2 = np.array([180, 255, 255])
        mask = cv2.bitwise_or(cv2.inRange(img_hsv, lower1, upper1), cv2.inRange(img_hsv, lower2, upper2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        v_cartesian_target = np.zeros(6)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            
            # Filtro de ruido y detección de "Esfera"
            if area > 100:
                # --- 1. Obtener Intrínsecos ---
                K = self.camera.get_intrinsics_matrix()
                fx = K[0, 0]; fy = K[1, 1]
                px = K[0, 2]; py = K[1, 2]

                # --- 2. Calcular Centroide y Radio ---
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                
                ((_, _), radius_px) = cv2.minEnclosingCircle(c)
                
                # --- 3. Estimar Profundidad (Z) ---
                Z_est = (fx * 0.03) / radius_px
                Z_est = np.clip(Z_est, 0.05, 1.5)
                Z_des = 0.2

                # --- 4. DEFINIR TRANSFORMACIONES ---
                # R_cv_to_usd: OpenCV (+Y Abajo) -> USD (+Y Arriba) -> Invierte signo Y
                R_cv_to_usd = np.array([
                    [1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]
                ])
                R_opt_world = R_cam @ R_cv_to_usd

                # --- 5. RECONSTRUCCIÓN DEL CENTRO ---
                X_opt = (cX - px) * Z_est / fx
                Y_opt = (cY - py) * Z_est / fy
                P_opt = np.array([X_opt, Y_opt, Z_est])
                P_center_world = p_cam + (R_opt_world @ P_opt)

                # --- 6. GENERACIÓN DE PUNTOS (CORREGIDO) ---
                # Objetivo: Generar puntos en orden [TL, TR, BR, BL] de la IMAGEN.
                
                off = 0.03
                world_offsets = [
                    # 1. Top-Left (TL): 
                    np.array([ off,  off, 0.0]), 
                    # 2. Top-Right (TR): 
                    np.array([ off, -off, 0.0]), 
                    # 3. Bottom-Right (BR): 
                    np.array([-off, -off, 0.0]), 
                    # 4. Bottom-Left (BL): 
                    np.array([-off,  off, 0.0])  
                ]
                
                points_projected = []
                for off_vec in world_offsets:
                    Pw = P_center_world + off_vec
                    Pc = R_opt_world.T @ (Pw - p_cam)
                    
                    if Pc[2] > 0.01: 
                        u = fx * (Pc[0] / Pc[2]) + px
                        v = fy * (Pc[1] / Pc[2]) + py
                        points_projected.append([u, v])
                    else:
                        points_projected.append([cX, cY])

                pts_sorted = np.array(points_projected, dtype="float32")

                # --- 7. DEBUG FRAME 100 ---
                if self.frame % 30 == 0:
                    print(f"[DEBUG] Guardando imagen comparativa Frame {self.frame}...")
                    debug_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    
                    # Dibujar puntos numerados para verificar orden
                    # 0: Rojo, 1: Verde, 2: Azul, 3: Amarillo
                    colors = [(0,0,255), (0,255,0), (255,0,0), (0,255,255)]
                    for i, pt in enumerate(pts_sorted):
                        cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 5, colors[i], -1)
                        cv2.putText(debug_img, str(i), (int(pt[0])+10, int(pt[1])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)

                    pts_int = pts_sorted.astype(np.int32)
                    cv2.polylines(debug_img, [pts_int], True, (0, 255, 0), 2)

                    half_side_px = (fx * (0.06 / 2.0)) / Z_des
                    
                    p_desired = np.array([
                        [px - half_side_px, py - half_side_px], 
                        [px + half_side_px, py - half_side_px], 
                        [px + half_side_px, py + half_side_px], 
                        [px - half_side_px, py + half_side_px]  
                    ])
                    
                    pts_des_int = p_desired.astype(np.int32)
                    cv2.polylines(debug_img, [pts_des_int], True, (255, 0, 0), 2)

                    if not os.path.exists("./graficas/funcion_propia"): os.makedirs("./graficas/funcion_propia")
                    cv2.imwrite(f"./graficas/funcion_propia/comparativa_frame_{self.frame}.png", debug_img)
                    
                # --- GUARDAR EN HISTORIAL ---
                self.features_history.append(pts_sorted.copy())
                
                # Control (usando 3 puntos)
                current_pts_px = pts_sorted[:3] 
                
                s = np.zeros((3, 2))
                s[:, 0] = (current_pts_px[:, 0] - px) / fx
                s[:, 1] = (current_pts_px[:, 1] - py) / fy
                
                dist_px = np.linalg.norm(pts_sorted[0] - pts_sorted[1])
                Z = np.clip((fx * 0.06) / dist_px, 0.05, 1.0)

                val = (0.06 / Z_des) / 2.0
                s_star = np.array([[-val, -val], [val, -val], [val, val]])
                
                error_vec = (s_star - s).flatten()
                
                L_stack = []
                for i in range(3):
                    u = pts_sorted[i][0] - s_star[i][0]  #es posible que este mal calculada s_star? calcular mediante px?
                    v = pts_sorted[i][1] - s_star[i][1]
                    z = Z

                    L_i = np.array([
                        [-fx / z, 0, u / z, u * v / fx, -(fx ** 2 + u ** 2) / fx, v],
                        [0, -fy / z, v / z, (fy ** 2 + v ** 2) / fy, -u * v / fy, -u]
                    ])
                    L_stack.append(L_i)
                L = np.vstack(L_stack)
                
                try:
                    L_inv = np.linalg.inv(L)
                except:
                    L_inv = np.linalg.pinv(L)

                if self.last_error is None:
                    Kp = 25_000.0
                    v_cam_optical = Kp * (L_inv @ error_vec)
                    self.last_error = error_vec
                else:
                    Kp = 50_000.0
                    Kd = 5_000.0
                    derivative = (error_vec - self.last_error) / dt
                    v_cam_optical = Kp * (L_inv @ error_vec) + Kd * (L_inv @ derivative)
                    self.last_error = error_vec
                
                R_opt_to_usd = np.array([
                    [ 1,  0,  0],
                    [ 0, -1,  0],
                    [ 0,  0, -1]
                ])
                v_lin_opt = v_cam_optical[:3]
                w_ang_opt = v_cam_optical[3:]

                v_lin_cam_world = R_cam @ (R_opt_to_usd @ v_lin_opt)
                w_cam_world     = R_cam @ (R_opt_to_usd @ w_ang_opt)
                

                # --- 2. GEOMETRÍA (ESTABILIZADOR DE HORIZONTE) ---
                # Calculamos el error de inclinación respecto al suelo
                
                # El eje Z de la cámara en el mundo es la 3ra columna de R_cam
                z_axis_cam = -1.0 * R_cam[:, 2] 
                
                # Queremos que mire perfectamente hacia abajo (Z mundo = -1)
                z_axis_des = np.array([0.0, 0.0, -1.0])
                
                # Producto Cruz: Nos da el eje y magnitud para alinear Z_cam con Z_des
                tilt_error = np.cross(z_axis_cam, z_axis_des)
                
                # Ganancia proporcional para la corrección (Alta para ser rígida)
                k_horizon = 4.0 
                w_horizon = k_horizon * tilt_error

                # --- 3. FUSIÓN DE CONTROLADORES ---
                w_final_world = np.zeros(3)
                
                # Ejes X e Y (Roll/Pitch): Manda la GEOMETRÍA (Mantener plano)
                # Esto "apaga" el ruido visual que intenta inclinar el robot
                w_final_world[0] = w_horizon[0]
                w_final_world[1] = w_horizon[1]
                
                # Eje Z (Yaw): Manda la VISIÓN (Alinear rotación del objeto)
                w_final_world[2] = w_cam_world[2]
                
                # --- 4. CORRECCIÓN BRAZO DE PALANCA (LEVER ARM) ---
                # Transportamos la velocidad del punto "Cámara" al punto "Mano"
                # v_hand = v_cam - (w x r)
                r_offset = p_cam - p_hand
                vel_induced = np.cross(w_final_world, r_offset)
                
                v_lin_hand = v_lin_cam_world - vel_induced
                w_hand     = w_final_world

                v_cartesian_target[:3] = v_lin_hand
                v_cartesian_target[3:] = w_hand
                
                if self.frame % 50 == 0:
                    print(f"Z: {Z:.2f} | Err: {np.linalg.norm(error_vec):.3f} | Err_vec: {np.round(error_vec,3)}")
                    print(f"Vel cam óptica: {np.round(v_lin_opt,3)}")
                    print(f"Vel cam mundo: {np.round(v_lin_cam_world,3)}")
                
                if self.frame == 400:
                    print("[INFO] Movemos pelotita...")
                    self.target_ball.set_linear_velocity(np.array([-0.1, -0.3, 0.0]))
                if self.frame == 500:
                    print("[INFO] Paramos pelotita...")
                    self.target_ball.set_linear_velocity(np.array([0.0, 0.0, 0.0]))
                    self.target_ball.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

        # Cinemática Inversa DLS
        lamb_dls = 0.1
        J_T = J_hand.T
        J_pinv_robot = J_T @ np.linalg.inv(J_hand @ J_T + lamb_dls**2 * np.eye(6))
        dq_target = J_pinv_robot @ v_cartesian_target
        q_next = q + (dq_target * dt)
        self.franka.apply_action(ArticulationAction(joint_positions=np.append(q_next, [0.04, 0.04])))
        self.frame += 1

    def start(self):
        update_stream = omni.kit.app.get_app().get_update_event_stream()
        self.subscription = update_stream.create_subscription_to_pop(
            self.on_update, name="franka_joint_impedance_control"
        )
        print("\n" + "="*60)
        print("SIMULACIÓN INICIADA")
        print("Haz click en la ventana de Isaac Sim y pulsa 'Q' para guardar y salir.")
        print("="*60 + "\n")

def main():
    sim = FrankaImpedanceSimulation()
    sim.start()
    while simulation_app.is_running():
        simulation_app.update()

if __name__ == "__main__":
    main()