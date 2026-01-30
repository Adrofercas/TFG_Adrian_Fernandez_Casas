import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import matplotlib
import time
import threading
import numpy as np
import os
from rl_package.auxiliar import pose_diff

# Usamos 'Agg' para guardar graficas sin necesitar interfaz grafica (evita errores en threads)
matplotlib.use('Agg')

class SimController(Node):

    def __init__(self):
        super().__init__('sim_controller_plotter')

        # --- CONFIGURACIÓN DE DATOS ---
        self.k_gains_list = [
            [1, 1, 1, 1, 1, 1, 1],
            [12_000, 12_000, 12_000, 12_000, 12_000, 12_000, 12_000], # Set 1
            [8_000, 8_000, 8_000, 8_000, 8_000, 8_000, 8_000],  # Set 2
            [5_000, 5_000, 5_000, 5_000, 5_000, 5_000, 5_000],   # Set 3
            [2_000, 2_000, 2_000, 2_000, 2_000, 2_000, 2_000],   # Set 4
            [600, 600, 600, 600, 600, 600, 600],     # Set 5
            [300, 300, 300, 300, 300, 300, 300],
        ]
        
        self.d_gains_list = [
            [1, 1, 1, 1, 1, 1, 1],
            [1_000, 1_000, 1_000, 1_000, 1_000, 1_000, 1_000], # Set 1
            [800, 800, 800, 800, 800, 800, 800], # Set 2
            [500, 500, 500, 500, 500, 500, 500], # Set 3
            [200, 200, 200, 200, 200, 200, 200], # Set 4
            [60, 60, 60, 60, 60, 60, 60],       # Set 5
            [30, 30, 30, 30, 30, 30, 30],
        ]

        self.goal_pose_1 = [1.0, -0.785, 0.2, -1.8, 0.0, 1.571, 0.785]
        self.goal_pose_2 = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.goal_sel = 0

        # --- PUBLICADORES ---
        self.pub_k = self.create_publisher(Float32MultiArray, '/k_gains', 10)
        self.pub_d = self.create_publisher(Float32MultiArray, '/d_gains', 10)
        self.pub_goal = self.create_publisher(Float32MultiArray, '/q_goals', 10)

        # --- SUSCRIPTORES ---
        self.create_subscription(Float32MultiArray, '/joint_pos', self.cb_pos, 10)
        self.create_subscription(Float32MultiArray, '/joint_vel', self.cb_vel, 10)
        self.create_subscription(Float32MultiArray, '/joint_jerks', self.cb_jerks, 10)
        self.create_subscription(Float32MultiArray, '/torques', self.cb_torques, 10)

        # --- ALMACENAMIENTO DE DATOS ---
        # MODIFICADO: Separamos 'pos' en 'dist' y 'rot'
        self.data_buffer = {
            'dist': [], 'rot': [], 'vel': [], 'jerks': [], 'torques': [], 'time': []
        }
        self.start_time = time.time()
        self.data_lock = threading.Lock() 

        # Crear carpeta base si no existe (la específica se crea al guardar)
        base_path = '/home/afernandez/colcon_ws/src/rl_package/graficas'
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)

        # --- HILO DE SECUENCIA ---
        self.control_thread = threading.Thread(target=self.run_sequence)
        self.control_thread.start()

    # --- CALLBACKS ---
    def record_data(self, key, msg):
        with self.data_lock:
            current_t = time.time() - self.start_time
            
            if key == 'pos':
                # Calculo de distancia y rotacion
                if self.goal_sel == 0:
                    dist, rot = pose_diff(msg.data, self.goal_pose_1)
                else:
                    dist, rot = pose_diff(msg.data, self.goal_pose_2)
                
                # MODIFICADO: Guardamos en listas separadas y como arrays de un elemento [x]
                # Esto es importante para que el shape sea (N, 1) y el bucle de graficado funcione igual
                self.data_buffer['dist'].append([dist]) 
                self.data_buffer['rot'].append([rot])
                
                # Usamos la llegada de posición como referencia de tiempo
                self.data_buffer['time'].append(current_t)
            else:
                self.data_buffer[key].append(msg.data)

    def cb_pos(self, msg): self.record_data('pos', msg)
    def cb_vel(self, msg): self.record_data('vel', msg)
    def cb_jerks(self, msg): self.record_data('jerks', msg)
    def cb_torques(self, msg): self.record_data('torques', msg)

    # --- LÓGICA DE ENVÍO ---
    def send_array(self, publisher, data_list):
        msg = Float32MultiArray()
        msg.data = [float(x) for x in data_list]
        publisher.publish(msg)

    # --- SECUENCIA PRINCIPAL ---
    def run_sequence(self):
        self.get_logger().info('Iniciando secuencia de control...')
        time.sleep(2) 

        idx = 0
        while rclpy.ok():
            if idx >= len(self.k_gains_list):
                print("\nPrograma finalizado todas las configuraciones.")
                break
            
            k_gain = self.k_gains_list[idx % len(self.k_gains_list)]
            d_gain = self.d_gains_list[idx % len(self.d_gains_list)]
            
            self.get_logger().info(f'--- NUEVA TRAYECTORIA: Configuración {idx} ---')
            
            # 1. Mandar Ganancias
            self.get_logger().info(f'Enviando Ganancias K: {k_gain}')
            self.send_array(self.pub_k, k_gain)
            self.send_array(self.pub_d, d_gain)
            
            # Reiniciar buffer
            with self.data_lock:
                for key in self.data_buffer:
                    self.data_buffer[key] = []
                self.start_time = time.time()

            # 2. Goal 1
            self.get_logger().info('Enviando Goal 1')
            self.goal_sel = 0
            k_gain = np.array([1,1,1,1,1,1,1])
            d_gain = np.array([1,1,1,1,1,1,1])
            self.send_array(self.pub_goal, self.goal_pose_1)
            
            for i in range(80):
                time.sleep(0.25)
                k_gain = k_gain + np.array([1,1,1,1,1,1,1])
                d_gain = d_gain + np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
                self.get_logger().info(f'Enviando Ganancias K: {k_gain}')
                self.send_array(self.pub_k, k_gain)
                self.send_array(self.pub_d, d_gain)
            
            for i in range(80):
                time.sleep(0.25)
                k_gain = k_gain - np.array([1,1,1,1,1,1,1])
                d_gain = d_gain - np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
                self.get_logger().info(f'Enviando Ganancias K: {k_gain}')
                self.send_array(self.pub_k, k_gain)
                self.send_array(self.pub_d, d_gain)
            
            time.sleep(10)

            # 3. Goal 2
            self.get_logger().info('Enviando Goal 2')
            self.send_array(self.pub_goal, self.goal_pose_2)
            self.goal_sel = 1

            k_gain = np.array([1,1,1,1,1,1,1])
            d_gain = np.array([1,1,1,1,1,1,1])
            for i in range(80):
                time.sleep(0.25)
                k_gain = k_gain + np.array([1,1,1,1,1,1,1])
                d_gain = d_gain + np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
                self.get_logger().info(f'Enviando Ganancias K: {k_gain}')
                self.send_array(self.pub_k, k_gain)
                self.send_array(self.pub_d, d_gain)
            
            for i in range(80):
                time.sleep(0.25)
                k_gain = k_gain - np.array([1,1,1,1,1,1,1])
                d_gain = d_gain - np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5])
                self.get_logger().info(f'Enviando Ganancias K: {k_gain}')
                self.send_array(self.pub_k, k_gain)
                self.send_array(self.pub_d, d_gain)

            """
            # 2. Goal 1
            self.get_logger().info('Enviando Goal 1')
            self.send_array(self.pub_goal, self.goal_pose_1)
            self.goal_sel = 0

            time.sleep(40.0)

            # 3. Goal 2
            self.get_logger().info('Enviando Goal 2')
            self.send_array(self.pub_goal, self.goal_pose_2)
            self.goal_sel = 1

            time.sleep(40.0)
            """
            # 4. Generar Gráficas
            self.get_logger().info('Fin de trayectoria. Generando gráficas...')
            self.save_plots(idx)
            
            idx += 1

    # --- GRAFICADO ---
    def save_plots(self, batch_id):
        with self.data_lock:
            data_snapshot = {
                'dist': list(self.data_buffer['dist']),
                'rot': list(self.data_buffer['rot']),
                'vel': list(self.data_buffer['vel']),
                'jerks': list(self.data_buffer['jerks']),
                'torques': list(self.data_buffer['torques']),
                'time': list(self.data_buffer['time'])
            }

        n_samples = len(data_snapshot['time'])
        if n_samples == 0:
            self.get_logger().warn("No se recibieron datos para graficar.")
            return

        # MODIFICADO: Mapeo actualizado para separar Distancia y Rotacion
        topics_map = {
            'Distancia': data_snapshot['dist'],
            'Rotacion': data_snapshot['rot'],
            'Velocidad': data_snapshot['vel'],
            'Jerks': data_snapshot['jerks'],
            'Torques': data_snapshot['torques']
        }

        # Asegurar que el directorio específico existe
        save_dir = f'/home/afernandez/colcon_ws/src/rl_package/graficas/save5/gains_{batch_id}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for name, dataset in topics_map.items():
            if not dataset: continue
            
            limit = min(len(data_snapshot['time']), len(dataset))
            t_axis = np.array(data_snapshot['time'][:limit])
            data_matrix = np.array(dataset[:limit]) # Shape (N, 7) o (N, 1)

            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Graficar: Si es Dist/Rot solo hará un ciclo (Joint 1), si es Vel hará 7
            for j in range(7):
                if data_matrix.shape[1] > j:
                    label_txt = f'Joint {j+1}' if data_matrix.shape[1] > 1 else 'Error Global'
                    ax.plot(t_axis, data_matrix[:, j], label=label_txt)

            ax.set_title(f'Configuracion {batch_id} - {name}')
            ax.set_xlabel('Tiempo (s)')
            ax.set_ylabel(name)
            ax.legend(loc='upper right', fontsize='small', ncol=2)
            ax.grid(True)

            filename = f'{save_dir}/{name.lower()}.png'
            plt.savefig(filename)
            plt.close(fig)
            
        self.get_logger().info(f'Gráficas guardadas en: {save_dir}')

def main(args=None):
    rclpy.init(args=args)
    node = SimController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()