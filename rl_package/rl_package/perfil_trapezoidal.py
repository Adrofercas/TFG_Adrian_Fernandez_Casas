import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from rl_package.auxiliar import CustomSAC as SAC
from rl_package.auxiliar import new_goal, pose_diff
import time
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
import threading
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import torch
from rclpy.qos import QoSProfile, ReliabilityPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

class AdaptiveEntropyCallback(BaseCallback):
    def __init__(self, check_freq=10000, decay=0.9, min_ent=0.01, verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.decay = decay
        self.min_ent = min_ent

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            new_ent = max(self.min_ent, float(self.model.ent_coef) * self.decay)
            self.model.ent_coef = new_ent
            print(f"[Entropy decay] step={self.n_calls}, new ent_coef={self.model.ent_coef:.4f}")
        
        self.logger.record("train/entropy_coef", float(self.model.ent_coef))
        return True
    
class CurriculumEntropyCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env
        # Target entropy base segun la dimension de la accion
        self.base_target_entropy = -14.0
        self.min_ent_coef = 0.05
        self.max_ent_coef = 1.5
        
    def _on_step(self):
        # 1. Obtener nivel de dificultad actual del entorno
        current_difficulty = self.training_env.get_attr("get_curriculum_level")[0]()
            
        # 2. Calcular nuevo Target Entropy
        # Si dificultad = 0 (fácil) -> Queremos MÁS entropía (ej. base + 5.0)
        # Si dificultad = 1 (difícil) -> Queremos entropía base (ej. base + 0.0)
        # entropy_boost = 5.0 * (1.0 - current_difficulty)
        # new_target = self.base_target_entropy + entropy_boost
        # self.model.target_entropy = new_target

        # Acotamos ent_coef entre min y max
        # current_ent_coef = float(self.model.ent_coef)
        # # Limitar ent_coef entre min y max
        # bounded_ent_coef = np.clip(current_ent_coef, self.min_ent_coef, self.max_ent_coef)
        # # Si fue acotado, forzar el nuevo valor
        # if bounded_ent_coef != current_ent_coef:
        #     self.model.ent_coef = bounded_ent_coef
        if current_difficulty == 1.0:
            self.model.ent_coef_min_value = 0.2
        elif current_difficulty == 2.0:
            self.model.ent_coef_min_value = 0.1
        elif current_difficulty == 3.0:
            self.model.ent_coef_min_value = 0.05
        # self.logger.record("train/target_entropy", new_target)
        self.logger.record("train/curriculum_level", current_difficulty)
        
        return True

class RosEnv(gym.Env):
    def __init__(self, a, b, c, d, max_steps=500):
        super().__init__()

        # === Inicialización ROS ===
        rclpy.init()
        self.node = Node('td3_env')
        self.node.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.rate = self.node.create_rate(1000)  # 1000 Hz

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.max_steps = max_steps

        # === Constantes de base ===
        self.base_pose = np.array([0., -0.785, 0., -2.356, 0., 1.571, 0.785], dtype=np.float32)
        self.default_k = np.array([300., 300., 300., 300., 300., 300., 300.], dtype=np.float32)
        self.default_d = 2*np.sqrt(self.default_k)  # critical damping
        self.joint_limits_lower = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
        self.joint_limits_upper = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])
        self.torques_limit = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])  # Nm

        # === Publishers / Subscribers ===
        # qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.node.create_subscription(Float32MultiArray, '/joint_pos', self.cb_pos, 1)
        self.node.create_subscription(Float32MultiArray, '/joint_vel', self.cb_vel, 1)
        self.node.create_subscription(Float32MultiArray, '/joint_jerks', self.cb_jerks, 1)
        self.node.create_subscription(Float32MultiArray, '/torques', self.cb_torques, 1)
        self.pub_k = self.node.create_publisher(Float32MultiArray, '/k_gains', 10)
        self.pub_d = self.node.create_publisher(Float32MultiArray, '/d_gains', 10)
        self.pub_goal = self.node.create_publisher(Float32MultiArray, '/q_goals', 10)
        self.pub_reset = self.node.create_publisher(Float32MultiArray, '/reset_sim', 10)
        self.node.create_subscription(Float32MultiArray, '/sim_reset_done', self.cb_reset_done, 1)
        self.node.create_subscription(Float32MultiArray, '/sim_reset_request', self.cb_reset_request, 1)
        self.sim_reset_done = False
        self.sim_reset_request = False

        # === Estados ===
        self.q_goal = np.zeros(7, dtype=np.float32)
        self.joint_pos = np.zeros(7, dtype=np.float32)
        self.joint_vel = np.zeros(7, dtype=np.float32)
        self.joint_jerks = np.zeros(7, dtype=np.float32)
        self.torques = np.zeros(7, dtype=np.float32)
        self.k_action = self.default_k.copy()
        self.d_action = self.default_d.copy()
        self.last_action = self.k_action.copy()
        self.current_step = 0
        self.dist_inicial = 0

        # === Espacios ===
        obs_high = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508, 2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508, *([10] * 7), *([750] * 7)], dtype=np.float32)
        obs_low = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508, -2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508, *([-10] * 7), *([0] * 7)], dtype=np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Acción en [0, 1] (más estable para algoritmos)
        self.act_high = np.array([750., 750., 750., 750., 750., 750., 750.], dtype=np.float32)
        self.action_space = spaces.Box(low=-np.ones(7, dtype=np.float32),
                                       high=np.ones(7, dtype=np.float32),
                                       dtype=np.float32)
        
        # === TensorBoard logger ===
        self.logger = configure("/home/afernandez/colcon_ws/src/rl_package/logs/custom_metrics", ["tensorboard"])
        self.total_eps = 0

        # lanzamos el spin en un hilo aparte
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

        self.mean_jerk = 0.0
        self.mean_count_k = 0.0
        self.mean_action = 0.0
        self.mean_torque = 0.0
        self.mean_stiffness = np.zeros(7, dtype=np.float32)
        self.mean_damping = np.zeros(7, dtype=np.float32)
        self.curriculum_level = 0.0  # nivel de dificultad actual
        self.dist_goal = (0.15, 0.10)  # distancias para new_goal
        self.rot_goal = (45, 15)  # rotaciones para new_goal
        self.dist_threshold = 0.05  # umbral para considerar alcanzada la meta
        self.angle_threshold = 5.0  # umbral angular para considerar alcanzada la meta
        self.rew_eps = np.array([])


    # === Callbacks ===
    def cb_pos(self, msg):
        self.joint_pos = np.array(msg.data, dtype=np.float32)

    def cb_vel(self, msg):
        self.joint_vel = np.array(msg.data, dtype=np.float32)

    def cb_jerks(self, msg):
        self.joint_jerks = np.array(msg.data, dtype=np.float32)
    
    def cb_torques(self, msg):
        self.torques =  np.array(msg.data, dtype=np.float32)

    def cb_reset_done(self, msg):
        if len(msg.data) > 0 and msg.data[0] == 1.0:
            self.sim_reset_done = True
    
    def cb_reset_request(self, msg):
        if len(msg.data) > 0 and msg.data[0] == 1.0:
            self.sim_reset_request = True

    def get_curriculum_level(self):
        return self.curriculum_level

    def wait_for_sim(self, timeout=300):
        print("[ROS ENV] Esperando a que Isaac Sim esté operativo...")
        start = time.time()

        while time.time() - start < timeout:
            # Si el tópico cambia, asumimos que IsaacSim está corriendo
            if self.sim_reset_done:
                print("[ROS ENV] Isaac Sim operativo.")
                self.sim_reset_done = False
                time.sleep(3.0)  # espera extra para asegurar que todo esté listo
                return True
            time.sleep(0.5)
        print("[ROS ENV] Isaac Sim no respondió en el tiempo esperado.")
        return False

    # === Episodio ===
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.joint_vel[:] = 0.0
        self.joint_jerks[:] = 0.0
        self.prev_err = None  # inicializamos el prev_err para shaping
        self.total_reward = 0.0
        self.mean_jerk = 0.0
        self.mean_count_k = 0.0
        self.mean_action = 0.0
        self.mean_torque = 0.0
        self.mean_stiffness = np.zeros(7, dtype=np.float32)
        self.mean_damping = np.zeros(7, dtype=np.float32)
        #genero nueva meta
        self.q_goal = new_goal(self.joint_pos, self.dist_goal, self.rot_goal, max_attempts=100000)
        
        print(f"Objetivo {self.q_goal}")

        # Publica metas y gains iniciales (usar default)
        self.pub_goal.publish(Float32MultiArray(data=self.q_goal.tolist()))
        self.pub_k.publish(Float32MultiArray(data=self.default_k.tolist()))
        self.pub_d.publish(Float32MultiArray(data=self.default_d.tolist()))

        # time.sleep(3.0)  # Esperar a que el robot lea la nueva meta

        dist2goal, rot2goal = pose_diff(self.joint_pos, self.q_goal)
        # guardamos la distancia total a la nueva pose
        self.dist_inicial = dist2goal

        obs = np.concatenate([
            self.joint_pos,
            self.q_goal,
            self.joint_vel,
            self.last_action
        ])
        return obs, {}

    def step(self, action):
        # Manejo de reinicio de simulación externo
        if self.sim_reset_request:
            self.sim_reset_request = False
            success = self.wait_for_sim()
            if success:
                print("[ROS ENV] Simulación reiniciada correctamente.")
                truncated = True
                terminated = False
                reward = self.total_reward
                reward -= 500.0  # penalización por reinicio
                # Espacio de observación
                obs = np.concatenate([
                    self.joint_pos,
                    self.q_goal,
                    self.joint_vel,
                    self.last_action
                ])
                print(f"Terminamos por colision: Step {self.current_step} -> Reward: {reward}")
                return obs, reward, terminated, truncated, {}
            else:
                print("[ROS ENV] Error al reiniciar la simulación.")
                exit(1)

        media_rew = 0.0
        if len(self.rew_eps) >= 250: #cuando llene el buffer de episodios, miro si ha conseguido aprender lo suficiente
            self.rew_eps = np.delete(self.rew_eps, 0) #mantengo tamanho 250
            media_rew = np.mean(self.rew_eps)
        
        # actualizo curriculum si ha superado umbrales
        if self.curriculum_level == 0.0 and media_rew >=0.5:
            self.rew_eps = np.array([])  # reseteo recompensas
            self.curriculum_level = 1.0
            self.dist_goal = (0.2, 0.15)
            self.rot_goal = (60, 20)
            media_rew = 0.0
        if self.curriculum_level == 1.0 and media_rew >=0.75:
            self.rew_eps = np.array([])  # reseteo recompensas
            self.curriculum_level = 2.0
            self.dist_goal = (0.25, 0.2)
            self.rot_goal = (75, 25)
            #hacemos mas exigente el umbral
            self.dist_threshold = 0.025
            self.angle_threshold = 2.5
            media_rew = 0.0
        if self.curriculum_level == 2.0 and media_rew >=0.75:
            self.rew_eps = np.array([])  # reseteo recompensas
            self.curriculum_level = 3.0
            self.dist_goal = (0.3, 0.2)
            self.rot_goal = (90, 30)
            #hacemos mas exigente el umbral
            self.dist_threshold = 0.01
            self.angle_threshold = 1
            media_rew = 0.0

        if self.current_step % 5 == 0:
            # === Mapear acción [-1,1] a los gains reales ===
            self.k_action = (action + 1) * (self.act_high / 2)

        # Publica gains
        self.pub_k.publish(Float32MultiArray(data=self.k_action.tolist()))
        self.d_action = 2 * np.sqrt(self.k_action)  # critical damping
        self.pub_d.publish(Float32MultiArray(data=self.d_action.tolist()))

        # Espacio de observación
        obs = np.concatenate([
            self.joint_pos,
            self.q_goal,
            self.joint_vel,
            self.last_action
        ])

        # mientras no acabe episodio reward 0
        reward = 0.0
        # Reward por step
        self.total_reward -= 0.1

        if np.mean(np.abs(self.joint_jerks)) > 10:
            self.total_reward -= 2.0  # penalización por jerk alto

        if np.any(np.abs(self.torques) > self.torques_limit):
            self.total_reward -= 2.0  # penalización por torque alto
        
        if np.mean(np.abs(self.joint_vel)) < 0.2:
            self.total_reward -= 0.5  # penalización por velocidad baja

        # Para incenctivar al agente a que aprenda a hacer el perfil trapezoidal, hacemos que si las ganancias no estan dentro de un cierto intervalo, se penaliza
        dist2goal, rot2goal = pose_diff(self.joint_pos, self.q_goal)

        #parte ascendente
        if dist2goal <= self.dist_inicial*0.3:
            #miramos si esta dentro de intervalo
            count_k = np.sum(self.k_action > self.act_high*0.2)
            self.total_reward -= 0.4 * count_k
        
        #parte estacionaria
        elif self.dist_inicial*0.3 < dist2goal < self.dist_inicial*0.6:
            #miramos si esta dentro de intervalo
            count_k = np.sum(self.k_action < self.act_high*0.2)
            self.total_reward -= 0.4 * count_k
        
        #parte descendente
        else:
            #miramos si esta dentro de intervalo
            count_k = np.sum(self.k_action > self.act_high*0.2)
            self.total_reward -= 0.4 * count_k

        count_delta_action = np.sum(np.abs(self.k_action - self.last_action) > 10)
        # actualizamos last_action
        self.last_action = self.k_action.copy()
        self.total_reward -= 0.1 * count_delta_action  # penalización por cambios bruscos en gains

        # Si el sistema está estable, termina el episodio con bonus
        terminated = False
        # Miramos si ha alcanzado la meta
        if dist2goal < self.dist_threshold and rot2goal < self.angle_threshold:
            self.total_reward += 1_000.0
            print(f"Meta alcanzada: Step {self.current_step} -> Reward: {self.total_reward}, Dist2Goal: {dist2goal:.3f}, Rot2Goal: {rot2goal:.3f}, Jerk: {np.mean(np.abs(self.joint_jerks)):.3f}")
            print("k_gains:", np.round(self.k_action,1), "\nd_gains:", np.round(self.d_action,1))
            terminated = True

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        #calculamos metricas para tensorboard
        jerk_scaled = np.log1p(np.mean(np.abs(self.joint_jerks)))  # log(1 + |jerk|)
        jerk_term = (jerk_scaled / np.log1p(5000.0)) ** 2   # normaliza a [0,1]
        self.mean_jerk += jerk_term
        self.mean_count_k += count_k
        self.mean_action += count_delta_action
        self.mean_torque += np.abs(self.torques)
        self.mean_stiffness += self.k_action
        self.mean_damping += self.d_action

        # Dump del logger cada episodio
        if truncated or terminated:
            reward = self.total_reward #paso el reward episodico
            # Logueo de métricas personalizadas
            self.logger.record(f"parameters/count_K", float(self.mean_count_k/self.current_step))
            self.logger.record(f"parameters/delta_action", float(self.mean_action*5/self.current_step))
            self.logger.record(f"parameters/jerk_term", float(self.mean_jerk/self.current_step))
            for i in range(7):
                self.logger.record(f"torques/joint{i+1}", float(self.mean_torque[i]/self.current_step))
                self.logger.record(f"stiffness/joint{i+1}", float(self.mean_stiffness[i]/self.current_step))
                self.logger.record(f"damping/joint{i+1}", float(self.mean_damping[i]/self.current_step))
            self.logger.record(f"parameters/reward", float(reward))
            self.logger.dump(self.total_eps)

            if (self.mean_jerk/self.current_step < 0.1) and terminated:
                self.rew_eps = np.append(self.rew_eps, 1) # añado recompensa episodio si alcanza meta y con poco jerk
            else:
                self.rew_eps = np.append(self.rew_eps, 0) # añado 0 si alcanza meta pero con jerk o si no la alcanza
            
            #reinicio medias
            self.total_eps += 1
            self.mean_jerk = 0.0
            self.mean_count_k = 0.0
            self.mean_action = 0.0
            self.mean_torque = 0.0
            self.mean_stiffness = np.zeros(7, dtype=np.float32)
            self.mean_damping = np.zeros(7, dtype=np.float32)


        # Muestro el reward cada 10 pasos
        if self.current_step % 10 == 0:
            print(f"Step {self.current_step} -> Reward: {self.total_reward:.1f}, Dist2Goal: {dist2goal:.3f}, Rot2Goal: {rot2goal:.3f}, Jerk: {np.mean(np.abs(self.joint_jerks)):.3f}")

        self.rate.sleep()  # asegurar dt constante
        return obs, reward, terminated, truncated, {}

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1)



# === Entrenamiento TD3 / SAC ===
def main():
    # guardar cada 5.000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="/home/afernandez/colcon_ws/src/rl_package/saves/simplificado",
        name_prefix="simplificado"
    )

    # entropyCallback = AdaptiveEntropyCallback(check_freq=8000, decay=0.97, min_ent=0.08)

    env = RosEnv(a=18, b=25, c=2, d=2, max_steps=3_000)
    env = Monitor(env)
    # DummyVecEnv espera una lista de funciones que crean entornos
    env = DummyVecEnv([lambda: env])
    # Aplicar VecNormalize para que normalice observaciones y recompensas
    env = VecNormalize(env, 
                       norm_obs=True, 
                       norm_reward=True, 
                       clip_obs=10., 
                       clip_reward=10., # Recortará picos extremos
                       gamma=0.98)      # Debe coincidir con el gamma de SAC

    run_name = "medidas_1"

    # Creamos callback de checkpoint y de entropía adaptativa
    entropyCallback = CurriculumEntropyCallback(env)
    callback = CallbackList([checkpoint_callback, entropyCallback])
    # unwrapped_env = env.envs[0].env  # Accedemos a entorno para pasar act_high
    entropy_parameters = [0.00003, 0.3, 0.6, 0.05] # max_change, min_value, max_value, loss_dampening

    nuevo_entrenamiento = False

    if nuevo_entrenamiento:
        # Iniciamos nuevo entrenamiento
        model = SAC("MlpPolicy", env, verbose=1,
                    learning_rate=3e-4, buffer_size=100_000, learning_starts=10_000, batch_size=256, tau=0.01, gamma=0.98, ent_coef='auto', train_freq=(5, 'step'),
                    policy_kwargs=dict(net_arch=[264, 264],activation_fn=torch.nn.ReLU), ent_param=entropy_parameters, target_entropy=-9.0,
                    tensorboard_log="/home/afernandez/colcon_ws/src/rl_package/logs/{0}".format(run_name))
        model.learn(total_timesteps=2_500_000, log_interval=1, callback=callback)

    else:
        # Seguimos con el entrenamiento anterior
        model = SAC.load(
            "/home/afernandez/colcon_ws/src/rl_package/saves/simplificado/simplificado_4000000_steps.zip",
            env=env,
            print_system_info=True,
            tensorboard_log="/home/afernandez/colcon_ws/src/rl_package/logs/{0}".format(run_name),
            ent_param=entropy_parameters
        )
        # #modificamos lr y ent_coef
        # model.replay_buffer.reset()   # reseteamos buffer para aplicar posibles cambios
        # model.learning_rate = 1e-4
        # model.ent_coef = 0.15

        model.learn(
            total_timesteps=10_000_000,
            reset_num_timesteps=False,    # continúa desde donde estaba
            log_interval=1,
            callback=callback
        )

    model.save("/home/afernandez/colcon_ws/src/rl_package/saves/model_final")
    env.close()


if __name__ == "__main__":
    main()
