import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float64MultiArray
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
import time
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.callbacks import CheckpointCallback
import threading


class RosEnv(gym.Env):
    def __init__(self, a, b, c, d, max_steps=500):
        super().__init__()

        # === Inicialización ROS ===
        rclpy.init()
        self.node = Node('td3_env')

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.max_steps = max_steps
        self.dt = 0.1  # 10 Hz

        # === Constantes de base ===
        self.base_pose = np.array([0., -0.785, 0., -2.356, 0., 1.571, 0.785], dtype=np.float32)
        self.default_k = np.array([600., 600., 600., 600., 250., 1500., 200.], dtype=np.float32)
        self.default_d = np.array([77.5, 77.5, 77.5, 77.5, 50., 5., 44.7], dtype=np.float32)
        self.joint_limits_lower = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
        self.joint_limits_upper = np.array([2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508])

        # === Publishers / Subscribers ===
        self.node.create_subscription(Float32MultiArray, '/joint_error', self.cb_error, 10)
        self.node.create_subscription(Float32MultiArray, '/joint_vel', self.cb_vel, 10)
        self.node.create_subscription(Float32MultiArray, '/joint_jerks', self.cb_jerks, 10)
        self.pub_k = self.node.create_publisher(Float32MultiArray, '/k_gains', 10)
        self.pub_d = self.node.create_publisher(Float32MultiArray, '/d_gains', 10)
        self.pub_goal = self.node.create_publisher(Float32MultiArray, '/q_goals', 10)
        self.pub_reset = self.node.create_publisher(Float32MultiArray, '/reset_sim', 10)
        self.node.create_subscription(Float32MultiArray, '/sim_reset_done', self.cb_reset_done, 10)
        self.sim_needs_reset = False

        # === Estados ===
        self.joint_error = np.zeros(7, dtype=np.float32)
        self.joint_vel = np.zeros(7, dtype=np.float32)
        self.joint_jerks = np.zeros(7, dtype=np.float32)
        self.prev_action = np.zeros(14, dtype=np.float32)
        self.current_joint = 5  # solo mueve la articulación 6 (índice 5)
        self.current_step = 0

        # === Espacios ===
        obs_high = np.ones(14, dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

        self.act_high = np.array([1300., 1300., 1300., 1300., 1300., 2000., 1300.,
                             1000., 1000., 1000., 1000., 1000., 1000., 1000.], dtype=np.float32)
        self.action_space = spaces.Box(low=np.zeros(14, dtype=np.float32),
                                       high=self.act_high, dtype=np.float32)
        
        # === TensorBoard logger ===
        self.logger = configure("/home/afernandez/colcon_ws/src/rl_package/logs/custom_metrics", ["tensorboard"])
        self.last_steps = np.zeros(7, dtype=np.int32)

        #lanzamos el spin en un hilo aparte
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()


    # === Callbacks ===
    def cb_error(self, msg):
        self.joint_error = np.array(msg.data, dtype=np.float32)
        # print(f"Joint errors: {self.joint_error}")

    def cb_vel(self, msg):
        self.joint_vel = np.array(msg.data, dtype=np.float32)

    def cb_jerks(self, msg):
        self.joint_jerks = np.array(msg.data, dtype=np.float32)

    def cb_reset_done(self, msg):
        if len(msg.data) > 0 and msg.data[0] == 1.0:
            self.sim_needs_reset = True

    # === Episodio ===
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.joint_error[:] = 0.0
        self.joint_vel[:] = 0.0
        self.joint_jerks[:] = 0.0
        self.prev_action[:] = 0.0

        # Calcula nueva meta (solo esa articulación se mueve)
        q_goal = self.base_pose.copy()
        delta = np.random.uniform(0.3, 0.9) * np.random.choice([-1, 1])
        q_goal[self.current_joint] += delta
        # si no esta dentro de los limites, recalcula hasta que esté
        while q_goal[self.current_joint] < self.joint_limits_lower[self.current_joint] or q_goal[self.current_joint] > self.joint_limits_upper[self.current_joint]:
            delta = np.random.uniform(0.3, 1.2) * np.random.choice([-1, 1])
            q_goal[self.current_joint] = self.base_pose[self.current_joint] + delta

        print(f"Nuevo episodio -> joint {self.current_joint+1}, objetivo {q_goal[self.current_joint]:.3f}")

        # Publica metas y gains iniciales
        self.pub_goal.publish(Float32MultiArray(data=q_goal.tolist()))
        self.pub_k.publish(Float32MultiArray(data=self.default_k.tolist()))
        self.pub_d.publish(Float32MultiArray(data=self.default_d.tolist()))


        time.sleep(1.0)  # Esperar a que el robot lea la nueva meta
        obs = np.concatenate([
            np.clip(self.joint_error / 7.0, -1, 1),
            np.clip(self.joint_jerks / 10.0, -1, 1)
        ])
        return obs, {}

    def step(self, action):
        start = time.time()

        if self.sim_needs_reset:
            self.sim_needs_reset = False
            time.sleep(3.0)
            return np.zeros(14, dtype=np.float32), -15.0, True, False, {}

        # Usa valores por defecto y cambia solo la articulación actual
        k_action = self.default_k.copy()
        d_action = self.default_d.copy()
        k_action[self.current_joint] = action[self.current_joint]
        d_action[self.current_joint] = action[self.current_joint + 7]

        self.pub_k.publish(Float32MultiArray(data=k_action.tolist()))
        self.pub_d.publish(Float32MultiArray(data=d_action.tolist()))
        # print(f"Published K: {k_action[self.current_joint]:.1f}")


        # Normaliza observaciones
        obs = np.concatenate([
            np.clip(self.joint_error / 7.0, -1, 1),
            np.clip(self.joint_jerks / 10.0, -1, 1)
        ])

        # Reward
        err_norm = np.abs(self.joint_error[self.current_joint])
        jerk_abs = np.abs(self.joint_jerks[self.current_joint])

        v_max = 3.0  # rad/s
        d_scale = 2.0  # rad

        v_des_mag = v_max * min(np.abs(self.joint_error[self.current_joint]) / d_scale, 1.0)
        v_des = np.sign(self.joint_error[self.current_joint]) * v_des_mag
        vel_dist = (self.joint_vel[self.current_joint] - v_des)**2

        # Diferencia de acción respecto al paso anterior
        norm_action = action / (self.act_high + 1e-8)
        norm_prev = self.prev_action / (self.act_high + 1e-8)
        delta_action = np.sum((norm_action - norm_prev) ** 2)  # ahora está en [0, ~4] típicamente
        if self.current_step <= 5:
            delta_action = 0.1  # casi no penalizo en los primeros pasos

        jerk_scaled = np.log1p(np.abs(self.joint_jerks[self.current_joint]))  # log(1 + |jerk|)
        jerk_term = (jerk_scaled / np.log1p(12000.0)) ** 2   # normaliza a [0,1]

        reward = - (self.a * err_norm + self.b * jerk_term + self.c * vel_dist + self.d * delta_action) #vel_dist se resta porque si es alta va en la direccion correcta

        # Si el sistema está estable, termina el episodio con bonus
        if err_norm < 0.3 and jerk_abs < 5.0:
            reward += 20.0
            print(f"Step {self.current_step} -> Reward: {reward*0.1:.3f}, Err: {err_norm:.3f}, Jerk term: {jerk_term:.3f}, Vel_dist: {vel_dist:.3f}, Delta: {delta_action:.3f}")
            terminated = True

        else:
            terminated = False

        reward*=0.1 # escala el reward para que no sea tan grande

        self.prev_action = action
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        # Logueo de métricas personalizadas
        self.logger.record(f"joint_{self.current_joint+1}/err_norm", float(err_norm))
        self.logger.record(f"joint_{self.current_joint+1}/jerk_term", float(jerk_term))
        self.logger.record(f"joint_{self.current_joint+1}/reward", float(reward))

        # Dump del logger cada episodio
        if truncated or terminated:
            self.logger.dump(self.last_steps[self.current_joint])
            #guardo cuantas veces se ha aprendido cada articulación
            self.last_steps[self.current_joint] += 1

        # Muestro el reward cada 100 pasos
        if self.current_step % 100 == 0:
            print(f"Step {self.current_step} -> Reward: {reward:.3f}, Err: {err_norm:.3f}, Jerk term: {jerk_term:.3f}, Vel/Dist: {vel_dist:.3f}, Delta: {delta_action:.3f}")


        elapsed = time.time() - start
        time.sleep(max(0, self.dt - elapsed))

        return obs, reward, terminated, truncated, {}

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1)



# === Entrenamiento TD3 ===
def main():
    traj_file="/home/afernandez/colcon_ws/src/rl_package/rl_package/trayectorias.txt"
    # guardar cada 50.000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,                      # cada 50k pasos
        save_path="/home/afernandez/colcon_ws/src/rl_package/saves/joint6-only",                 # carpeta donde guardar
        name_prefix="joint6-only"            # prefijo de los archivos
    )
    env = RosEnv(a=13, b=25, c=2, d=2, max_steps=1500)
    env = Monitor(env)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))

    run_name = "medidas_1"

    model = SAC("MlpPolicy", env, action_noise=action_noise, verbose=1,
                learning_rate=1e-3, buffer_size=1_000_000, batch_size=256, tau=0.005, ent_coef=0.2,
                tensorboard_log="/home/afernandez/colcon_ws/src/rl_package/logs/{0}".format(run_name))
    model.learn(total_timesteps=1_000_000, log_interval=1, callback=checkpoint_callback)
    model.save("/home/afernandez/colcon_ws/src/rl_package/saves/model_td3_improved")
    env.close()


if __name__ == "__main__":
    main()
