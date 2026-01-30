import torch as th
import torch.nn.functional as F
from stable_baselines3.common.utils import polyak_update
from stable_baselines3 import SAC
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque

class CustomSAC(SAC):
    """
    SAC con control más fino sobre cómo aprende ent_coef.
    Permite limitar la velocidad de cambio y aplicar regularización.
    """
    
    def __init__(self, *args, ent_param=[0.01, 0.1, 0.5, 0.3], **kwargs):
        super().__init__(*args, **kwargs)
        
        # Guardar parámetros de control
        self.ent_coef_max_change = ent_param[0]
        self.ent_coef_min_value = ent_param[1]
        self.ent_coef_max_value = ent_param[2]
        self.ent_loss_dampening = ent_param[3]
        self.prev_log_ent_coef = None
        
        print(f"[CustomSAC] Entropía configurada:")
        print(f"  - Max change per step: {self.ent_coef_max_change}")
        print(f"  - Rango: [{self.ent_coef_min_value}, {self.ent_coef_max_value}]")
        print(f"  - Loss dampening: {self.ent_loss_dampening}")

    @classmethod
    def load(cls, path, env=None, device="auto", print_system_info=False, ent_param=None, **kwargs):
        """
        Load compatible con SB3. Construye manualmente el modelo para evitar errores de atributos faltantes.
        """
        from stable_baselines3.sac.sac import SAC as SAC_Original

        custom_objects = {"ent_coef": "auto"}

        # 1. Cargar metadatos usando la clase original
        base_model = SAC_Original.load(
            path,
            env=None, 
            device=device,
            print_system_info=print_system_info,
            custom_objects=custom_objects,
            **kwargs
        )

        # 2. Determinar configuración de entropía
        if ent_param is None:
            ent_param = getattr(base_model, "ent_param", [0.01, 0.1, 0.5, 0.3])
        
        print(f"[CustomSAC] Cargando con ent_param: {ent_param}")

        tb_log = kwargs.get("tensorboard_log", None)

        # 3. Crear instancia vacía de CustomSAC
        model = cls(
            policy=base_model.policy_class,
            env=None,
            learning_rate=base_model.learning_rate,
            buffer_size=base_model.buffer_size,
            learning_starts=base_model.learning_starts,
            batch_size=base_model.batch_size,
            tau=base_model.tau,
            gamma=base_model.gamma,
            train_freq=base_model.train_freq,
            gradient_steps=base_model.gradient_steps,
            action_noise=base_model.action_noise,
            replay_buffer_class=base_model.replay_buffer_class,
            replay_buffer_kwargs=base_model.replay_buffer_kwargs,
            optimize_memory_usage=base_model.optimize_memory_usage,
            ent_coef=base_model.ent_coef,
            target_update_interval=base_model.target_update_interval,
            target_entropy=base_model.target_entropy,
            use_sde=base_model.use_sde,
            sde_sample_freq=base_model.sde_sample_freq,
            use_sde_at_warmup=base_model.use_sde_at_warmup,
            policy_kwargs=base_model.policy_kwargs,
            verbose=base_model.verbose,
            device=device,
            tensorboard_log=tb_log,
            ent_param=ent_param,
            _init_setup_model=False 
        )

        # 4. Configuración Manual del Modelo
        # Copiamos espacios necesarios para setup
        model.observation_space = base_model.observation_space
        model.action_space = base_model.action_space
        
        # --- FIX: Asignamos n_envs explícitamente ---
        model.n_envs = base_model.n_envs
        # --------------------------------------------

        # Ahora construimos las redes (policy, actor, critic)
        model._setup_model() 

        # 5. Cargar los pesos
        model.set_parameters(base_model.get_parameters(), exact_match=False)

        # 6. Configurar el Entorno real (si se pasa)
        if env is not None:
            model.set_env(env)

        # 7. Restaurar estado del entrenamiento (timesteps y buffers)
        model.num_timesteps = base_model.num_timesteps
        model._n_updates = base_model._n_updates
        
        if hasattr(base_model, "ep_info_buffer") and base_model.ep_info_buffer is not None:
            model.ep_info_buffer = base_model.ep_info_buffer
        else:
            model.ep_info_buffer = deque(maxlen=100)
            
        if hasattr(base_model, "ep_success_buffer") and base_model.ep_success_buffer is not None:
            model.ep_success_buffer = base_model.ep_success_buffer
        else:
             model.ep_success_buffer = deque(maxlen=100)

        return model


    def train(self, gradient_steps: int, batch_size: int = 200) -> None:
        """
        Igual que SAC.train() pero con update de entropía customizado
        """
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample a batch from the replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Compute loss
                # Important: detach log_prob so we don't change the actor by this loss
                # ← AQUÍ EMPIEZA LA MODIFICACIÓN
                
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                
                # PASO 1: Aplicar dampening al loss (reduce magnitud)
                ent_coef_loss = ent_coef_loss * self.ent_loss_dampening
                
                ent_coef_losses.append(ent_coef_loss.item())

                # Backward
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                
                # PASO 2: Clipear gradientes antes de aplicarlos
                th.nn.utils.clip_grad_norm_(
                    self.log_ent_coef.parameters(), 
                    max_norm=0.1  # Limitar magnitud del gradiente
                ) if hasattr(self.log_ent_coef, 'parameters') else None
                
                self.ent_coef_optimizer.step()

                # PASO 3: Guardar log_ent_coef anterior y aplicar clipping de cambio
                if self.prev_log_ent_coef is None:
                    self.prev_log_ent_coef = self.log_ent_coef.clone().detach()
                
                # Calcular cambio en log_ent_coef
                log_ent_change = self.log_ent_coef - self.prev_log_ent_coef
                
                # Clipear el cambio
                max_change_log = th.log(
                    th.tensor(1.0 + self.ent_coef_max_change, device=self.log_ent_coef.device)
                )
                log_ent_change = th.clamp(log_ent_change, -max_change_log, max_change_log)
                
                # Aplicar cambio limitado
                self.log_ent_coef.data = self.prev_log_ent_coef + log_ent_change
                self.prev_log_ent_coef = self.log_ent_coef.clone().detach()
                
                # PASO 4: Asegurar que ent_coef está dentro de rango
                self.ent_coef = self.log_ent_coef.exp().detach()
                ent_coef_clamped = th.clamp(
                    self.ent_coef,
                    min=self.ent_coef_min_value,
                    max=self.ent_coef_max_value
                )
                
                if (ent_coef_clamped != self.ent_coef).any():
                    # Si fue clipeado, actualizar log_ent_coef
                    self.log_ent_coef.data = th.log(ent_coef_clamped)
                    self.ent_coef = ent_coef_clamped
                
                # ← FIN DE LA MODIFICACIÓN
                ent_coefs.append(self.ent_coef.item())

            else:
                ent_coefs.append(self.ent_coef_static)

            # Optimize critic
            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - self.ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (
                    1 - replay_data.dones
                ) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # the n_critics is known because the critic is a Parallel structure.
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor-twin delayed update. Computation depends on `gradient_steps` and `target_update_interval`
            if gradient_step % self.target_update_interval == 0:
                # Compute actor loss
                pi_actions, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)
                qf_pi = th.cat(self.critic(replay_data.observations, pi_actions), dim=1)
                min_qf_pi = th.min(qf_pi, dim=1, keepdim=True)[0]
                actor_loss = (self.ent_coef * log_prob - min_qf_pi).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                # Update target networks
                polyak_update(
                    self.critic_target.parameters(), self.critic.parameters(), self.tau
                )

        # Unpack returns
        self._n_updates += gradient_steps
        logger = self.logger
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/ent_coef", self.ent_coef.item() if isinstance(self.ent_coef, th.Tensor) else self.ent_coef)
        if len(ent_coef_losses) > 0:
            logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))
        logger.record("train/actor_loss", np.mean(actor_losses))

# Importa esto si usas polyak_update
from stable_baselines3.common.preprocessing import preprocess_obs
import numpy as np

def polyak_update(params, target_params, tau: float) -> None:
    """
    Perform a Polyak average update: target_params = (1 - tau) * target_params + tau * params
    :param params: Iterable of parameters from the source network
    :param target_params: Iterable of parameters from the target network
    :param tau: the soft update coefficient ("Polyak averaging"), should be between 0 and 1.
    """
    for param, target_param in zip(params, target_params):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    """
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        max_attempts = 50
        attempts = 0

        # Loop de Rejection Sampling
        while attempts < max_attempts:
            # 1. Generar acción (Warmup vs Modelo)
            if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
                # Sample aleatorio. Output shape: (n_envs, 14)
                unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
            else:
                # Sample del actor. Output shape: (n_envs, 14)
                unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

            # 2. Rescalar/Clipping
            clipped_action = unscaled_action.copy()
            if isinstance(self.action_space, spaces.Box):
                # predict devuelve valores que pueden salirse ligeramente, clippeamos a [-1, 1] antes de escalar
                scaled_action = np.clip(unscaled_action, -1, 1)
                
                if action_noise is not None:
                    # El ruido se suma. Ojo con broadcasting si noise es (14,) y scaled es (1,14)
                    scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
                
                # Transformar de [-1, 1] al espacio real del env
                clipped_action = scaled_action * (self.action_space.high - self.action_space.low) / 2 + (
                    self.action_space.high + self.action_space.low
                ) / 2

            # Acción que se guarda en el buffer (dependiendo de si el actor usa squash o no)
            buffer_action = unscaled_action if self.actor.squash_output else clipped_action

            # 3. Validar acción (Torque check)
            # Pasamos clipped_action que tiene forma (n_envs, 14)
            if self.is_valid_action(clipped_action, self._last_obs):
                return clipped_action, buffer_action

            attempts += 1
            # print(f"[Rechazo] Acción rechazada (intento {attempts}/{max_attempts}). Remuestreando...")

        # 4. Fallback: Si fallan todos los intentos, usar acción segura.
        # CORRECCIÓN: Debemos devolver la forma (n_envs, 14), no (14,)
        print(f"[Rechazo] Máximo de intentos excedido. Usando acción default.")
        
        # Expandimos default_action para que coincida con (n_envs, 14)
        safe_action = np.tile(self.default_action, (n_envs, 1))
        
        # En el caso de SAC normal (con squash), buffer_action suele ser la acción raw (pre-tanh) 
        # o la misma si está clippeada. Devolvemos la safe_action en ambos slots para asegurar estabilidad.
        return safe_action, safe_action

    def is_valid_action(self, action: np.ndarray, obs: np.ndarray) -> bool:
        # Aplanar batch dim si existe (común en SB3 para n_envs=1)
        if action.ndim == 2:
            action = action[0]  # Toma el primer (y único) elemento del batch
        if obs.ndim == 2:
            obs = obs[0]

        # Extrae del obs
        q = obs[:7]      # Posiciones actuales
        q_goal = obs[7:14]  # Posiciones objetivo
        dq = obs[14:21]  # Velocidades

        # Escala acción a gains reales
        # k = action[:7] * self.act_high[:7]  # Gains de rigidez
        # d = action[7:] * self.act_high[7:]  # Gains de amortiguamiento

        k = np.array([600., 600., 500., 500., 250., 200., 200.], dtype=np.float32)
        d = np.array([77.5, 77.5, 77.5, 77.5, 50., 44.7, 44.7], dtype=np.float32)

        # Calcula error
        position_error = q_goal - q

        M = self.modelo.inertia(q)
        C = self.modelo.coriolis(q, dq)
        G = self.modelo.gravload(q)

        torques = M @ (k * position_error - d * dq) + C @ dq + G
        print("observación:")
        print(obs)
        print("Torques calculados:")
        print(torques)
        # Verifica si algún torque supera el umbral elemento a elemento
        if np.any(np.abs(torques) > self.torque_threshold):
            return False  # Inválido: Torques demasiado altos
        return True  # Válido
    """

def get_fr3_fk(q):
    """
    Calcula la posición (x, y, z) del End-Effector (Flange) del Franka FR3
    usando parámetros DH modificados estándar.
    """
    # Parámetros DH Modificados para Franka FR3 / Panda
    # a (metros), d (metros), alpha (radianes)
    dh_params = [
        [0,       0.333, 0],       # Joint 1
        [0,       0.0,   -np.pi/2],# Joint 2
        [0,       0.316, np.pi/2], # Joint 3
        [0.0825,  0.0,   np.pi/2], # Joint 4
        [-0.0825, 0.384, -np.pi/2],# Joint 5
        [0,       0.0,   np.pi/2], # Joint 6
        [0.088,   0.2104, np.pi/2]  # Joint 7 (Flange y Hand)
    ]
    
    T_total = np.eye(4)
    
    for i, (a, d, alpha) in enumerate(dh_params):
        theta = q[i]
        
        # Matriz de transformación DH Modificada
        # T (i-1 -> i) = Rot_x(alpha) * Trans_x(a) * Rot_z(theta) * Trans_z(d)
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        T_i = np.array([
            [ct,    -st,    0,      a],
            [st*ca, ct*ca, -sa,    -sa*d],
            [st*sa, ct*sa,  ca,     ca*d],
            [0,     0,      0,      1]
        ])
        
        T_total = T_total @ T_i

    R_total= T_total[0:3, 0:3]
    R_quat = R.from_matrix(R_total)

    # Retorno posicion y orientacion en cuaternion
    return T_total[:3, 3], R_total

def new_goal(pose_actual, dist = (0.15, 0.10), rot = (45, 15), max_attempts=1000):
    # --- 1. Configuración ---
    base_pose = np.array([0., -0.785, 0., -2.356, 0., 1.571, 0.785], dtype=np.float32)
    joint_limits_lower = np.array([-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508])
    joint_limits_upper = np.array([ 2.9007,  1.8361,  2.9007, -0.1169,  2.8763,  4.6216,  3.0508])
    
    # Calculamos la posición cartesiana de referencia (Base Pose y Actual)
    pos_base, rot_base = get_fr3_fk(base_pose)
    rot_base = R.from_matrix(rot_base)
    pos_actual, rot_actual = get_fr3_fk(pose_actual)
    rot_actual = R.from_matrix(rot_actual)

    # --- 2. Búsqueda por muestreo (Rejection Sampling) ---
    for _ in range(max_attempts):
        # ESTRATEGIA: En lugar de buscar en todo el espacio (que es enorme),
        # buscamos cerca de la base_pose añadiendo ruido. Esto hace mucho más probable
        # encontrar un punto a menos de 20cm de la base.
        
        # Ruido aleatorio (ajustar desviación estándar según necesidad, 0.5 rad es razonable)
        noise = np.random.uniform(-0.6, 0.6, size=7)
        candidate_q = base_pose + noise
        
        # Clip para asegurar que estamos dentro de los límites físicos del robot
        candidate_q = np.clip(candidate_q, joint_limits_lower, joint_limits_upper)
        
        # Calculamos FK del candidato
        pos_candidate, rot_candidate = get_fr3_fk(candidate_q)
        
        # --- 3. Verificación de Restricciones ---
        
        # A. Distancia a la Base Pose
        dist_to_base = np.linalg.norm(pos_candidate - pos_base)
        if dist_to_base > dist[0]:
            continue # Descartar y probar otro
            
        # B. Distancia a la Pose Actual
        dist_to_actual = np.linalg.norm(pos_candidate - pos_actual)
        if dist_to_actual < dist[1]:
            continue # Descartar y probar otro
        
        # C. Rotación con respecto a Base
        rot_candidate = R.from_matrix(rot_candidate)
        rot_diff = rot_base * rot_candidate.inv()
        ang_rad_base = rot_diff.magnitude()
        ang_deg_base = np.degrees(ang_rad_base)
        if ang_deg_base > rot[0]:
            continue # Descartar y probar otro

        # D. Rotación con respecto a Actual
        rot_diff_actual = rot_actual * rot_candidate.inv()
        ang_rad_actual = rot_diff_actual.magnitude()
        ang_deg_actual = np.degrees(ang_rad_actual)
        if ang_deg_actual < rot[1]:
            continue # Descartar y probar otro

        # ¡Éxito! Encontramos una pose válida
        return candidate_q.astype(np.float32)

    # Si fallamos tras muchos intentos, devolvemos base_pose o lanzamos error
    print("Warning: No se encontró pose válida, devolviendo base_pose")
    return base_pose.astype(np.float32)

def pose_diff(pose1, pose2):
    """
    Calcula la diferencia linear y angular entre 2 poses dadas en configuracion articular
    """
    pos1 , rot1 = get_fr3_fk(pose1)
    pos2 , rot2 = get_fr3_fk(pose2)

    # Diferencia lineal
    dist = np.linalg.norm(pos1 - pos2)

    # Diferencia angular
    r1 = R.from_matrix(rot1)
    r2 = R.from_matrix(rot2)

    # Rotación relativa
    r_rel = r1 * r2.inv()
    # Ángulo de la rotación relativa
    ang_rad = r_rel.magnitude()
    # Convertir a grados
    ang_deg = np.degrees(ang_rad)
    
    return dist, ang_deg