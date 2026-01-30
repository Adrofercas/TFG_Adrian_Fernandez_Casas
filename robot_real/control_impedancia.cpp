// Copyright (c) 2023 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <array>
#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <thread>
 
#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>
#include <Eigen/Geometry>
#include <chrono>
 
#include "examples_common.h"
 
namespace {
template <class T, size_t N>
std::ostream& operator<<(std::ostream& ostream, const std::array<T, N>& array) {
  ostream << "[";
  std::copy(array.cbegin(), array.cend() - 1, std::ostream_iterator<T>(ostream, ","));
  std::copy(array.cend() - 1, array.cend(), std::ostream_iterator<T>(ostream));
  ostream << "]";
  return ostream;
}
}  // anonymous namespace
 
int main() {
  // Set print rate for comparing commanded vs. measured torques.
  const double print_rate = 10.0;
 
  double vel_current = 0.0;
  double angle = 0.0;
  double time = 0.0;
 
  // Initialize data fields for the print thread.
  struct {
    std::mutex mutex;
    bool has_data;
    std::array<double, 7> tau_d_last;
    franka::RobotState robot_state;
    std::array<double, 7> gravity;
  } print_data{};
  std::atomic_bool running{true};
 
  // Start print thread.
  std::thread print_thread([print_rate, &print_data, &running]() {
    while (running) {
      // Sleep to achieve the desired print rate.
      std::this_thread::sleep_for(
          std::chrono::milliseconds(static_cast<int>((1.0 / print_rate * 1000.0))));
 
      // Try to lock data to avoid read write collisions.
      if (print_data.mutex.try_lock()) {
        if (print_data.has_data) {
          std::array<double, 7> tau_error{};
          double error_rms(0.0);
          std::array<double, 7> tau_d_actual{};
          for (size_t i = 0; i < 7; ++i) {
            tau_d_actual[i] = print_data.tau_d_last[i] + print_data.gravity[i];
            tau_error[i] = tau_d_actual[i] - print_data.robot_state.tau_J[i];
            error_rms += std::pow(tau_error[i], 2.0) / tau_error.size();
          }
          error_rms = std::sqrt(error_rms);
 
          // Print data to console
          std::cout << "tau_error [Nm]: " << tau_error << std::endl
                    << "tau_commanded [Nm]: " << tau_d_actual << std::endl
                    << "tau_measured [Nm]: " << print_data.robot_state.tau_J << std::endl
                    << "root mean square of tau_error [Nm]: " << error_rms << std::endl
                    << "-----------------------" << std::endl;
          print_data.has_data = false;
        }
        print_data.mutex.unlock();
      }
    }
  });
 
  try {
    // Connect to robot.
    franka::Robot robot("172.16.0.2");
    setDefaultBehavior(robot);
 
    // First move the robot to a suitable joint configuration
    // std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    // MotionGenerator motion_generator(0.5, q_goal);
    // std::cout << "WARNING: This example will move the robot! "
    //           << "Please make sure to have the user stop button at hand!" << std::endl
    //           << "Press Enter to continue..." << std::endl;
    // std::cin.ignore();
    // robot.control(motion_generator);
    // std::cout << "Finished moving to initial joint configuration." << std::endl;
 
    // Set additional parameters always before the control loop, NEVER in the control loop!
    // Set collision behavior.
    robot.setCollisionBehavior(
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
 
    // Load the kinematics and dynamics model.
    franka::Model model = robot.loadModel();

    // Definimos pose objetivo global y leemos la inicial
    std::array<double, 3> goal_pose = {0.55, 0.0, 0.4866};
    std::array<double, 16> pose_initial = robot.readOnce().O_T_EE;

    // Calculamos vector relativo de inicial a objetivo
    std::array<double, 3> delta = {
        goal_pose[0] - pose_initial[12],
        goal_pose[1] - pose_initial[13],
        goal_pose[2] - pose_initial[14]
    };

    // Calculamos distancia total y dirección
    double dist_total = std::sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
    std::array<double, 3> dir = {0.0,0.0,0.0};
    if (dist_total > 1e-6) {
      dir = { delta[0]/dist_total, delta[1]/dist_total, delta[2]/dist_total };
    }

    // Leemos la rotación inicial
    Eigen::Matrix3d R_initial;
    R_initial << pose_initial[0], pose_initial[4], pose_initial[8],
        pose_initial[1], pose_initial[5], pose_initial[9],
        pose_initial[2], pose_initial[6], pose_initial[10];

    // Definimos la rotación objetivo como rotaciones sucesivas en X, Y y Z
    Eigen::AngleAxisd rot_x(M_PI, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd rot_y(0.0, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rot_z(0.0, Eigen::Vector3d::UnitZ());

    // Creamos la matriz de rotación objetivo
    Eigen::Matrix3d R_goal = rot_z.toRotationMatrix()
                            * rot_y.toRotationMatrix()
                            * rot_x.toRotationMatrix();


    // Calculo la matriz de error, es decir, cuánto rota respecto a la inicial
    Eigen::Matrix3d R_err = R_initial.transpose() * R_goal;
    // La pasamos a angle axis
    Eigen::AngleAxisd aa(R_err);
    Eigen::Vector3d axis = aa.axis();
    double angle = aa.angle();

    double time = 0.0;
    const double v_max = 0.2;    // [m/s]
    const double a_max = 0.4;    // [m/s^2]
    const double w_max = M_PI_4;  // [rad/s]
    const double eps = 1e-9;
 
    // Define callback function to send Cartesian pose goals to get inverse kinematics solved.
    auto cartesian_pose_callback = [=, &time, &vel_current, &running, &angle](
                                       const franka::RobotState& robot_state,
                                       franka::Duration period) -> franka::CartesianPose {
     auto start = std::chrono::high_resolution_clock::now();
 
      static bool first_call = true;
      static std::array<double, 16> pose_initial_cb;
      static Eigen::Matrix3d R_initial;
      static double dist_total_cb = 0.0;
      static std::array<double, 3> dir_cb = {0.0, 0.0, 0.0};
      static Eigen::Vector3d axis_cb = Eigen::Vector3d::UnitX();
      static double angle_cb = 0.0;
      static double time = 0.0;
      static double T_total_transl = 0.0;
      static double T_total_rot = 0.0;
      static std::array<double,16> pose_filtered = {};
      static bool filter_initialized = false;
 
      double dt = period.toSec();
 
      if (first_call) {
        pose_initial_cb = robot_state.O_T_EE;
        pose_filtered = pose_initial_cb;
        filter_initialized = true;
 
        std::array<double, 3> delta = {
            goal_pose[0] - pose_initial_cb[12],
            goal_pose[1] - pose_initial_cb[13],
            goal_pose[2] - pose_initial_cb[14]};
        double dist_total = std::sqrt(delta[0] * delta[0] + delta[1] * delta[1] +
                                      delta[2] * delta[2]);
        dist_total_cb = dist_total;
        if (dist_total > eps) {
          dir_cb = {delta[0] / dist_total, delta[1] / dist_total,
                    delta[2] / dist_total};
        } else {
          dir_cb = {0.0, 0.0, 0.0};
        }
 
        R_initial << pose_initial_cb[0], pose_initial_cb[4], pose_initial_cb[8],
            pose_initial_cb[1], pose_initial_cb[5], pose_initial_cb[9],
            pose_initial_cb[2], pose_initial_cb[6], pose_initial_cb[10];
 
        Eigen::AngleAxisd rot_x(M_PI, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd rot_y(0.0, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd rot_z(0.0, Eigen::Vector3d::UnitZ());
        Eigen::Matrix3d R_goal =
            rot_z.toRotationMatrix() * rot_y.toRotationMatrix() * rot_x.toRotationMatrix();
 
        Eigen::Matrix3d R_err = R_initial.transpose() * R_goal;
        Eigen::AngleAxisd aa(R_err);
        axis_cb = aa.axis();
        angle_cb = aa.angle();
 
        // Calcular tiempos totales
        if (dist_total_cb > eps) {
          double t_acc = v_max / a_max;
          double dist_acc = 0.5 * a_max * t_acc * t_acc;
          
          if (2 * dist_acc > dist_total_cb) {
            t_acc = std::sqrt(dist_total_cb / a_max);
            T_total_transl = 2 * t_acc;
          } else {
            double dist_cruise = dist_total_cb - 2 * dist_acc;
            double t_cruise = dist_cruise / v_max;
            T_total_transl = 2 * t_acc + t_cruise;
          }
        }
 
        if (std::abs(angle_cb) > 1e-8) {
          double t_acc_ang = w_max / a_max;
          double ang_acc = 0.5 * a_max * t_acc_ang * t_acc_ang;
          
          if (2 * ang_acc > std::abs(angle_cb)) {
            t_acc_ang = std::sqrt(std::abs(angle_cb) / a_max);
            T_total_rot = 2 * t_acc_ang;
          } else {
            double ang_cruise = std::abs(angle_cb) - 2 * ang_acc;
            double t_cruise_ang = ang_cruise / w_max;
            T_total_rot = 2 * t_acc_ang + t_cruise_ang;
          }
        }
 
        time = 0.0;
        first_call = false;
        std::cout << "Trayectoria iniciada. Distancia: " << dist_total_cb
                  << "m, Ángulo: " << angle_cb << " rad" << std::endl;
        std::cout << "T_transl: " << T_total_transl << "s, T_rot: " << T_total_rot << "s" << std::endl;
        return pose_initial_cb;
      }
 
      if (!std::isfinite(dt) || dt <= 0.0) {
        dt = 1e-3;
      }
      time += dt;
 
      // Determinar si el movimiento ha terminado ANTES de calcular nuevas poses
      double T_max = std::max(T_total_transl, T_total_rot);
      bool motion_complete = (time >= T_max + 0.01); // 10ms de margen
 
      if (motion_complete) {
        // Construir pose final exacta
        std::array<double, 16> final_pose = pose_initial_cb;
        
        final_pose[12] = goal_pose[0];
        final_pose[13] = goal_pose[1];
        final_pose[14] = goal_pose[2];
 
        if (std::abs(angle_cb) > 1e-8) {
          Eigen::AngleAxisd aa_final(angle_cb, axis_cb.normalized());
          Eigen::Matrix3d R_final = R_initial * aa_final.toRotationMatrix();
          
          final_pose[0]  = R_final(0,0);
          final_pose[1]  = R_final(1,0);
          final_pose[2]  = R_final(2,0);
          final_pose[4]  = R_final(0,1);
          final_pose[5]  = R_final(1,1);
          final_pose[6]  = R_final(2,1);
          final_pose[8]  = R_final(0,2);
          final_pose[9]  = R_final(1,2);
          final_pose[10] = R_final(2,2);
        }
 
        std::cout << "¡Meta alcanzada! Tiempo total: " << time << "s" << std::endl;
        return franka::MotionFinished(final_pose);
      }
 
      // ----------------- Perfil trapezoidal (traslación) -----------------
      double step = 0.0;
      if (dist_total_cb > eps) {
        double t_acc = v_max / a_max;
        double dist_acc = 0.5 * a_max * t_acc * t_acc;
        double t_cruise = 0.0;
 
        if (2 * dist_acc > dist_total_cb) {
          t_acc = std::sqrt(dist_total_cb / a_max);
        } else {
          double dist_cruise = dist_total_cb - 2 * dist_acc;
          t_cruise = dist_cruise / v_max;
        }
 
        double t = std::min(time, T_total_transl);
        double s_t = 0.0;
 
        if (t < t_acc) {
          s_t = 0.5 * a_max * t * t;
        } else if (t < (T_total_transl - t_acc)) {
          double t_in_cruise = t - t_acc;
          s_t = dist_acc + v_max * t_in_cruise;
        } else {
          double t_dec = t - (T_total_transl - t_acc);
          s_t = dist_acc + v_max * t_cruise + (v_max * t_dec - 0.5 * a_max * t_dec * t_dec);
        }
 
        step = std::clamp(s_t, 0.0, dist_total_cb);
      }
 
      // ----------------- Perfil trapezoidal (rotación) -----------------
      double theta_t = 0.0;
      if (std::abs(angle_cb) > 1e-8) {
        double t_acc_ang = w_max / a_max;
        double ang_acc = 0.5 * a_max * t_acc_ang * t_acc_ang;
        double t_cruise_ang = 0.0;
 
        if (2 * ang_acc > std::abs(angle_cb)) {
          t_acc_ang = std::sqrt(std::abs(angle_cb) / a_max);
        } else {
          double ang_cruise = std::abs(angle_cb) - 2 * ang_acc;
          t_cruise_ang = ang_cruise / w_max;
        }
 
        double t_ang = std::min(time, T_total_rot);
        double theta_abs = 0.0;
 
        if (t_ang < t_acc_ang) {
          theta_abs = 0.5 * a_max * t_ang * t_ang;
        } else if (t_ang < (T_total_rot - t_acc_ang)) {
          double t_in_cruise = t_ang - t_acc_ang;
          theta_abs = ang_acc + w_max * t_in_cruise;
        } else {
          double t_dec = t_ang - (T_total_rot - t_acc_ang);
          theta_abs = std::abs(angle_cb) - 0.5 * a_max * (t_acc_ang - t_dec) * (t_acc_ang - t_dec);
        }
 
        theta_t = std::copysign(std::clamp(theta_abs, 0.0, std::abs(angle_cb)), angle_cb);
      }
 
      // ----------------- Construir nueva pose -----------------
      std::array<double, 16> new_pose = pose_initial_cb;
 
      new_pose[12] += dir_cb[0] * step;
      new_pose[13] += dir_cb[1] * step;
      new_pose[14] += dir_cb[2] * step;
 
      Eigen::AngleAxisd aa_t(theta_t, axis_cb.normalized());
      Eigen::Matrix3d R_new = R_initial * aa_t.toRotationMatrix();
 
      new_pose[0]  = R_new(0,0);
      new_pose[1]  = R_new(1,0);
      new_pose[2]  = R_new(2,0);
      new_pose[4]  = R_new(0,1);
      new_pose[5]  = R_new(1,1);
      new_pose[6]  = R_new(2,1);
      new_pose[8]  = R_new(0,2);
      new_pose[9]  = R_new(1,2);
      new_pose[10] = R_new(2,2);
 
      // Verificación de NaN/inf
      for (double &v : new_pose) {
        if (!std::isfinite(v)) {
          std::cerr << "Valor no finito detectado. Abortando." << std::endl;
          return franka::MotionFinished(pose_initial_cb);
        }
      }
 
      // Aplicar filtro SOLO durante el movimiento, no al final
      static const double alpha_filter = 0.3;
      
      // Calcular qué tan cerca estamos del final (0.0 = inicio, 1.0 = final)
      double progress = time / T_max;
      
      // Reducir filtrado cerca del final para permitir convergencia exacta
      double alpha_dynamic = alpha_filter;
      if (progress > 0.9) {
        // En el último 10% del movimiento, reducir filtrado gradualmente
        alpha_dynamic = alpha_filter + (1.0 - alpha_filter) * ((progress - 0.9) / 0.1);
      }
      
      for (size_t i = 0; i < 16; i++) {
        pose_filtered[i] = alpha_dynamic * new_pose[i] + (1.0 - alpha_dynamic) * pose_filtered[i];
      }
 
      return pose_filtered;
    };
 
    // Set gains for the joint impedance control.
    // Stiffness
    Eigen::Matrix<double,6,1> k_gains;
    k_gains << 50.0, 500.0, 500.0, 500.0, 500.0, 500.0;  // tus ganancias para x, y, z, roll, pitch, yaw

    Eigen::Matrix<double,6,1> d_gains;
    d_gains << 100.0, 10.0, 10.0, 10.0, 10.0, 10.0;

    // Velocidad articular previa, iniciada a 0
    Eigen::Matrix<double, 7, 1> prev_dq = Eigen::Matrix<double, 7, 1>::Zero();

 
    // Define callback for the joint torque control loop.
    // Define callback for the joint torque control loop.
std::function<franka::Torques(const franka::RobotState&, franka::Duration)>
    impedance_control_callback =
        [&print_data, &model, k_gains, d_gains](
            const franka::RobotState& state, franka::Duration period) -> franka::Torques {
  
  // 1) Obtener Jacobiano 6x7
  std::array<double, 42> jacobian_array =
      model.zeroJacobian(franka::Frame::kEndEffector, state);
  Eigen::Map<const Eigen::Matrix<double,6,7>> J(jacobian_array.data());

  // 2) Obtener matriz de masa articular 7x7
  std::array<double, 49> mass_array = model.mass(state);
  Eigen::Map<const Eigen::Matrix<double,7,7>> M(mass_array.data());

  // 3) Calcular matriz de masa cartesiana (Lambda)
  Eigen::Matrix<double,6,6> Lambda;
  Lambda = (J * M.inverse() * J.transpose()).inverse();

  // 4) Obtener gravedad y Coriolis
  std::array<double, 7> g = model.gravity(state);
  std::array<double, 7> C = model.coriolis(state);
  Eigen::Map<Eigen::Matrix<double,7,1>> tau_g(g.data());
  Eigen::Map<Eigen::Matrix<double,7,1>> tau_c(C.data());

  // 5) Velocidades articulares
  Eigen::Matrix<double, 7, 1> dq(state.dq.data());
  Eigen::Matrix<double, 7, 1> dq_d(state.dq_d.data());
  
  // 6) Pose actual
  std::array<double, 16> pose = state.O_T_EE;
  Eigen::Map<const Eigen::Matrix4d> T(pose.data());
  Eigen::Vector3d position = T.block<3,1>(0,3);
  Eigen::Matrix3d rotation = T.block<3,3>(0,0);
  
  // 7) Pose deseada
  std::array<double, 16> pose_d = state.O_T_EE_d;
  Eigen::Map<const Eigen::Matrix4d> T_d(pose_d.data());
  Eigen::Vector3d position_d = T_d.block<3,1>(0,3);
  Eigen::Matrix3d rotation_d = T_d.block<3,3>(0,0);

  // 8) Error de posición
  Eigen::Vector3d e_pos = position_d - position;

  // 9) Error de orientación usando quaternions (más estable)
  Eigen::Quaterniond q_current(rotation);
  Eigen::Quaterniond q_desired(rotation_d);
  
  // Normalizar para evitar problemas numéricos
  q_current.normalize();
  q_desired.normalize();
  
  // Error de orientación
  Eigen::Quaterniond q_err = q_desired * q_current.inverse();
  if (q_err.w() < 0) {
    q_err.coeffs() *= -1;  // Elegir camino más corto
  }
  Eigen::Vector3d e_rot = 2.0 * q_err.vec();

  // 10) Error cartesiano completo (6D)
  Eigen::Matrix<double,6,1> e_x;
  e_x.head<3>() = e_pos;
  e_x.tail<3>() = e_rot;

  // 11) Velocidades cartesianas
  Eigen::Matrix<double,6,1> dx = J * dq;
  Eigen::Matrix<double,6,1> dx_d = J * dq_d;
  Eigen::Matrix<double,6,1> e_dx = dx_d - dx;

  // Debug (opcional)
  std::cout << "Error pos: " << e_pos.transpose() << std::endl;
  std::cout << "Error rot: " << e_rot.transpose() << std::endl;

  // 12) Fuerza cartesiana (control PD en espacio cartesiano)
  Eigen::Matrix<double,6,1> F_cartesian = 
      k_gains.cwiseProduct(e_x) + d_gains.cwiseProduct(e_dx);

  // 13) Torque articular = J^T * Lambda * F + compensación dinámica
  Eigen::Matrix<double,7,1> tau_task = J.transpose() * Lambda * F_cartesian;
  Eigen::Matrix<double,7,1> tau_d = tau_task + tau_c + tau_g;

  // 14) Convertir a std::array para Franka
  std::array<double, 7> tau_d_array;
  Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;

  // 15) Rate limiting (importante para seguridad)
  std::array<double, 7> tau_d_rate_limited =
      franka::limitRate(franka::kMaxTorqueRate, tau_d_array, state.tau_J_d);

  // 16) Update data to print (opcional)
  if (print_data.mutex.try_lock()) {
    print_data.has_data = true;
    print_data.robot_state = state;
    print_data.tau_d_last = tau_d_rate_limited;
    print_data.gravity = g;
    print_data.mutex.unlock();
  }

  // 17) Retornar torques
  return tau_d_rate_limited;
};
 
    // Start real-time control loop.
    robot.control(impedance_control_callback, cartesian_pose_callback);
 
  } catch (const franka::Exception& ex) {
    running = false;
    std::cerr << ex.what() << std::endl;
  }
 
  if (print_thread.joinable()) {
    print_thread.join();
  }
  return 0;
}
