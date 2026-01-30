#include "cartesianSave.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <franka/exception.h>
#include <franka/robot.h>
#include "examples_common.h"
#include <Eigen/Geometry>
#include <iomanip>
#include <ctime>
#include <chrono>

// Funcion auxiliar pasar de grados a radianes
std::array<double, 3> deg2rad_array(const std::array<double, 3>& rot_deg) {
    std::array<double, 3> rot_rad;
    for (size_t i = 0; i < rot_deg.size(); i++) {
        rot_rad[i] = rot_deg[i] * M_PI / 180.0;
    }
    return rot_rad;
}

std::string build_robot_data_json(const std::array<double, 16>& pose,
                                  const std::array<double, 7>& tau,
                                  const std::array<double, 7>& dtau,
                                  const std::array<double, 7>& q,
                                  const std::array<double, 7>& dq) {
    // Obtener hora local con milisegundos
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) % 1000;

    std::tm bt{};
    localtime_r(&in_time_t, &bt);

    std::ostringstream oss;
    oss << std::put_time(&bt, "%H:%M:%S")
        << "." << std::setfill('0') << std::setw(3) << ms.count();
    std::string hora = oss.str();

    // Construir JSON como string
    std::ostringstream json_ss;
    json_ss << "{";
    json_ss << "\"time\":\"" << hora << "\"";
    json_ss << ",\"pose\":[";
    for (size_t i = 0; i < pose.size(); i++) {
        json_ss << pose[i];
        if (i < pose.size() - 1) json_ss << ",";
    }
    json_ss << "]";
    json_ss << ",\"tau\":[";
    for (size_t i = 0; i < tau.size(); i++) {
        json_ss << tau[i];
        if (i < tau.size() - 1) json_ss << ",";
    }
    json_ss << "]";
    json_ss << ",\"dtau\":[";
    for (size_t i = 0; i < dtau.size(); i++) {
        json_ss << dtau[i];
        if (i < dtau.size() - 1) json_ss << ",";
    }
    json_ss << "]";
    json_ss << ",\"q\":[";
    for (size_t i = 0; i < q.size(); i++) {
        json_ss << q[i];
        if (i < q.size() - 1) json_ss << ",";
    }
    json_ss << "]";
    json_ss << ",\"dq\":[";
    for (size_t i = 0; i < dq.size(); i++) {
        json_ss << dq[i];
        if (i < dq.size() - 1) json_ss << ",";
    }
    json_ss << "]";
    json_ss << "}" << std::endl;

    return json_ss.str();
}

bool go_home() {
  try {
      // Movemos el robot a home por joint space
    franka::Robot robot("172.16.0.2");
    setDefaultBehavior(robot);
    std::array<double, 7> q_goal = {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    MotionGenerator motion_generator(0.5, q_goal);
    robot.control(motion_generator);
    std::cout << "Hemos llegado a home" << std::endl;
    return true; // éxito
  }

  catch (const franka::Exception& e) {
    std::cout << e.what() << std::endl;
    return false; // fallo
  }
}

bool control_cart(std::array<double, 3> goal_pos,
                  std::array<double, 3> goal_orient,
                  float v_max_param,
                  float w_max_param,
                  bool deg,
                  bool save_data) {
    // Variables para threading (declaradas fuera del try)
    std::queue<std::string> data_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::atomic<bool> done_saving{false};
    std::thread writer_thread;
    std::string filename = "videos/robot_state.txt";

    try {
        franka::Robot robot("172.16.0.2");
        setDefaultBehavior(robot);

        // Leemos pose inicial
        std::array<double, 16> pose_initial = robot.readOnce().O_T_EE;

        // Si la entrada está en grados, la convertimos a radianes
        if (deg) {
            goal_orient = deg2rad_array(goal_orient);
        }
        
        // Definimos la rotación objetivo como rotaciones sucesivas en X, Y y Z
        Eigen::AngleAxisd rot_x(goal_orient[0], Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd rot_y(goal_orient[1], Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd rot_z(goal_orient[2], Eigen::Vector3d::UnitZ());

        // Creamos la matriz de rotación objetivo FUERA del callback
        Eigen::Matrix3d R_goal = rot_z.toRotationMatrix()
                                * rot_y.toRotationMatrix()
                                * rot_x.toRotationMatrix();

        if (save_data) {
            writer_thread = std::thread([&]() {
                std::ofstream file(filename, std::ios::app);
                if (!file.is_open()) {
                    std::cerr << "Error al abrir el archivo " << filename << std::endl;
                    return;
                }

                while (true) {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    queue_cv.wait(lock, [&] { return !data_queue.empty() || done_saving.load(); });

                    if (data_queue.empty() && done_saving.load()) {
                        break;
                    }

                    while (!data_queue.empty()) {
                        std::string json = std::move(data_queue.front());
                        data_queue.pop();
                        lock.unlock(); // Desbloquea mientras escribe
                        file << json;
                        lock.lock();
                    }
                }

                file.close();
            });
        }

        double last_save_time = 0.0;
        double save_interval = 1.0 / 60.0; // 60 Hz
        const double v_max = v_max_param;    // [m/s]
        const double a_max = 0.2;    // [m/s^2]
        const double w_max = w_max_param;  // [rad/s]
        const double eps = 1e-9;
        bool first_call = true;
        std::array<double, 16> pose_initial_cb;
        Eigen::Matrix3d R_initial;
        Eigen::Matrix3d R_goal_cb;
        double dist_total_cb = 0.0;
        std::array<double, 3> dir_cb = {0.0, 0.0, 0.0};
        Eigen::Vector3d axis_cb = Eigen::Vector3d::UnitX();
        double angle_cb = 0.0;
        double time = 0.0;
        double T_total_transl = 0.0;
        double T_total_rot = 0.0;
        std::array<double,16> pose_filtered = {};
        double last_save_time_cb = 0.0;

        robot.control([&](const franka::RobotState& robot_state, franka::Duration period)
                        -> franka::CartesianPose {
    
        auto start = std::chrono::high_resolution_clock::now();
        double dt = period.toSec();
    
        if (first_call) {
            pose_initial_cb = robot_state.O_T_EE;
            pose_filtered = pose_initial_cb;
    
            std::array<double, 3> delta = {
                goal_pos[0] - pose_initial_cb[12],
                goal_pos[1] - pose_initial_cb[13],
                goal_pos[2] - pose_initial_cb[14]};
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
    
            R_goal_cb = R_goal;
    
            Eigen::Matrix3d R_err = R_initial.transpose() * R_goal_cb;
            Eigen::AngleAxisd aa(R_err);
            axis_cb = aa.axis();
            angle_cb = aa.angle();
    
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
            last_save_time_cb = 0.0;
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
    
        double T_max = std::max(T_total_transl, T_total_rot);
        bool motion_complete = (time >= T_max + 0.1);
    
        if (motion_complete) {
            std::array<double, 16> final_pose = pose_initial_cb;
            
            final_pose[12] = goal_pos[0];
            final_pose[13] = goal_pos[1];
            final_pose[14] = goal_pos[2];
    
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
    
        for (double &v : new_pose) {
            if (!std::isfinite(v)) {
            std::cerr << "Valor no finito detectado. Abortando." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            return franka::MotionFinished(pose_initial_cb);
            }
        }
    
        static const double alpha_filter = 0.3;
        double progress = time / T_max;
        double alpha_dynamic = alpha_filter;
        if (progress > 0.9) {
            alpha_dynamic = alpha_filter + (1.0 - alpha_filter) * ((progress - 0.9) / 0.1);
        }
        
        for (size_t i = 0; i < 16; i++) {
            pose_filtered[i] = alpha_dynamic * new_pose[i] + (1.0 - alpha_dynamic) * pose_filtered[i];
        }
        
        if ((time - last_save_time_cb >= save_interval) && save_data) {
            std::string json = build_robot_data_json(robot_state.O_T_EE, robot_state.tau_J, 
                                                    robot_state.dtau_J, robot_state.q, robot_state.dq);
            if (!json.empty()) {
                std::unique_lock<std::mutex> lock(queue_mutex);
                data_queue.push(std::move(json));
                lock.unlock();
                queue_cv.notify_one();
            }
            last_save_time_cb = time;

            // Detén el cronómetro
            auto end = std::chrono::high_resolution_clock::now();
            // Calcula la duración en microsegundos
            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "Tiempo transcurrido al guardar: " << duration_us.count()/1000.0 << " ms" << std::endl;
        }

        return pose_filtered;
        });

        // Cleanup: Signal done and join writer thread
        if (save_data && writer_thread.joinable()) {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                done_saving.store(true);
            }
            queue_cv.notify_one();
            writer_thread.join();
        }

        return true;
    } catch (const franka::Exception& e) {
        std::cout << e.what() << std::endl;
        // Ensure thread cleanup on exception
        if (save_data && writer_thread.joinable()) {
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                done_saving.store(true);
            }
            queue_cv.notify_one();
            writer_thread.join();
        }
        return false;
    }
}