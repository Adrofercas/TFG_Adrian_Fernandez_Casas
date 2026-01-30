#include <cmath>
#include <iostream>
#include <array>

#include <franka/exception.h>
#include <franka/robot.h>
#include "examples_common.h"
#include <Eigen/Geometry>

// Funcion auxiliar pasar de grados a radianes
std::array<double, 7> deg2rad_array(const std::array<double, 7>& q_deg) {
    std::array<double, 7> q_rad;
    for (size_t i = 0; i < q_deg.size(); i++) {
        q_rad[i] = q_deg[i] * M_PI / 180.0;
    }
    return q_rad;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
    return -1;
  }

  try {
    franka::Robot robot(argv[1]);
    setDefaultBehavior(robot);

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
    Eigen::AngleAxisd rot_x(-M_PI, Eigen::Vector3d::UnitX());
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
    const double v_max = 0.1;   // velocidad máxima [m/s]
    const double w_max = M_PI_4; // velocidad angular máxima [rad/s]
    double slowdown_factor = 2.0; // factor de ralentización para tener en cuenta aceleración y desaceleración

    robot.control([&](const franka::RobotState& robot_state, franka::Duration period) -> franka::CartesianPose {
    time += period.toSec();

    // Duración total del movimiento lineal
    double T_total = dist_total / v_max * slowdown_factor;
    // Duración total de la rotación
    double T_rot = angle / w_max * slowdown_factor;

    // Tiempo normalizado para lineal
    double t_norm = time / T_total;
    if (t_norm > 1.0) t_norm = 1.0;

    // Tiempo normalizado para angular
    double t_norm_ang = time / T_rot;
    if (t_norm_ang > 1.0) t_norm_ang = 1.0;

    // Perfil seno (suave en inicio y fin): desplazamiento absoluto
    // s va de 0 -> dist_total
    double s_prev = 0;
    double s = dist_total * 0.5 * (1 - cos(M_PI * t_norm));
    double step = s - s_prev; // paso incremental en este control
    s_prev = s;

    double theta_t = angle * 0.5 * (1 - cos(M_PI * t_norm_ang));

    // Lo volvemos a pasar a matriz de rotación
    Eigen::AngleAxisd aa_t(theta_t, axis);
    axis.normalize();
    Eigen::Matrix3d R_new = R_initial * aa_t.toRotationMatrix();

    // Creamos la nueva pose
    std::array<double, 16> new_pose = pose_initial;
    new_pose[12] += dir[0] * step;
    new_pose[13] += dir[1] * step;
    new_pose[14] += dir[2] * step;

    // Actualizamos la rotación
    new_pose[0]  = R_new(0,0);
    new_pose[1]  = R_new(1,0);
    new_pose[2]  = R_new(2,0);

    new_pose[4]  = R_new(0,1);
    new_pose[5]  = R_new(1,1);
    new_pose[6]  = R_new(2,1);

    new_pose[8]  = R_new(0,2);
    new_pose[9]  = R_new(1,2);
    new_pose[10] = R_new(2,2);


    // Terminar cuando hemos llegado, tanto a nivel de posición como de orientación
    if (t_norm >= 1.0 && t_norm_ang >= 1.0) {
        std::cout << "Meta alcanzada!" << std::endl;
        return franka::MotionFinished(new_pose);
    }

    return new_pose;
});

  } catch (const franka::Exception& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}
