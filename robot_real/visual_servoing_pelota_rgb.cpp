#include <iostream>

#include <visp3/core/vpConfig.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpXmlParserCamera.h>
#include <visp3/gui/vpDisplayFactory.h>
#include <visp3/gui/vpPlot.h>
#include <visp3/io/vpImageIo.h>
#include <visp3/robot/vpRobotFranka.h>
#include <visp3/sensor/vpRealSense2.h>
#include <visp3/visual_features/vpFeatureBuilder.h>
#include <visp3/visual_features/vpFeaturePoint.h>
#include <visp3/vs/vpServo.h>
#include <visp3/vs/vpServoDisplay.h>
#include <visp3/vision/vpPose.h>
#include <visp3/core/vpPixelMeterConversion.h>
#include <deque>
#include <cmath>

#if defined(VISP_HAVE_REALSENSE2) && defined(VISP_HAVE_DISPLAY) && defined(VISP_HAVE_FRANKA) && defined(VISP_HAVE_PUGIXML)

#ifdef ENABLE_VISP_NAMESPACE
using namespace VISP_NAMESPACE_NAME;
#endif

void display_point_trajectory(const vpImage<vpRGBa> &I, const std::vector<vpImagePoint> &vip,
                              std::vector<vpImagePoint> *traj_vip)
{
  for (size_t i = 0; i < vip.size(); ++i) {
    if (traj_vip[i].size()) {
      // Add the point only if distance with the previous > 1 pixel
      if (vpImagePoint::distance(vip[i], traj_vip[i].back()) > 1.) {
        traj_vip[i].push_back(vip[i]);
      }
    }
    else {
      traj_vip[i].push_back(vip[i]);
    }
  }
  for (size_t i = 0; i < vip.size(); ++i) {
    for (size_t j = 1; j < traj_vip[i].size(); j++) {
      vpDisplay::displayLine(I, traj_vip[i][j - 1], traj_vip[i][j], vpColor::green, 2);
    }
  }
}


void sortCornersForTarget(std::vector<vpImagePoint>& corners) {
    if (corners.size() != 4) return;

    // 1. Paso intermedio: Clasificar geométricamente quién es quién
    // Ordenamos primero por altura (v) para separar los de Arriba (Top) y Abajo (Bottom)
    std::sort(corners.begin(), corners.end(), [](const vpImagePoint& a, const vpImagePoint& b) {
        return a.get_v() < b.get_v();
    });

    // Ahora corners[0] y corners[1] son los de ARRIBA (Top)
    // corners[2] y corners[3] son los de ABAJO (Bottom)

    // Ordenamos los de ARRIBA entre ellos por izquierda/derecha (u)
    if (corners[0].get_u() > corners[1].get_u()) std::swap(corners[0], corners[1]);
    // Ahora corners[0] es TL (Top-Left), corners[1] es TR (Top-Right)

    // Ordenamos los de ABAJO entre ellos por izquierda/derecha (u)
    if (corners[2].get_u() > corners[3].get_u()) std::swap(corners[2], corners[3]);
    // Ahora corners[2] es BL (Bottom-Left), corners[3] es BR (Bottom-Right)

    // Guardamos los puntos geométricos temporales
    vpImagePoint tl = corners[0];
    vpImagePoint tr = corners[1];
    vpImagePoint bl = corners[2];
    vpImagePoint br = corners[3];

    // 2. Paso Final: Asignar al vector en el orden que TUS PUNTOS ROJOS (META) piden.
    // Mirando tu imagen 'image_2f230c.png':
    // El 0 rojo está Abajo-Izquierda (BL)
    // El 1 rojo está Abajo-Derecha (BR)
    // El 2 rojo está Arriba-Derecha (TR)
    // El 3 rojo está Arriba-Izquierda (TL)
    
    corners[0] = bl; // Índice 0 -> Bottom-Left
    corners[1] = br; // Índice 1 -> Bottom-Right
    corners[2] = tr; // Índice 2 -> Top-Right
    corners[3] = tl; // Índice 3 -> Top-Left
}



bool detectBlueObject(vpImage<vpRGBa> &I, 
                      unsigned char minBlue, 
                      unsigned char maxRed, 
                      unsigned char maxGreen, 
                      std::vector<vpImagePoint> &corners) 
{
    static std::deque<std::vector<vpImagePoint>> history;
    const size_t history_size = 15; 

    corners.clear();
    
    unsigned int width = I.getWidth();
    unsigned int height = I.getHeight();
    
    std::vector<int> col_hist(width, 0);
    std::vector<int> row_hist(height, 0);
    int pixel_count = 0;

    // 1. Detección de color
    for (unsigned int r = 0; r < height; r++) {
        for (unsigned int c = 0; c < width; c++) {
            vpRGBa val = I[r][c];
            if (val.B > minBlue && val.R < maxRed && val.G < maxGreen) {
                col_hist[c]++;
                row_hist[r]++;
                pixel_count++;
            }
        }
    }

    if (pixel_count < 200) {
        history.clear(); 
        return false; 
    }

    // 2. Encontrar límites con Histogramas
    auto max_x_it = std::max_element(col_hist.begin(), col_hist.end());
    int center_x = (int)std::distance(col_hist.begin(), max_x_it);
    
    auto max_y_it = std::max_element(row_hist.begin(), row_hist.end());
    int center_y = (int)std::distance(row_hist.begin(), max_y_it);

    int min_x = center_x; int max_x = center_x;
    int min_y = center_y; int max_y = center_y;
    int umbral_ruido = 2; 

    while(min_x > 0 && col_hist[min_x] > umbral_ruido) min_x--;
    while(max_x < (int)width - 1 && col_hist[max_x] > umbral_ruido) max_x++;
    while(min_y > 0 && row_hist[min_y] > umbral_ruido) min_y--;
    while(max_y < (int)height - 1 && row_hist[max_y] > umbral_ruido) max_y++;

    // ---------------------------------------------------------
    // 3. NUEVO: VERIFICACIÓN GEOMÉTRICA (Aspect Ratio)
    // ---------------------------------------------------------
    double w = (double)(max_x - min_x);
    double h = (double)(max_y - min_y);
    
    // Evitar divisiones por cero o cajas minúsculas
    if (w < 5 || h < 5) {
        history.clear();
        return false;
    }

    // Calculamos la relación de aspecto
    // Para una esfera perfecta w == h (ratio 1.0)
    // Permitimos cierta distorsión (ej. de 0.6 a 1.5)
    // Si w es mucho mayor que h, o h mucho mayor que w, rechazamos.
    
    double ratio = w / h;
    double max_ratio_diff = 1.5; // Umbral: Un lado no puede ser más de 1.5 veces el otro

    if (ratio > max_ratio_diff || ratio < (1.0 / max_ratio_diff)) {
        // Es demasiado rectangular -> Lo consideramos ruido o error
        // Limpiamos historial porque no es una detección válida del objeto
        history.clear(); 
        return false; 
    }
    // ---------------------------------------------------------

    // 4. Crear puntos CRUDOS (Raw)
    std::vector<vpImagePoint> raw_corners;
    raw_corners.push_back(vpImagePoint(min_y, max_x)); // TR
    raw_corners.push_back(vpImagePoint(min_y, min_x)); // TL
    raw_corners.push_back(vpImagePoint(max_y, min_x)); // BL
    raw_corners.push_back(vpImagePoint(max_y, max_x)); // BR

    // 5. LÓGICA DE MEDIANA
    history.push_back(raw_corners);
    if (history.size() > history_size) {
        history.pop_front();
    }

    for (int i = 0; i < 4; i++) {
        std::vector<double> u_vals;
        std::vector<double> v_vals;

        for (const auto& frame_pts : history) {
            u_vals.push_back(frame_pts[i].get_u());
            v_vals.push_back(frame_pts[i].get_v());
        }

        std::sort(u_vals.begin(), u_vals.end());
        std::sort(v_vals.begin(), v_vals.end());

        size_t mid_idx = u_vals.size() / 2;
        corners.push_back(vpImagePoint(v_vals[mid_idx], u_vals[mid_idx]));
    }

    return true;
}

bool detector(vpImage<vpRGBa> &I, double tag_size, const vpCameraParameters &cam,
              vpHomogeneousMatrix &c_M_o, std::vector<vpImagePoint> &corners)
{
  bool ret;

  ret = detectBlueObject(I, 70, 10, 100, corners);
  if (ret) {
    // 1. Instanciar el estimador de pose
    vpPose pose;
    sortCornersForTarget(corners);
    // 2. Definir el modelo 3D (4 esquinas de un cuadrado virtual)
    // Asumimos un sistema donde X es derecha, Y es abajo (coherente con imagen)
    // tag_size debe estar en METROS (ej. 0.10 para 10cm)
    double r = tag_size / 2.0; 
    
    // Crear 4 puntos. El orden DEBE coincidir con el orden de tu vector 'corners'
    // Tu vector 'corners' era: TL, TR, BR, BL
    vpPoint P[4];
    P[0].setWorldCoordinates( r, -r, 0); // Top-Right    (X pos, Y neg)
    P[1].setWorldCoordinates(-r, -r, 0); // Top-Left     (X neg, Y neg)
    P[2].setWorldCoordinates(-r,  r, 0); // Bottom-Left  (X neg, Y pos)
    P[3].setWorldCoordinates( r,  r, 0); // Bottom-Right (X pos, Y pos)

    // 3. Asociar los puntos 2D detectados a los puntos 3D
    for (int i = 0; i < 4; i++) {
        // Obtener coordenadas en píxeles
        double u = corners[i].get_u();
        double v = corners[i].get_v();

        // Convertir píxeles a coordenadas normalizadas (x = X/Z, y = Y/Z)
        // Esto es CRUCIAL para vpPose
        double x, y;
        vpPixelMeterConversion::convertPoint(cam, u, v, x, y);

        // Asignar al punto
        P[i].set_x(x);
        P[i].set_y(y);
        
        // Añadir el punto al estimador
        pose.addPoint(P[i]);
    }

    // 4. Calcular la pose
    // VIRTUAL_VS es robusto y refina una estimación inicial, pero necesita una buena inicialización
    // LAGRANGE puede usarse para obtener una primera estimación
    pose.computePose(vpPose::LAGRANGE, c_M_o);
    pose.computePose(vpPose::VIRTUAL_VS, c_M_o);
    
    std::cout << "Translation: " << c_M_o.getTranslationVector().t() << std::endl;
}

  return ret;
}

int main(int argc, char **argv)
{
  double pelota_size = 0.06; // [m]
  bool opt_tag_z_aligned = false;
  std::string opt_robot_ip = "172.16.0.2";
  std::string opt_eMc_filename = "output_eMc.yaml";
  std::string opt_intrinsic_filename = "franka_camera.xml";
  std::string opt_camera_name = "Camera";
  bool display_tag = true;
  int opt_quad_decimate = 2;
  bool opt_verbose = true;
  bool opt_plot = false;
  bool opt_adaptive_gain = true;
  bool opt_task_sequencing = false;
  double convergence_threshold = 0.00005;

  vpRealSense2 rs;
  rs2::config config;
  unsigned int width = 1280, height = 720;
  config.disable_stream(RS2_STREAM_DEPTH);
  config.disable_stream(RS2_STREAM_INFRARED);
  config.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGBA8, 30);
  rs.open(config);

  vpImage<vpRGBa> I(height, width);

#if (VISP_CXX_STANDARD >= VISP_CXX_STANDARD_11)
  std::shared_ptr<vpDisplay> display = vpDisplayFactory::createDisplay(I, 10, 10, "Current image");
#else
  vpDisplay *display = vpDisplayFactory::allocateDisplay(I, 10, 10, "Current image");
#endif

  std::cout << "Parameters:" << std::endl;
  std::cout << "  Apriltag                  " << std::endl;
  std::cout << "    Size [m]              : " << pelota_size << std::endl;
  std::cout << "    Z aligned             : " << (opt_tag_z_aligned ? "true" : "false") << std::endl;
  std::cout << "  Camera intrinsics         " << std::endl;
  std::cout << "    Factory parameters    : " << (opt_intrinsic_filename.empty() ? "yes" : "no") << std::endl;

  // Get camera intrinsics
  vpCameraParameters cam;
  if (opt_intrinsic_filename.empty()) {
    std::cout << "Use Realsense camera intrinsic factory parameters: " << std::endl;
    cam = rs.getCameraParameters(RS2_STREAM_COLOR, vpCameraParameters::perspectiveProjWithDistortion);
    std::cout << "cam:\n" << cam << std::endl;
  }
  else if (!vpIoTools::checkFilename(opt_intrinsic_filename)) {
    std::cout << "Camera parameters file " << opt_intrinsic_filename << " doesn't exist." << std::endl;
    return EXIT_FAILURE;
  }
  else {
    vpXmlParserCamera parser;
    if (!opt_camera_name.empty()) {

      std::cout << "    Param file name [.xml]: " << opt_intrinsic_filename << std::endl;
      std::cout << "    Camera name           : " << opt_camera_name << std::endl;

      if (parser.parse(cam, opt_intrinsic_filename, opt_camera_name, vpCameraParameters::perspectiveProjWithDistortion) !=
        vpXmlParserCamera::SEQUENCE_OK) {
        std::cout << "Unable to parse parameters with distortion for camera \"" << opt_camera_name << "\" from "
          << opt_intrinsic_filename << " file" << std::endl;
        std::cout << "Attempt to find parameters without distortion" << std::endl;

        if (parser.parse(cam, opt_intrinsic_filename, opt_camera_name,
                         vpCameraParameters::perspectiveProjWithoutDistortion) != vpXmlParserCamera::SEQUENCE_OK) {
          std::cout << "Unable to parse parameters without distortion for camera \"" << opt_camera_name << "\" from "
            << opt_intrinsic_filename << " file" << std::endl;
          return EXIT_FAILURE;
        }
      }
    }
  }

  std::cout << "Camera parameters used to compute the pose:\n" << cam << std::endl;

  // Setup camera extrinsics
  vpPoseVector e_P_c;
  // Set camera extrinsics default values
  e_P_c[0] = 0.0337731;
  e_P_c[1] = -0.00535012;
  e_P_c[2] = -0.0523339;
  e_P_c[3] = -0.247294;
  e_P_c[4] = -0.306729;
  e_P_c[5] = 1.53055;

  // If provided, read camera extrinsics from --eMc <file>
  if (!opt_eMc_filename.empty()) {
    e_P_c.loadYAML(opt_eMc_filename, e_P_c);
  }
  else {
    std::cout << "Warning, opt_eMc_filename is empty! Use hard coded values." << std::endl;
  }
  vpHomogeneousMatrix e_M_c(e_P_c);
  std::cout << "e_M_c:\n" << e_M_c << std::endl;

  // Set desired pose used to compute the desired features
  vpHomogeneousMatrix cd_M_o(vpTranslationVector(0, 0, pelota_size * 3), // queremos que la altura sea 3 veces el tamanho de la pelota
                             vpRotationMatrix({ 1, 0, 0, 0, -1, 0, 0, 0, -1 }));

  vpRobotFranka robot;

  try {
    robot.connect(opt_robot_ip);

    // Create visual features
    std::vector<vpFeaturePoint> p(4), pd(4); // We use 4 points

    // Definimos los 4 puntos de la meta
    std::vector<vpPoint> point(4);
    point[0].setWorldCoordinates(-pelota_size / 2., -pelota_size / 2., 0);
    point[1].setWorldCoordinates(+pelota_size / 2., -pelota_size / 2., 0);
    point[2].setWorldCoordinates(+pelota_size / 2., +pelota_size / 2., 0);
    point[3].setWorldCoordinates(-pelota_size / 2., +pelota_size / 2., 0);

    // Setup IBVS
    vpServo task;
    // Add the 4 visual feature points
    for (size_t i = 0; i < p.size(); ++i) {
      task.addFeature(p[i], pd[i]);
    }
    task.setServo(vpServo::EYEINHAND_CAMERA);
    task.setInteractionMatrixType(vpServo::CURRENT);

    if (opt_adaptive_gain) {
      vpAdaptiveGain lambda(1.5, 0.4, 30); // lambda(0)=4, lambda(oo)=0.4 and lambda'(0)=30
      task.setLambda(lambda);
    }
    else {
      task.setLambda(0.5);
    }

    vpPlot *plotter = nullptr;
    int iter_plot = 0;

    if (opt_plot) {
      plotter = new vpPlot(2, static_cast<int>(250 * 2), 500, static_cast<int>(I.getWidth()) + 80, 10,
                           "Real time curves plotter");
      plotter->setTitle(0, "Visual features error");
      plotter->setTitle(1, "Camera velocities");
      plotter->initGraph(0, 8);
      plotter->initGraph(1, 6);
      plotter->setLegend(0, 0, "error_feat_p1_x");
      plotter->setLegend(0, 1, "error_feat_p1_y");
      plotter->setLegend(0, 2, "error_feat_p2_x");
      plotter->setLegend(0, 3, "error_feat_p2_y");
      plotter->setLegend(0, 4, "error_feat_p3_x");
      plotter->setLegend(0, 5, "error_feat_p3_y");
      plotter->setLegend(0, 6, "error_feat_p4_x");
      plotter->setLegend(0, 7, "error_feat_p4_y");
      plotter->setLegend(1, 0, "vc_x");
      plotter->setLegend(1, 1, "vc_y");
      plotter->setLegend(1, 2, "vc_z");
      plotter->setLegend(1, 3, "wc_x");
      plotter->setLegend(1, 4, "wc_y");
      plotter->setLegend(1, 5, "wc_z");
    }

    bool final_quit = false;
    bool has_converged = false;
    bool send_velocities = false;
    bool servo_started = false;
    std::vector<vpImagePoint> *traj_corners = nullptr; // To memorize point trajectory

    static double t_init_servo = vpTime::measureTimeMs();

    robot.set_eMc(e_M_c); // Set location of the camera wrt end-effector frame
    robot.setRobotState(vpRobot::STATE_VELOCITY_CONTROL);

    vpHomogeneousMatrix cd_M_c, c_M_o, o_M_o;
    std::vector<vpImagePoint> corners;

    while (!has_converged && !final_quit) {
      double t_start = vpTime::measureTimeMs();

      rs.acquire(I);

      vpDisplay::display(I);

      bool ret = detector(I, pelota_size, cam, c_M_o, corners);

      {
        std::stringstream ss;
        ss << "Left click to " << (send_velocities ? "stop the robot" : "servo the robot") << ", right click to quit.";
        vpDisplay::displayText(I, 20, 20, ss.str(), vpColor::red);
      }

      vpColVector v_c(6);

      // Miramos si se ha detectado la pelota
      if (ret) {

        static bool first_time = true;
        if (first_time) {
          // Introduce security wrt tag positioning in order to avoid PI rotation
        //   std::vector<vpHomogeneousMatrix> secure_o_M_o(2), secure_cd_M_c(2);
        //   secure_o_M_o[1].buildFrom(0, 0, 0, 0, 0, M_PI);
        //   for (size_t i = 0; i < 2; ++i) {
        //     secure_cd_M_c[i] = cd_M_o * secure_o_M_o[i] * c_M_o.inverse();
        //   }
        //   if (std::fabs(secure_cd_M_c[0].getThetaUVector().getTheta()) < std::fabs(secure_cd_M_c[1].getThetaUVector().getTheta())) {
        //     o_M_o = secure_o_M_o[0];
        //   }
        //   else {
        //     std::cout << "Desired frame modified to avoid PI rotation of the camera" << std::endl;
        //     o_M_o = secure_o_M_o[1]; // Introduce PI rotation
        //   }

          // Compute the desired position of the features from the desired pose
          for (size_t i = 0; i < point.size(); ++i) {
            vpColVector c_P, p;
            point[i].changeFrame(cd_M_o, c_P);   //point[i].changeFrame(cd_M_o * o_M_o, c_P);
            point[i].projection(c_P, p);

            pd[i].set_x(p[0]);
            pd[i].set_y(p[1]);
            pd[i].set_Z(c_P[2]);
          }
        }

        // Update visual features
        for (size_t i = 0; i < corners.size(); ++i) {
          // Update the point feature from the tag corners location
          vpFeatureBuilder::create(p[i], cam, corners[i]);
          // Set the feature Z coordinate from the pose
          vpColVector c_P;
          point[i].changeFrame(c_M_o, c_P);

          p[i].set_Z(c_P[2]);
        //   std::cout << "Z de la pelota: " << c_P[2] << std::endl;
        }

        if (opt_task_sequencing) {
          if (!servo_started) {
            if (send_velocities) {
              servo_started = true;
            }
            t_init_servo = vpTime::measureTimeMs();
          }
          v_c = task.computeControlLaw((vpTime::measureTimeMs() - t_init_servo) / 1000.);
        }
        else {
          v_c = task.computeControlLaw();
        }

        // Display the current and desired feature points in the image display
        vpServoDisplay::display(task, cam, I);
        for (size_t i = 0; i < corners.size(); ++i) {
          std::stringstream ss;
          ss << i;
          // Display current point indexes
          vpDisplay::displayText(I, corners[i] + vpImagePoint(15, 15), ss.str(), vpColor::red);
          // Display desired point indexes
          vpImagePoint ip;
          vpMeterPixelConversion::convertPoint(cam, pd[i].get_x(), pd[i].get_y(), ip);
          vpDisplay::displayText(I, ip + vpImagePoint(15, 15), ss.str(), vpColor::red);
        }
        if (first_time) {
          traj_corners = new std::vector<vpImagePoint>[corners.size()];
        }
        // Display the trajectory of the points used as features
        display_point_trajectory(I, corners, traj_corners);

        if (opt_plot) {
          plotter->plot(0, iter_plot, task.getError());
          plotter->plot(1, iter_plot, v_c);
          iter_plot++;
        }

        if (opt_verbose) {
          std::cout << "v_c: " << v_c.t() << std::endl;
        }

        double error = task.getError().sumSquare();
        std::stringstream ss;
        ss << "error: " << error;
        vpDisplay::displayText(I, 20, static_cast<int>(I.getWidth()) - 150, ss.str(), vpColor::red);

        if (opt_verbose)
          std::cout << "error: " << error << std::endl;

        if (error < convergence_threshold) {
          has_converged = true;
          std::cout << "Servo task has converged" << std::endl;
          vpDisplay::displayText(I, 100, 20, "Servo task has converged", vpColor::red);
        }
        if (first_time) {
          first_time = false;
        }
      } // end if (c_M_o_vec.size() == 1)
      else {
        v_c = 0;
      }

      if (!send_velocities) {
        v_c = 0;
      }
      
      // Escalamos la velocidad para que sea segura
      double max_vel_trans = 0.2; // Límite traslacional: 10 cm/s
      vpColVector v_trans = v_c.extract(0, 3);
      //calculamos norma
      double n_trans = v_trans.frobeniusNorm(); // Raíz(vx^2 + vy^2 + vz^2)
      if (n_trans > max_vel_trans) {
        // Factor de reducción (ej: si voy a 0.2 y el max es 0.1, factor = 0.5)
        double factor = max_vel_trans / n_trans;
        v_c *= factor;
      }

      // Send to the robot
      //elimino velocidad rotacional, ya que al ser una esfera no me interesa
      v_c[3] = 0;
      v_c[4] = 0;
      v_c[5] = 0;
      robot.setVelocity(vpRobot::CAMERA_FRAME, v_c);
    //   std::cout << "v_c: " << v_c.t() << std::endl;

    
    // Exemplo do que deberiamos de obter (obtido con aprilTag):
    //v_c: -0.1099096374  -0.08464309085  0.1506319744  -0.003932486808  0.01986559139  0.01668936651
    //error: 0.480896606


      {
        std::stringstream ss;
        ss << "Loop time: " << vpTime::measureTimeMs() - t_start << " ms";
        vpDisplay::displayText(I, 40, 20, ss.str(), vpColor::red);
      }
      vpDisplay::flush(I);

      vpMouseButton::vpMouseButtonType button;
      if (vpDisplay::getClick(I, button, false)) {
        switch (button) {
        case vpMouseButton::button1:
          send_velocities = !send_velocities;
          break;

        case vpMouseButton::button3:
          final_quit = true;
          v_c = 0;
          break;

        default:
          break;
        }
      }
    }
    std::cout << "Stop the robot " << std::endl;
    robot.setRobotState(vpRobot::STATE_STOP);

    if (opt_plot && plotter != nullptr) {
      delete plotter;
      plotter = nullptr;
    }

    if (!final_quit) {
      while (!final_quit) {
        rs.acquire(I);
        vpDisplay::display(I);

        vpDisplay::displayText(I, 20, 20, "Click to quit the program.", vpColor::red);
        vpDisplay::displayText(I, 40, 20, "Visual servo converged.", vpColor::red);

        if (vpDisplay::getClick(I, false)) {
          final_quit = true;
        }

        vpDisplay::flush(I);
      }
    }
    if (traj_corners) {
      delete[] traj_corners;
    }
  }
  catch (const vpException &e) {
    std::cout << "ViSP exception: " << e.what() << std::endl;
    std::cout << "Stop the robot " << std::endl;
    robot.setRobotState(vpRobot::STATE_STOP);
#if (VISP_CXX_STANDARD < VISP_CXX_STANDARD_11)
    if (display != nullptr) {
      delete display;
    }
#endif
    return EXIT_FAILURE;
  }
  catch (const franka::NetworkException &e) {
    std::cout << "Franka network exception: " << e.what() << std::endl;
    std::cout << "Check if you are connected to the Franka robot"
      << " or if you specified the right IP using --ip command line option set by default to 192.168.1.1. "
      << std::endl;
#if (VISP_CXX_STANDARD < VISP_CXX_STANDARD_11)
    if (display != nullptr) {
      delete display;
    }
#endif
    return EXIT_FAILURE;
  }
  catch (const std::exception &e) {
    std::cout << "Franka exception: " << e.what() << std::endl;
#if (VISP_CXX_STANDARD < VISP_CXX_STANDARD_11)
    if (display != nullptr) {
      delete display;
    }
#endif
    return EXIT_FAILURE;
  }

#if (VISP_CXX_STANDARD < VISP_CXX_STANDARD_11)
  if (display != nullptr) {
    delete display;
  }
#endif

  return EXIT_SUCCESS;
}
#else
int main()
{
#if !defined(VISP_HAVE_REALSENSE2)
  std::cout << "Install librealsense-2.x and rebuild ViSP." << std::endl;
#endif
#if !defined(VISP_HAVE_FRANKA)
  std::cout << "Install libfranka and rebuild ViSP." << std::endl;
#endif
#if !defined(VISP_HAVE_PUGIXML)
  std::cout << "Build ViSP with pugixml support enabled." << std::endl;
#endif
  return EXIT_SUCCESS;
}
#endif