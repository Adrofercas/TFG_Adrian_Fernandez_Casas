#include <iostream>

#include <visp3/core/vpConfig.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpXmlParserCamera.h>
#include <visp3/detection/vpDetectorAprilTag.h>
#include <visp3/gui/vpDisplayFactory.h>
#include <visp3/gui/vpPlot.h>
#include <visp3/io/vpImageIo.h>
#include <visp3/robot/vpRobotFranka.h>
#include <visp3/sensor/vpRealSense2.h>
#include <visp3/visual_features/vpFeatureBuilder.h>
#include <visp3/visual_features/vpFeaturePoint.h>
#include <visp3/vs/vpServo.h>
#include <visp3/vs/vpServoDisplay.h>

#if defined(VISP_HAVE_REALSENSE2) && defined(VISP_HAVE_DISPLAY) && defined(VISP_HAVE_FRANKA) && defined(VISP_HAVE_PUGIXML)

#ifdef ENABLE_VISP_NAMESPACE
using namespace VISP_NAMESPACE_NAME;
#endif

void display_point_trajectory(const vpImage<unsigned char> &I, const std::vector<vpImagePoint> &vip,
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

int main(int argc, char **argv)
{
  double opt_tag_size = 0.09557;
  bool opt_tag_z_aligned = false;
  std::string opt_robot_ip = "172.16.0.2";
  std::string opt_eMc_filename = "output_eMc.yaml";
  std::string opt_intrinsic_filename = "franka_camera.xml";
  std::string opt_camera_name = "Camera";
  bool display_tag = true;
  int opt_quad_decimate = 2;
  bool opt_verbose = true;
  bool opt_plot = false;
  bool opt_adaptive_gain = false;
  bool opt_task_sequencing = false;
  double convergence_threshold = 0.00005;

  vpRealSense2 rs;
  rs2::config config;
  unsigned int width = 1280, height = 720;
  config.disable_stream(RS2_STREAM_DEPTH);
  config.disable_stream(RS2_STREAM_INFRARED);
  config.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGBA8, 30);
  rs.open(config);

  vpImage<unsigned char> I(height, width);

#if (VISP_CXX_STANDARD >= VISP_CXX_STANDARD_11)
  std::shared_ptr<vpDisplay> display = vpDisplayFactory::createDisplay(I, 10, 10, "Current image");
#else
  vpDisplay *display = vpDisplayFactory::allocateDisplay(I, 10, 10, "Current image");
#endif

  std::cout << "Parameters:" << std::endl;
  std::cout << "  Apriltag                  " << std::endl;
  std::cout << "    Size [m]              : " << opt_tag_size << std::endl;
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

  // Setup Apriltag detector
  vpDetectorAprilTag::vpAprilTagFamily tagFamily = vpDetectorAprilTag::TAG_36h11;
  vpDetectorAprilTag::vpPoseEstimationMethod poseEstimationMethod = vpDetectorAprilTag::HOMOGRAPHY_VIRTUAL_VS;
  // vpDetectorAprilTag::vpPoseEstimationMethod poseEstimationMethod = vpDetectorAprilTag::BEST_RESIDUAL_VIRTUAL_VS;
  vpDetectorAprilTag detector(tagFamily);
  detector.setAprilTagPoseEstimationMethod(poseEstimationMethod);
  detector.setDisplayTag(display_tag);
  detector.setAprilTagQuadDecimate(opt_quad_decimate);
  detector.setZAlignedWithCameraAxis(opt_tag_z_aligned);

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
  vpHomogeneousMatrix cd_M_o(vpTranslationVector(0, 0, opt_tag_size * 3), // 3 times tag with along camera z axis
                             vpRotationMatrix({ 1, 0, 0, 0, -1, 0, 0, 0, -1 }));

  vpRobotFranka robot;

  try {
    robot.connect(opt_robot_ip);

    // Create visual features
    std::vector<vpFeaturePoint> p(4), pd(4); // We use 4 points

    // Define 4 3D points corresponding to the CAD model of the Apriltag
    std::vector<vpPoint> point(4);
    point[0].setWorldCoordinates(-opt_tag_size / 2., -opt_tag_size / 2., 0);
    point[1].setWorldCoordinates(+opt_tag_size / 2., -opt_tag_size / 2., 0);
    point[2].setWorldCoordinates(+opt_tag_size / 2., +opt_tag_size / 2., 0);
    point[3].setWorldCoordinates(-opt_tag_size / 2., +opt_tag_size / 2., 0);

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

    while (!has_converged && !final_quit) {
      double t_start = vpTime::measureTimeMs();

      rs.acquire(I);

      vpDisplay::display(I);

      std::vector<vpHomogeneousMatrix> c_M_o_vec;
      bool ret = detector.detect(I, opt_tag_size, cam, c_M_o_vec);

      {
        std::stringstream ss;
        ss << "Left click to " << (send_velocities ? "stop the robot" : "servo the robot") << ", right click to quit.";
        vpDisplay::displayText(I, 20, 20, ss.str(), vpColor::red);
      }

      vpColVector v_c(6);

      // Only one tag is detected
      if (ret && (c_M_o_vec.size() == 1)) {
        c_M_o = c_M_o_vec[0];

        static bool first_time = true;
        if (first_time) {
          // Introduce security wrt tag positioning in order to avoid PI rotation
          std::vector<vpHomogeneousMatrix> secure_o_M_o(2), secure_cd_M_c(2);
          secure_o_M_o[1].buildFrom(0, 0, 0, 0, 0, M_PI);
          for (size_t i = 0; i < 2; ++i) {
            secure_cd_M_c[i] = cd_M_o * secure_o_M_o[i] * c_M_o.inverse();
          }
          if (std::fabs(secure_cd_M_c[0].getThetaUVector().getTheta()) < std::fabs(secure_cd_M_c[1].getThetaUVector().getTheta())) {
            o_M_o = secure_o_M_o[0];
          }
          else {
            std::cout << "Desired frame modified to avoid PI rotation of the camera" << std::endl;
            o_M_o = secure_o_M_o[1]; // Introduce PI rotation
          }

          // Compute the desired position of the features from the desired pose
          for (size_t i = 0; i < point.size(); ++i) {
            vpColVector c_P, p;
            point[i].changeFrame(cd_M_o * o_M_o, c_P);
            point[i].projection(c_P, p);

            pd[i].set_x(p[0]);
            pd[i].set_y(p[1]);
            pd[i].set_Z(c_P[2]);
          }
        }

        // Get tag corners
        std::vector<vpImagePoint> corners = detector.getPolygon(0);
        std::cout << "Translation: " << c_M_o.getTranslationVector().t() << std::endl;
        // Update visual features
        for (size_t i = 0; i < corners.size(); ++i) {
          // Update the point feature from the tag corners location
          vpFeatureBuilder::create(p[i], cam, corners[i]);
          // Set the feature Z coordinate from the pose
          vpColVector c_P;
          point[i].changeFrame(c_M_o, c_P);

          p[i].set_Z(c_P[2]);
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

      // Send to the robot
      robot.setVelocity(vpRobot::CAMERA_FRAME, v_c);

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