#include <iostream>
#include <visp3/core/vpConfig.h>
#include <visp3/robot/vpRobotFranka.h>
#include <visp3/core/vpColVector.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpPoseVector.h>
#include <cmath>
#include <franka/gripper.h>

int main(int argc, char **argv)
{
    std::string robot_ip = "172.16.0.2";

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--ip" && i + 1 < argc) {
            robot_ip = std::string(argv[i + 1]);
        }
        else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout << argv[0] << " [--ip 192.168.1.1] [--help] [-h]" << std::endl;
            return EXIT_SUCCESS;
        }
    }

    try {
        vpRobotFranka robot;
        robot.connect(robot_ip);

        std::cout << "WARNING: This example will move the robot! "
                  << "Please make sure to have the user stop button at hand!" << std::endl
                  << "Press Enter to continue..." << std::endl;
        std::cin.ignore();

        franka::Gripper gripper(robot_ip); // misma IP del robot
        gripper.move(0.0, 0.1); // cerrar completamente, velocidad 0.1 m/s

        // Mover a posición articular home
        vpColVector q(7, 0);
        q[1] = -M_PI_4;
        q[3] = -3 * M_PI_4;
        q[5] = M_PI_2;
        q[6] = M_PI_4;
        robot.setRobotState(vpRobot::STATE_POSITION_CONTROL);
        std::cout << "Move to joint position: " << q.t() << std::endl;
        robot.setPosition(vpRobot::JOINT_STATE, q);

    }
    catch (const vpException &e) {
        std::cout << "ViSP exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (const franka::NetworkException &e) {
        std::cout << "Franka network exception: " << e.what() << std::endl;
        std::cout << "Check if you are connected to the Franka robot"
                  << " or if you specified the right IP using --ip command"
                  << " line option set by default to 192.168.1.1. " << std::endl;
        return EXIT_FAILURE;
    }
    catch (const std::exception &e) {
        std::cout << "Franka exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "The end" << std::endl;
    return EXIT_SUCCESS;
}
