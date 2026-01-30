#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <vector>

#include <franka/exception.h>
#include <franka/robot.h>
#include "examples_common.h"
#include <Eigen/Geometry>
#include "acquire-data.h"
#include "eye-in-hand.h"
#include "chessboard.h"

// Funcion auxiliar pasar de grados a radianes
std::array<double, 7> deg2rad_array(const std::array<double, 7>& q_deg) {
    std::array<double, 7> q_rad;
    for (size_t i = 0; i < q_deg.size(); i++) {
        q_rad[i] = q_deg[i] * M_PI / 180.0;
    }
    return q_rad;
}

// Leer goals desde un archivo de texto
// Cada línea debe tener 7 números (en grados), separados por comas
std::vector<std::array<double, 7>> load_goals_from_file(const std::string& filename) {
    std::vector<std::array<double, 7>> goals;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::array<double, 7> q_deg;
        size_t pos = 0;
        for (size_t i = 0; i < 7; i++) {
            pos = line.find(',');
            std::string token;
            if (pos != std::string::npos) {
                token = line.substr(0, pos);
                line.erase(0, pos + 1); // eliminar hasta la coma
            } else {
                token = line; // último número
                line.clear();
            }
            try {
                q_deg[i] = std::stod(token);
            } catch (...) {
                throw std::runtime_error("Formato incorrecto en goals.txt");
            }
        }
        goals.push_back(deg2rad_array(q_deg));
    }
    return goals;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
        return -1;
    }

    try {
        franka::Robot robot(argv[1]);
        setDefaultBehavior(robot);

        std::string output_folder = "data_calibraciones/data-eye-in-hand11";
        std::cout << "Be sure to have the correct empty directory: "<<output_folder<<". Press Enter to continue" << std::endl;
        std::cin.ignore();
        //Inicializamos la funcion de calibracion
        FrankaCalibCapture calibrator("172.16.0.2", output_folder);
    
        if (!calibrator.initialize()) {
            return -1;
        }

        // Leer los goals desde archivo
        std::vector<std::array<double, 7>> goals = load_goals_from_file("src/puntos_joint.txt");

        for (size_t i = 0; i < goals.size(); i++) {
            robot.control(MotionGenerator(0.2, goals[i]));

            std::cout << "Point reached: " << i + 1 << std::endl;
            std::array<double, 16> pose = robot.readOnce().O_T_EE;
            vpPoseVector robot_pose;

            // Primero crear la matriz homogénea desde tu array
            vpHomogeneousMatrix M;
            for(int i = 0; i < 4; i++) {
                for(int j = 0; j < 4; j++) {
                    M[i][j] = pose[j * 4 + i];  // column-major
                }
            }

            // Convertir la matriz homogénea a vpPoseVector
            robot_pose.buildFrom(M);
            calibrator.capture(robot_pose);
        }
        robot.control(MotionGenerator(0.2, {{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}}));
        std::cout << "Home reached, movement finalized!\n\nComputing calibration" << std::endl;

        //Calibracion

        bool ok = ChessboardPoseLib::computeChessboardPosesFromDataset(
            output_folder+"/franka_image-%d.png",   // --input
            output_folder+"/franka_camera.xml",     // --intrinsic
            "Camera",                                 // nombre en el XML
            output_folder+"/franka_pose_cPo_%d.yaml", // --output
            9,                                        // -w
            6,                                        // -h
            0.026                                    // --square-size
        );

        if (ok) std::cout << "Poses calculated and saved\n";
        else    std::cout << "Error processing dataset\n";

        EyeInHandCalibration calib(output_folder,
                                "franka_pose_rPe_%d.yaml",
                                "franka_pose_cPo_%d.yaml");
        calib.loadData();

        if (!calib.calibrate()) {
            std::cerr << "Calibration failed!" << std::endl;
            return EXIT_FAILURE;
        }

        calib.savePose(calib.getCameraPose(), "output_eMc.yaml");
        calib.savePose(calib.getObjectPose(), "output_rMo.yaml");

        std::cout << "Calibration completed successfully!" << std::endl;

    } catch (const franka::Exception& e) {
        std::cout << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
