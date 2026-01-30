#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <array>
#include <vector>
#include <string>
#include <thread>
#include <atomic>

#include <franka/exception.h>
#include <franka/robot.h>
#include "examples_common.h"

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

// ===== Leer trayectorias desde archivo (líneas vacías separan trayectorias) =====
std::vector<std::vector<std::array<double, 7>>> load_trajectories(const std::string& filename) {
    std::vector<std::vector<std::array<double, 7>>> trajectories;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo: " + filename);
    }

    std::string line;
    std::vector<std::array<double, 7>> current_traj;
    while (std::getline(file, line)) {
        if (line.empty()) {
            if (!current_traj.empty()) {
                trajectories.push_back(current_traj);
                current_traj.clear();
            }
            continue;
        }
        std::array<double, 7> q_rad;
        std::stringstream ss(line);
        std::string token;
        for (size_t i = 0; i < 7; i++) {
            if (!std::getline(ss, token, ',')) {
                throw std::runtime_error("Formato incorrecto en src/trayectorias.txt");
            }
            q_rad[i] = std::stod(token);
        }
        current_traj.push_back(q_rad);
    }
    if (!current_traj.empty()) {
        trajectories.push_back(current_traj);
    }
    return trajectories;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <robot-hostname>" << std::endl;
        return -1;
    }

    try {
        franka::Robot robot(argv[1]);
        setDefaultBehavior(robot);

        // === Inicializar Realsense ===
        rs2::pipeline pipe;
        rs2::config cfg;
        int width = 1280, height = 720, fps = 30;
        cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
        pipe.start(cfg);
        pipe.wait_for_frames();

        // Cargar trayectorias
        auto trajectories = load_trajectories("src/trayectorias.txt");
        std::cout << "Trayectorias cargadas: " << trajectories.size() << std::endl;

        // Ejecutar trayectorias
        for (size_t t = 0; t < trajectories.size(); t++) {
            std::cout << "Ejecutando trayectoria " << t+1 << std::endl;

            // Ejecutar la **primera configuración**
            robot.control(MotionGenerator(0.2, trajectories[t][0]));
            std::cout << "Primera configuración alcanzada." << std::endl;

            // === Iniciar grabación desde aquí ===
            std::string filename = "videos/trayectoria_" + std::to_string(t+1) + ".avi";
            int fourcc = cv::VideoWriter::fourcc('X','V','I','D');
            cv::VideoWriter writer(filename, fourcc, fps, cv::Size(width, height));
            if (!writer.isOpened()) {
                throw std::runtime_error("No se pudo abrir VideoWriter");
            }

            std::atomic<bool> recording(true);

            // Hilo para grabar vídeo
            std::thread video_thread([&]() {
                while (recording) {
                    rs2::frameset frames = pipe.wait_for_frames();
                    rs2::video_frame color = frames.get_color_frame();
                    cv::Mat img(cv::Size(width, height), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
                    writer.write(img);
                    cv::imshow("Realsense", img);

                    if ((cv::waitKey(1) & 0xFF) == 'q') {
                        recording = false;
                    }
                }
            });

            // Mover a las configuraciones restantes
            for (size_t i = 1; i < trajectories[t].size(); i++) {
                robot.control(MotionGenerator(0.2, trajectories[t][i]));
                std::cout << "Config alcanzada: " << i+1 << std::endl;
            }

            // Terminar grabación
            recording = false;
            video_thread.join();
            writer.release();
            std::cout << "Vídeo guardado: " << filename << std::endl;
        }

        pipe.stop();
        std::cout << "Todas las trayectorias ejecutadas." << std::endl;

    } catch (const franka::Exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
