#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <thread>
#include <atomic>

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include "cartesianSave.h" // tu función control_cart

#include <chrono>
#include <iomanip>
#include <ctime>

void print_hora() {
    // Obtener hora actual
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) % 1000;

    std::tm bt{};
    localtime_r(&in_time_t, &bt); // versión segura en Linux

    // Imprimir en formato HH:MM:SS.mmm
    std::cout << "Capturamos video: "<<std::put_time(&bt, "%H:%M:%S")
              << "." << std::setfill('0') << std::setw(3) << ms.count()
              << std::endl;
}


// ===== Leer trayectorias de poses cartesianas =====
std::vector<std::vector<std::array<double, 6>>> load_cartesian_trajectories(const std::string& filename) {
    std::vector<std::vector<std::array<double, 6>>> trajectories;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo: " + filename);
    }

    std::string line;
    std::vector<std::array<double, 6>> current_traj;
    while (std::getline(file, line)) {
        if (line.empty()) {
            if (!current_traj.empty()) {
                trajectories.push_back(current_traj);
                current_traj.clear();
            }
            continue;
        }

        std::array<double, 6> pose;
        std::stringstream ss(line);
        std::string token;
        for (size_t i = 0; i < 6; i++) {
            if (!std::getline(ss, token, ',')) {
                throw std::runtime_error("Formato incorrecto en " + filename);
            }
            pose[i] = std::stod(token);
        }
        current_traj.push_back(pose);
    }
    if (!current_traj.empty()) {
        trajectories.push_back(current_traj);
    }
    return trajectories;
}

int main() {
    try {
        // === Inicializar Realsense ===
        rs2::pipeline pipe;
        rs2::config cfg;
        int width = 1280, height = 720, fps = 30;
        cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
        pipe.start(cfg);
        pipe.wait_for_frames();

        // Cargar trayectorias
        auto trajectories = load_cartesian_trajectories("src/trayectorias.txt");
        std::cout << "Trayectorias cargadas: " << trajectories.size() << std::endl;

        for (size_t t = 0; t < trajectories.size(); t++) {
            std::cout << "Ejecutando trayectoria " << t+1 << std::endl;

            // === Preparar VideoWriter ===
            std::string filename = "videos/trayectoria_" + std::to_string(t+1) + ".avi";
            int fourcc = cv::VideoWriter::fourcc('X','V','I','D');
            cv::VideoWriter writer(filename, fourcc, fps, cv::Size(width, height));
            if (!writer.isOpened()) {
                throw std::runtime_error("No se pudo abrir VideoWriter");
            }

            //me muevo al primer punto de la trayectoria
            std::array<double,3> pos = {trajectories[t][0][0], trajectories[t][0][1], trajectories[t][0][2]};
            std::array<double,3> orient = {trajectories[t][0][3], trajectories[t][0][4], trajectories[t][0][5]};
            
            bool success = control_cart(pos, orient, 0.05, 0.2, true, false); // tu función
            if (!success) {
                std::cerr << "Error moviendo a la primera pose"<< std::endl;
            } else {
                std::cout << "Primera pose alcanzada" << std::endl;
            }
            // std::cout<<"Durmiendo"<<std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            std::atomic<bool> recording(true);
            // === Hilo para grabar vídeo ===
            std::thread video_thread([&]() {
                print_hora();
                while (recording) {
                    rs2::frameset frames = pipe.wait_for_frames();
                    rs2::video_frame color = frames.get_color_frame();
                    cv::Mat img(cv::Size(width, height), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
                    writer.write(img);
                    cv::imshow("Realsense", img);

                    if ((cv::waitKey(1) & 0xFF) == 'q' or cv::waitKey(1) == 27) {
                        recording = false;
                    }
                }
            });

            // === Ejecutar la trayectoria ===
            for (size_t i = 1; i < trajectories[t].size(); i++) {
                std::array<double,3> pos = {trajectories[t][i][0], trajectories[t][i][1], trajectories[t][i][2]};
                std::array<double,3> orient = {trajectories[t][i][3], trajectories[t][i][4], trajectories[t][i][5]};

                bool success = control_cart(pos, orient, 0.05, 0.71, true, true); // tu función
                if (!success) {
                    std::cerr << "Error moviendo a la pose " << i+1 << std::endl;
                } else {
                    std::cout << "Pose alcanzada: " << i+1 << std::endl;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }

            // Terminar grabación
            recording = false;
            video_thread.join();
            writer.release();
            std::cout << "Vídeo guardado: " << filename << std::endl;
        }

        pipe.stop();
        std::cout << "Todas las trayectorias ejecutadas." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
