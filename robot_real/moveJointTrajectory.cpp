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
                throw std::runtime_error("Formato incorrecto en src/puntos_joint2.txt");
            }
        }
        goals.push_back(deg2rad_array(q_deg)); // Convertimos a radianes
        // goals.push_back(q_deg);             // Si se quieren en grados, usar esta línea en su lugar
    }
    return goals;
}

int main() {
    try {
        franka::Robot robot("172.16.0.2");
        setDefaultBehavior(robot);

        std::cout << "Iniciamos trayectoria. Press Enter to continue" << std::endl;
        std::cin.ignore();

        // Leer los goals desde archivo
        std::vector<std::array<double, 7>> goals = load_goals_from_file("src/puntos_joint2.txt");

        for (size_t i = 0; i < goals.size(); i++) {
            robot.control(MotionGenerator(0.7, goals[i]));
        }

        std::cout << "Trayectoria completada!" << std::endl;

    } catch (const franka::Exception& e) {
        std::cout << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
