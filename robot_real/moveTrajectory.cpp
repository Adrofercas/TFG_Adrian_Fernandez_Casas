#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <string>
#include <thread>
#include <chrono>

#include "cartesianSave.h" // Asegúrate de que este header existe y contiene control_cart

// ===== Leer trayectorias de poses cartesianas (Igual que en saveVideo.cpp) =====
std::vector<std::vector<std::array<double, 6>>> load_cartesian_trajectories(const std::string& filename) {
    std::vector<std::vector<std::array<double, 6>>> trajectories;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo: " + filename);
    }

    std::string line;
    std::vector<std::array<double, 6>> current_traj;
    while (std::getline(file, line)) {
        // Manejo de líneas vacías para separar trayectorias
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
        // Parsear 6 valores separados por comas
        for (size_t i = 0; i < 6; i++) {
            if (!std::getline(ss, token, ',')) {
                // Si falla la lectura por comas, intentamos leer toda la línea (caso de formatos mixtos)
                // aunque saveVideo.cpp usa estrictamente comas.
                 throw std::runtime_error("Formato incorrecto en línea (se esperaban comas): " + filename);
            }
            try {
                pose[i] = std::stod(token);
            } catch (...) {
                 throw std::runtime_error("Error convirtiendo a número: " + token);
            }
        }
        current_traj.push_back(pose);
    }
    // Añadir la última trayectoria si el archivo no termina en línea vacía
    if (!current_traj.empty()) {
        trajectories.push_back(current_traj);
    }
    return trajectories;
}

int main() {
    try {
        std::cout << "Iniciando ejecución de trayectorias (Estilo saveVideo)..." << std::endl;

        // Cargar trayectorias
        // Asegúrate de que el nombre del archivo es correcto. En saveVideo usas "src/trayectorias.txt".
        // Aquí pongo "trayectorias.txt" asumiendo que está en local, cámbialo si es necesario.
        auto trajectories = load_cartesian_trajectories("src/trayectorias.txt");
        std::cout << "Trayectorias cargadas: " << trajectories.size() << std::endl;

        // Movemos a punto inicial
        std::array<double,3> pos = {trajectories[0][0][0], trajectories[0][0][1], trajectories[0][0][2]};
        std::array<double,3> orient = {trajectories[0][0][3], trajectories[0][0][4], trajectories[0][0][5]};
        
        bool success = control_cart(pos, orient, 0.15, 0.5, true, false); 
        
        if (!success) {
            std::cerr << "Error moviendo a la primera pose de la trayectoria 1"<< std::endl;
        } else {
            std::cout << "Posición inicial alcanzada." << std::endl;
        }
        
        std::cout << "\nPresiona Enter para comenzar secuencia" << std::endl;
        std::cin.ignore();

        // Bucle principal: Recorre cada bloque de trayectorias
        for (size_t t = 0; t < trajectories.size(); t++) {
            // std::cout << "\nPresiona Enter para ir al primer punto" << std::endl;
            // std::cin.ignore();
            std::cout << "--- Ejecutando trayectoria " << t+1 << " ---" << std::endl;
            if (t!= 0) {
                // 1. Mover al PRIMER punto de la trayectoria (Aproximación lenta)
                std::array<double,3> pos = {trajectories[t][0][0], trajectories[t][0][1], trajectories[t][0][2]};
                std::array<double,3> orient = {trajectories[t][0][3], trajectories[t][0][4], trajectories[t][0][5]};
                
                bool success = control_cart(pos, orient, 0.15, 0.5, true, false); 
                
                if (!success) {
                    std::cerr << "Error moviendo a la primera pose de la trayectoria " << t+1 << std::endl;
                } else {
                    std::cout << "Posición inicial alcanzada." << std::endl;
                }
            }
            
            // Espera de estabilización
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            // std::cout << "\nPress Enter para empezar trayectoria" << std::endl;
            // std::cin.ignore();

            // 2. Ejecutar el RESTO de la trayectoria (Movimiento continuo)
            for (size_t i = 1; i < trajectories[t].size(); i++) {
                std::array<double,3> next_pos = {trajectories[t][i][0], trajectories[t][i][1], trajectories[t][i][2]};
                std::array<double,3> next_orient = {trajectories[t][i][3], trajectories[t][i][4], trajectories[t][i][5]};

                success = control_cart(next_pos, next_orient, 0.15, 0.5, true, false); 
                
                if (!success) {
                    std::cerr << "Error en punto " << i+1 << " de trayectoria " << t+1 << std::endl;
                } else {
                    std::cout << "Punto " << i+1 << " alcanzado." << std::endl;
                }
                
                // Espera entre puntos
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
            
            std::cout << "Trayectoria " << t+1 << " finalizada." << std::endl;
        }

        std::cout << "Todas las trayectorias completadas." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Excepción fatal: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}