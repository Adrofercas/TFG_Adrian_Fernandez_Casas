#include <iostream>
#include <visp3/core/vpConfig.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/robot/vpRobotFranka.h>
#include <visp3/core/vpTime.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <sstream>
#include <fcntl.h>
 
using namespace std;
 
// Conexion basica por socket
int connectToServer(const char *ip, int port) {
    int sock = 0;
    struct sockaddr_in serv_addr;
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) return -1;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0) return -1;
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) return -1;
 
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
    return sock;
}
 
int main() {
    // Definimos el centro deseado de la imagen (basado en resolucion 640x480 o similar)
    const double u_dest = 640.0; 
    const double v_dest = 360.0;
    const double ganancia = 0.0005; // Ajustar segun respuesta deseada
    int sock = connectToServer("127.0.0.1", 5006);
    vpRobotFranka robot;
    try {
        robot.connect("172.16.0.2");
        robot.setRobotState(vpRobot::STATE_VELOCITY_CONTROL);
    } catch (...) {
        cout << "Error de conexion con el robot." << endl;
        return -1;
    }
 
    vpColVector v(6); 
    v = 0;
 
    cout << "Iniciando control proporcional simple..." << endl;
 
    while (true) {
        double t_start = vpTime::measureTimeMs();
 
        // 1. Leer datos del socket
        char buffer[1024];
        string latest_message = "";
        ssize_t bytes_read;
        while ((bytes_read = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
            buffer[bytes_read] = '\0';
            latest_message = buffer;
        }
 
        // 2. Si hay datos, calcular error y velocidad
        if (!latest_message.empty()) {
            stringstream ss(latest_message);
            string segment;
            vector<double> coords;
            while (getline(ss, segment, ',')) {
                try { coords.push_back(stod(segment)); } catch (...) {}
            }
 
            if (coords.size() >= 4) {
                double u_c, v_c;
                if (coords.size() == 4) { // Formato x,y,w,h
                    u_c = coords[0] + coords[2]/2.0;
                    v_c = coords[1] + coords[3]/2.0;
                } else { // Formato esquinas
                    u_c = (coords[0] + coords[2] + coords[4] + coords[6]) / 4.0;
                    v_c = (coords[1] + coords[3] + coords[5] + coords[7]) / 4.0;
                }
 
                // Error = Posicion actual - Posicion deseada
                double error_u = u_c - u_dest;
                double error_v = v_c - v_dest;
 
                // Control Proporcional Simple (v = K * error)
                // Se asignan a los ejes X e Y de la camara
                v[0] = ganancia * error_u; 
                v[1] = ganancia * error_v;
 
                // Limite de seguridad simple (0.15 m/s)
                if (v.frobeniusNorm() > 0.05) {
                    v *= (0.05 / v.frobeniusNorm());
                }

                // std::cout << "Error (u,v): (" << error_u << ", " << error_v << ") -> Velocidad (vx, vy): (" << v[0] << ", " << v[1] << ")" << std::endl;
            }
        }
 
        // 3. Aplicar velocidad al robot
        // float vy = static_cast<float>(v[1]);
        // v[1] = v[0];
        // v[0] = -vy;
        vpColVector v_final(6);
        v_final = 0;
        v_final[0] = -v[1]; // Mover en X de robot seg
        v_final[1] = v[0]; // Mover en Y de robot segun camara
        // std::cout << "Velocidad (vx, vy): (" << v[0] << ", " << v[1] << ")" << std::endl;
        robot.setVelocity(vpRobot::CAMERA_FRAME, v_final);
 
        // 4. Bucle a 1kHz
        double t_exec = vpTime::measureTimeMs() - t_start;
        if (t_exec < 1.0) vpTime::wait(1.0 - t_exec);
    }
 
    return 0;
}