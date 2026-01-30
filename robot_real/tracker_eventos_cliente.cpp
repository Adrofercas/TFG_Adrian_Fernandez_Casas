#include <iostream>
#include <visp3/core/vpConfig.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/robot/vpRobotFranka.h>
#include <visp3/visual_features/vpFeatureBuilder.h>
#include <visp3/visual_features/vpFeaturePoint.h>
#include <visp3/vs/vpServo.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <sstream>
#include <fcntl.h>

using namespace std;

int connectToServer(const char *ip, int port) {
    int sock = 0;
    struct sockaddr_in serv_addr;
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) return -1;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0) return -1;
    if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) return -1;

    // Configurar socket como no bloqueante para vaciado rápido
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);
    return sock;
}

int main() {
    vpCameraParameters cam(720, 720, 640, 360); 
    int sock = connectToServer("127.0.0.1", 5006);
    
    vpRobotFranka robot;
    try {
        robot.connect("172.16.0.2");
    } catch (...) {
        cout << "Error conectando con el robot Franka." << endl;
        return -1;
    }

    vpServo task;
    vpFeaturePoint p, pd; 
    pd.set_x(0); pd.set_y(0); pd.set_Z(0.5); // Objetivo: centro a 0.5m

    task.addFeature(p, pd);
    task.setServo(vpServo::EYEINHAND_CAMERA);
    task.setInteractionMatrixType(vpServo::CURRENT);
    task.setLambda(0.5); // Ganancia moderada para evitar oscilaciones a 1kHz

    robot.setRobotState(vpRobot::STATE_VELOCITY_CONTROL);
    
    // v_last almacenará la última velocidad calculada para repetirla si no hay frame nuevo
    vpColVector v_last(6); 
    v_last = 0;
    
    cout << "Bucle de control híbrido (Fisica: 1kHz / Visión: Variable)" << endl;

    while (true) {
        double t_start = vpTime::measureTimeMs();

        // 1. Intentar obtener el ÚLTIMO frame disponible del socket
        char buffer[1024];
        string latest_message = "";
        ssize_t bytes_read;
        
        // Vaciamos el buffer TCP por completo para quedarnos con el dato más reciente
        while ((bytes_read = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
            buffer[bytes_read] = '\0';
            latest_message = buffer;
        }

        // 2. Si ha llegado un frame nuevo, recalculamos la ley de control
        if (!latest_message.empty()) {
            stringstream ss(latest_message);
            string segment;
            vector<double> coords;
            while (getline(ss, segment, ',')) {
                try { coords.push_back(stod(segment)); } catch (...) {}
            }

            if (coords.size() >= 8) {
                // Cálculo de centroide
                double u_c = (coords[0] + coords[2] + coords[4] + coords[6]) / 4.0;
                double v_c = (coords[1] + coords[3] + coords[5] + coords[7]) / 4.0;
                
                std::cout << "Centroide detectado en (u,v): (" << u_c << ", " << v_c << ")" << std::endl;
                vpImagePoint centroide(v_c, u_c);
                vpFeatureBuilder::create(p, cam, centroide);
                p.set_Z(0.5); // Escala necesaria para la matriz de interacción

                // Calculamos la nueva velocidad
                vpColVector v_c_task = task.computeControlLaw();
                
                // Aplicamos seguridad y actualizamos v_last
                v_last = 0;
                // // Deadzone para evitar ruido cuando el error es despreciable
                // if (task.getError().frobeniusNorm() > 0.005) {
                //     v_last[0] = v_c_task[0]; 
                //     v_last[1] = v_c_task[1];
                // }

                // // Saturación de velocidad máxima (0.15 m/s)
                // if (v_last.frobeniusNorm() > 0.15) {
                //     v_last *= (0.15 / v_last.frobeniusNorm());
                // }
            }
        }

        // 3. ENVIAR CONTRASIGNA AL ROBOT (Siempre, a 1kHz)
        // Si no hubo frame nuevo, v_last mantiene el valor del ciclo anterior
        robot.setVelocity(vpRobot::CAMERA_FRAME, v_last);

        // 4. Sincronización a ~1ms (1000Hz)
        double t_exec = vpTime::measureTimeMs() - t_start;
        if (t_exec < 1.0) {
            vpTime::wait(1.0 - t_exec);
        }
    }

    return 0;
}