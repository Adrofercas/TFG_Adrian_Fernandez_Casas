#ifndef FRANKA_CALIB_CAPTURE_H
#define FRANKA_CALIB_CAPTURE_H

#include <string>
#include <visp3/core/vpConfig.h>

#if defined(VISP_HAVE_REALSENSE2) && defined(VISP_HAVE_FRANKA) && defined(VISP_HAVE_PUGIXML)

#include <visp3/core/vpImage.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/robot/vpRobotFranka.h>
#include <visp3/sensor/vpRealSense2.h>
#include <visp3/core/vpPoseVector.h>

#if defined(ENABLE_VISP_NAMESPACE)
using namespace VISP_NAMESPACE_NAME;
#endif

/**
 * @brief Clase para captura automática de datos de calibración para robot Franka
 */
class FrankaCalibCapture {
private:
    vpRobotFranka robot;
    vpRealSense2 camera;
    std::string output_folder;
    std::string robot_ip;
    bool is_initialized;
    unsigned int image_counter;

    /**
     * @brief Encuentra el siguiente número disponible para los archivos
     * @return Número del siguiente archivo disponible
     */
    unsigned int getNextFileNumber();

    /**
     * @brief Verifica si existe un archivo con el nombre dado
     * @param filename Nombre del archivo a verificar
     * @return true si existe, false en caso contrario
     */
    bool fileExists(const std::string& filename);

public:
    /**
     * @brief Constructor de la clase
     * @param robot_ip_address Dirección IP del robot Franka
     * @param output_directory Directorio donde guardar los archivos
     */
    FrankaCalibCapture(const std::string& robot_ip_address = "192.16.0.2",
                       const std::string& output_directory = "data-eye-in-hand");

    /**
     * @brief Destructor
     */
    ~FrankaCalibCapture();

    /**
     * @brief Inicializa la conexión con el robot y la cámara
     * @return true si la inicialización fue exitosa
     */
    bool initialize();

    /**
     * @brief Captura una imagen y obtiene la pose del robot
     * @return Número del archivo capturado, 0 si hay error
     */
    unsigned int capture(vpPoseVector &robot_pose);

    /**
     * @brief Captura múltiples imágenes con un intervalo de tiempo
     * @param num_captures Número de capturas a realizar
     * @param delay_ms Retraso en milisegundos entre capturas
     * @return Número de capturas exitosas
     */
    unsigned int captureSequence(unsigned int num_captures, unsigned int delay_ms = 1000);

    /**
     * @brief Verifica si el sistema está inicializado
     * @return true si está inicializado
     */
    bool isInitialized() const { return is_initialized; }

    /**
     * @brief Obtiene el directorio de salida configurado
     * @return Ruta del directorio de salida
     */
    std::string getOutputFolder() const { return output_folder; }

    /**
     * @brief Cambia el directorio de salida
     * @param new_folder Nueva ruta del directorio
     */
    void setOutputFolder(const std::string& new_folder);
};

#endif // defined(VISP_HAVE_REALSENSE2) && defined(VISP_HAVE_FRANKA) && defined(VISP_HAVE_PUGIXML)

#endif // FRANKA_CALIB_CAPTURE_H