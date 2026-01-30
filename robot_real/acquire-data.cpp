#include "acquire-data.h"

#if defined(VISP_HAVE_REALSENSE2) && defined(VISP_HAVE_FRANKA) && defined(VISP_HAVE_PUGIXML)

#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>
#include <chrono>
#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpXmlParserCamera.h>
#include <visp3/io/vpImageIo.h>

FrankaCalibCapture::FrankaCalibCapture(const std::string& robot_ip_address, 
                                       const std::string& output_directory)
    : robot_ip(robot_ip_address), output_folder(output_directory), 
      is_initialized(false), image_counter(0)
{
    // Asegurar que el directorio termine con '/'
    if (!output_folder.empty() && output_folder.back() != '/') {
        output_folder += "/";
    }
}

FrankaCalibCapture::~FrankaCalibCapture()
{
    // El destructor se encarga de limpiar automáticamente
}

bool FrankaCalibCapture::fileExists(const std::string& filename)
{
    std::ifstream file(filename);
    return file.good();
}

unsigned int FrankaCalibCapture::getNextFileNumber()
{
    unsigned int counter = 1;
    std::string img_filename, pose_filename;
    
    do {
        std::stringstream ss_img, ss_pos;
        ss_img << output_folder << "franka_image-" << counter << ".png";
        ss_pos << output_folder << "franka_pose_rPe_" << counter << ".yaml";
        
        img_filename = ss_img.str();
        pose_filename = ss_pos.str();
        
        if (!fileExists(img_filename) && !fileExists(pose_filename)) {
            break;
        }
        counter++;
    } while (counter < 100); // Límite de seguridad
    
    return counter;
}

bool FrankaCalibCapture::initialize()
{
    try {
        // Crear directorio de salida si no existe
        if (!vpIoTools::checkDirectory(output_folder)) {
            std::cout << "Creating output directory: " << output_folder << std::endl;
            vpIoTools::makeDirectory(output_folder);
        }
        
        
        // Configurar la cámara RealSense
        std::cout << "Configuring RealSense camera..." << std::endl;
        rs2::config config;
        config.disable_stream(RS2_STREAM_DEPTH);
        config.disable_stream(RS2_STREAM_INFRARED);
        config.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGBA8, 30);
        camera.open(config);
        
        // Capturar una imagen de prueba para verificar que funciona
        vpImage<unsigned char> test_image;
        camera.acquire(test_image);
        
        std::cout << "Image seize: " << test_image.getWidth() << " x " << test_image.getHeight() << std::endl;
        
        // Guardar parámetros intrínsecos de la cámara
        vpCameraParameters cam = camera.getCameraParameters(RS2_STREAM_COLOR, vpCameraParameters::perspectiveProjWithDistortion);
        vpXmlParserCamera xml_camera;
        std::string camera_params_file = output_folder + "franka_camera.xml";
        xml_camera.save(cam, camera_params_file, "Camera", test_image.getWidth(), test_image.getHeight());
        
        std::cout << "Camera parameters saved in: " << camera_params_file << std::endl;
        
        is_initialized = true;
        std::cout << "Initialization completed succesfully." << std::endl;
        return true;
        
    } catch (const vpException& e) {
        std::cerr << "Error de ViSP durante la inicialización: " << e.what() << std::endl;
        is_initialized = false;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error durante la inicialización: " << e.what() << std::endl;
        is_initialized = false;
        return false;
    }
}

unsigned int FrankaCalibCapture::capture(vpPoseVector& robot_pose)
{
    if (!is_initialized) {
        std::cerr << "Error: El sistema no está inicializado. Llame a initialize() primero." << std::endl;
        return 0;
    }
    
    try {
        // Obtener el próximo número de archivo disponible
        unsigned int file_number = getNextFileNumber();
        
        // Capturar imagen
        vpImage<unsigned char> image;
        camera.acquire(image);
        
        // Generar nombres de archivos
        std::stringstream ss_img, ss_pos;
        ss_img << output_folder << "franka_image-" << file_number << ".png";
        ss_pos << output_folder << "franka_pose_rPe_" << file_number << ".yaml";
        
        std::string img_filename = ss_img.str();
        std::string pose_filename = ss_pos.str();
        
        // Guardar imagen
        vpImageIo::write(image, img_filename);
        
        // Guardar pose
        robot_pose.saveYAML(pose_filename, robot_pose);
        
        std::cout << "Frame " << file_number << " saved:" << std::endl;
        std::cout << "  - Image: " << img_filename << std::endl;
        std::cout << "  - Pose: " << pose_filename << std::endl;
        
        return file_number;
        
    } catch (const vpException& e) {
        std::cerr << "Error de ViSP durante la captura: " << e.what() << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error durante la captura: " << e.what() << std::endl;
        return 0;
    }
}



void FrankaCalibCapture::setOutputFolder(const std::string& new_folder)
{
    output_folder = new_folder;
    // Asegurar que el directorio termine con '/'
    if (!output_folder.empty() && output_folder.back() != '/') {
        output_folder += "/";
    }
}

#endif // defined(VISP_HAVE_REALSENSE2) && defined(VISP_HAVE_FRANKA) && defined(VISP_HAVE_PUGIXML)