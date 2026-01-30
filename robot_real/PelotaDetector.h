#ifndef PELOTA_DETECTOR_H
#define PELOTA_DETECTOR_H

#include <visp3/core/vpImage.h>
#include <visp3/core/vpImagePoint.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <deque> // <--- NUEVO

class PelotaDetector {
private:
    cv::dnn::Net net;
    std::vector<std::string> outputNames;
    float confThreshold;
    float nmsThreshold;

    // --- NUEVO: BUFFER PARA EL FILTRO DE MEDIANA ---
    // Guardamos un historial de vectores de esquinas
    // std::deque es más eficiente que vector para quitar elementos del principio (FIFO)
    std::deque<std::vector<vpImagePoint>> history;
    const size_t historySize = 30; // Tamaño de la ventana (15 muestras)

    // Función auxiliar para calcular la mediana de un vector de doubles
    double getMedian(std::vector<double>& values);

public:
    PelotaDetector(const std::string& modelConfig, const std::string& modelWeights, float conf = 0.5, float nms = 0.4);
    
    // La firma de la función no cambia, pero internamente hará magia
    bool detectPelota(vpImage<vpRGBa> &I, std::vector<vpImagePoint> &corners);
};

#endif