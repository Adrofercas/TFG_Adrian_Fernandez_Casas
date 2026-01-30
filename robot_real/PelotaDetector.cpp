#include "PelotaDetector.h"
#include <visp3/core/vpImageConvert.h>
#include <algorithm>

using namespace cv;
using namespace cv::dnn;
using namespace std;

PelotaDetector::PelotaDetector(const string& modelConfig, const string& modelWeights, float conf, float nms) 
    : confThreshold(conf), nmsThreshold(nms) 
{
    // 1. Cargar la red (solo ocurre al inicio)
    net = readNetFromDarknet(modelConfig, modelWeights);
    
    // Opcional: Activar backend optimizado (CPU o CUDA)
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    
    outputNames = net.getUnconnectedOutLayersNames();
}

double PelotaDetector::getMedian(std::vector<double>& values) {
    if (values.empty()) return 0.0;
    size_t n = values.size();
    std::sort(values.begin(), values.end());
    if (n % 2 == 0) {
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;
    } else {
        return values[n / 2];
    }
}

bool PelotaDetector::detectPelota(vpImage<vpRGBa> &I, vector<vpImagePoint> &corners) {
    // Limpiar vector de salida
    corners.clear();

    // 1. Convertir ViSP (vpImage<vpRGBa>) a OpenCV (cv::Mat)
    cv::Mat frame;
    vpImageConvert::convert(I, frame); // Esto crea un Mat en formato BGRA o RGBA

    // OpenCV DNN suele trabajar mejor con BGR, convertimos si es necesario
    // ViSP vpRGBa -> OpenCV (tiene canal alpha). Quitamos el canal alpha y aseguramos BGR.
    cv::cvtColor(frame, frame, cv::COLOR_RGBA2BGR);

    // 2. Preprocesamiento para YOLO
    Mat blob;
    // IMPORTANTE: swapRB=true porque pasamos de BGR a RGB que es lo que espera YOLO
    blobFromImage(frame, blob, 1/255.0, Size(416, 416), Scalar(0,0,0), true, false);
    net.setInput(blob);

    // 3. Inferencia
    vector<Mat> outs;
    net.forward(outs, outputNames);

    // 4. Procesar salidas
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // 5. Non-Maximum Suppression (NMS) para quitar duplicados
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    vector<vpImagePoint> currentCorners;

    // 6. Procesar detección cruda
    if (!indices.empty()) {
        int idx = indices[0];
        Rect box = boxes[idx];
        
        double x = box.x;
        double y = box.y;
        double w = box.width;
        double h = box.height;

        // Definir las 4 esquinas crudas (Raw corners)
        currentCorners.push_back(vpImagePoint(y + h, x));     // BL
        currentCorners.push_back(vpImagePoint(y + h, x + w)); // BR
        currentCorners.push_back(vpImagePoint(y, x + w));     // TR
        currentCorners.push_back(vpImagePoint(y, x));         // TL

        // --- LÓGICA DEL FILTRO DE MEDIANA ---
        
        // Añadir al historial
        history.push_back(currentCorners);
        
        // Mantener tamaño fijo
        if (history.size() > historySize) {
            history.pop_front();
        }

    } else {
        // OPCIÓN A: Si no detecta nada, vaciamos historial (Reiniciar)
        // Esto evita que la pelota se quede "congelada" si desaparece de la imagen
        // history.clear();
        return false; 
    }

    // Si el historial está vacío (caso raro), salimos
    if (history.empty()) return false;

    // Calcular la mediana para CADA coordenada de CADA esquina
    // Tenemos 4 esquinas, cada una con (u, v) -> 8 valores a filtrar
    corners.clear();
    
    // Iterar sobre las 4 esquinas (0 a 3)
    for (int i = 0; i < 4; ++i) {
        std::vector<double> u_values;
        std::vector<double> v_values;

        // Recorrer el historial para extraer los valores de la esquina 'i'
        for (const auto& sample : history) {
            u_values.push_back(sample[i].get_u());
            v_values.push_back(sample[i].get_v());
        }

        double u_median = getMedian(u_values);
        double v_median = getMedian(v_values);

        corners.push_back(vpImagePoint(v_median, u_median)); // vpImagePoint(row, col) = (v, u)
    }

    return true;
}