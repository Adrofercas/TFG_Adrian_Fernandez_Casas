#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cv;
using namespace dnn;

int main() {
    // ---------------------------------------------------------
    // 1. CONFIGURACIÓN: CAMBIAMOS A LOS ARCHIVOS DE TU MODELO
    // ---------------------------------------------------------
    // Asegúrate de que estos archivos estén en la carpeta de tu proyecto
    string modelConfig = "src/modelo_yolo/yolov4-custom/yolov4-tiny-custom.cfg";
    string modelWeights = "src/modelo_yolo/yolov4-custom/yolov4-tiny-custom_best.weights";
    string classesFile = "src/modelo_yolo/yolov4-custom/obj.names";

    // 2. Cargar nombres de clases
    vector<string> classNames;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classNames.push_back(line);

    // 3. Cargar la red neuronal
    // Nota: Si usas GPU, descomenta las líneas de CUDA abajo
    Net net = readNetFromDarknet(modelConfig, modelWeights);
    
    // net.setPreferableBackend(DNN_BACKEND_CUDA);
    // net.setPreferableTarget(DNN_TARGET_CUDA);
    // O si usas CPU:
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // 4. Iniciar captura de video
    // OJO: Mantengo tu índice '4', cámbialo a '0' si usas la webcam integrada
    VideoCapture cap(4); 
    if (!cap.isOpened()) {
        cerr << "No se pudo abrir la cámara" << endl;
        return -1;
    }

    // --- CONFIGURACIÓN DE UMBRALES ---
    // Como tu modelo tiene un IOU de 0.96, podemos ser más exigentes con la confianza
    float confThreshold = 0.7; // Confianza mínima del 50%
    float nmsThreshold = 0.4;  // Umbral para limpiar cajas duplicadas

    Mat frame, blob;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // YOLOv4-tiny suele usar 416x416. Si cambiaste esto en el .cfg, cámbialo aquí también.
        blobFromImage(frame, blob, 1/255.0, Size(416, 416), Scalar(0,0,0), true, false);
        net.setInput(blob);

        vector<Mat> outs;
        net.forward(outs, net.getUnconnectedOutLayersNames());

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

                    // Como solo hay una clase, guardamos todo
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }

        vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Rect box = boxes[idx];
            
            // Dibujar rectángulo verde
            rectangle(frame, box, Scalar(0, 255, 0), 2);
            
            // Etiqueta: "pelota: 0.98"
            string label = format("%.2f", confidences[idx]);
            if (!classNames.empty()) {
                label = classNames[classIds[idx]] + ": " + label;
            }
            
            putText(frame, label, Point(box.x, box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }

        imshow("Detector Custom YOLO", frame);
        if (waitKey(1) == 27) break; 
    }

    return 0;
}