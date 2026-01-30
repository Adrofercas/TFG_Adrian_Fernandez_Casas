#include "eye-in-hand.h"
#include <visp3/core/vpIoTools.h>
#include <iostream>
#include <map>
#include <fstream>

EyeInHandCalibration::EyeInHandCalibration(const std::string &data_path,
                                           const std::string &rPe_pattern,
                                           const std::string &cPo_pattern)
    : data_path(data_path), rPe_pattern(rPe_pattern), cPo_pattern(cPo_pattern) {}

void EyeInHandCalibration::loadData() {
    rMe.clear();
    cMo.clear();

    std::map<long, std::string> map_rPe_files;
    std::map<long, std::string> map_cPo_files;
    std::vector<std::string> files = vpIoTools::getDirFiles(data_path);

    for (auto &file : files) {
        long index_rPe = vpIoTools::getIndex(file, rPe_pattern);
        long index_cPo = vpIoTools::getIndex(file, cPo_pattern);
        if (index_rPe != -1) map_rPe_files[index_rPe] = file;
        if (index_cPo != -1) map_cPo_files[index_cPo] = file;
    }

    if (map_rPe_files.empty() || map_cPo_files.empty()) {
        throw std::runtime_error("No data files found for calibration.");
    }

    for (auto it_rPe = map_rPe_files.begin(); it_rPe != map_rPe_files.end(); ++it_rPe) {
        auto it_cPo = map_cPo_files.find(it_rPe->first);
        if (it_cPo != map_cPo_files.end()) {
            vpPoseVector rPe;
            if (!rPe.loadYAML(vpIoTools::createFilePath(data_path, it_rPe->second), rPe)) continue;

            vpPoseVector cPo;
            if (!cPo.loadYAML(vpIoTools::createFilePath(data_path, it_cPo->second), cPo)) continue;

            rMe.push_back(vpHomogeneousMatrix(rPe));
            cMo.push_back(vpHomogeneousMatrix(cPo));
        }
    }

    if (rMe.size() < 3) {
        throw std::runtime_error("Not enough data pairs found.");
    }
}

bool EyeInHandCalibration::calibrate() {
    int ret = vpHandEyeCalibration::calibrate(cMo, rMe, eMc, rMo);
    return ret == 0;
}

void EyeInHandCalibration::savePose(const vpHomogeneousMatrix &matrix, const std::string &filename) {
    vpPoseVector pose_vec(matrix);
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    pose_vec.saveYAML(filename, pose_vec);
    std::cout << "Saved pose to " << filename << std::endl;
}
