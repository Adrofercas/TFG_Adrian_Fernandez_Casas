#pragma once

#include <string>
#include <vector>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpPoseVector.h>
#include <visp3/vision/vpHandEyeCalibration.h>

class EyeInHandCalibration {
public:
    EyeInHandCalibration(const std::string &data_path = "./",
                         const std::string &rPe_pattern = "pose_rPe_%d.yaml",
                         const std::string &cPo_pattern = "pose_cPo_%d.yaml");

    void loadData();
    bool calibrate();
    void savePose(const vpHomogeneousMatrix &matrix, const std::string &filename);

    vpHomogeneousMatrix getCameraPose() const { return eMc; }
    vpHomogeneousMatrix getObjectPose() const { return rMo; }

private:
    std::string data_path;
    std::string rPe_pattern;
    std::string cPo_pattern;

    std::vector<vpHomogeneousMatrix> rMe;
    std::vector<vpHomogeneousMatrix> cMo;

    vpHomogeneousMatrix eMc;
    vpHomogeneousMatrix rMo;
};
