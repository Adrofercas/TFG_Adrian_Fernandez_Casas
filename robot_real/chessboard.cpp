#include "chessboard.h"

#include <vector>
#include <string>
#include <iostream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <visp3/core/vpConfig.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/core/vpPixelMeterConversion.h>
#include <visp3/core/vpPoint.h>
#include <visp3/core/vpXmlParserCamera.h>
#include <visp3/io/vpVideoReader.h>
#include <visp3/vision/vpPose.h>
#include <visp3/core/vpImageConvert.h>
#include <visp3/core/vpImage.h>
#include <visp3/core/vpRGBa.h>

#if defined(ENABLE_VISP_NAMESPACE)
using namespace VISP_NAMESPACE_NAME;
#endif

namespace ChessboardPoseLib {

static void calcChessboardCorners(int width, int height, double squareSize, std::vector<vpPoint>& corners)
{
    corners.clear();
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            vpPoint pt;
            pt.set_oX(j * squareSize);
            pt.set_oY(i * squareSize);
            pt.set_oZ(0.0);
            corners.push_back(pt);
        }
    }
}

bool computeChessboardPosesFromDataset(
    const std::string& input_pattern,
    const std::string& intrinsic_file,
    const std::string& camera_name,
    const std::string& output_pattern,
    int chessboard_width,
    int chessboard_height,
    double square_size
)
{
    if (!vpIoTools::checkFilename(intrinsic_file)) {
        std::cerr << "Camera intrinsic file " << intrinsic_file << " not found.\n";
        return false;
    }

    if (input_pattern.empty()) {
        std::cerr << "Input image pattern is empty.\n";
        return false;
    }

    try {
        vpVideoReader reader;
        reader.setFileName(input_pattern);
        vpImage<vpRGBa> I;
        reader.open(I);

        std::vector<vpPoint> corners_pts;
        calcChessboardCorners(chessboard_width, chessboard_height, square_size, corners_pts);

        vpCameraParameters cam;
        vpXmlParserCamera parser;
        if (parser.parse(cam, intrinsic_file, camera_name,
                         vpCameraParameters::perspectiveProjWithDistortion) != vpXmlParserCamera::SEQUENCE_OK)
        {
            if (parser.parse(cam, intrinsic_file, camera_name,
                             vpCameraParameters::perspectiveProjWithoutDistortion) != vpXmlParserCamera::SEQUENCE_OK)
            {
                std::cerr << "Unable to parse camera parameters.\n";
                return false;
            }
        }

        do {
            reader.acquire(I);
            cv::Mat matImg;
            vpImageConvert::convert(I, matImg);

            cv::Size chessboardSize(chessboard_width, chessboard_height);
            std::vector<cv::Point2f> corners2D;
            bool found = cv::findChessboardCorners(matImg, chessboardSize, corners2D,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

            if (found) {
                cv::Mat gray;
                cv::cvtColor(matImg, gray, cv::COLOR_BGR2GRAY);
                cv::cornerSubPix(gray, corners2D, cv::Size(11,11), cv::Size(-1,-1),
                                 cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

                for (size_t i = 0; i < corners_pts.size(); ++i) {
                    vpImagePoint imPt(corners2D[i].y, corners2D[i].x);
                    double x, y;
                    vpPixelMeterConversion::convertPoint(cam, imPt, x, y);
                    corners_pts[i].set_x(x);
                    corners_pts[i].set_y(y);
                }

                vpHomogeneousMatrix cMo;
                vpPose pose;
                pose.addPoints(corners_pts);
                if (!pose.computePose(vpPose::DEMENTHON_LAGRANGE_VIRTUAL_VS, cMo)) {
                    std::cerr << "Failed to compute pose.\n";
                    return false;
                }

                // Generar nombre de archivo evitando sobrescribir
                std::string filename = vpIoTools::formatString(output_pattern, reader.getFrameIndex());
                int counter = 1;
                while (vpIoTools::checkFilename(filename)) {
                    std::ostringstream oss;
                    oss << vpIoTools::formatString(output_pattern, reader.getFrameIndex()) << "_" << counter << ".yaml";
                    filename = oss.str();
                    counter++;
                }

                vpPoseVector pose_vec(cMo);
                pose_vec.saveYAML(filename, pose_vec);
            }
        } while (!reader.end());
    }
    catch (const vpException& e) {
        std::cerr << "ViSP exception: " << e.getMessage() << std::endl;
        return false;
    }

    return true;
}

} // namespace ChessboardPoseLib
