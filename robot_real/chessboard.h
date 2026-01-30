#pragma once

#include <string>

namespace ChessboardPoseLib {

    /**
     * @brief Compute the chessboard poses from a dataset of images.
     *
     * @param input_pattern Pattern for the input images (e.g., "image-%d.png")
     * @param intrinsic_file XML file with camera intrinsic parameters
     * @param camera_name Name of the camera in the XML file
     * @param output_pattern Pattern for the output YAML pose files (e.g., "pose-%d.yaml")
     * @param chessboard_width Width of the chessboard (number of corners in X)
     * @param chessboard_height Height of the chessboard (number of corners in Y)
     * @param square_size Size of one square in meters
     * @return true if all poses were computed successfully, false otherwise
     */
    bool computeChessboardPosesFromDataset(
        const std::string& input_pattern,
        const std::string& intrinsic_file,
        const std::string& camera_name,
        const std::string& output_pattern,
        int chessboard_width = 9,
        int chessboard_height = 6,
        double square_size = 0.026
    );

} // namespace ChessboardPoseLib
