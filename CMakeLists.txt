cmake_minimum_required(VERSION 3.10)

project(visp_cmake_project)

set(CMAKE_CXX_STANDARD 14)

find_package(pinocchio REQUIRED)
find_package(VISP REQUIRED)
find_package(Franka REQUIRED)

find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)

add_library(examples_common STATIC
  src/examples_common.cpp
)
target_link_libraries(examples_common PUBLIC Franka::Franka)

add_library(acquire-data STATIC
  src/acquire-data.cpp
)
target_link_libraries(acquire-data PUBLIC Franka::Franka ${VISP_LIBRARIES})

add_library(eye-in-hand STATIC
  src/eye-in-hand.cpp
)
target_link_libraries(eye-in-hand PUBLIC ${VISP_LIBRARIES})

add_library(chessboard STATIC
  src/chessboard.cpp
)
target_link_libraries(chessboard PUBLIC ${VISP_LIBRARIES})


add_library(cartesianSave STATIC
  src/cartesianSave.cpp
)
target_link_libraries(cartesianSave PUBLIC Franka::Franka examples_common)

add_library(PelotaDetector STATIC
  src/PelotaDetector.cpp
)
target_link_libraries(PelotaDetector PUBLIC ${VISP_LIBRARIES} ${OpenCV_LIBS})

include_directories(${VISP_INCLUDE_DIRS}
                    ${PINOCCHIO_INCLUDE_DIRS})

add_executable(go_home src/go_home.cpp)
add_executable(control_cartesiano src/controlFranka.cpp)
add_executable(control_impedancia src/control_impedancia.cpp)
add_executable(calibracion_joint src/calibracion_joint.cpp)
add_executable(saveVideo_joint src/saveVideo_joint.cpp)
add_executable(saveVideo src/saveVideo.cpp)
add_executable(moveTrajectory src/moveTrajectory.cpp)
add_executable(visual_servoing_apriltag src/visual_servoing_apriltag.cpp)
add_executable(visual_servoing_pelota_rgb src/visual_servoing_pelota_rgb.cpp)
find_package(VISP REQUIRED COMPONENTS core gui sensor vision dnn_tracker)
add_executable(visual_servoing_pelota_yolo src/visual_servoing_pelota_yolo.cpp)
add_executable(moveJointTrajectory src/moveJointTrajectory.cpp)
add_executable(visual_servoing_pelota_eventos_cliente src/visual_servoing_pelota_eventos_cliente.cpp)

target_link_libraries(go_home ${VISP_LIBRARIES})
target_link_libraries(control_cartesiano Franka::Franka examples_common pinocchio::pinocchio)
target_link_libraries(control_impedancia Franka::Franka examples_common pinocchio::pinocchio)
target_link_libraries(calibracion_joint Franka::Franka examples_common acquire-data eye-in-hand chessboard pinocchio::pinocchio)
target_link_libraries(saveVideo_joint ${realsense2_LIBRARY} ${OpenCV_LIBS} Franka::Franka examples_common)
target_link_libraries(saveVideo ${realsense2_LIBRARY} ${OpenCV_LIBS} Franka::Franka examples_common cartesianSave)
target_link_libraries(moveTrajectory ${realsense2_LIBRARY} ${OpenCV_LIBS} Franka::Franka examples_common cartesianSave)
target_link_libraries(visual_servoing_apriltag ${VISP_LIBRARIES})
target_link_libraries(visual_servoing_pelota_rgb ${VISP_LIBRARIES})
target_link_libraries(prueba_det_pelota ${VISP_LIBRARIES})
target_link_libraries(visual_servoing_pelota_yolo PelotaDetector ${VISP_LIBRARIES})
target_link_libraries(moveJointTrajectory Franka::Franka examples_common pinocchio::pinocchio)
target_link_libraries(visual_servoing_pelota_eventos_cliente ${VISP_LIBRARIES})