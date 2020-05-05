cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS}")

project(camera_fusion)

find_package(OpenCV 4.1 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executable for top view point coloring exercise
add_executable (q61_show_lidar_top_view src/quizzes/q61_show_lidar_top_view.cpp src/quizzes/structIO.cpp)
target_link_libraries (q61_show_lidar_top_view ${OpenCV_LIBRARIES})

# Executable for point projection exercise
add_executable (q62_project_lidar_to_camera src/quizzes/q62_project_lidar_to_camera.cpp src/quizzes/structIO.cpp)
target_link_libraries (q62_project_lidar_to_camera ${OpenCV_LIBRARIES})

# Executable for object detection with YOLO exercise
add_executable (q63_detect_objects src/quizzes/q63_detect_objects.cpp)
target_link_libraries (q63_detect_objects ${OpenCV_LIBRARIES})


# Executable for final project
add_executable (3D_object_tracking src/camFusion_Student.cpp src/FinalProject_Camera.cpp src/lidarData.cpp src/matching2D_Student.cpp src/objectDetection2D.cpp)
target_link_libraries (3D_object_tracking ${OpenCV_LIBRARIES})