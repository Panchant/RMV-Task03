cmake_minimum_required(VERSION 3.0.0)
project(task)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
# 查找 Eigen3 包
find_package(Eigen3 REQUIRED)
# 查找 Ceres Solver 包
find_package(Ceres REQUIRED)

aux_source_directory(./src ALL_SRCS)
include_directories(include ${OpenCV_INCLUDE_DIRS})
# 包含 Eigen3 头文件
include_directories(${EIGEN3_INCLUDE_DIR})

# 包含 Ceres 头文件
include_directories(${CERES_INCLUDE_DIRS})


add_executable(task main.cpp ${ALL_SRCS})
target_link_libraries(task ${OpenCV_LIBS})
# 链接 Ceres 库
target_link_libraries(task ${CERES_LIBRARIES})