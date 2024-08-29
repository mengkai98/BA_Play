cmake_minimum_required(VERSION 3.0)
project(Optimation_Test)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
include_directories(
    ./
    ${EIGEN3_INCLUDE_DIRS}
    ${Ceres_INCLUDE_DIRS}
)

add_executable(ot main.cpp BASolver.cpp)
target_link_libraries(ot Ceres::ceres  )
