cmake_minimum_required(VERSION 3.10)
project(cuda_ndt_matcher_launch)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
