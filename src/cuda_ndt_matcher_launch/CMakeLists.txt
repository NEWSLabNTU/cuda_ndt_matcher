cmake_minimum_required(VERSION 3.10)
project(cuda_ndt_matcher_launch)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # Every value shared with Autoware's ndt_scan_matcher is Autoware's.
  ament_add_pytest_test(test_stock_defaults test/test_stock_defaults.py)
endif()

ament_package()
