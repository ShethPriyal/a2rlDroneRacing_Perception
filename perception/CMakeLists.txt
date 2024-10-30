cmake_minimum_required(VERSION 3.8)
project(perception)

# Set C++ standard and compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(drone_racing_msgs REQUIRED)

# Add C++ executable
add_executable(perception_node src/perception_node.cpp)
target_include_directories(perception_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(perception_node PUBLIC c_std_99 cxx_std_17)

# Install C++ executable
install(TARGETS perception_node
  DESTINATION lib/${PROJECT_NAME})

# Install Python modules (detector_node.py assumed to be inside the src/ directory)
install(
  DIRECTORY src/
  DESTINATION lib/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

install(
  DIRECTORY model/
  DESTINATION lib/${PROJECT_NAME}/Detector/models
  USE_SOURCE_PERMISSIONS
)

# Uncomment if you have launch files to install
 install(
  DIRECTORY launch/
   DESTINATION share/${PROJECT_NAME}/launch
   USE_SOURCE_PERMISSIONS
 )

# Ensure the Python package has access to drone_racing_msgs
ament_python_install_package(${PROJECT_NAME})

# Build testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_lint_auto_FOUND TRUE)  # Skip cpplint and copyright check
  ament_lint_auto_find_test_dependencies()
endif()

# Mark this package as ament-based
ament_package()
