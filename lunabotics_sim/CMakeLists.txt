cmake_minimum_required(VERSION 3.8)
project(lunabotics_sim)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(rock_crater_detector_node
  src/rock_crater_detector_node.cpp
)

target_include_directories(rock_crater_detector_node PRIVATE
  include
  ${EIGEN3_INCLUDE_DIR}
)

ament_target_dependencies(rock_crater_detector_node
  rclcpp
  sensor_msgs
  std_msgs
  visualization_msgs
  cv_bridge
  OpenCV
)

install(TARGETS
  rock_crater_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/collect_camera_images.py
  scripts/live_camera_view.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  worlds
  models
  urdf
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
