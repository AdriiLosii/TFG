cmake_minimum_required(VERSION 3.5)
project(aravis_driver_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(image_transport REQUIRED)

message(test "${OpenCV_INCLUDE_DIRS}")

find_package(PkgConfig REQUIRED)
pkg_check_modules(ARAVIS REQUIRED aravis-0.8)
pkg_check_modules(GLIB REQUIRED glib-2.0)

include_directories(
	include
	${catkin_INCLUDE_DIRS}
	${ARAVIS_INCLUDE_DIRS}
	)
	
link_libraries(
	${ARAVIS_LIBRARIES}
	)
	
link_directories(
	${ARAVIS_LIBRARY_DIRS}
	)

add_executable(aravis_camera_node src/aravis_camera_node.cpp)
ament_target_dependencies(aravis_camera_node
	rclcpp sensor_msgs std_msgs image_transport)

target_link_libraries(aravis_camera_node ${catkin_LIBRARIES} ${ARAVIS_LIBRARIES} ${image_transport_LIBRARIES})

target_include_directories(aravis_camera_node PUBLIC
  sensor_msgs
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(aravis_camera_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS aravis_camera_node
  DESTINATION lib/${PROJECT_NAME})
  
install(DIRECTORY launch
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
