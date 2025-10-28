cmake_minimum_required(VERSION 3.12)
project(g1_controller)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/ocs2/include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
  ${PROJECT_SOURCE_DIR}/../../thirdparty/cnpy        # <<< 新增: cnpy 头文件
)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread                 # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0
  z                                                  # <<< 新增: zlib（cnpy 需要）
)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# <<< 新增: 把 cnpy.cpp 一起编进库里（最简单可靠）
list(APPEND ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/../../thirdparty/cnpy/cnpy.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(g1_ctrl main.cpp)
# （可选）编一个小demo验证能读共享内存深度
add_executable(depth_ipc_demo src/vision/depth_ipc_demo.cpp)
target_link_libraries(depth_ipc_demo ${PROJECT_NAME}_lib)