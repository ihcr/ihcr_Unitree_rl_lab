cmake_minimum_required(VERSION 3.12)
project(g1_controller CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # 保险一点

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

# --- cnpy 静态库 ---
add_library(cnpy STATIC
  ${CMAKE_CURRENT_LIST_DIR}/../../thirdparty/cnpy/cnpy.cpp
)
target_include_directories(cnpy PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/../../thirdparty/cnpy
)
target_link_libraries(cnpy PUBLIC z)   # 需要 zlib

# --- 你的业务库 ---
file(GLOB ADD_SRC_LIST
  ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)
add_library(g1_controller_lib ${ADD_SRC_LIST})
target_include_directories(g1_controller_lib PRIVATE
  /usr/include/eigen3
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  include
  ${CMAKE_CURRENT_LIST_DIR}/../../thirdparty/ocs2/include
  ${CMAKE_CURRENT_LIST_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${CMAKE_CURRENT_LIST_DIR}/../../include
  ${CMAKE_CURRENT_LIST_DIR}/../../thirdparty/cnpy
)
target_link_libraries(g1_controller_lib
  PUBLIC
    unitree_sdk2 ddsc ddscxx rt pthread
    Boost::program_options
    yaml-cpp
    ${CMAKE_CURRENT_LIST_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so
  PRIVATE
    cnpy    # ← 把 cnpy 链接到你的库
)

# --- 可执行文件 ---
add_executable(g1_ctrl main.cpp)
target_link_libraries(g1_ctrl PRIVATE g1_controller_lib cnpy)  # ← 也显式连一次 cnpy