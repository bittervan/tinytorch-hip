cmake_minimum_required(VERSION 3.26)

# （可选方案B）把 C++ 编译器切到 ROCm clang++，避免 g++ 不识别 offload 参数
if(EXISTS "/opt/rocm/llvm/bin/clang++")
  set(CMAKE_CXX_COMPILER "/opt/rocm/llvm/bin/clang++" CACHE STRING "" FORCE)
endif()

project(tinytorch_hip LANGUAGES CXX HIP)

# 让 pybind11 用 FindPython（去掉那个 CMP0148 警告）
set(PYBIND11_FINDPYTHON ON)

# 指定 HIP 架构（推荐）：避免奇怪的全局 offload flag 注入
# 你的 GPU 是 RDNA3（gfx1201），按需修改
set(CMAKE_HIP_ARCHITECTURES gfx1201)

find_package(pybind11 CONFIG REQUIRED)

# HIP 包名兼容处理
set(_HIP_TARGET "")
find_package(hip CONFIG)
if (hip_FOUND)
  set(_HIP_TARGET hip::device)
else()
  find_package(HIP)
  if (HIP_FOUND)
    set(_HIP_TARGET HIP::device)
  else()
    message(FATAL_ERROR "HIP not found; set ROCM_PATH or ensure /opt/rocm.")
  endif()
endif()

pybind11_add_module(_core MODULE
  cpp/bindings.cpp
  cpp/add_kernel.hip
)

set_source_files_properties(cpp/bindings.cpp PROPERTIES LANGUAGE CXX)
set_source_files_properties(cpp/add_kernel.hip PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

target_link_libraries(_core PRIVATE ${_HIP_TARGET})
target_compile_features(_core PRIVATE cxx_std_17)
target_compile_options(_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:HIP>:-O3>
)

# 运行时能找到 ROCm 动态库（可选，更稳）
set_target_properties(_core PROPERTIES
  BUILD_RPATH "/opt/rocm/lib"
  INSTALL_RPATH "/opt/rocm/lib"
)

install(TARGETS _core LIBRARY DESTINATION tinytorch_hip)
