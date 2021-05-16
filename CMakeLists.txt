# cmake 版本
cmake_minimum_required(VERSION 3.1)
# 项目名
project(trt_proj)
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    MESSAGE("\n===This is Linux,CMAKE_SYSTEM_NAME:${CMAKE_SYSTEM_NAME}")
ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Windows")
    MESSAGE("===This is Windows,CMAKE_SYSTEM_NAME:${CMAKE_SYSTEM_NAME}")
ELSE()
    MESSAGE("===This platform:${CMAKE_SYSTEM_NAME}")
ENDIF()

# 编译增加C++11 支持
add_definitions(-std=c++11)
# 设置C++标准为 C++ 11
set(CMAKE_CXX_STANDARD 11)
#指定生成的版本
set(CMAKE_BUILD_TYPE Debug)
#设置可执行文件输出目录
#SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
#设置库输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib)

#option 选项开关
#启用后，将在中使用CUDA运行时库的静态版本CUDA_LIBRARIES。
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# -D_MWAITXINTRIN_H_INCLUDED for solving error: identifier "__builtin_ia32_mwaitx" is undefined
##-Wall  #-Wfatal-errors 
#指定编译选项
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Ofast -D_MWAITXINTRIN_H_INCLUDED")

include_directories(${PROJECT_SOURCE_DIR}/include/common/)
include_directories(${PROJECT_SOURCE_DIR}/include/yolov4/)
include_directories(${PROJECT_SOURCE_DIR}/include/yolov5/)
include_directories(${PROJECT_SOURCE_DIR}/)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
    # setup CUDA
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
	
    # tensorrt
    include_directories(/usr/include/aarch64-linux-gnu/)
    link_directories(/usr/lib/aarch64-linux-gnu/)
    # setup opencv
    find_package(OpenCV)
	include_directories(${OpenCV_INCLUDE_DIRS})
ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Windows")
	# setup CUDA
	set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11; -g; -G;-gencode; arch=compute_75;code=sm_75)
	find_package(CUDA REQUIRED)
	include_directories(${CUDA_INCLUDE_DIRS})
	enable_language(CUDA)  # add this line, then no need to setup cuda path in vs
	
	# tensorrt
	set(TRT_DIR "C://tools//TensorRT-7.0.0.11//")  #3
	include_directories(${TRT_DIR}\\include)
	link_directories(${TRT_DIR}\\lib)
	
	# setup opencv
	set(OpenCV_DIR "C:/tools/opencv/build/")  #2
    find_package(OpenCV REQUIRED)
	include_directories(${OpenCV_INCLUDE_DIRS})
	# Threads
	set(THREADS_PREFER_PTHREAD_FLAG ON)
	find_package(Threads REQUIRED)
	message(STATUS "Threads library status: ")
	message(STATUS "    Threads_FOUND: ${Threads_FOUND}")
	message(STATUS "    include path: ${Threads_FIND_QUIETLY}")
ENDIF()

#打印信息
message("===message=========================================")
message(STATUS "CUDA library status:")
message(STATUS "    CUDA_FOUND: ${CUDA_FOUND}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "OpenCV library status: ")
message(STATUS "    OpenCV_FOUND: ${OpenCV_FOUND}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message("===message=========================================")

# common/
FILE(GLOB COMMON_INC_FILES "include/common/*.h")
# yolov4.so libraries
set(YOLOV4_LIB_NAME "yolov4")
FILE(GLOB YOLOV4_SRC_FILES "src/yolov4/*.cpp")
FILE(GLOB YOLOV4_SRC_CU_FILES "src/yolov4/*.cu")
FILE(GLOB YOLOV4_INC_FILES "include/yolov4/*.h")
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
	cuda_add_library(${YOLOV4_LIB_NAME} SHARED ${YOLOV4_SRC_FILES}${YOLOV4_SRC_CU_FILES} ${YOLOV4_INC_FILES} ${COMMON_INC_FILES})
	add_definitions(-O2 -pthread)
ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Windows")
	add_library(${YOLOV4_LIB_NAME} ${YOLOV4_SRC_FILES} ${YOLOV4_SRC_CU_FILES} ${YOLOV4_INC_FILES} ${COMMON_INC_FILES})
	target_link_libraries(${YOLOV4_LIB_NAME} nvinfer cudart nvinfer_plugin Threads::Threads)
ENDIF()


# yolov5.so libraries
set(YOLOV5_LIB_NAME "yolov5")
FILE(GLOB YOLOV5_SRC_FILES "src/yolov5/*.cpp")
FILE(GLOB YOLOV5_SRC_CU_FILES "src/yolov5/*.cu")
FILE(GLOB YOLOV5_INC_FILES "include/yolov5/*.h")
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
	cuda_add_library(${YOLOV5_LIB_NAME} SHARED ${YOLOV5_SRC_FILES} ${YOLOV5_SRC_CU_FILES} ${YOLOV5_INC_FILES} ${COMMON_INC_FILES})
	add_definitions(-O2 -pthread)
ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Windows")
	add_library(${YOLOV5_LIB_NAME} ${YOLOV5_SRC_FILES} ${YOLOV5_SRC_CU_FILES} ${YOLOV5_INC_FILES} ${COMMON_INC_FILES})
	target_link_libraries(${YOLOV5_LIB_NAME} nvinfer cudart nvinfer_plugin Threads::Threads)
ENDIF()
#生成目标文件
add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/main.cpp)

target_link_libraries(${PROJECT_NAME} ${YOLOV4_LIB_NAME})
target_link_libraries(${PROJECT_NAME} ${YOLOV5_LIB_NAME})
target_link_libraries(${PROJECT_NAME} nvinfer cudart ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} Threads::Threads)
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
	add_definitions(-O2 -pthread)
ELSEIF(CMAKE_SYSTEM_NAME MATCHES "Windows")
ENDIF()

