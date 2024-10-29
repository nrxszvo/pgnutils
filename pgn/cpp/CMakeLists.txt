cmake_minimum_required(VERSION 3.22)
project(PgnProcessor)
add_executable(downloadAndParse downloadAndParse.cpp)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_subdirectory(lib)
target_link_libraries(downloadAndParse PUBLIC PgnParser)
target_include_directories(downloadAndParse PUBLIC "${PROJECT_SOURCE_DIR}/lib")

