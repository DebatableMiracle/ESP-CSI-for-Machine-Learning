
# The following lines of boilerplate have to be in your project's CMakeLists
# in this exact order for cmake to work correctlycmake_minimum_required(VERSION 3.5)
cmake_minimum_required(VERSION 3.5)
add_compile_options(-fdiagnostics-color=always)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

string(REGEX REPLACE ".*/\(.*\)" "\\1" CURDIR ${CMAKE_CURRENT_SOURCE_DIR})
project(esp_bt)

git_describe(PROJECT_VERSION ${COMPONENT_DIR})
message("Project commit: " ${PROJECT_VERSION})
