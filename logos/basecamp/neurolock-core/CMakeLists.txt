cmake_minimum_required(VERSION 3.14)
project(NeurolockPopPlugin LANGUAGES CXX)

if(DEFINED ENV{LOGOS_MODULE_BUILDER_ROOT})
    include($ENV{LOGOS_MODULE_BUILDER_ROOT}/cmake/LogosModule.cmake)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/LogosModule.cmake")
    include(cmake/LogosModule.cmake)
else()
    message(FATAL_ERROR "LogosModule.cmake not found")
endif()

find_package(Qt6 REQUIRED COMPONENTS Network)

logos_module(
    NAME neurolock_pop
    SOURCES
        src/neurolock_pop_interface.h
        src/neurolock_pop_plugin.h
        src/neurolock_pop_plugin.cpp
)
target_link_libraries(neurolock_pop_module_plugin PRIVATE Qt6::Network)
