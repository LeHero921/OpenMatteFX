cmake_minimum_required(VERSION 3.20)
project(MagicMaskPlugin LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform check for Apple Silicon
if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    message(FATAL_ERROR "This project requires Apple Silicon (ARM64)")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(PkgConfig REQUIRED)

# OpenFX SDK
set(OFX_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include/openfx")
if(NOT EXISTS ${OFX_INCLUDE_DIRS})
    message(FATAL_ERROR "OpenFX SDK not found at ${OFX_INCLUDE_DIRS}")
endif()

# ONNX Runtime
find_package(onnxruntime CONFIG REQUIRED)

# Apple Frameworks
find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation) # Wichtig für Metal!

set_source_files_properties(src/PluginRegistration.cpp src/InferenceEngine.cpp PROPERTIES LANGUAGE OBJCXX)

# 1. ÄNDERUNG: MODULE statt SHARED nutzen!
add_library(${PROJECT_NAME} MODULE
    src/PluginRegistration.cpp
    src/InferenceEngine.cpp
)

# 2. ÄNDERUNG: Native macOS Bundle-Struktur erzwingen
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUNDLE TRUE
    BUNDLE_EXTENSION "ofx.bundle"
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/Info.plist" # Du brauchst eine Info.plist in deinem Hauptordner!
    MACOSX_RPATH ON
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    OFX_SUPPORTS_OPENGL=0
    OFX_SUPPORTS_INTERACT=0
    OFX_SUPPORTS_MULTITHREAD=1
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${OFX_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/libOfxSupport.a
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    onnxruntime::onnxruntime
)

# Metal Shader Kompilierung
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/Resources/shaders
    COMMAND xcrun metal -c ${CMAKE_CURRENT_SOURCE_DIR}/shaders/MetalTensorConverter.metal 
                        -o ${CMAKE_CURRENT_BINARY_DIR}/MetalTensorConverter.air
                        -std=metal3.0
                        -Wno-unused-function
    COMMAND xcrun metallib ${CMAKE_CURRENT_BINARY_DIR}/MetalTensorConverter.air
                           -o ${CMAKE_CURRENT_BINARY_DIR}/Resources/shaders/MetalTensorConverter.metallib
    COMMENT "Compiling Metal shaders"
)

# 3. ÄNDERUNG: Ressourcen direkt in das Bundle kopieren lassen (anstatt über install())
# ONNX Models in Resources
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/Resources/models
    $<TARGET_BUNDLE_CONTENT_DIR:${PROJECT_NAME}>/Resources/models
)

# Shader in Resources
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_BUNDLE_CONTENT_DIR:${PROJECT_NAME}>/Resources/shaders
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/Resources/shaders/MetalTensorConverter.metallib
    $<TARGET_BUNDLE_CONTENT_DIR:${PROJECT_NAME}>/Resources/shaders/
)