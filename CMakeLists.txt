cmake_minimum_required(VERSION 3.12)
project(LBM_Particle)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/imgui
    ${PROJECT_SOURCE_DIR}/src
)

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)
find_package(Ktx REQUIRED)

find_program(GLSLANG_VALIDATOR_EXECUTABLE NAMES glslangValidator glslangValidator.exe)
if(NOT GLSLANG_VALIDATOR_EXECUTABLE)
    message(FATAL_ERROR "glslangValidator not found.")
endif()

set(SHADERS_SOURCE_DIR ${PROJECT_SOURCE_DIR}/shaders)

file(GLOB SHADER_HEADER_FILES CONFIGURE_DEPENDS "${SHADERS_SOURCE_DIR}/*.glsl")

file(GLOB_RECURSE SHADER_COMPILE_FILES CONFIGURE_DEPENDS
    "${SHADERS_SOURCE_DIR}/*.comp"
    "${SHADERS_SOURCE_DIR}/*.vert"
    "${SHADERS_SOURCE_DIR}/*.frag"
)

set(COMPILED_SHADERS)
foreach(SHADER_FILE IN LISTS SHADER_COMPILE_FILES)
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER_FILE} EXT)
    get_filename_component(SHADER_DIR ${SHADER_FILE} DIRECTORY)
    
    file(RELATIVE_PATH SHADER_REL_DIR "${SHADERS_SOURCE_DIR}" "${SHADER_DIR}")
    
    string(REPLACE "." "" SHADER_EXT_NO_DOT ${SHADER_EXT})

    set(OUTPUT_SPV_FILE "${SHADERS_SOURCE_DIR}/${SHADER_REL_DIR}/${SHADER_NAME}_${SHADER_EXT_NO_DOT}.spv")

    add_custom_command(
        OUTPUT ${OUTPUT_SPV_FILE}
        COMMAND ${GLSLANG_VALIDATOR_EXECUTABLE} -g0 -gVS -V -I"${SHADERS_SOURCE_DIR}" -o "${OUTPUT_SPV_FILE}" "${SHADER_FILE}"
        DEPENDS 
            ${SHADER_FILE} 
            ${SHADER_HEADER_FILES}
        COMMENT "Compiling shader ${SHADER_FILE}"
    )
    list(APPEND COMPILED_SHADERS ${OUTPUT_SPV_FILE})
endforeach()

add_custom_target(shaders_target ALL DEPENDS ${COMPILED_SHADERS})

file(GLOB_RECURSE APP_SRC_FILES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/*.h"
    "${PROJECT_SOURCE_DIR}/src/*.hpp"
)

file(GLOB_RECURSE APP_INC_FILES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/*.h"
    "${PROJECT_SOURCE_DIR}/include/*.hpp"
)
list(FILTER APP_INC_FILES EXCLUDE REGEX ".*include/imgui/.*")

file(GLOB IMGUI_SOURCE_FILES CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/imgui/*.cpp"
    "${PROJECT_SOURCE_DIR}/include/imgui/*.h"
)

set(ALL_FILES 
    ${APP_SRC_FILES} 
    ${APP_INC_FILES}
    ${IMGUI_SOURCE_FILES}
    ${SHADER_HEADER_FILES}
    ${SHADER_COMPILE_FILES}
)

add_executable(LBM_Particle ${ALL_FILES})

if(MSVC)
    add_compile_options(/Zi) 
    
    add_compile_options(/Od) 
    
    set_target_properties(${PROJECT_NAME} PROPERTIES 
        LINK_FLAGS_DEBUG "/INCREMENTAL:NO /DEBUG"
    )
    
    # (可选) 启用静态运行时，避免缺少 vcruntime140d.dll 等问题
    # set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDebug")
endif()

add_dependencies(LBM_Particle shaders_target)

if(APP_SRC_FILES)
    source_group(TREE "${PROJECT_SOURCE_DIR}/src" PREFIX "Source Files" FILES ${APP_SRC_FILES})
endif()
if(APP_INC_FILES)
    source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${APP_INC_FILES})
endif()

source_group("ImGui" FILES ${IMGUI_SOURCE_FILES})

if(SHADER_COMPILE_FILES)
    source_group(TREE "${PROJECT_SOURCE_DIR}/shaders" PREFIX "Shaders" FILES ${SHADER_COMPILE_FILES})
endif()
if(SHADER_HEADER_FILES)
    source_group("Shaders" FILES ${SHADER_HEADER_FILES})
endif()

target_link_libraries(LBM_Particle PRIVATE
    glfw
    Vulkan::Vulkan
    KTX::ktx
)

set_property(TARGET LBM_Particle PROPERTY RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
set_property(TARGET LBM_Particle PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
