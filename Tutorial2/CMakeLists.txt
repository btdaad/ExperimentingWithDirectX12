cmake_minimum_required( VERSION 3.10.1 ) # Latest version of CMake when this file was created.

set( HEADER_FILES
    inc/Camera.h
    inc/Tutorial2.h
)

set( SRC_FILES
    src/main.cpp
    src/tinygltf_impl.cpp
    src/Camera.cpp
    src/Tutorial2.cpp
)

set( SHADER_FILES
    shaders/VertexShader.hlsl
    shaders/PixelShader.hlsl
)

source_group( "Resources\\Shaders" FILES ${SHADER_FILES} )

set_source_files_properties( shaders/VertexShader.hlsl PROPERTIES 
    VS_SHADER_TYPE Vertex
    VS_SHADER_MODEL 5.1
)

set_source_files_properties( shaders/PixelShader.hlsl PROPERTIES 
    VS_SHADER_TYPE Pixel
    VS_SHADER_MODEL 5.1
)

add_executable( Tutorial2 WIN32
    ${HEADER_FILES} 
    ${SRC_FILES}
    ${SHADER_FILES}
)

add_custom_command(
    TARGET Tutorial2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/extern/Resources
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/Resources
)

target_include_directories( Tutorial2
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_link_libraries( Tutorial2
    DX12Lib
    D3DCompiler.lib
    Shlwapi.lib
    extern
)
