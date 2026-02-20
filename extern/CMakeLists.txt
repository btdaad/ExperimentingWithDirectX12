cmake_minimum_required( VERSION 3.10.1 )

add_library( extern INTERFACE )

add_subdirectory( DirectXTex )

set_target_properties(DirectXTex PROPERTIES FOLDER "extern/DirectXTex")
target_include_directories( extern
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)

set( RESOURCE_FILES 
    Resources/shiba/scene.gltf
    Resources/shiba/textures/default_baseColor.png
)

source_group( "extern\\Resources" FILES ${RESOURCE_FILES} )
set_source_files_properties(
    ${RESOURCE_FILES}
    PROPERTIES HEADER_FILE_ONLY TRUE
)

get_property(DIRECTXTEX_TARGETS DIRECTORY DirectXTex PROPERTY BUILDSYSTEM_TARGETS)
foreach(target ${DIRECTXTEX_TARGETS})
    set_target_properties(${target} PROPERTIES FOLDER "extern/DirectXTex")
endforeach()