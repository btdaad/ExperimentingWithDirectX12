cmake_minimum_required( VERSION 3.10.1 )

add_library( extern INTERFACE )

target_include_directories( extern
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)

set( RESOURCE_FILES 
    Resources/shiba/scene.gltf
    Resources/shiba/textures/default_baseColor.png
)

source_group( "Resources\\Icon" FILES ${RESOURCE_FILES} )
set_source_files_properties(
    ${RESOURCE_FILES}
    PROPERTIES HEADER_FILE_ONLY TRUE
)

