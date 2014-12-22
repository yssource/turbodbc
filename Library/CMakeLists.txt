file(GLOB_RECURSE LIBRARY_FILES "src/*.cpp")

add_library(pydbc_intern ${LIBRARY_FILES})

set_target_properties(pydbc_intern PROPERTIES 
    PREFIX ""
    INSTALL_RPATH "../../.."
)

add_dependencies(pydbc_intern
    refresh_cmake_configuration
)

target_link_libraries(pydbc_intern
	${Boost_LIBRARIES}
	${Odbc_LIBRARIES}
	${PYTHON_LIBRARIES}
	cpp_odbc
)



install(
    TARGETS pydbc_intern
    LIBRARY DESTINATION lib
)

install(
	DIRECTORY pydbc_intern
	DESTINATION include
	FILES_MATCHING PATTERN "*.h"
	PATTERN ".git" EXCLUDE
)

add_custom_command(TARGET pydbc_intern POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                   ${CMAKE_CURRENT_BINARY_DIR}/pydbc_intern.so ${CMAKE_BINARY_DIR}/pydbc)
