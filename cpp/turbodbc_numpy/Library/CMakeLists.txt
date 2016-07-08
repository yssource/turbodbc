file(GLOB_RECURSE LIBRARY_FILES "src/*.cpp")

add_library(turbodbc_numpy ${LIBRARY_FILES})

set_target_properties(turbodbc_numpy PROPERTIES 
    PREFIX ""
    INSTALL_RPATH "../../.."
)

add_dependencies(turbodbc_numpy
    refresh_cmake_configuration
)

target_link_libraries(turbodbc_numpy
	${Boost_LIBRARIES}
	${Odbc_LIBRARIES}
	${PYTHON_LIBRARIES}
	cpp_odbc
	turbodbc_intern
)

add_custom_command(TARGET turbodbc_numpy POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy
                   ${CMAKE_CURRENT_BINARY_DIR}/turbodbc_numpy.so ${CMAKE_BINARY_DIR})

install(
    TARGETS turbodbc_numpy
    LIBRARY DESTINATION lib
)

install(
	DIRECTORY turbodbc_numpy
	DESTINATION include
	FILES_MATCHING PATTERN "*.h"
	PATTERN ".git" EXCLUDE
)

install(
	DIRECTORY src/
	DESTINATION src/turbodbc_numpy
	FILES_MATCHING PATTERN "*.cpp"
	PATTERN ".git" EXCLUDE
)
