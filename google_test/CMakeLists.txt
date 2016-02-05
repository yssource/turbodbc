add_custom_command(
	OUTPUT googletest_built
	COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/build_google_test.sh
	COMMAND touch googletest_built
)
add_custom_target(googletest_build ALL DEPENDS googletest_built)

set(ENV{GOOGLETEST_LIB_DIR} "${CMAKE_CURRENT_BINARY_DIR}/dist/lib")
set(ENV{GOOGLETEST_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/dist/include)
