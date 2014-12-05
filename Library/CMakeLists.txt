file(GLOB_RECURSE LIBRARY_FILES "src/*.cpp")

add_library(pydbc ${LIBRARY_FILES})

add_dependencies(pydbc
    refresh_cmake_configuration
)

target_link_libraries(pydbc
	${Boost_LIBRARIES}
	${Odbc_LIBRARIES}
	${Psapp_core_LIB}
)


install(
    TARGETS pydbc
    LIBRARY DESTINATION lib
)

install(
	DIRECTORY pydbc
	DESTINATION include
	FILES_MATCHING PATTERN "*.h"
	PATTERN ".git" EXCLUDE
)

file(GLOB USE_SCRIPTS "use_*.sh")
install(
    FILES ${USE_SCRIPTS}
    DESTINATION .
)
