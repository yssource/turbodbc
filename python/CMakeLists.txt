if(BUILD_TESTING)
    add_subdirectory(turbodbc_test turbodbc_test)
endif()

file(COPY turbodbc DESTINATION ${CMAKE_BINARY_DIR})

install(
	DIRECTORY turbodbc
	DESTINATION .
	FILES_MATCHING PATTERN "*.py"
	PATTERN ".git" EXCLUDE
)