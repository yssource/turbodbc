if (WIN32)
    include (GenerateExportHeader)
endif()

file(GLOB_RECURSE LIBRARY_FILES "src/*.cpp")

add_library(cpp_odbc ${LIBRARY_FILES})

add_dependencies(cpp_odbc
    refresh_cmake_configuration
)

if (WIN32)
    GENERATE_EXPORT_HEADER( cpp_odbc
        BASE_NAME cpp_odbc
        EXPORT_MACRO_NAME cpp_odbc_EXPORT
        EXPORT_FILE_NAME cpp_odbc_Export.h
        STATIC_DEFINE cpp_odbc_BUILT_AS_STATIC
        )
endif()

target_link_libraries(cpp_odbc
    ${Odbc_LIBRARIES}
)


install(
    TARGETS cpp_odbc
    DESTINATION lib
)

install(
    DIRECTORY cpp_odbc
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
    PATTERN ".git" EXCLUDE
)

install(
    DIRECTORY src/
    DESTINATION src/cpp_odbc
    FILES_MATCHING PATTERN "*.cpp"
    PATTERN ".git" EXCLUDE
)

file(GLOB USE_SCRIPTS "use_*.sh")
install(
    FILES ${USE_SCRIPTS}
    DESTINATION .
)
