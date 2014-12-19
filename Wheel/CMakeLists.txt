configure_file(MANIFEST.template MANIFEST.in @ONLY)
configure_file(setup.template setup.py @ONLY)
configure_file(build_wheel.sh build_wheel.sh @ONLY)

file(COPY "${CMAKE_SOURCE_DIR}/pydbc" DESTINATION ".")

add_custom_target(wheel
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_INSTALL_PREFIX}/lib/pydbc_intern.so ${CMAKE_CURRENT_BINARY_DIR}/pydbc
    COMMAND ./build_wheel.sh
)