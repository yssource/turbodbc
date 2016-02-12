configure_file(MANIFEST.template MANIFEST.in @ONLY)
configure_file(setup.template setup.py @ONLY)
configure_file(build_wheel.sh build_wheel.sh @ONLY)

add_custom_target(wheel COMMAND ./build_wheel.sh)