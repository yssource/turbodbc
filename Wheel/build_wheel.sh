#!/bin/bash

VENV_DIR=/tmp/build_cpp_odbc_wheel_venv_${label}

virtualenv ${VENV_DIR}
source ${VENV_DIR}/bin/activate
pip install wheel

python setup.py bdist_wheel --dist-dir @CMAKE_INSTALL_PREFIX@/dist

deactivate
rm -r ${VENV_DIR}