export BY_HOME_BASE=/home/localdata/${USER}

export CPP_ODBC_LIB_DIR=${BY_HOME_BASE}/workspace/cpp_odbc_build//Library
export CPP_ODBC_INCLUDE_DIR=${BY_HOME_BASE}/workspace/cpp_odbc/Library/
source /data/software/foundation/release/cppunit_toolbox/${label}/cppunit_toolbox_0_3_0/use_cppunit_toolbox.sh
source /data/software/${label}/unixODBC/unixODBC-2.2.14/use_unixodbc.sh

export PYTHONPATH=${BY_HOME_BASE}/workspace/pydbc_build/
