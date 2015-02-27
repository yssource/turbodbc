BY_HOME_BASE=/home/localdata/${USER}

if [ ! -n "$CPP_ODBC_LIB_DIR" ]; then
	export CPP_ODBC_LIB_DIR=${BY_HOME_BASE}/workspace/cpp_odbc_build/Library
	export CPP_ODBC_INCLUDE_DIR=${BY_HOME_BASE}/workspace/cpp_odbc/Library/
fi

if [ -z "$label" ]; then 
        if grep -q Ubuntu /etc/issue; then
                label=Ubuntu_12.04
        elif grep -q Enterprise /etc/issue; then
                label=SUSE_LINUX_11
        elif grep -q -i debian /etc/issue; then
                label=Debian_7
        else 
                echo "Couldn't determine distribution and 'label' is not set"
                exit 1
        fi
fi

CENTRAL_SOFTWARE_REPOSITORY=/data/software/${label}

if [ ! -n "$ENVIRONMENT_ROOT" ]
then
	ENVIRONMENT_ROOT=${CENTRAL_SOFTWARE_REPOSITORY}
fi

source $ENVIRONMENT_ROOT/gcc/gcc4.7.2/use_gcc.sh
source $ENVIRONMENT_ROOT/boost/1.55_gcc4.7.2_python2.7.3/use_boost.sh
source $CENTRAL_SOFTWARE_REPOSITORY/python/2.7.3/use_python.sh

source /data/software/foundation/release/cppunit_toolbox/${label}/cppunit_toolbox_0_3_0/use_cppunit_toolbox.sh
source /data/software/${label}/unixODBC/unixODBC-2.2.14/use_unixodbc.sh

export PYTHONPATH=${BY_HOME_BASE}/workspace/pydbc_build/
