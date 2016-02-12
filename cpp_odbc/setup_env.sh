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

source ${CENTRAL_SOFTWARE_REPOSITORY}/cppunit_toolbox/0_3/use_cppunit_toolbox.sh
source ${CENTRAL_SOFTWARE_REPOSITORY}/unixODBC/unixODBC-2.2.14/use_unixodbc.sh
