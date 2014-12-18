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
source ${CENTRAL_SOFTWARE_REPOSITORY}/gcc/gcc4.7.2/use_gcc.sh
source ${CENTRAL_SOFTWARE_REPOSITORY}/boost/1.55_gcc4.7.2_python2.7.3/use_boost.sh
source /data/software/foundation/release/cppunit_toolbox/${label}/cppunit_toolbox_0_3_0/use_cppunit_toolbox.sh
source ${CENTRAL_SOFTWARE_REPOSITORY}/unixODBC/unixODBC-2.2.14/use_unixodbc.sh