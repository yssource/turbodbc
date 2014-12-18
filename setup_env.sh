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
source /data/software/foundation/release/cppunit_toolbox/${label}/cppunit_toolbox_0_3_0/use_cppunit_toolbox.sh
source /data/software/${label}/unixODBC/unixODBC-2.2.14/use_unixodbc.sh