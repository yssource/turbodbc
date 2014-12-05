/**
 *  @file dummy.cpp
 *  @date 05.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include <boost/python.hpp>

namespace bp = boost::python;

namespace pydbc {

	int connect()
	{
		return 42;
	}

}

BOOST_PYTHON_MODULE(pydbc)
{
    bp::def("connect", pydbc::connect);
}
