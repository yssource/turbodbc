/**
 *  @file module.cpp
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include <boost/python/module.hpp>

namespace pydbc { namespace bindings {

	void for_connect();
	void for_connection();
	void for_cursor();
	void for_error();

} }

BOOST_PYTHON_MODULE(pydbc_intern)
{
	using namespace pydbc;
	bindings::for_connect();
	bindings::for_connection();
	bindings::for_cursor();
	bindings::for_error();
}
