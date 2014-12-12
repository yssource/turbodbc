/**
 *  @file cursor.h
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include <pydbc/cursor.h>

#include <boost/python/class.hpp>

namespace pydbc { namespace bindings {

void for_cursor()
{
	boost::python::class_<pydbc::py_cursor>("Cursor", boost::python::no_init)
    		.def("execute", &pydbc::py_cursor::execute)
    	;
}

} }
