/**
 *  @file connection.h
 *  @date 12.12.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include <pydbc/connection.h>

#include <boost/python/class.hpp>

namespace pydbc { namespace bindings {

void for_connection()
{
	boost::python::class_<pydbc::py_connection>("Connection", boost::python::no_init)
    		.def("commit", &pydbc::py_connection::commit)
    		.def("cursor", &pydbc::py_connection::cursor)
    	;
}

} }
