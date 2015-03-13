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
	boost::python::class_<pydbc::cursor>("Cursor", boost::python::no_init)
    		.def("execute", &pydbc::cursor::execute)
    		.def("execute_many", &pydbc::cursor::execute_many)
    		.def("fetchone", &pydbc::cursor::fetch_one)
    		.def("get_rowcount", &pydbc::cursor::get_rowcount)
    	;
}

} }
