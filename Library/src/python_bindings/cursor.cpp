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
			.def("prepare", &pydbc::cursor::prepare)
    		.def("execute", &pydbc::cursor::execute)
    		.def("add_parameter_set", &pydbc::cursor::add_parameter_set)
    		.def("fetchone", &pydbc::cursor::fetch_one)
    		.def("get_row_count", &pydbc::cursor::get_row_count)
    		.def("get_result_set_info", &pydbc::cursor::get_result_set_info)
    	;
}

} }
