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
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace pydbc { namespace bindings {

void for_cursor()
{
	boost::python::class_<std::vector<int>>("vector_of_ints")
    .def(boost::python::vector_indexing_suite<std::vector<int>>() );

	boost::python::class_<pydbc::cursor>("Cursor", boost::python::no_init)
    		.def("execute", &pydbc::cursor::execute)
    		.def("fetchone", &pydbc::cursor::fetch_one)
    	;
}

} }
