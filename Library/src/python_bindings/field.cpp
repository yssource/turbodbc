/**
 *  @file field.cpp
 *  @date 19.12.2014
 *  @author mkoenig
 *  @brief
 *
 *  $LastChangedDate$
 *  $LastChangedBy$
 *  $LastChangedRevision$
 *
 */

#include <pydbc/field.h>

#include <boost/python/to_python_converter.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <vector>

namespace pydbc { namespace bindings {

struct field_to_object : boost::static_visitor<PyObject *> {
	static result_type convert(field const & f) {
		return apply_visitor(field_to_object(), f);
	}

	template<typename Value>
	result_type operator()(Value const & value) const {
		return boost::python::incref(boost::python::object(value).ptr());
	}
};


void for_field()
{
	boost::python::to_python_converter<field, field_to_object>();
	boost::python::implicitly_convertible<long, field>();

	bool const disable_proxies = true;
	boost::python::class_<std::vector<field>>("vector_of_fields")
    	.def(boost::python::vector_indexing_suite<std::vector<field>, disable_proxies>() );
}

} }


