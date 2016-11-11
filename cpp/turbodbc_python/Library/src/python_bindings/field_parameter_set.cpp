#include <turbodbc/parameter_sets/field_parameter_set.h>
#include <turbodbc/cursor.h>

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>

namespace turbodbc { namespace bindings {


field_parameter_set make_parameter_set(cursor & cursor)
{
	return {cursor.get_command()->get_parameters()};
}


void for_field_parameter_set()
{
	boost::python::class_<field_parameter_set>("ParameterSet", boost::python::no_init)
			.def("add_set", &field_parameter_set::add_parameter_set)
			.def("flush", &field_parameter_set::flush)
		;

	boost::python::def("make_parameter_set", make_parameter_set);
}

} }
