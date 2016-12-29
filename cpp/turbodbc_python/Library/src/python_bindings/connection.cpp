#include <turbodbc/connection.h>

#include <pybind11/pybind11.h>


namespace turbodbc { namespace bindings {

void for_connection(pybind11::module & module)
{
	pybind11::class_<turbodbc::connection>(module, "Connection")
			.def("commit", &turbodbc::connection::commit)
			.def("rollback", &turbodbc::connection::rollback)
			.def("cursor", &turbodbc::connection::make_cursor)
			.def("get_buffer_size", &turbodbc::connection::get_buffer_size)
			.def("set_buffer_size", &turbodbc::connection::set_buffer_size)
			.def_readwrite("parameter_sets_to_buffer", &turbodbc::connection::parameter_sets_to_buffer)
			.def_readwrite("use_async_io", &turbodbc::connection::use_async_io)
			;
}

} }
