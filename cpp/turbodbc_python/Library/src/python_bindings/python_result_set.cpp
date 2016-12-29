#include <turbodbc_python/python_result_set.h>

#include <pybind11/pybind11.h>


using turbodbc::result_sets::python_result_set;

namespace turbodbc { namespace bindings {


bool has_result_set(std::shared_ptr<turbodbc::result_sets::result_set> result_set_pointer)
{
	return static_cast<bool>(result_set_pointer);
}

python_result_set make_python_result_set(std::shared_ptr<turbodbc::result_sets::result_set> result_set_pointer)
{
	return python_result_set(*result_set_pointer);
}


void for_python_result_set(pybind11::module & module)
{
	pybind11::class_<python_result_set>(module, "ResultSet")
			.def("get_column_info", &python_result_set::get_column_info)
			.def("fetch_row", &python_result_set::fetch_row)
		;

	pybind11::class_<std::shared_ptr<turbodbc::result_sets::result_set>>(module, "RawResultSetPointer");

	module.def("make_row_based_result_set", make_python_result_set);
	module.def("has_result_set", has_result_set);
}

} }
