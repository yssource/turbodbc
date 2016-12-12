#include <turbodbc_arrow/arrow_result_set.h>

#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// compare http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
// as to why this define is necessary
#define PY_ARRAY_UNIQUE_SYMBOL turbodbc_arrow_API

using turbodbc_arrow::arrow_result_set;

namespace {

  // TODO: Replace this file with Cython code
arrow_result_set make_arrow_result_set(std::shared_ptr<turbodbc::result_sets::result_set> result_set_pointer)
{
	return arrow_result_set(*result_set_pointer);
}

}


BOOST_PYTHON_MODULE(turbodbc_arrow_support)
{
	boost::python::class_<arrow_result_set>("ArrowResultSet", boost::python::no_init)
			.def("fetch_all", &arrow_result_set::fetch_all)
		;

	boost::python::def("make_arrow_result_set", make_arrow_result_set);
}
