#include <turbodbc_numpy/numpy_result_set.h>

#include <boost/python/list.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// compare http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
// as to why these defines are necessary
#define PY_ARRAY_UNIQUE_SYMBOL turbodbc_numpy_API
#define NO_IMPORT_ARRAY
#include <numpy/ndarrayobject.h>

namespace turbodbc { namespace result_sets {

const int one_dimensional = 1;

numpy_result_set::numpy_result_set(result_set & base) :
	base_result_(base)
{
}


boost::python::object numpy_result_set::fetch_all()
{
	boost::python::list columns;
	npy_intp size = 0;
	int const type_code = NPY_INT64;
	int const type_size = 8;
	int const flags = 0;
	boost::python::object column{boost::python::handle<>(PyArray_New(&PyArray_Type,
	                                                                 one_dimensional,
	                                                                 &size,
	                                                                 type_code,
	                                                                 nullptr,
	                                                                 nullptr,
	                                                                 type_size,
	                                                                 flags,
	                                                                 nullptr))};
	columns.append(column);
	return columns;
}



} }
