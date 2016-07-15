#include <turbodbc_numpy/numpy_result_set.h>

#include <boost/python/list.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// compare http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
// as to why these defines are necessary
#define PY_ARRAY_UNIQUE_SYMBOL turbodbc_numpy_API
#define NO_IMPORT_ARRAY
#include <numpy/ndarrayobject.h>

namespace turbodbc { namespace result_sets {



namespace {

	struct numpy_type {
		int code;
		int size;
	};

	numpy_type const numpy_int_type = {NPY_INT64, 8};

	boost::python::object make_numpy_array(npy_intp elements, numpy_type type)
	{
		int const flags = 0;
		int const one_dimensional = 1;
		// __extension__ needed because of some C/C++ incompatibility.
		// see issue https://github.com/numpy/numpy/issues/2539
		return boost::python::object{boost::python::handle<>(__extension__ PyArray_New(&PyArray_Type,
		                                                                               one_dimensional,
		                                                                               &elements,
		                                                                               type.code,
		                                                                               nullptr,
		                                                                               nullptr,
		                                                                               type.size,
		                                                                               flags,
		                                                                               nullptr))};
	}

}

numpy_result_set::numpy_result_set(result_set & base) :
	base_result_(base)
{
}


boost::python::object numpy_result_set::fetch_all()
{
	boost::python::list columns;
	npy_intp elements = 0;

	columns.append(make_numpy_array(elements, numpy_int_type));
	return columns;
}



} }
