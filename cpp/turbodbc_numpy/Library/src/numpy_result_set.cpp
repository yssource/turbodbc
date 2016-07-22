#include <turbodbc_numpy/numpy_result_set.h>

#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
// compare http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
// as to why these defines are necessary
#define PY_ARRAY_UNIQUE_SYMBOL turbodbc_numpy_API
#define NO_IMPORT_ARRAY
#include <numpy/ndarrayobject.h>

#include <Python.h>

#include <cstring>
#include <vector>

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

	PyArrayObject * get_array_ptr(boost::python::object & object)
	{
		return reinterpret_cast<PyArrayObject *>(object.ptr());
	}

	void resize_numpy_array(boost::python::object & array, npy_intp new_size)
	{
		PyArray_Dims new_dimensions = {&new_size, 1};
		int const no_reference_check = 0;
		__extension__ PyArray_Resize(get_array_ptr(array), &new_dimensions, no_reference_check, NPY_ANYORDER);
	}

	long * get_numpy_data_pointer(boost::python::object & numpy_object)
	{
		return static_cast<long *>(PyArray_DATA(get_array_ptr(numpy_object)));
	}

	boost::python::list as_python_list(std::vector<boost::python::object> const & objects)
	{
		boost::python::list result;
		for (auto const & object : objects) {
			result.append(boost::python::make_tuple(object, boost::python::object(false)));
		}
		return result;
	}
}

numpy_result_set::numpy_result_set(result_set & base) :
	base_result_(base)
{
}


boost::python::object numpy_result_set::fetch_all()
{
	std::size_t processed_rows = 0;
	std::size_t rows_in_batch = base_result_.fetch_next_batch();
	auto const n_columns = base_result_.get_column_info().size();

	std::vector<boost::python::object> columns;
	for (std::size_t i = 0; i != n_columns; ++i) {
		auto column = make_numpy_array(rows_in_batch, numpy_int_type);
		columns.push_back(column);
	}

	do {
		auto const buffers = base_result_.get_buffers();

		for (std::size_t i = 0; i != n_columns; ++i) {
			resize_numpy_array(columns[i], processed_rows + rows_in_batch);

			std::memcpy(get_numpy_data_pointer(columns[i]) + processed_rows,
						buffers[i].get().data_pointer(),
						rows_in_batch * numpy_int_type.size);
		}
		processed_rows += rows_in_batch;
		rows_in_batch = base_result_.fetch_next_batch();
	} while (rows_in_batch != 0);

	return as_python_list(columns);
}



} }
