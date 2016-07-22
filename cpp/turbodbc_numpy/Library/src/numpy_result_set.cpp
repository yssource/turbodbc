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
#include <sql.h>

namespace turbodbc { namespace result_sets {



namespace {

	PyArrayObject * get_array_ptr(boost::python::object & object)
	{
		return reinterpret_cast<PyArrayObject *>(object.ptr());
	}

	struct numpy_type {
		int code;
		int size;
	};

	numpy_type const numpy_int_type = {NPY_INT64, 8};
	numpy_type const numpy_double_type = {NPY_FLOAT64, 8};
	numpy_type const numpy_bool_type = {NPY_BOOL, 1};

	numpy_type as_numpy_type(type_code type)
	{
		switch (type) {
			case type_code::floating_point:
				return numpy_double_type;
			default:
				return numpy_int_type;
		}
	}

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

	struct masked_column {
		boost::python::object data;
		boost::python::object mask;

		masked_column(numpy_type const & type) :
			data(make_numpy_array(0, type)),
			mask(make_numpy_array(0, numpy_bool_type))
		{}

		// pointer to 64bit data type is okay for integer and double
		std::int64_t * get_data_pointer()
		{
			return static_cast<int64_t *>(PyArray_DATA(get_array_ptr(data)));
		}

		std::int8_t * get_mask_pointer()
		{
			return static_cast<std::int8_t *>(PyArray_DATA(get_array_ptr(mask)));
		}

		void resize(npy_intp new_size)
		{
			PyArray_Dims new_dimensions = {&new_size, 1};
			int const no_reference_check = 0;
			__extension__ PyArray_Resize(get_array_ptr(data), &new_dimensions, no_reference_check, NPY_ANYORDER);
			__extension__ PyArray_Resize(get_array_ptr(mask), &new_dimensions, no_reference_check, NPY_ANYORDER);
		}

		void fill_values_from_buffer(cpp_odbc::multi_value_buffer const & buffer,
			                         std::size_t n_values,
			                         std::size_t offset)
		{
			std::memcpy(get_data_pointer() + offset,
			            buffer.data_pointer(),
			            n_values * numpy_int_type.size);

			auto const mask_pointer = get_mask_pointer() + offset;
			std::memset(mask_pointer, 0, n_values);

			auto const indicator_pointer = buffer.indicator_pointer();
			for (std::size_t element = 0; element != n_values; ++element) {
				if (indicator_pointer[element] == SQL_NULL_DATA) {
					mask_pointer[element] = 1;
				}
			}
		}
	};


	boost::python::list as_python_list(std::vector<masked_column> const & objects)
	{
		boost::python::list result;
		for (auto const & object : objects) {
			result.append(boost::python::make_tuple(object.data, object.mask));
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
	auto const column_info = base_result_.get_column_info();
	auto const n_columns = column_info.size();

	std::vector<masked_column> columns;
	for (std::size_t i = 0; i != n_columns; ++i) {
		columns.emplace_back(as_numpy_type(column_info[i].type));
	}

	do {
		auto const buffers = base_result_.get_buffers();

		for (std::size_t i = 0; i != n_columns; ++i) {
			columns[i].resize(processed_rows + rows_in_batch);
			columns[i].fill_values_from_buffer(buffers[i].get(), rows_in_batch, processed_rows);
		}
		processed_rows += rows_in_batch;
		rows_in_batch = base_result_.fetch_next_batch();
	} while (rows_in_batch != 0);

	return as_python_list(columns);
}



} }
