#include <turbodbc_numpy/datetime_column.h>
#include <turbodbc_numpy/ndarrayobject.h>
#include <turbodbc_numpy/make_numpy_array.h>

#include <Python.h>

#include <sql.h>
#include <cstring>
#include <chrono>

namespace turbodbc_numpy {

namespace {

	using std::chrono::system_clock;
	typedef std::chrono::duration<long, std::micro> microseconds;

	std::tm to_tm(SQL_TIMESTAMP_STRUCT const & value)
	{
		std::tm time;
		time.tm_sec = value.second;
		time.tm_min = value.minute;
		time.tm_hour = value.hour;
		time.tm_mday = value.day;
		time.tm_mon = value.month - 1;
		time.tm_year = value.year - 1900;
		return time;
	}

	long microseconds_since_epoch(SQL_TIMESTAMP_STRUCT const & value)
	{
		std::tm time = to_tm(value);
		auto const duration_since_epoch = system_clock::from_time_t(std::mktime(&time)).time_since_epoch();
		return microseconds(duration_since_epoch).count();
	}

	PyArrayObject * get_array_ptr(boost::python::object & object)
	{
		return reinterpret_cast<PyArrayObject *>(object.ptr());
	}

}

datetime_column::datetime_column() :
	data_(make_empty_numpy_array("datetime64[us]")),
	mask_(make_empty_numpy_array(numpy_bool_type)),
	size_(0)
{
}


datetime_column::~datetime_column() = default;

void datetime_column::do_append(cpp_odbc::multi_value_buffer const & buffer, std::size_t n_values)
{
	auto const old_size = size_;
	resize(old_size + n_values);

	auto const data_pointer = static_cast<int64_t *>(PyArray_DATA(get_array_ptr(data_))) + old_size;
	auto const mask_pointer = static_cast<std::int8_t *>(PyArray_DATA(get_array_ptr(mask_))) + old_size;
	std::memset(mask_pointer, 0, n_values);

	for (std::size_t i = 0; i != n_values; ++i) {
		auto element = buffer[i];
		if (element.indicator == SQL_NULL_DATA) {
			mask_pointer[i] = 1;
		} else {
			reinterpret_cast<long *>(data_pointer)[i] =
					microseconds_since_epoch(*reinterpret_cast<SQL_TIMESTAMP_STRUCT const *>(element.data_pointer));
		}
	}
}

boost::python::object datetime_column::do_get_data()
{
	return data_;
}

boost::python::object datetime_column::do_get_mask()
{
	return mask_;
}

void datetime_column::resize(std::size_t new_size)
{
	npy_intp size = new_size;
	PyArray_Dims new_dimensions = {&size, 1};
	int const no_reference_check = 0;
	__extension__ PyArray_Resize(get_array_ptr(data_), &new_dimensions, no_reference_check, NPY_ANYORDER);
	__extension__ PyArray_Resize(get_array_ptr(mask_), &new_dimensions, no_reference_check, NPY_ANYORDER);
	size_ = new_size;
}


}

