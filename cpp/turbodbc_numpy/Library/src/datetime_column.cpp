#include <turbodbc_numpy/datetime_column.h>
#include <turbodbc_numpy/ndarrayobject.h>
#include <turbodbc_numpy/make_numpy_array.h>

#include <Python.h>

#include <boost/date_time/gregorian/gregorian_types.hpp>

#include <sql.h>
#include <cstring>
#include <chrono>

namespace turbodbc_numpy {

namespace {

	using std::chrono::system_clock;
	typedef std::chrono::duration<long, std::micro> microseconds;

	std::string get_type_descriptor(turbodbc::type_code type)
	{
		if (type == turbodbc::type_code::timestamp) {
			return "datetime64[us]";
		} else {
			return "datetime64[D]";
		}
	}

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
		long const microsecond_fraction = value.fraction / 1000;
		auto const duration_since_epoch = system_clock::from_time_t(std::mktime(&time)).time_since_epoch();
		return std::chrono::duration_cast<microseconds>(duration_since_epoch).count() + microsecond_fraction;
	}

	long timestamp_to_microseconds(char const * data_pointer)
	{
		return microseconds_since_epoch(*reinterpret_cast<SQL_TIMESTAMP_STRUCT const *>(data_pointer));
	}

	boost::gregorian::date date_epoch(1970, 1, 1);

	long date_to_days(char const * data_pointer)
	{
		auto & sql_date = *reinterpret_cast<SQL_DATE_STRUCT const *>(data_pointer);
		boost::gregorian::date const date(sql_date.year, sql_date.month, sql_date.day);
		return (date - date_epoch).days();
	}

	datetime_column::converter make_converter(turbodbc::type_code type)
	{
		if (type == turbodbc::type_code::timestamp) {
			return timestamp_to_microseconds;
		} else {
			return date_to_days;
		}
	}

	PyArrayObject * get_array_ptr(boost::python::object & object)
	{
		return reinterpret_cast<PyArrayObject *>(object.ptr());
	}

}

datetime_column::datetime_column(turbodbc::type_code type) :
	type_(type),
	data_(make_empty_numpy_array(get_type_descriptor(type_))),
	mask_(make_empty_numpy_array(numpy_bool_type)),
	size_(0),
	converter_(make_converter(type_))
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
			reinterpret_cast<long *>(data_pointer)[i] = converter_(element.data_pointer);
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

