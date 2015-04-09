#include <pydbc/descriptions/number_description.h>

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <boost/variant/get.hpp>
#include <sqlext.h>

namespace pydbc {

number_description::number_description() = default;
number_description::~number_description() = default;

namespace {

	boost::multiprecision::checked_cpp_int extract_signed_mantissa(SQL_NUMERIC_STRUCT const & value)
	{
		boost::multiprecision::checked_cpp_int mantissa = 0;
		for (unsigned int i = SQL_MAX_NUMERIC_LEN; i != 0; --i) {
			unsigned char const byte = value.val[i - 1];
			mantissa <<= 8;
			mantissa += byte;
		}

		if (value.sign != 1) {
			mantissa *= -1;
		}

		return mantissa;
	}

	void set_signed_mantissa(SQL_NUMERIC_STRUCT & value, boost::multiprecision::cpp_int & mantissa)
	{
		boost::multiprecision::cpp_int const byte_limit(256);
		std::memset(value.val, 0, SQL_MAX_NUMERIC_LEN);

		if (mantissa.sign() < 0) {
			mantissa *= -1;
			value.sign = 0;
		} else {
			value.sign = 1;
		}

		for (unsigned int i = 0; i != SQL_MAX_NUMERIC_LEN; ++i) {
			unsigned char & byte = value.val[i];
			boost::multiprecision::cpp_int const diff = (mantissa % byte_limit);
			byte = diff.convert_to<long>();
			mantissa >>= 8;
		}
	}

	using floating = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<36>>;

	floating extract_exponent(SQL_NUMERIC_STRUCT const & value)
	{
		return boost::multiprecision::pow(floating(0.1), static_cast<int>(value.scale));
	}

	bool is_integer(SQL_NUMERIC_STRUCT const & value)
	{
		return value.scale == 0;
	}

}

std::size_t number_description::do_element_size() const
{
	return sizeof(SQL_NUMERIC_STRUCT);
}

SQLSMALLINT number_description::do_column_c_type() const
{
	return SQL_C_NUMERIC;
}

SQLSMALLINT number_description::do_column_sql_type() const
{
	return SQL_NUMERIC;
}

field number_description::do_make_field(char const * data_pointer) const
{
	auto numeric_ptr = reinterpret_cast<SQL_NUMERIC_STRUCT const *>(data_pointer);

	auto const mantissa = extract_signed_mantissa(*numeric_ptr);

	if (is_integer(*numeric_ptr)) {
		return {mantissa.convert_to<long>()};
	} else {
		auto exponent = extract_exponent(*numeric_ptr);
		exponent *= static_cast<floating>(mantissa);
		return {exponent.convert_to<double>()};
	}
}

void number_description::do_set_field(cpp_odbc::writable_buffer_element & element, field const & value) const
{
	auto numeric_ptr = reinterpret_cast<SQL_NUMERIC_STRUCT *>(element.data_pointer);
	boost::multiprecision::cpp_int mantissa(boost::get<long>(value));
	set_signed_mantissa(*numeric_ptr, mantissa);
	element.indicator = element_size();
}

}
