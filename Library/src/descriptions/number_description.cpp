#include <pydbc/descriptions/number_description.h>

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

#include <sqlext.h>

namespace pydbc {

number_description::number_description() = default;
number_description::~number_description() = default;

std::size_t number_description::do_element_size() const
{
	return sizeof(SQL_NUMERIC_STRUCT);
}

SQLSMALLINT number_description::do_column_type() const
{
	return SQL_NUMERIC;
}

// TODO extract bits and pieces! very low-level!
field number_description::do_make_field(char const * data_pointer) const
{
	auto numeric_ptr = reinterpret_cast<SQL_NUMERIC_STRUCT const *>(data_pointer);

	boost::multiprecision::checked_cpp_int mantissa = 0;
	for (unsigned int i = SQL_MAX_NUMERIC_LEN; i != 0; --i) {
		unsigned char const byte = numeric_ptr->val[i - 1];
		mantissa <<= 8;
		mantissa += byte;
	}

	if (numeric_ptr->sign != 1) {
		mantissa *= -1;
	}

	if (numeric_ptr->scale == 0) {
		return {mantissa.convert_to<long>()};
	} else {
		using floating = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<36>>;
		floating exponent = boost::multiprecision::pow(floating(0.1), static_cast<int>(numeric_ptr->scale));
		exponent *= static_cast<floating>(mantissa);
		return {exponent.convert_to<double>()};
	}
}

}
