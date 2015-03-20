#include <pydbc/descriptions/number_description.h>

#include <sqlext.h>
#include <algorithm>
#include <cstring>

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

field number_description::do_make_field(char const * data_pointer) const
{
	auto numeric_ptr = reinterpret_cast<SQL_NUMERIC_STRUCT const *>(data_pointer);
//	std::cout << "***********************" << std::endl;
//	std::cout << "precision = " << static_cast<int>(converted->precision) << std::endl;
//	std::cout << "scale = " << static_cast<int>(converted->scale) << std::endl;
//	std::cout << "sign = " << static_cast<int>(converted->sign) << std::endl;
//
//	std::cout << "value = ";
//	for (unsigned int i = 0; i != 16; ++i) {
//		unsigned char const v = converted->val[i];
//		std::cout << static_cast<unsigned int>(v) << " ";
//	}
//	std::cout << std::endl;
	if (numeric_ptr->scale == 0) {
		long converted = 0;
		std::memcpy(&converted, numeric_ptr->val, sizeof(converted));
		if (numeric_ptr->sign != 1) {
			converted *= -1;
		}
		return {converted};
	} else {
		return {0.0};
	}
}

}
