#include <pydbc/description_types.h>

#include <sqlext.h>

namespace pydbc {

integer_description::integer_description() = default;
integer_description::~integer_description() = default;

std::size_t integer_description::do_element_size() const
{
	return sizeof(long);
}

SQLSMALLINT integer_description::do_column_type() const
{
	return SQL_C_SBIGINT;
}

field integer_description::do_make_field(char const * data_pointer) const
{
	return {*reinterpret_cast<long const *>(data_pointer)};
}




string_description::string_description(std::size_t maximum_length) :
		maximum_length_(maximum_length)
{
}

string_description::~string_description() = default;

std::size_t string_description::do_element_size() const
{
	return maximum_length_ + 1;
}

SQLSMALLINT string_description::do_column_type() const
{
	return SQL_CHAR;
}

field string_description::do_make_field(char const * data_pointer) const
{
	return {std::string(data_pointer)};
}


}
