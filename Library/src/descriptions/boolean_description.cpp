#include <pydbc/descriptions/boolean_description.h>

#include <sqlext.h>

namespace pydbc {

boolean_description::boolean_description() = default;
boolean_description::~boolean_description() = default;

std::size_t boolean_description::do_element_size() const
{
	return sizeof(char);
}

SQLSMALLINT boolean_description::do_column_type() const
{
	return SQL_C_BIT;
}

field boolean_description::do_make_field(char const * data_pointer) const
{
	return {static_cast<bool>(*data_pointer)};
}

}
