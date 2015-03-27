#include <pydbc/descriptions/integer_description.h>

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

SQLSMALLINT integer_description::do_column_sql_type() const
{
	return SQL_BIGINT;
}

field integer_description::do_make_field(char const * data_pointer) const
{
	return {*reinterpret_cast<long const *>(data_pointer)};
}

}
