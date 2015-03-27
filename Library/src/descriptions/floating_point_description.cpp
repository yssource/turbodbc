#include <pydbc/descriptions/floating_point_description.h>

#include <sqlext.h>

namespace pydbc {

floating_point_description::floating_point_description() = default;
floating_point_description::~floating_point_description() = default;

std::size_t floating_point_description::do_element_size() const
{
	return sizeof(double);
}

SQLSMALLINT floating_point_description::do_column_type() const
{
	return SQL_C_DOUBLE;
}

SQLSMALLINT floating_point_description::do_column_sql_type() const
{
	return SQL_DOUBLE;
}

field floating_point_description::do_make_field(char const * data_pointer) const
{
	return {*reinterpret_cast<double const *>(data_pointer)};
}

}
