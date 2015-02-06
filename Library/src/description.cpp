#include <pydbc/description.h>

namespace pydbc {

description::description() = default;
description::~description() = default;

std::size_t description::element_size() const
{
	return do_element_size();
}

SQLSMALLINT description::column_type() const
{
	return do_column_type();
}

field description::make_field(char const * data_pointer) const
{
	return do_make_field(data_pointer);
}

}
