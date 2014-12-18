/**
 *  @file statement.cpp
 *  @date 16.05.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 11:59:59 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21206 $
 *
 */

#include "cpp_odbc/statement.h"

namespace cpp_odbc {


statement::statement() = default;
statement::~statement() = default;

long statement::get_integer_statement_attribute(SQLINTEGER attribute) const
{
	return do_get_integer_statement_attribute(attribute);
}

void statement::set_statement_attribute(SQLINTEGER attribute, long value) const
{
	do_set_statement_attribute(attribute, value);
}

void statement::execute(std::string const & sql) const
{
	do_execute(sql);
}

void statement::prepare(std::string const & sql) const
{
	do_prepare(sql);
}

void statement::bind_input_parameter(SQLUSMALLINT parameter_id, SQLSMALLINT value_type, SQLSMALLINT parameter_type, cpp_odbc::multi_value_buffer & parameter_values) const
{
	do_bind_input_parameter(parameter_id, value_type, parameter_type, parameter_values);
}

void statement::execute_prepared() const
{
	do_execute_prepared();
}

short int statement::number_of_columns() const
{
	return do_number_of_columns();
}

void statement::bind_column(SQLUSMALLINT column_id, SQLSMALLINT column_type, cpp_odbc::multi_value_buffer & column_buffer) const
{
	do_bind_column(column_id, column_type, column_buffer);
}

bool statement::fetch_next() const
{
	return do_fetch_next();
}

void statement::close_cursor() const
{
	do_close_cursor();
}

long statement::get_integer_column_attribute(SQLUSMALLINT column_id, SQLUSMALLINT field_identifier) const
{
	return do_get_integer_column_attribute(column_id, field_identifier);
}

std::string statement::get_string_column_attribute(SQLUSMALLINT column_id, SQLUSMALLINT field_identifier) const
{
	return do_get_string_column_attribute(column_id, field_identifier);
}

SQLLEN statement::row_count() const
{
	return do_row_count();
}

}
