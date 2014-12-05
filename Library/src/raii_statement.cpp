/**
 *  @file raii_statement.cpp
 *  @date 23.05.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 11:59:59 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21206 $
 *
 */

#include "cpp_odbc/raii_statement.h"

#include "cpp_odbc/level2/api.h"

#include "sql.h"

namespace cpp_odbc {

raii_statement::raii_statement(psapp::valid_ptr<cpp_odbc::level2::api const> api, cpp_odbc::level2::connection_handle const & connection) :
		api_(api),
		handle_(api->allocate_statement_handle(connection))
{
}

raii_statement::~raii_statement()
{
	api_->free_handle(handle_);
}

long raii_statement::do_get_integer_statement_attribute(SQLINTEGER attribute) const
{
	return api_->get_integer_statement_attribute(handle_, attribute);
}

void raii_statement::do_set_statement_attribute(SQLINTEGER attribute, long value) const
{
	api_->set_statement_attribute(handle_, attribute, value);
}

void raii_statement::do_execute(std::string const & sql) const
{
	api_->execute_statement(handle_, sql);
}

void raii_statement::do_prepare(std::string const & sql) const
{
	api_->prepare_statement(handle_, sql);
}

void raii_statement::do_bind_input_parameter(SQLUSMALLINT parameter_id, SQLSMALLINT value_type, SQLSMALLINT parameter_type, cpp_odbc::multi_value_buffer & parameter_values) const
{
	api_->bind_input_parameter(handle_, parameter_id, value_type, parameter_type, parameter_values);
}

void raii_statement::do_execute_prepared() const
{
	api_->execute_prepared_statement(handle_);
}

short int raii_statement::do_number_of_columns() const
{
	return api_->number_of_result_columns(handle_);
}

void raii_statement::do_bind_column(SQLUSMALLINT column_id, SQLSMALLINT column_type, cpp_odbc::multi_value_buffer & column_buffer) const
{
	api_->bind_column(handle_, column_id, column_type, column_buffer);
}

void raii_statement::do_fetch_next() const
{
	SQLSMALLINT const offset = 0;
	api_->fetch_scroll(handle_, SQL_FETCH_NEXT, offset);
}

void raii_statement::do_close_cursor() const
{
	api_->free_statement(handle_, SQL_CLOSE);
}

long raii_statement::do_get_integer_column_attribute(SQLUSMALLINT column_id, SQLUSMALLINT field_identifier) const
{
	return api_->get_integer_column_attribute(handle_, column_id, field_identifier);
}

std::string raii_statement::do_get_string_column_attribute(SQLUSMALLINT column_id, SQLUSMALLINT field_identifier) const
{
	return api_->get_string_column_attribute(handle_, column_id, field_identifier);
}

}
