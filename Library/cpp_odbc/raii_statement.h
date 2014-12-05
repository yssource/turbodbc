#pragma once
/**
 *  @file raii_statement.h
 *  @date 23.05.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 11:59:59 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21206 $
 *
 */

#include "cpp_odbc/statement.h"

#include "cpp_odbc/level2/handles.h"

#include "psapp/valid_ptr_core.h"

namespace cpp_odbc { namespace level2 {
	class api;
} }

namespace cpp_odbc {

class raii_statement : public statement {
public:
	raii_statement(psapp::valid_ptr<cpp_odbc::level2::api const> api, cpp_odbc::level2::connection_handle const & connection);

	virtual ~raii_statement();
private:
	long do_get_integer_statement_attribute(SQLINTEGER attribute) const final;
	void do_set_statement_attribute(SQLINTEGER attribute, long value) const final;
	void do_execute(std::string const & sql) const final;
	void do_prepare(std::string const & sql) const final;
	void do_bind_input_parameter(SQLUSMALLINT parameter_id, SQLSMALLINT value_type, SQLSMALLINT parameter_type, cpp_odbc::multi_value_buffer & parameter_values) const final;
	void do_execute_prepared() const final;

	short int do_number_of_columns() const final;
	void do_bind_column(SQLUSMALLINT column_id, SQLSMALLINT column_type, cpp_odbc::multi_value_buffer & column_buffer) const final;
	void do_fetch_next() const final;
	void do_close_cursor() const final;

	long do_get_integer_column_attribute(SQLUSMALLINT column_id, SQLUSMALLINT field_identifier) const final;
	std::string do_get_string_column_attribute(SQLUSMALLINT column_id, SQLUSMALLINT field_identifier) const final;

	psapp::valid_ptr<level2::api const> api_;
	level2::statement_handle handle_;
};


}
