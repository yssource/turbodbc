#pragma once
/**
 *  @file mock_statement.h
 *  @date 16.05.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 15:26:51 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21210 $
 *
 */

#include "cpp_odbc/statement.h"
#include "gmock/gmock.h"

namespace cpp_odbc_test {

class mock_statement : public cpp_odbc::statement {
public:
	MOCK_CONST_METHOD1( do_get_integer_statement_attribute, long(SQLINTEGER));
	MOCK_CONST_METHOD2( do_set_statement_attribute, void(SQLINTEGER, long));
	MOCK_CONST_METHOD1( do_execute, void(std::string const &));
	MOCK_CONST_METHOD1( do_prepare, void(std::string const &));
	MOCK_CONST_METHOD4( do_bind_input_parameter, void(SQLUSMALLINT, SQLSMALLINT, SQLSMALLINT, cpp_odbc::multi_value_buffer &));
	MOCK_CONST_METHOD0( do_execute_prepared, void());
	MOCK_CONST_METHOD0( do_number_of_columns, short int());
	MOCK_CONST_METHOD3( do_bind_column, void(SQLUSMALLINT, SQLSMALLINT, cpp_odbc::multi_value_buffer &));
	MOCK_CONST_METHOD0( do_fetch_next, bool());
	MOCK_CONST_METHOD0( do_close_cursor, void());
	MOCK_CONST_METHOD2( do_get_integer_column_attribute, long(SQLUSMALLINT, SQLUSMALLINT));
	MOCK_CONST_METHOD2( do_get_string_column_attribute, std::string(SQLUSMALLINT, SQLUSMALLINT));
	MOCK_CONST_METHOD0( do_row_count, SQLLEN());
};

}
