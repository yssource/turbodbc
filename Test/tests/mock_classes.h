#pragma once
/*
 * mock_classes.h
 *
 *  Created on: 19.12.2014
 *      Author: mwarsinsky
 */


#include "gmock/gmock.h"

#include "cpp_odbc/connection.h"
#include "pydbc/column.h"


namespace pydbc_test {

	class mock_connection : public cpp_odbc::connection {
	public:
		mock_connection();
		~mock_connection();
		MOCK_CONST_METHOD0( do_make_statement, std::shared_ptr<cpp_odbc::statement const>());
		MOCK_CONST_METHOD2( do_set_attribute, void(SQLINTEGER, long));
		MOCK_CONST_METHOD0( do_commit, void());
		MOCK_CONST_METHOD0( do_rollback, void());
		MOCK_CONST_METHOD1( do_get_string_info, std::string(SQLUSMALLINT info_type));
	};


	class mock_statement : public cpp_odbc::statement {
	public:
		mock_statement();
		~mock_statement();
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



