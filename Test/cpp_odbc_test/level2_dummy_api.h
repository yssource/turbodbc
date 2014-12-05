#pragma once
/**
 *  @file level2_dummy_api.h
 *  @date 21.03.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 15:26:51 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21210 $
 *
 */

#include "cpp_odbc/level2/api.h"

#include "gmock/gmock.h"

namespace cpp_odbc_test {

	struct level2_dummy_api : public cpp_odbc::level2::api {
		level2_dummy_api();
		virtual ~level2_dummy_api();

		cpp_odbc::level2::statement_handle do_allocate_statement_handle(cpp_odbc::level2::connection_handle const & input_handle) const final;
		cpp_odbc::level2::connection_handle do_allocate_connection_handle(cpp_odbc::level2::environment_handle const & input_handle) const final;
		cpp_odbc::level2::environment_handle do_allocate_environment_handle() const final;

		void do_free_handle(cpp_odbc::level2::statement_handle & handle) const final;
		void do_free_handle(cpp_odbc::level2::connection_handle & handle) const final;
		void do_free_handle(cpp_odbc::level2::environment_handle & handle) const final;

		MOCK_CONST_METHOD1(do_get_diagnostic_record, cpp_odbc::level2::diagnostic_record(cpp_odbc::level2::statement_handle const &));
		MOCK_CONST_METHOD1(do_get_diagnostic_record, cpp_odbc::level2::diagnostic_record(cpp_odbc::level2::connection_handle const &));
		MOCK_CONST_METHOD1(do_get_diagnostic_record, cpp_odbc::level2::diagnostic_record(cpp_odbc::level2::environment_handle const &));

		void do_set_environment_attribute(cpp_odbc::level2::environment_handle const & handle, SQLINTEGER attribute, long value) const final;
		void do_set_connection_attribute(cpp_odbc::level2::connection_handle const & handle, SQLINTEGER attribute, long value) const final;
		void do_establish_connection(cpp_odbc::level2::connection_handle & handle, std::string const & connection_string) const final;
		void do_disconnect(cpp_odbc::level2::connection_handle & handle) const final;
		void do_end_transaction(cpp_odbc::level2::connection_handle const & handle, SQLSMALLINT completion_type) const final;
		std::string do_get_string_connection_info(cpp_odbc::level2::connection_handle const & handle, SQLUSMALLINT info_type) const final;
		MOCK_CONST_METHOD4(do_bind_column, void(cpp_odbc::level2::statement_handle const &, SQLUSMALLINT, SQLSMALLINT, cpp_odbc::multi_value_buffer &));
		MOCK_CONST_METHOD5(do_bind_input_parameter, void(cpp_odbc::level2::statement_handle const &, SQLUSMALLINT, SQLSMALLINT, SQLSMALLINT, cpp_odbc::multi_value_buffer &));
		void do_execute_prepared_statement(cpp_odbc::level2::statement_handle const & handle) const final;
		void do_execute_statement(cpp_odbc::level2::statement_handle const & handle, std::string const & sql) const final;
		void do_fetch_scroll(cpp_odbc::level2::statement_handle const &, SQLSMALLINT, SQLLEN) const final;
		void do_free_statement(cpp_odbc::level2::statement_handle const & handle, SQLUSMALLINT option) const final;
		MOCK_CONST_METHOD3(do_get_integer_column_attribute, long(cpp_odbc::level2::statement_handle const & handle, SQLUSMALLINT column_id, SQLUSMALLINT field_identifier));
		MOCK_CONST_METHOD2(do_get_integer_statement_attribute, long(cpp_odbc::level2::statement_handle const &, SQLINTEGER));
		MOCK_CONST_METHOD3(do_get_string_column_attribute, std::string(cpp_odbc::level2::statement_handle const & handle, SQLUSMALLINT column_id, SQLUSMALLINT field_identifier));
		MOCK_CONST_METHOD1(do_number_of_result_columns, short int(cpp_odbc::level2::statement_handle const &));
		void do_prepare_statement(cpp_odbc::level2::statement_handle const & handle, std::string const & sql) const final;
		MOCK_CONST_METHOD3(do_set_statement_attribute, void(cpp_odbc::level2::statement_handle const &, SQLINTEGER, long));
	};

}
