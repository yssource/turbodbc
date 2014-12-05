/**
 *  @file level1_api_test.cpp
 *  @date 03.03.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 15:26:51 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21210 $
 *
 */


#include <cppunit/extensions/HelperMacros.h>
#include "cppunit_toolbox/helpers/is_abstract_base_class.h"

#include "cpp_odbc/level1/api.h"
#include "cpp_odbc_test/level1_mock_api.h"

#include "gmock/gmock.h"

#include <array>

class level1_api_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( level1_api_test );

	CPPUNIT_TEST( abstract_base );
	CPPUNIT_TEST( allocate_handle_forwards );
	CPPUNIT_TEST( free_handle_forwards );
	CPPUNIT_TEST( get_diagnostic_record_forwards );

	CPPUNIT_TEST( set_environment_attribute_forwards );

	CPPUNIT_TEST( set_connection_attribute_forwards );
	CPPUNIT_TEST( establish_connection_forwards );
	CPPUNIT_TEST( disconnect_forwards );
	CPPUNIT_TEST( end_transaction_forwards );
	CPPUNIT_TEST( get_connection_info_forwards );

	CPPUNIT_TEST( bind_column_forwards );
	CPPUNIT_TEST( bind_parameter_forwards );
	CPPUNIT_TEST( column_attribute_forwards );
	CPPUNIT_TEST( execute_prepared_statement_forwards );
	CPPUNIT_TEST( execute_statement_forwards );
	CPPUNIT_TEST( fetch_scroll_forwards );
	CPPUNIT_TEST( free_statement_forwards );
	CPPUNIT_TEST( get_statement_attribute_forwards );
	CPPUNIT_TEST( number_of_result_columns_forwards );
	CPPUNIT_TEST( prepare_statement_forwards );
	CPPUNIT_TEST( set_statement_attribute_forwards );

CPPUNIT_TEST_SUITE_END();

public:

	void abstract_base();
	void allocate_handle_forwards();
	void free_handle_forwards();
	void get_diagnostic_record_forwards();

	void set_environment_attribute_forwards();

	void set_connection_attribute_forwards();
	void establish_connection_forwards();
	void disconnect_forwards();
	void end_transaction_forwards();
	void get_connection_info_forwards();

	void bind_column_forwards();
	void bind_parameter_forwards();
	void column_attribute_forwards();
	void execute_prepared_statement_forwards();
	void execute_statement_forwards();
	void fetch_scroll_forwards();
	void free_statement_forwards();
	void get_statement_attribute_forwards();
	void number_of_result_columns_forwards();
	void prepare_statement_forwards();
	void set_statement_attribute_forwards();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( level1_api_test );

namespace level1 = cpp_odbc::level1;
using cpp_odbc_test::level1_mock_api;

namespace {

	// destinations for pointers
	int value_a = 23;
	int value_b = 17;

}

void level1_api_test::abstract_base()
{
	bool const is_abstract_base = cppunit_toolbox::is_abstract_base_class<level1::api>::value;
	CPPUNIT_ASSERT( is_abstract_base );
}

void level1_api_test::allocate_handle_forwards()
{
	SQLRETURN const expected = 42;
	SQLSMALLINT const handle_type = 17;
	SQLHANDLE const input_handle = &value_a;
	SQLHANDLE output_handle = &value_b;

	level1_mock_api api;
	EXPECT_CALL(api, do_allocate_handle(handle_type, input_handle, &output_handle))
		.WillOnce(testing::Return(expected));

	auto const actual = api.allocate_handle(handle_type, input_handle, &output_handle);
	CPPUNIT_ASSERT_EQUAL( expected, actual );
}

void level1_api_test::free_handle_forwards()
{
	SQLRETURN const expected = 42;
	SQLSMALLINT const handle_type = 17;
	SQLHANDLE const input_handle = &value_a;

	level1_mock_api api;
	EXPECT_CALL(api, do_free_handle(handle_type, input_handle))
		.WillOnce(testing::Return(expected));

	auto const actual = api.free_handle(handle_type, input_handle);
	CPPUNIT_ASSERT_EQUAL( expected, actual );
}

void level1_api_test::get_diagnostic_record_forwards()
{
	SQLRETURN const expected = 1;
	SQLSMALLINT const handle_type = 42;
	SQLHANDLE const handle = &value_a;
	SQLSMALLINT const record_id = 17;
	std::array<unsigned char, 5> status_code;
	SQLINTEGER native_error_ptr = 23;
	std::array<unsigned char, 6> message_text;
	SQLSMALLINT const buffer_length = 123;
	SQLSMALLINT text_length = 95;

	level1_mock_api api;
	EXPECT_CALL(api, do_get_diagnostic_record(handle_type, handle, record_id, status_code.data(), &native_error_ptr, message_text.data(), buffer_length, &text_length))
		.WillOnce(testing::Return(expected));

	auto const actual = api.get_diagnostic_record(handle_type, handle, record_id, status_code.data(), &native_error_ptr, message_text.data(), buffer_length, &text_length);
	CPPUNIT_ASSERT_EQUAL( expected, actual );
}

void level1_api_test::set_environment_attribute_forwards()
{
	SQLRETURN const expected = 1;
	SQLHENV const handle = &value_a;
	SQLINTEGER const attribute = 17;
	std::array<unsigned char, 6> value;

	level1_mock_api api;
	EXPECT_CALL(api, do_set_environment_attribute(handle, attribute, value.data(), value.size()))
		.WillOnce(testing::Return(expected));

	auto const actual = api.set_environment_attribute(handle, attribute, value.data(), value.size());
	CPPUNIT_ASSERT_EQUAL( expected, actual );
}

void level1_api_test::set_connection_attribute_forwards()
{
	SQLRETURN const expected = 1;
	SQLHDBC const handle = &value_a;
	SQLINTEGER const attribute = 17;
	std::array<unsigned char, 6> value;

	level1_mock_api api;
	EXPECT_CALL(api, do_set_connection_attribute(handle, attribute, value.data(), value.size()))
		.WillOnce(testing::Return(expected));

	auto const actual = api.set_connection_attribute(handle, attribute, value.data(), value.size());
	CPPUNIT_ASSERT_EQUAL( expected, actual );
}

void level1_api_test::establish_connection_forwards()
{
	SQLRETURN const expected = 27;
	SQLHDBC connection_handle = &value_a;
	SQLHWND window_handle = &value_b;
	std::array<unsigned char, 6> input_string;
	std::array<unsigned char, 7> output_string;
	SQLSMALLINT output_string_length = 123;
	SQLUSMALLINT const driver_completion = 24;

	level1_mock_api api;
	EXPECT_CALL(api, do_establish_connection(connection_handle, window_handle, input_string.data(), input_string.size(), output_string.data(), output_string.size(), &output_string_length, driver_completion))
		.WillOnce(testing::Return(expected));

	auto const actual = api.establish_connection(connection_handle, window_handle, input_string.data(), input_string.size(), output_string.data(), output_string.size(), &output_string_length, driver_completion);
	CPPUNIT_ASSERT_EQUAL( expected, actual );
}

void level1_api_test::disconnect_forwards()
{
	SQLHDBC connection_handle = &value_a;
	SQLRETURN const expected = 27;

	level1_mock_api api;
	EXPECT_CALL(api, do_disconnect(connection_handle))
		.WillOnce(testing::Return(expected));

	api.disconnect(connection_handle);
}

void level1_api_test::end_transaction_forwards()
{
	SQLRETURN const expected = 23;
	SQLSMALLINT const handle_type = 17;
	SQLHANDLE handle = &value_a;
	SQLSMALLINT const completion_type = 42;

	level1_mock_api api;
	EXPECT_CALL(api, do_end_transaction(handle_type, handle, completion_type))
		.WillOnce(testing::Return(expected));

	auto const actual = api.end_transaction(handle_type, handle, completion_type);
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}

void level1_api_test::get_connection_info_forwards()
{
	SQLRETURN const expected = 23;
	SQLHDBC handle = &value_a;
	SQLUSMALLINT const info_type = 17;
	std::vector<char> buffer(10);
	SQLSMALLINT string_length = 0;

	level1_mock_api api;
	EXPECT_CALL(api, do_get_connection_info(handle, info_type, buffer.data(), buffer.size(), &string_length))
		.WillOnce(testing::Return(expected));

	auto const actual = api.get_connection_info(handle, info_type, buffer.data(), buffer.size(), &string_length);
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}

void level1_api_test::bind_column_forwards()
{
	SQLRETURN const expected = 23;
	SQLHDBC handle = &value_a;
	SQLUSMALLINT const column_id = 17;
	SQLSMALLINT const target_type = 42;
	std::vector<char> buffer(100);
	SQLLEN buffer_length;

	level1_mock_api api;
	EXPECT_CALL(api, do_bind_column(handle, column_id, target_type, buffer.data(), buffer.size(), &buffer_length))
		.WillOnce(testing::Return(expected));

	auto const actual = api.bind_column(handle, column_id, target_type, buffer.data(), buffer.size(), &buffer_length);
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}

void level1_api_test::bind_parameter_forwards()
{
	SQLRETURN const expected = 23;
	SQLHDBC handle = &value_a;
	SQLUSMALLINT const parameter_id = 17;
	SQLSMALLINT const input_output_type = 42;
	SQLSMALLINT const value_type = 33;
	SQLSMALLINT const parameter_type = 91;

	SQLULEN const column_size = 5;
	SQLSMALLINT const decimal_digits = 6;
	std::vector<char> buffer(100);
	SQLLEN buffer_length;

	level1_mock_api api;
	EXPECT_CALL(api, do_bind_parameter(handle, parameter_id, input_output_type, value_type, parameter_type, column_size, decimal_digits, buffer.data(), buffer.size(), &buffer_length))
		.WillOnce(testing::Return(expected));

	auto const actual = api.bind_parameter(handle, parameter_id, input_output_type, value_type, parameter_type, column_size, decimal_digits, buffer.data(), buffer.size(), &buffer_length);
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}

void level1_api_test::column_attribute_forwards()
{
	SQLRETURN const expected = 23;
	SQLHDBC handle = &value_a;
	SQLUSMALLINT const column_id = 1;
	SQLUSMALLINT const field_identifier = 2;
	std::vector<char> buffer(100);
	SQLSMALLINT buffer_length;
	SQLLEN numeric_attribute;

	level1_mock_api api;
	EXPECT_CALL(api, do_column_attribute(handle, column_id, field_identifier, buffer.data(), buffer.size(), &buffer_length, &numeric_attribute))
		.WillOnce(testing::Return(expected));

	auto const actual = api.column_attribute(handle, column_id, field_identifier, buffer.data(), buffer.size(), &buffer_length, &numeric_attribute);
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}


void level1_api_test::execute_prepared_statement_forwards()
{
	SQLRETURN const expected = 23;
	SQLHDBC handle = &value_a;

	level1_mock_api api;
	EXPECT_CALL(api, do_execute_prepared_statement(handle))
		.WillOnce(testing::Return(expected));

	auto const actual = api.execute_prepared_statement(handle);
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}


void level1_api_test::execute_statement_forwards()
{
	SQLRETURN const expected = 23;
	SQLHDBC handle = &value_a;
	std::array<unsigned char, 5> statement;

	level1_mock_api api;
	EXPECT_CALL(api, do_execute_statement(handle, statement.data(), statement.size()))
		.WillOnce(testing::Return(expected));

	auto const actual = api.execute_statement(handle, statement.data(), statement.size());
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}


void level1_api_test::fetch_scroll_forwards()
{
	SQLRETURN const expected = 23;
	SQLHDBC handle = &value_a;
	SQLSMALLINT const fetch_orientation = 17;
	SQLLEN const fetch_offset = 42;

	level1_mock_api api;
	EXPECT_CALL(api, do_fetch_scroll(handle, fetch_orientation, fetch_offset))
		.WillOnce(testing::Return(expected));

	auto const actual = api.fetch_scroll(handle, fetch_orientation, fetch_offset);
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}


void level1_api_test::free_statement_forwards()
{
	SQLRETURN const expected = 23;
	SQLHSTMT handle = &value_a;
	SQLUSMALLINT const option = 42;

	level1_mock_api api;
	EXPECT_CALL(api, do_free_statement(handle, option))
		.WillOnce(testing::Return(expected));

	auto const actual = api.free_statement(handle, option);
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}


void level1_api_test::get_statement_attribute_forwards()
{
	SQLRETURN const expected = 23;
	SQLHSTMT handle = &value_a;
	SQLINTEGER const attribute = 42;
	std::vector<char> buffer(100);
	SQLINTEGER string_length = 1;

	level1_mock_api api;
	EXPECT_CALL(api, do_get_statement_attribute(handle, attribute, buffer.data(), buffer.size(), &string_length))
		.WillOnce(testing::Return(expected));

	auto const actual = api.get_statement_attribute(handle, attribute, buffer.data(), buffer.size(), &string_length);
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}


void level1_api_test::number_of_result_columns_forwards()
{
	SQLRETURN const expected = 23;
	SQLHSTMT handle = &value_a;
	SQLSMALLINT number_of_columns = 0;

	level1_mock_api api;
	EXPECT_CALL(api, do_number_of_result_columns(handle, &number_of_columns))
		.WillOnce(testing::Return(expected));

	auto const actual = api.number_of_result_columns(handle, &number_of_columns);
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}

void level1_api_test::prepare_statement_forwards()
{
	SQLRETURN const expected = 23;
	SQLHDBC handle = &value_a;
	std::array<unsigned char, 5> statement;

	level1_mock_api api;
	EXPECT_CALL(api, do_prepare_statement(handle, statement.data(), statement.size()))
		.WillOnce(testing::Return(expected));

	auto const actual = api.prepare_statement(handle, statement.data(), statement.size());
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}

void level1_api_test::set_statement_attribute_forwards()
{
	SQLRETURN const expected = 23;
	SQLHSTMT handle = &value_a;
	SQLINTEGER const attribute = 42;
	std::vector<char> buffer(100);

	level1_mock_api api;
	EXPECT_CALL(api, do_set_statement_attribute(handle, attribute, buffer.data(), buffer.size()))
		.WillOnce(testing::Return(expected));

	auto const actual = api.set_statement_attribute(handle, attribute, buffer.data(), buffer.size());
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}












