/**
 *  @file level1_connector_test.cpp
 *  @date 07.03.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 15:26:51 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21210 $
 *
 */


#include <cppunit/extensions/HelperMacros.h>

#include "cpp_odbc/level2/level1_connector.h"

#include "cpp_odbc/error.h"

#include "cpp_odbc_test/level1_mock_api.h"

#include "sqlext.h"

#include <type_traits>

class level1_connector_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( level1_connector_test );

	CPPUNIT_TEST( is_handle_api );

	CPPUNIT_TEST( allocate_statement_handle_calls_api );
	CPPUNIT_TEST( allocate_connection_handle_calls_api );
	CPPUNIT_TEST( allocate_environment_handle_calls_api );

	CPPUNIT_TEST( allocate_statement_handle_fails );
	CPPUNIT_TEST( allocate_connection_handle_fails );
	CPPUNIT_TEST( allocate_environment_handle_fails );

	CPPUNIT_TEST( free_statement_handle_calls_api );
	CPPUNIT_TEST( free_connection_handle_calls_api );
	CPPUNIT_TEST( free_environment_handle_calls_api );

	CPPUNIT_TEST( free_statement_handle_fails );
	CPPUNIT_TEST( free_connection_handle_fails );
	CPPUNIT_TEST( free_environment_handle_fails );

	CPPUNIT_TEST( get_statement_diagnostic_record_calls_api );
	CPPUNIT_TEST( get_connection_diagnostic_record_calls_api );
	CPPUNIT_TEST( get_environment_diagnostic_record_calls_api );

	CPPUNIT_TEST( get_statement_diagnostic_record_fails );
	CPPUNIT_TEST( get_connection_diagnostic_record_fails );
	CPPUNIT_TEST( get_environment_diagnostic_record_fails );

	CPPUNIT_TEST( set_environment_attribute_calls_api );
	CPPUNIT_TEST( set_environment_attribute_fails );

	CPPUNIT_TEST( set_connection_attribute_calls_api );
	CPPUNIT_TEST( set_connection_attribute_fails );

	CPPUNIT_TEST( establish_connection_calls_api );
	CPPUNIT_TEST( establish_connection_fails );

	CPPUNIT_TEST( disconnect_calls_api );
	CPPUNIT_TEST( disconnect_fails );

	CPPUNIT_TEST( end_transaction_calls_api );
	CPPUNIT_TEST( end_transaction_fails );

	CPPUNIT_TEST( get_string_connection_info_calls_api );
	CPPUNIT_TEST( get_string_connection_info_fails );

	CPPUNIT_TEST( bind_column_calls_api );
	CPPUNIT_TEST( bind_column_fails );
	CPPUNIT_TEST( bind_input_parameter_calls_api );
	CPPUNIT_TEST( bind_input_parameter_fails );
	CPPUNIT_TEST( execute_prepared_statement_calls_api );
	CPPUNIT_TEST( execute_prepared_statement_fails );
	CPPUNIT_TEST( execute_statement_calls_api );
	CPPUNIT_TEST( execute_statement_fails );
	CPPUNIT_TEST( fetch_scroll_calls_api );
	CPPUNIT_TEST( fetch_scroll_fails );
	CPPUNIT_TEST( fetch_scroll_has_no_more_data );
	CPPUNIT_TEST( free_statement_calls_api );
	CPPUNIT_TEST( free_statement_fails );
	CPPUNIT_TEST( get_integer_column_attribute_calls_api );
	CPPUNIT_TEST( get_integer_column_attribute_fails );
	CPPUNIT_TEST( get_integer_statement_attribute_calls_api );
	CPPUNIT_TEST( get_integer_statement_attribute_fails );
	CPPUNIT_TEST( get_string_column_attribute_calls_api );
	CPPUNIT_TEST( get_string_column_attribute_fails );
	CPPUNIT_TEST( number_of_result_columns_calls_api );
	CPPUNIT_TEST( number_of_result_columns_fails );
	CPPUNIT_TEST( prepare_statement_calls_api );
	CPPUNIT_TEST( prepare_statement_fails );
	CPPUNIT_TEST( set_long_statement_attribute_calls_api );
	CPPUNIT_TEST( set_long_statement_attribute_fails );
	CPPUNIT_TEST( set_pointer_statement_attribute_calls_api );
	CPPUNIT_TEST( set_pointer_statement_attribute_fails );
	CPPUNIT_TEST( row_count_calls_api );
	CPPUNIT_TEST( row_count_fails );
	CPPUNIT_TEST( describe_column_calls_api );
	CPPUNIT_TEST( describe_column_fails );
	CPPUNIT_TEST( describe_parameter_calls_api );
	CPPUNIT_TEST( describe_parameter_fails );

CPPUNIT_TEST_SUITE_END();

public:

	void is_handle_api();

	void allocate_statement_handle_calls_api();
	void allocate_connection_handle_calls_api();
	void allocate_environment_handle_calls_api();

	void allocate_statement_handle_fails();
	void allocate_connection_handle_fails();
	void allocate_environment_handle_fails();

	void free_statement_handle_calls_api();
	void free_connection_handle_calls_api();
	void free_environment_handle_calls_api();

	void free_statement_handle_fails();
	void free_connection_handle_fails();
	void free_environment_handle_fails();

	void get_statement_diagnostic_record_calls_api();
	void get_connection_diagnostic_record_calls_api();
	void get_environment_diagnostic_record_calls_api();

	void get_statement_diagnostic_record_fails();
	void get_connection_diagnostic_record_fails();
	void get_environment_diagnostic_record_fails();

	void set_environment_attribute_calls_api();
	void set_environment_attribute_fails();

	void set_connection_attribute_calls_api();
	void set_connection_attribute_fails();

	void establish_connection_calls_api();
	void establish_connection_fails();

	void disconnect_calls_api();
	void disconnect_fails();

	void end_transaction_calls_api();
	void end_transaction_fails();

	void get_string_connection_info_calls_api();
	void get_string_connection_info_fails();

	void bind_column_calls_api();
	void bind_column_fails();
	void bind_input_parameter_calls_api();
	void bind_input_parameter_fails();
	void execute_prepared_statement_calls_api();
	void execute_prepared_statement_fails();
	void execute_statement_calls_api();
	void execute_statement_fails();
	void fetch_scroll_calls_api();
	void fetch_scroll_fails();
	void fetch_scroll_has_no_more_data();
	void free_statement_calls_api();
	void free_statement_fails();
	void get_integer_column_attribute_calls_api();
	void get_integer_column_attribute_fails();
	void get_integer_statement_attribute_calls_api();
	void get_integer_statement_attribute_fails();
	void get_string_column_attribute_calls_api();
	void get_string_column_attribute_fails();
	void number_of_result_columns_calls_api();
	void number_of_result_columns_fails();
	void prepare_statement_calls_api();
	void prepare_statement_fails();
	void set_long_statement_attribute_calls_api();
	void set_long_statement_attribute_fails();
	void set_pointer_statement_attribute_calls_api();
	void set_pointer_statement_attribute_fails();

	void row_count_calls_api();
	void row_count_fails();

	void describe_column_calls_api();
	void describe_column_fails();

	void describe_parameter_calls_api();
	void describe_parameter_fails();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( level1_connector_test );

namespace level2 = cpp_odbc::level2;
using level2::level1_connector;

namespace {

	// destinations for pointers, values irrelevant
	int value_a = 17;
	int value_b = 23;

	level2::diagnostic_record const expected_error = {"ABCDE", 23, "This is a test error message"};

	void expect_error(cpp_odbc_test::level1_mock_api const & mock, level2::diagnostic_record const & expected)
	{
		EXPECT_CALL(mock, do_get_diagnostic_record(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
			.WillOnce(testing::DoAll(
						testing::SetArrayArgument<3>(expected.odbc_status_code.begin(), expected.odbc_status_code.end()),
						testing::SetArgPointee<4>(expected.native_error_code),
						testing::SetArrayArgument<5>(expected.message.begin(), expected.message.end()),
						testing::SetArgPointee<7>(expected.message.size()),
						testing::Return(SQL_SUCCESS)
					));
	}

}

void level1_connector_test::is_handle_api()
{
	bool const implements_handle_api = std::is_base_of<cpp_odbc::level2::api, level1_connector>::value;
	CPPUNIT_ASSERT( implements_handle_api );
}

void level1_connector_test::allocate_statement_handle_calls_api()
{
	level2::connection_handle input_handle = {&value_a};
	level2::statement_handle const expected_output_handle = {&value_b};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_allocate_handle(SQL_HANDLE_STMT, input_handle.handle, testing::_) )
		.WillOnce(testing::DoAll(
				testing::SetArgPointee<2>(expected_output_handle.handle),
				testing::Return(SQL_SUCCESS)
		));

	level1_connector const connector(api);

	auto const actual_output_handle = connector.allocate_statement_handle(input_handle);
	CPPUNIT_ASSERT( expected_output_handle == actual_output_handle );
}

void level1_connector_test::allocate_connection_handle_calls_api()
{
	level2::environment_handle input_handle = {&value_a};
	level2::connection_handle const expected_output_handle = {&value_b};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_allocate_handle(SQL_HANDLE_DBC, input_handle.handle, testing::_) )
		.WillOnce(testing::DoAll(
				testing::SetArgPointee<2>(expected_output_handle.handle),
				testing::Return(SQL_SUCCESS)
		));

	level1_connector const connector(api);

	auto const actual_output_handle = connector.allocate_connection_handle(input_handle);
	CPPUNIT_ASSERT( expected_output_handle == actual_output_handle );
}

void level1_connector_test::allocate_environment_handle_calls_api()
{
	level2::environment_handle const expected_output_handle = {&value_a};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_allocate_handle(SQL_HANDLE_ENV, nullptr, testing::_) )
		.WillOnce(testing::DoAll(
				testing::SetArgPointee<2>(expected_output_handle.handle),
				testing::Return(SQL_SUCCESS)
		));

	level1_connector const connector(api);

	auto const actual_output_handle = connector.allocate_environment_handle();
	CPPUNIT_ASSERT( expected_output_handle == actual_output_handle );
}


void level1_connector_test::allocate_statement_handle_fails()
{
	level2::connection_handle input_handle = {&value_a};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_allocate_handle(testing::_, testing::_, testing::_) )
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);

	CPPUNIT_ASSERT_THROW( connector.allocate_statement_handle(input_handle), cpp_odbc::error );
}

void level1_connector_test::allocate_connection_handle_fails()
{
	level2::environment_handle input_handle = {&value_a};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_allocate_handle(testing::_, testing::_, testing::_) )
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);

	CPPUNIT_ASSERT_THROW( connector.allocate_connection_handle(input_handle), cpp_odbc::error );
}

void level1_connector_test::allocate_environment_handle_fails()
{
	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_allocate_handle(testing::_, testing::_, testing::_) )
		.WillOnce(testing::Return(SQL_ERROR));

	level1_connector const connector(api);

	CPPUNIT_ASSERT_THROW( connector.allocate_environment_handle(), cpp_odbc::error );
}

namespace {
	template <typename Handle>
	void test_free_handle_calls_api(short signed int expected_handle_type)
	{
		Handle handle = {&value_a};

		auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
		EXPECT_CALL(*api, do_free_handle(expected_handle_type, handle.handle) )
			.WillOnce(testing::Return(SQL_SUCCESS));

		level1_connector const connector(api);

		connector.free_handle(handle);
	}
}

void level1_connector_test::free_statement_handle_calls_api()
{
	test_free_handle_calls_api<level2::statement_handle>(SQL_HANDLE_STMT);
}

void level1_connector_test::free_connection_handle_calls_api()
{
	test_free_handle_calls_api<level2::connection_handle>(SQL_HANDLE_DBC);
}

void level1_connector_test::free_environment_handle_calls_api()
{
	test_free_handle_calls_api<level2::environment_handle>(SQL_HANDLE_ENV);
}

namespace {
	template <typename Handle>
	void test_free_handle_fails()
	{
		Handle handle = {&value_a};

		auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
		EXPECT_CALL(*api, do_free_handle(testing::_, testing::_) )
			.WillOnce(testing::Return(SQL_ERROR));
		expect_error(*api, expected_error);

		level1_connector const connector(api);

		CPPUNIT_ASSERT_THROW( connector.free_handle(handle), cpp_odbc::error);
	}
}

void level1_connector_test::free_statement_handle_fails()
{
	test_free_handle_fails<level2::statement_handle>();
}

void level1_connector_test::free_connection_handle_fails()
{
	test_free_handle_fails<level2::connection_handle>();
}

void level1_connector_test::free_environment_handle_fails()
{
	test_free_handle_fails<level2::environment_handle>();
}

namespace {

	template <typename Handle>
	void test_diagnostic_record_calls_api(signed short int expected_type)
	{
		Handle const handle = {&value_a};
		std::string const expected_status_code("ABCDE");
		SQLINTEGER const expected_native_error = 23;
		std::string const expected_message = "This is a test error message";

		auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
		EXPECT_CALL(*api, do_get_diagnostic_record(expected_type, handle.handle, 1, testing::_, testing::_, testing::_, 1024, testing::_))
			.WillOnce(testing::DoAll(
						testing::SetArrayArgument<3>(expected_status_code.begin(), expected_status_code.end()),
						testing::SetArgPointee<4>(expected_native_error),
						testing::SetArrayArgument<5>(expected_message.begin(), expected_message.end()),
						testing::SetArgPointee<7>(expected_message.size()),
						testing::Return(SQL_SUCCESS)
					));

		level1_connector const connector(api);

		auto const actual = connector.get_diagnostic_record(handle);

		CPPUNIT_ASSERT_EQUAL( expected_status_code, actual.odbc_status_code );
		CPPUNIT_ASSERT_EQUAL( expected_native_error, actual.native_error_code );
		CPPUNIT_ASSERT_EQUAL( expected_message, actual.message );
	}

}

void level1_connector_test::get_statement_diagnostic_record_calls_api()
{
	test_diagnostic_record_calls_api<level2::statement_handle>(SQL_HANDLE_STMT);
}

void level1_connector_test::get_connection_diagnostic_record_calls_api()
{
	test_diagnostic_record_calls_api<level2::connection_handle>(SQL_HANDLE_DBC);
}

void level1_connector_test::get_environment_diagnostic_record_calls_api()
{
	test_diagnostic_record_calls_api<level2::environment_handle>(SQL_HANDLE_ENV);
}

namespace {

	template <typename Handle>
	void test_diagnostic_record_fails()
	{
		Handle const handle = {&value_a};

		auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
		EXPECT_CALL(*api, do_get_diagnostic_record(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
			.WillOnce(testing::Return(SQL_ERROR));

		level1_connector const connector(api);

		CPPUNIT_ASSERT_THROW( connector.get_diagnostic_record(handle), cpp_odbc::error );
	}

}

void level1_connector_test::get_statement_diagnostic_record_fails()
{
	test_diagnostic_record_fails<level2::statement_handle>();
}

void level1_connector_test::get_connection_diagnostic_record_fails()
{
	test_diagnostic_record_fails<level2::connection_handle>();
}

void level1_connector_test::get_environment_diagnostic_record_fails()
{
	test_diagnostic_record_fails<level2::environment_handle>();
}

void level1_connector_test::set_environment_attribute_calls_api()
{
	level2::environment_handle const handle = {&value_a};
	SQLINTEGER const attribute = SQL_ATTR_ODBC_VERSION;
	long const value = 42;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_set_environment_attribute(handle.handle, attribute, reinterpret_cast<SQLPOINTER>(value), 0))
		.WillOnce(testing::Return(SQL_SUCCESS));

	level1_connector const connector(api);
	connector.set_environment_attribute(handle, attribute, value);
}

void level1_connector_test::set_environment_attribute_fails()
{
	level2::environment_handle const handle = {&value_a};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_set_environment_attribute(testing::_, testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.set_environment_attribute(handle, 0, 0), cpp_odbc::error );
}

void level1_connector_test::set_connection_attribute_calls_api()
{
	level2::connection_handle const handle = {&value_a};
	SQLINTEGER const attribute = SQL_ATTR_AUTOCOMMIT;
	long const value = 42;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_set_connection_attribute(handle.handle, attribute, reinterpret_cast<SQLPOINTER>(value), 0))
		.WillOnce(testing::Return(SQL_SUCCESS));

	level1_connector const connector(api);
	connector.set_connection_attribute(handle, attribute, value);
}

void level1_connector_test::set_connection_attribute_fails()
{
	level2::connection_handle const handle = {&value_a};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_set_connection_attribute(testing::_, testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.set_connection_attribute(handle, 0, 0), cpp_odbc::error );
}

void level1_connector_test::establish_connection_calls_api()
{
	level2::connection_handle handle = {&value_a};
	std::string const connection_string = "dummy connection string";

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_establish_connection(handle.handle, nullptr, testing::_, connection_string.length(), testing::_, 1024, testing::_, SQL_DRIVER_NOPROMPT))
		.WillOnce(testing::Return(SQL_SUCCESS));

	level1_connector const connector(api);
	connector.establish_connection(handle, connection_string);
}

void level1_connector_test::establish_connection_fails()
{
	level2::connection_handle handle = {&value_a};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_establish_connection(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.establish_connection(handle, "dummy connection string"), cpp_odbc::error);
}

void level1_connector_test::disconnect_calls_api()
{
	level2::connection_handle handle = {&value_a};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_disconnect(handle.handle))
		.WillOnce(testing::Return(SQL_SUCCESS));

	level1_connector const connector(api);
	connector.disconnect(handle);
}

void level1_connector_test::disconnect_fails()
{
	level2::connection_handle handle = {&value_a};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_disconnect(testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.disconnect(handle), cpp_odbc::error);
}

void level1_connector_test::end_transaction_calls_api()
{
	level2::connection_handle handle = {&value_a};
	SQLSMALLINT const completion_type = SQL_COMMIT;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_end_transaction(SQL_HANDLE_DBC, handle.handle, completion_type))
		.WillOnce(testing::Return(SQL_SUCCESS));

	level1_connector const connector(api);
	connector.end_transaction(handle, completion_type);
}

void level1_connector_test::end_transaction_fails()
{
	level2::connection_handle handle = {&value_a};
	SQLSMALLINT const completion_type = SQL_COMMIT;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_end_transaction(testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.end_transaction(handle, completion_type), cpp_odbc::error);
}

void level1_connector_test::get_string_connection_info_calls_api()
{
	level2::connection_handle handle = {&value_a};
	SQLUSMALLINT const info_type = SQL_ODBC_VER;
	std::string const expected_info = "test info";

	auto copy_string_to_void_pointer = [&expected_info](testing::Unused, testing::Unused, void * destination, testing::Unused, testing::Unused) {
		memcpy(destination, expected_info.data(), expected_info.size());
	};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_get_connection_info(handle.handle, info_type, testing::_, 1024, testing::_))
		.WillOnce(testing::DoAll(
					testing::Invoke(copy_string_to_void_pointer),
					testing::SetArgPointee<4>(expected_info.size()),
					testing::Return(SQL_SUCCESS)
				));

	level1_connector const connector(api);
	CPPUNIT_ASSERT_EQUAL(expected_info, connector.get_string_connection_info(handle, info_type));
}

void level1_connector_test::get_string_connection_info_fails()
{
	level2::connection_handle handle = {&value_a};
	SQLUSMALLINT const info_type = SQL_ODBC_VER;
	std::string const expected_info = "test info";

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_get_connection_info(testing::_, testing::_, testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW(connector.get_string_connection_info(handle, info_type), cpp_odbc::error);
}

void level1_connector_test::bind_column_calls_api()
{
	level2::statement_handle handle = {&value_a};
	SQLUSMALLINT const column_id = 42;
	SQLSMALLINT const column_type = 17;
	cpp_odbc::multi_value_buffer column_buffer(23, 2);

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_bind_column(handle.handle, column_id, column_type, column_buffer.data_pointer(), column_buffer.capacity_per_element(), column_buffer.indicator_pointer()))
		.WillOnce(testing::Return(SQL_SUCCESS));

	level1_connector const connector(api);
	connector.bind_column(handle, column_id, column_type, column_buffer);
}
void level1_connector_test::bind_column_fails()
{
	level2::statement_handle handle = {&value_a};
	SQLUSMALLINT const column_id = 42;
	SQLSMALLINT const column_type = 17;
	cpp_odbc::multi_value_buffer column_buffer(23, 2);

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_bind_column(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW(connector.bind_column(handle, column_id, column_type, column_buffer), cpp_odbc::error);
}
void level1_connector_test::bind_input_parameter_calls_api()
{
	level2::statement_handle handle = {&value_a};
	SQLUSMALLINT const parameter_id = 42;
	SQLSMALLINT const value_type = 17;
	SQLSMALLINT const parameter_type = 51;
	cpp_odbc::multi_value_buffer buffer(23, 2);

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_bind_parameter(handle.handle, parameter_id, SQL_PARAM_INPUT, value_type, parameter_type, buffer.capacity_per_element(), 0, buffer.data_pointer(), buffer.capacity_per_element(), buffer.indicator_pointer()))
		.WillOnce(testing::Return(SQL_SUCCESS));

	level1_connector const connector(api);
	connector.bind_input_parameter(handle, parameter_id, value_type, parameter_type, buffer);
}
void level1_connector_test::bind_input_parameter_fails()
{
	level2::statement_handle handle = {&value_a};
	SQLUSMALLINT const parameter_id = 42;
	SQLSMALLINT const value_type = 17;
	SQLSMALLINT const parameter_type = 51;
	cpp_odbc::multi_value_buffer buffer(23, 2);

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_bind_parameter(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.bind_input_parameter(handle, parameter_id, value_type, parameter_type, buffer), cpp_odbc::error);
}

void level1_connector_test::execute_prepared_statement_calls_api()
{
	level2::statement_handle handle = {&value_a};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_execute_prepared_statement(handle.handle))
		.WillOnce(testing::Return(SQL_SUCCESS));

	level1_connector const connector(api);
	connector.execute_prepared_statement(handle);
}

void level1_connector_test::execute_prepared_statement_fails()
{
	level2::statement_handle handle = {&value_a};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_execute_prepared_statement(testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.execute_prepared_statement(handle), cpp_odbc::error);
}

namespace {

	// useful functor for comparing unsigned char * with strings
	struct matches_string {
		matches_string(std::string matchee) :
			matchee(std::move(matchee))
		{
		}

		bool operator()(unsigned char * pointer) const
		{
			return (memcmp(pointer, matchee.c_str(), matchee.size()) == 0);
		}

		std::string matchee;
	};

}

void level1_connector_test::execute_statement_calls_api()
{
	level2::statement_handle handle = {&value_a};
	std::string const sql = "XXX";

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_execute_statement(handle.handle, testing::Truly(matches_string(sql)), sql.size()))
		.WillOnce(testing::Return(SQL_SUCCESS));

	level1_connector const connector(api);
	connector.execute_statement(handle, sql);
}

void level1_connector_test::execute_statement_fails()
{
	level2::statement_handle handle = {&value_a};
	std::string const sql = "XXX";

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_execute_statement(testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.execute_statement(handle, sql), cpp_odbc::error );
}

void level1_connector_test::fetch_scroll_calls_api()
{
	level2::statement_handle handle = {&value_a};
	SQLSMALLINT const orientation = 42;
	SQLLEN const offset = 17;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_fetch_scroll(handle.handle, orientation, offset))
		.WillOnce(testing::Return(SQL_SUCCESS));

	level1_connector const connector(api);
	CPPUNIT_ASSERT(connector.fetch_scroll(handle, orientation, offset));
}

void level1_connector_test::fetch_scroll_fails()
{
	level2::statement_handle handle = {&value_a};
	SQLSMALLINT const orientation = 42;
	SQLLEN const offset = 17;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_fetch_scroll(testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.fetch_scroll(handle, orientation, offset), cpp_odbc::error );
}

void level1_connector_test::fetch_scroll_has_no_more_data()
{
	level2::statement_handle handle = {&value_a};
	SQLSMALLINT const orientation = 42;
	SQLLEN const offset = 17;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_fetch_scroll(handle.handle, orientation, offset))
		.WillOnce(testing::Return(SQL_NO_DATA));

	level1_connector const connector(api);
	CPPUNIT_ASSERT(not connector.fetch_scroll(handle, orientation, offset));
}

void level1_connector_test::free_statement_calls_api()
{
	level2::statement_handle handle = {&value_a};
	SQLUSMALLINT const option = 42;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_free_statement(handle.handle, option))
		.WillOnce(testing::Return(SQL_SUCCESS));

	level1_connector const connector(api);
	connector.free_statement(handle, option);
}

void level1_connector_test::free_statement_fails()
{
	level2::statement_handle handle = {&value_a};
	SQLUSMALLINT const option = 42;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_free_statement(testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.free_statement(handle, option), cpp_odbc::error );
}

void level1_connector_test::get_integer_column_attribute_calls_api()
{
	level2::statement_handle handle = {&value_a};
	SQLUSMALLINT const column_id = 17;
	SQLUSMALLINT const field_identifier = 23;
	long const expected = 12345;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_column_attribute(handle.handle, column_id, field_identifier, nullptr, 0, nullptr, testing::_))
		.WillOnce(testing::DoAll(
					testing::SetArgPointee<6>(expected),
					testing::Return(SQL_SUCCESS)
				));

	level1_connector const connector(api);
	CPPUNIT_ASSERT_EQUAL( expected, connector.get_integer_column_attribute(handle, column_id, field_identifier) );
}

void level1_connector_test::get_integer_column_attribute_fails()
{
	level2::statement_handle handle = {&value_a};
	SQLUSMALLINT const column_id = 17;
	SQLUSMALLINT const field_identifier = 23;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_column_attribute(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.get_integer_column_attribute(handle, column_id, field_identifier), cpp_odbc::error );
}

void level1_connector_test::get_integer_statement_attribute_calls_api()
{
	level2::statement_handle handle = {&value_a};
	SQLINTEGER const attribute = 23;
	long const expected = 12345;

	auto copy_long_to_void_pointer = [&expected](testing::Unused, testing::Unused, void * destination, testing::Unused, testing::Unused) {
		*reinterpret_cast<long *>(destination) = expected;
	};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_get_statement_attribute(handle.handle, attribute, testing::_, 0, nullptr))
		.WillOnce(testing::DoAll(
					testing::Invoke(copy_long_to_void_pointer),
					testing::Return(SQL_SUCCESS)
				));

	level1_connector const connector(api);
	CPPUNIT_ASSERT_EQUAL( expected, connector.get_integer_statement_attribute(handle, attribute) );
}

void level1_connector_test::get_integer_statement_attribute_fails()
{
	level2::statement_handle handle = {&value_a};
	SQLINTEGER const attribute = 23;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_get_statement_attribute(testing::_, testing::_, testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.get_integer_statement_attribute(handle, attribute), cpp_odbc::error );
}

void level1_connector_test::get_string_column_attribute_calls_api()
{
	level2::statement_handle handle = {&value_a};
	SQLUSMALLINT const column_id = 17;
	SQLUSMALLINT const field_identifier = 23;
	std::string const expected = "value";

	auto copy_string_to_void_pointer = [&expected](testing::Unused, testing::Unused, testing::Unused, void * destination, testing::Unused, testing::Unused, testing::Unused) {
		memcpy(destination, expected.data(), expected.size());
	};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_column_attribute(handle.handle, column_id, field_identifier, testing::_, 1024, testing::_, nullptr))
		.WillOnce(testing::DoAll(
					testing::Invoke(copy_string_to_void_pointer),
					testing::SetArgPointee<5>(expected.size()),
					testing::Return(SQL_SUCCESS)
				));

	level1_connector const connector(api);
	CPPUNIT_ASSERT_EQUAL( expected, connector.get_string_column_attribute(handle, column_id, field_identifier) );
}

void level1_connector_test::get_string_column_attribute_fails()
{
	level2::statement_handle handle = {&value_a};
	SQLUSMALLINT const column_id = 17;
	SQLUSMALLINT const field_identifier = 23;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_column_attribute(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.get_string_column_attribute(handle, column_id, field_identifier), cpp_odbc::error );
}

void level1_connector_test::number_of_result_columns_calls_api()
{
	level2::statement_handle handle = {&value_a};
	short int const expected = 42;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_number_of_result_columns(handle.handle, testing::_))
		.WillOnce(testing::DoAll(
					testing::SetArgPointee<1>(expected),
					testing::Return(SQL_SUCCESS)
				));

	level1_connector const connector(api);
	CPPUNIT_ASSERT_EQUAL( expected, connector.number_of_result_columns(handle) );
}

void level1_connector_test::number_of_result_columns_fails()
{
	level2::statement_handle handle = {&value_a};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_number_of_result_columns(testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.number_of_result_columns(handle), cpp_odbc::error );
}

void level1_connector_test::prepare_statement_calls_api()
{
	level2::statement_handle handle = {&value_a};
	std::string const sql = "XXX";

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_prepare_statement(handle.handle, testing::Truly(matches_string(sql)), sql.size()))
		.WillOnce(testing::Return(SQL_SUCCESS));

	level1_connector const connector(api);
	connector.prepare_statement(handle, sql);
}

void level1_connector_test::prepare_statement_fails()
{
	level2::statement_handle handle = {&value_a};
	std::string const sql = "XXX";

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_prepare_statement(testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.prepare_statement(handle, sql), cpp_odbc::error );
}

void level1_connector_test::set_long_statement_attribute_calls_api()
{
	level2::statement_handle handle = {&value_a};
	SQLINTEGER const attribute = 42;
	long const value = 23;

	auto matches_pointer_as_value = [&value](void * pointer) {
		return reinterpret_cast<long>(pointer) == value;
	};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_set_statement_attribute(handle.handle, attribute, testing::Truly(matches_pointer_as_value), SQL_IS_INTEGER))
		.WillOnce(testing::Return(SQL_SUCCESS));

	level1_connector const connector(api);
	connector.set_statement_attribute(handle, attribute, value);
}

void level1_connector_test::set_long_statement_attribute_fails()
{
	level2::statement_handle handle = {&value_a};
	SQLINTEGER const attribute = 42;
	long const value = 23;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_set_statement_attribute(testing::_, testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.set_statement_attribute(handle, attribute, value), cpp_odbc::error );
}

void level1_connector_test::set_pointer_statement_attribute_calls_api()
{
	level2::statement_handle handle = {&value_a};
	SQLINTEGER const attribute = 42;
	SQLULEN value = 23;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_set_statement_attribute(handle.handle, attribute, &value, SQL_IS_POINTER))
		.WillOnce(testing::Return(SQL_SUCCESS));

	level1_connector const connector(api);
	connector.set_statement_attribute(handle, attribute, &value);
}

void level1_connector_test::set_pointer_statement_attribute_fails()
{
	level2::statement_handle handle = {&value_a};
	SQLINTEGER const attribute = 42;
	SQLULEN value = 23;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_set_statement_attribute(testing::_, testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.set_statement_attribute(handle, attribute, &value), cpp_odbc::error );
}

void level1_connector_test::row_count_calls_api()
{
	level2::statement_handle handle = {&value_a};
	SQLLEN const expected = 42;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_row_count(handle.handle, testing::_))
		.WillOnce(testing::DoAll(
					testing::SetArgPointee<1>(expected),
					testing::Return(SQL_SUCCESS)
				));

	level1_connector const connector(api);
	CPPUNIT_ASSERT_EQUAL( expected, connector.row_count(handle) );
}

void level1_connector_test::row_count_fails()
{
	level2::statement_handle handle = {&value_a};

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_row_count(testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.row_count(handle), cpp_odbc::error );
}

namespace {

	void test_describe_column(bool expected_nullable, SQLSMALLINT sql_nullable, std::string const & message)
	{
		level2::statement_handle handle = {&value_a};
		SQLUSMALLINT const column_id = 17;

		cpp_odbc::column_description const expected = {"value", 123, 456, 666, expected_nullable};

		auto copy_string_to_void_pointer = [&expected](testing::Unused, testing::Unused, void * destination, testing::Unused, testing::Unused, testing::Unused, testing::Unused, testing::Unused, testing::Unused) {
			memcpy(destination, expected.name.data(), expected.name.size());
		};

		auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
		EXPECT_CALL(*api, do_describe_column(handle.handle, column_id, testing::_, 256, testing::_, testing::_, testing::_, testing::_, testing::_))
			.WillOnce(testing::DoAll(
						testing::Invoke(copy_string_to_void_pointer),
						testing::SetArgPointee<4>(expected.name.size()),
						testing::SetArgPointee<5>(expected.data_type),
						testing::SetArgPointee<6>(expected.size),
						testing::SetArgPointee<7>(expected.decimal_digits),
						testing::SetArgPointee<8>(sql_nullable),
						testing::Return(SQL_SUCCESS)
					));

		level1_connector const connector(api);
		CPPUNIT_ASSERT_MESSAGE( message, expected == connector.describe_column(handle, column_id));
	}

}

void level1_connector_test::describe_column_calls_api()
{
	test_describe_column(true, SQL_NULLABLE, "SQL_NULLABLE");
	test_describe_column(false, SQL_NO_NULLS, "SQL_NO_NULLS");
	test_describe_column(true, SQL_NULLABLE_UNKNOWN, "SQL_NULLABLE_UNKNOWN");
}

void level1_connector_test::describe_column_fails()
{
	level2::statement_handle handle = {&value_a};
	SQLUSMALLINT const column_id = 17;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_describe_column(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.describe_column(handle, column_id), cpp_odbc::error );
}



namespace {

	void test_describe_parameter(bool expected_nullable, SQLSMALLINT sql_nullable, std::string const & message)
	{
		level2::statement_handle handle = {&value_a};
		SQLUSMALLINT const parameter_id = 17;

		cpp_odbc::column_description const expected = {"parameter_17", 123, 456, 666, expected_nullable};

		auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
		EXPECT_CALL(*api, do_describe_parameter(handle.handle, parameter_id, testing::_, testing::_, testing::_, testing::_))
			.WillOnce(testing::DoAll(
						testing::SetArgPointee<2>(expected.data_type),
						testing::SetArgPointee<3>(expected.size),
						testing::SetArgPointee<4>(expected.decimal_digits),
						testing::SetArgPointee<5>(sql_nullable),
						testing::Return(SQL_SUCCESS)
					));

		level1_connector const connector(api);
		CPPUNIT_ASSERT_MESSAGE( message, expected == connector.describe_parameter(handle, parameter_id));
	}

}

void level1_connector_test::describe_parameter_calls_api()
{
	test_describe_parameter(true, SQL_NULLABLE, "SQL_NULLABLE");
	test_describe_parameter(false, SQL_NO_NULLS, "SQL_NO_NULLS");
	test_describe_parameter(true, SQL_NULLABLE_UNKNOWN, "SQL_NULLABLE_UNKNOWN");
}

void level1_connector_test::describe_parameter_fails()
{
	level2::statement_handle handle = {&value_a};
	SQLUSMALLINT const parameter_id = 17;

	auto api = std::make_shared<cpp_odbc_test::level1_mock_api const>();
	EXPECT_CALL(*api, do_describe_parameter(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
		.WillOnce(testing::Return(SQL_ERROR));
	expect_error(*api, expected_error);

	level1_connector const connector(api);
	CPPUNIT_ASSERT_THROW( connector.describe_parameter(handle, parameter_id), cpp_odbc::error );
}
