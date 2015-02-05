/**
 *  @file level2_api_test.cpp
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

#include "cpp_odbc/level2/api.h"

#include "cpp_odbc_test/level2_mock_api.h"

#include "sqlext.h"

class level2_api_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( level2_api_test );

	CPPUNIT_TEST( abstract_base );
	CPPUNIT_TEST( allocate_statement_handle_forwards );
	CPPUNIT_TEST( allocate_connection_handle_forwards );
	CPPUNIT_TEST( allocate_environment_handle_forwards );
	CPPUNIT_TEST( free_statement_handle_forwards );
	CPPUNIT_TEST( free_connection_handle_forwards );
	CPPUNIT_TEST( free_environment_handle_forwards );
	CPPUNIT_TEST( get_statement_diagnostic_record_forwards );
	CPPUNIT_TEST( get_connection_diagnostic_record_forwards );
	CPPUNIT_TEST( get_environment_diagnostic_record_forwards );
	CPPUNIT_TEST( set_environment_attribute_forwards );
	CPPUNIT_TEST( set_connection_attribute_forwards );
	CPPUNIT_TEST( establish_connection_forwards );
	CPPUNIT_TEST( disconnect_forwards );
	CPPUNIT_TEST( end_transaction_forwards );
	CPPUNIT_TEST( get_string_connection_info_forwards );
	CPPUNIT_TEST( bind_column_forwards );
	CPPUNIT_TEST( bind_input_parameter_forwards );
	CPPUNIT_TEST( get_string_column_attribute_forwards );
	CPPUNIT_TEST( get_integer_column_attribute_forwards );
	CPPUNIT_TEST( execute_prepared_statement_forwards );
	CPPUNIT_TEST( execute_statement_forwards );
	CPPUNIT_TEST( fetch_scroll_forwards );
	CPPUNIT_TEST( free_statement_forwards );
	CPPUNIT_TEST( get_integer_statement_attribute_forwards );
	CPPUNIT_TEST( number_of_result_columns_forwards );
	CPPUNIT_TEST( prepare_statement_forwards );
	CPPUNIT_TEST( set_statement_attribute_forwards );
	CPPUNIT_TEST( row_count_forwards );
	CPPUNIT_TEST( describe_column_forwards );

CPPUNIT_TEST_SUITE_END();

public:

	void abstract_base();
	void allocate_statement_handle_forwards();
	void allocate_connection_handle_forwards();
	void allocate_environment_handle_forwards();
	void free_statement_handle_forwards();
	void free_connection_handle_forwards();
	void free_environment_handle_forwards();
	void get_statement_diagnostic_record_forwards();
	void get_connection_diagnostic_record_forwards();
	void get_environment_diagnostic_record_forwards();
	void set_environment_attribute_forwards();
	void set_connection_attribute_forwards();
	void establish_connection_forwards();
	void disconnect_forwards();
	void end_transaction_forwards();
	void get_string_connection_info_forwards();
	void bind_column_forwards();
	void bind_input_parameter_forwards();
	void get_string_column_attribute_forwards();
	void get_integer_column_attribute_forwards();
	void execute_prepared_statement_forwards();
	void execute_statement_forwards();
	void fetch_scroll_forwards();
	void free_statement_forwards();
	void get_integer_statement_attribute_forwards();
	void number_of_result_columns_forwards();
	void prepare_statement_forwards();
	void set_statement_attribute_forwards();
	void row_count_forwards();
	void describe_column_forwards();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( level2_api_test );

namespace level2 = cpp_odbc::level2;

namespace {

	// destinations for pointers, values irrelevant
	int value_a = 17;
	int value_b = 23;

}

using cpp_odbc_test::level2_mock_api;

void level2_api_test::abstract_base()
{
	bool const is_abstract_base = cppunit_toolbox::is_abstract_base_class<level2::api>::value;
	CPPUNIT_ASSERT( is_abstract_base );
}

void level2_api_test::allocate_statement_handle_forwards()
{
	level2::connection_handle input_handle = {&value_a};
	level2::statement_handle expected = {&value_b};

	level2_mock_api api;
	EXPECT_CALL(api, do_allocate_statement_handle(input_handle))
		.WillOnce(testing::Return(expected));

	auto const actual = api.allocate_statement_handle(input_handle);
	CPPUNIT_ASSERT(expected == actual);
}

void level2_api_test::allocate_connection_handle_forwards()
{
	level2::environment_handle input_handle = {&value_a};
	level2::connection_handle expected = {&value_b};

	level2_mock_api api;
	EXPECT_CALL(api, do_allocate_connection_handle(input_handle))
		.WillOnce(testing::Return(expected));

	auto const actual = api.allocate_connection_handle(input_handle);
	CPPUNIT_ASSERT(expected == actual);
}

void level2_api_test::allocate_environment_handle_forwards()
{
	level2::environment_handle expected = {&value_b};

	level2_mock_api api;
	EXPECT_CALL(api, do_allocate_environment_handle())
		.WillOnce(testing::Return(expected));

	auto const actual = api.allocate_environment_handle();
	CPPUNIT_ASSERT(expected == actual);
}

void level2_api_test::free_statement_handle_forwards()
{
	level2::statement_handle handle = {&value_a};

	level2_mock_api api;
	EXPECT_CALL(api, do_free_handle(handle)).Times(1);

	api.free_handle(handle);
}

void level2_api_test::free_connection_handle_forwards()
{
	level2::connection_handle handle = {&value_a};

	level2_mock_api api;
	EXPECT_CALL(api, do_free_handle(handle)).Times(1);

	api.free_handle(handle);
}

void level2_api_test::free_environment_handle_forwards()
{
	level2::environment_handle handle = {&value_a};

	level2_mock_api api;
	EXPECT_CALL(api, do_free_handle(handle)).Times(1);

	api.free_handle(handle);
}

namespace {

	template <typename Handle>
	void test_get_diagnostic_record_forwards()
	{
		Handle const handle = {&value_a};
		level2::diagnostic_record const expected = {"abcde", 17, "test"};

		level2_mock_api api;
		EXPECT_CALL(api, do_get_diagnostic_record(handle))
			.WillOnce(testing::Return(expected));

		auto actual = api.get_diagnostic_record(handle);

		CPPUNIT_ASSERT_EQUAL(expected.odbc_status_code, actual.odbc_status_code);
		CPPUNIT_ASSERT_EQUAL(expected.native_error_code, actual.native_error_code);
		CPPUNIT_ASSERT_EQUAL(expected.message, actual.message);
	}

}

void level2_api_test::get_statement_diagnostic_record_forwards()
{
	test_get_diagnostic_record_forwards<level2::statement_handle>();
}

void level2_api_test::get_connection_diagnostic_record_forwards()
{
	test_get_diagnostic_record_forwards<level2::connection_handle>();
}

void level2_api_test::get_environment_diagnostic_record_forwards()
{
	test_get_diagnostic_record_forwards<level2::environment_handle>();
}

void level2_api_test::set_environment_attribute_forwards()
{
	level2::environment_handle const handle = {&value_a};
	SQLINTEGER const attribute = 42;
	long const value = 17;

	level2_mock_api api;
	EXPECT_CALL(api, do_set_environment_attribute(handle, attribute, value))
		.Times(1);

	api.set_environment_attribute(handle, attribute, value);
}

void level2_api_test::set_connection_attribute_forwards()
{
	level2::connection_handle const handle = {&value_a};
	SQLINTEGER const attribute = 42;
	long const value = 17;

	level2_mock_api api;
	EXPECT_CALL(api, do_set_connection_attribute(handle, attribute, value))
		.Times(1);

	api.set_connection_attribute(handle, attribute, value);
}

void level2_api_test::establish_connection_forwards()
{
	level2::connection_handle handle = {&value_a};
	std::string const connection_string = "My fancy database";

	level2_mock_api api;
	EXPECT_CALL(api, do_establish_connection(handle, connection_string))
		.Times(1);

	api.establish_connection(handle, connection_string);
}

void level2_api_test::disconnect_forwards()
{
	level2::connection_handle handle = {&value_a};

	level2_mock_api api;
	EXPECT_CALL(api, do_disconnect(handle))
		.Times(1);

	api.disconnect(handle);
}

void level2_api_test::end_transaction_forwards()
{
	level2::connection_handle handle = {&value_a};
	SQLSMALLINT const completion_type = SQL_COMMIT;

	level2_mock_api api;
	EXPECT_CALL(api, do_end_transaction(handle, completion_type))
		.Times(1);

	api.end_transaction(handle, completion_type);
}

void level2_api_test::get_string_connection_info_forwards()
{
	level2::connection_handle handle = {&value_a};
	SQLUSMALLINT const info_type = SQL_DRIVER_ODBC_VER;
	std::string const expected_info("dummy");

	level2_mock_api api;
	EXPECT_CALL(api, do_get_string_connection_info(handle, info_type))
		.WillOnce(testing::Return(expected_info));

	CPPUNIT_ASSERT_EQUAL(expected_info, api.get_string_connection_info(handle, info_type));
}

void level2_api_test::bind_column_forwards()
{
	level2::statement_handle const handle = {&value_a};
	SQLUSMALLINT column_id = 17;
	SQLSMALLINT column_type = 42;
	cpp_odbc::multi_value_buffer column_buffer(2,3);

	level2_mock_api api;
	EXPECT_CALL(api, do_bind_column(handle, column_id, column_type, testing::Ref(column_buffer))).Times(1);

	api.bind_column(handle, column_id, column_type, column_buffer);
}

void level2_api_test::bind_input_parameter_forwards()
{
	level2::statement_handle const handle = {&value_a};
	SQLUSMALLINT parameter_id = 17;
	SQLSMALLINT c_data_type = 42;
	SQLSMALLINT sql_data_type = 23;
	cpp_odbc::multi_value_buffer column_buffer(2,3);

	level2_mock_api api;
	EXPECT_CALL(api, do_bind_input_parameter(handle, parameter_id, c_data_type, sql_data_type, testing::Ref(column_buffer))).Times(1);

	api.bind_input_parameter(handle, parameter_id, c_data_type, sql_data_type, column_buffer);
}

void level2_api_test::get_string_column_attribute_forwards()
{
	level2::statement_handle const handle = {&value_a};
	SQLUSMALLINT column_id = 17;
	SQLUSMALLINT field_identifier = 23;
	std::string const expected("value");

	level2_mock_api api;
	EXPECT_CALL(api, do_get_string_column_attribute(handle, column_id, field_identifier))
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT_EQUAL( expected, api.get_string_column_attribute(handle, column_id, field_identifier) );
}

void level2_api_test::get_integer_column_attribute_forwards()
{
	level2::statement_handle const handle = {&value_a};
	SQLUSMALLINT column_id = 17;
	SQLUSMALLINT field_identifier = 23;
	long const expected = 42;

	level2_mock_api api;
	EXPECT_CALL(api, do_get_integer_column_attribute(handle, column_id, field_identifier))
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT_EQUAL( expected, api.get_integer_column_attribute(handle, column_id, field_identifier) );
}

void level2_api_test::execute_prepared_statement_forwards()
{
	level2::statement_handle const handle = {&value_a};

	level2_mock_api api;
	EXPECT_CALL(api, do_execute_prepared_statement(handle)).Times(1);

	api.execute_prepared_statement(handle);
}

void level2_api_test::execute_statement_forwards()
{
	level2::statement_handle const handle = {&value_a};
	std::string const query("SELECT * FROM table");

	level2_mock_api api;
	EXPECT_CALL(api, do_execute_statement(handle, query)).Times(1);

	api.execute_statement(handle, query);
}

void level2_api_test::fetch_scroll_forwards()
{
	level2::statement_handle const handle = {&value_a};
	SQLSMALLINT const orientation = SQL_FETCH_NEXT;
	SQLLEN const offset = 17;
	bool const expected = true;

	level2_mock_api api;
	EXPECT_CALL(api, do_fetch_scroll(handle, orientation, offset)).WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT_EQUAL(expected, api.fetch_scroll(handle, orientation, offset));
}

void level2_api_test::free_statement_forwards()
{
	level2::statement_handle handle = {&value_a};
	SQLUSMALLINT const option = SQL_CLOSE;

	level2_mock_api api;
	EXPECT_CALL(api, do_free_statement(handle, option)).Times(1);

	api.free_statement(handle, option);
}

void level2_api_test::get_integer_statement_attribute_forwards()
{
	level2::statement_handle const handle = {&value_a};
	SQLINTEGER const attribute = 42;
	long const expected = 17;

	level2_mock_api api;
	EXPECT_CALL(api, do_get_integer_statement_attribute(handle, attribute))
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT_EQUAL( expected, api.get_integer_statement_attribute(handle, attribute));
}

void level2_api_test::number_of_result_columns_forwards()
{
	level2::statement_handle const handle = {&value_a};
	short int const expected = 42;

	level2_mock_api api;
	EXPECT_CALL(api, do_number_of_result_columns(handle))
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT_EQUAL( expected, api.number_of_result_columns(handle));
}

void level2_api_test::prepare_statement_forwards()
{
	level2::statement_handle const handle = {&value_a};
	std::string const query("SELECT * FROM table");

	level2_mock_api api;
	EXPECT_CALL(api, do_prepare_statement(handle, query)).Times(1);

	api.prepare_statement(handle, query);
}

void level2_api_test::set_statement_attribute_forwards()
{
	level2::statement_handle const handle = {&value_a};
	SQLINTEGER const attribute = 23;
	long const value = 42;

	level2_mock_api api;
	EXPECT_CALL(api, do_set_statement_attribute(handle, attribute, value)).Times(1);

	api.set_statement_attribute(handle, attribute, value);
}

void level2_api_test::row_count_forwards()
{
	level2::statement_handle const handle = {&value_a};
	SQLLEN const expected = 23;

	level2_mock_api api;
	EXPECT_CALL(api, do_row_count(handle)).WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT_EQUAL(expected, api.row_count(handle));
}

void level2_api_test::describe_column_forwards()
{
	level2::statement_handle const handle = {&value_a};
	SQLUSMALLINT const column_id = 42;
	cpp_odbc::column_description const expected = {"dummy", 1, 2, 3, false};

	level2_mock_api api;
	EXPECT_CALL(api, do_describe_column(handle, column_id)).WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT(expected == api.describe_column(handle, column_id));
}
