/**
 *  @file raii_statement_test.cpp
 *  @date 23.05.2014
 *  @author mkoenig
 *  @brief 
 *
 *  $LastChangedDate: 2014-11-28 15:26:51 +0100 (Fr, 28 Nov 2014) $
 *  $LastChangedBy: mkoenig $
 *  $LastChangedRevision: 21210 $
 *
 */

#include "cpp_odbc/raii_statement.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>

#include "cpp_odbc/raii_connection.h"
#include "cpp_odbc_test/level2_mock_api.h"
#include "psapp/valid_ptr_core.h"

#include <type_traits>

class raii_statement_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( raii_statement_test );

	CPPUNIT_TEST( is_statement );
	CPPUNIT_TEST( resource_management );
	CPPUNIT_TEST( keeps_connection_alive );
	CPPUNIT_TEST( get_integer_statement_attribute );
	CPPUNIT_TEST( set_integer_statement_attribute );
	CPPUNIT_TEST( execute );
	CPPUNIT_TEST( prepare );
	CPPUNIT_TEST( bind_input_parameter );
	CPPUNIT_TEST( execute_prepared );
	CPPUNIT_TEST( number_of_columns );
	CPPUNIT_TEST( bind_column );
	CPPUNIT_TEST( fetch_next );
	CPPUNIT_TEST( close_cursor );
	CPPUNIT_TEST( get_integer_column_attribute );
	CPPUNIT_TEST( get_string_column_attribute );

CPPUNIT_TEST_SUITE_END();

public:

	void is_statement();
	void resource_management();
	void keeps_connection_alive();
	void get_integer_statement_attribute();
	void set_integer_statement_attribute();
	void execute();
	void prepare();
	void bind_input_parameter();
	void execute_prepared();
	void number_of_columns();
	void bind_column();
	void fetch_next();
	void close_cursor();
	void get_integer_column_attribute();
	void get_string_column_attribute();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( raii_statement_test );

using cpp_odbc::raii_statement;
using cpp_odbc::raii_connection;
using cpp_odbc_test::level2_mock_api;
using cpp_odbc::level2::environment_handle;
using cpp_odbc::level2::connection_handle;
using cpp_odbc::level2::statement_handle;

namespace {

	// destinations for pointers, values irrelevant
	int value_a = 17;
	int value_b = 23;
	int value_c = 17;

	environment_handle const default_e_handle = {&value_a};
	connection_handle const default_c_handle = {&value_b};
	statement_handle const default_s_handle = {&value_c};

	psapp::valid_ptr<testing::NiceMock<level2_mock_api> const> make_default_api()
	{
		auto api = psapp::make_valid_ptr<testing::NiceMock<level2_mock_api>>();

		ON_CALL(*api, do_allocate_connection_handle(testing::_))
			.WillByDefault(testing::Return(default_c_handle));
		ON_CALL(*api, do_allocate_statement_handle(testing::_))
			.WillByDefault(testing::Return(default_s_handle));

		return api;
	}

}

void raii_statement_test::is_statement()
{
	bool const derived_from_statement = std::is_base_of<cpp_odbc::statement, raii_statement>::value;
	CPPUNIT_ASSERT( derived_from_statement );
}

void raii_statement_test::resource_management()
{
	auto api = psapp::make_valid_ptr<testing::NiceMock<level2_mock_api> const>();
	ON_CALL(*api, do_allocate_connection_handle(testing::_))
		.WillByDefault(testing::Return(default_c_handle));
	auto connection = psapp::make_valid_ptr<raii_connection const>(api, default_e_handle, "dummy");

	statement_handle s_handle = {&value_b};

	EXPECT_CALL(*api, do_allocate_statement_handle(default_c_handle))
		.WillOnce(testing::Return(s_handle));

	{
		raii_statement statement(connection);

		// free handle on destruction
		EXPECT_CALL(*api, do_free_handle(s_handle)).Times(1);
	}
}

void raii_statement_test::keeps_connection_alive()
{
	auto api = make_default_api();
	auto connection = psapp::make_valid_ptr<raii_connection const>(api, default_e_handle, "dummy");

	auto const use_count_before = connection.get().use_count();
	raii_statement statement(connection);
	auto const use_count_after = connection.get().use_count();

	CPPUNIT_ASSERT_EQUAL(1, use_count_after - use_count_before);
}

void raii_statement_test::get_integer_statement_attribute()
{
	SQLINTEGER const attribute = 42;
	long const expected = 12345;

	auto api = make_default_api();
	auto connection = psapp::make_valid_ptr<raii_connection const>(api, default_e_handle, "dummy");
	EXPECT_CALL(*api, do_get_integer_statement_attribute(default_s_handle, attribute))
		.WillOnce(testing::Return(expected));

	raii_statement statement(connection);
	CPPUNIT_ASSERT_EQUAL( expected, statement.get_integer_statement_attribute(attribute));
}

void raii_statement_test::set_integer_statement_attribute()
{
	SQLINTEGER const attribute = 42;
	long const value = 12345;

	auto api = make_default_api();
	auto connection = psapp::make_valid_ptr<raii_connection const>(api, default_e_handle, "dummy");
	EXPECT_CALL(*api, do_set_statement_attribute(default_s_handle, attribute, value)).Times(1);

	raii_statement statement(connection);
	statement.set_statement_attribute(attribute, value);
}

void raii_statement_test::execute()
{
	std::string const sql = "SELECT dummy FROM test";

	auto api = make_default_api();
	auto connection = psapp::make_valid_ptr<raii_connection const>(api, default_e_handle, "dummy");
	EXPECT_CALL(*api, do_execute_statement(default_s_handle, sql)).Times(1);

	raii_statement statement(connection);
	statement.execute(sql);
}

void raii_statement_test::prepare()
{
	std::string const sql = "SELECT dummy FROM test";

	auto api = make_default_api();
	auto connection = psapp::make_valid_ptr<raii_connection const>(api, default_e_handle, "dummy");
	EXPECT_CALL(*api, do_prepare_statement(default_s_handle, sql)).Times(1);

	raii_statement statement(connection);
	statement.prepare(sql);
}

void raii_statement_test::bind_input_parameter()
{
	SQLUSMALLINT const parameter_id = 17;
	SQLSMALLINT const value_type = 23;
	SQLSMALLINT const parameter_type = 42;
	cpp_odbc::multi_value_buffer parameter_values(3, 4);

	auto api = make_default_api();
	auto connection = psapp::make_valid_ptr<raii_connection const>(api, default_e_handle, "dummy");
	EXPECT_CALL(*api, do_bind_input_parameter(default_s_handle, parameter_id, value_type, parameter_type, testing::Ref(parameter_values))).Times(1);

	raii_statement statement(connection);
	statement.bind_input_parameter(parameter_id, value_type, parameter_type, parameter_values);
}

void raii_statement_test::execute_prepared()
{
	auto api = make_default_api();
	auto connection = psapp::make_valid_ptr<raii_connection const>(api, default_e_handle, "dummy");
	EXPECT_CALL(*api, do_execute_prepared_statement(default_s_handle)).Times(1);

	raii_statement statement(connection);
	statement.execute_prepared();
}

void raii_statement_test::number_of_columns()
{
	short int const expected = 23;

	auto api = make_default_api();
	auto connection = psapp::make_valid_ptr<raii_connection const>(api, default_e_handle, "dummy");
	EXPECT_CALL(*api, do_number_of_result_columns(default_s_handle))
		.WillOnce(testing::Return(expected));

	raii_statement statement(connection);
	CPPUNIT_ASSERT_EQUAL( expected, statement.number_of_columns());
}

void raii_statement_test::bind_column()
{
	SQLUSMALLINT const column_id = 17;
	SQLSMALLINT const column_type = 23;
	cpp_odbc::multi_value_buffer column_buffer(3, 4);

	auto api = make_default_api();
	auto connection = psapp::make_valid_ptr<raii_connection const>(api, default_e_handle, "dummy");
	EXPECT_CALL(*api, do_bind_column(default_s_handle, column_id, column_type, testing::Ref(column_buffer))).Times(1);

	raii_statement statement(connection);
	statement.bind_column(column_id, column_type, column_buffer);
}

void raii_statement_test::fetch_next()
{
	auto api = make_default_api();
	auto connection = psapp::make_valid_ptr<raii_connection const>(api, default_e_handle, "dummy");
	EXPECT_CALL(*api, do_fetch_scroll(default_s_handle, SQL_FETCH_NEXT, 0)).Times(1);

	raii_statement statement(connection);
	statement.fetch_next();
}

void raii_statement_test::close_cursor()
{
	auto api = make_default_api();
	auto connection = psapp::make_valid_ptr<raii_connection const>(api, default_e_handle, "dummy");
	EXPECT_CALL(*api, do_free_statement(default_s_handle, SQL_CLOSE)).Times(1);

	raii_statement statement(connection);
	statement.close_cursor();
}

void raii_statement_test::get_integer_column_attribute()
{
	SQLUSMALLINT const column_id = 17;
	SQLUSMALLINT const field_identifier = 42;
	long const expected = 23;

	auto api = make_default_api();
	auto connection = psapp::make_valid_ptr<raii_connection const>(api, default_e_handle, "dummy");
	EXPECT_CALL(*api, do_get_integer_column_attribute(default_s_handle, column_id, field_identifier))
		.WillOnce(testing::Return(expected));

	raii_statement statement(connection);
	CPPUNIT_ASSERT_EQUAL( expected, statement.get_integer_column_attribute(column_id, field_identifier));
}

void raii_statement_test::get_string_column_attribute()
{
	SQLUSMALLINT const column_id = 17;
	SQLUSMALLINT const field_identifier = 42;
	std::string const expected = "test value";

	auto api = make_default_api();
	auto connection = psapp::make_valid_ptr<raii_connection const>(api, default_e_handle, "dummy");
	EXPECT_CALL(*api, do_get_string_column_attribute(default_s_handle, column_id, field_identifier))
		.WillOnce(testing::Return(expected));

	raii_statement statement(connection);
	CPPUNIT_ASSERT_EQUAL( expected, statement.get_string_column_attribute(column_id, field_identifier));
}
