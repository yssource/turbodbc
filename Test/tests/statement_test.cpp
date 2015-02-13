/**
 *  @file statement_test.cpp
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

#include <cppunit/extensions/HelperMacros.h>
#include "cppunit_toolbox/helpers/is_abstract_base_class.h"

#include "cpp_odbc_test/mock_statement.h"

#include "gmock/gmock.h"

class statement_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( statement_test );

	CPPUNIT_TEST( is_suitable_as_base_class );
	CPPUNIT_TEST( get_integer_attribute_forwards );
	CPPUNIT_TEST( set_integer_attribute_forwards );
	CPPUNIT_TEST( set_pointer_attribute_forwards );
	CPPUNIT_TEST( execute_forwards );
	CPPUNIT_TEST( prepare_forwards );
	CPPUNIT_TEST( bind_input_parameter_forwards );
	CPPUNIT_TEST( execute_prepared_forwards );
	CPPUNIT_TEST( number_of_columns_forwards );
	CPPUNIT_TEST( bind_column_forwards );
	CPPUNIT_TEST( fetch_next_forwards );
	CPPUNIT_TEST( close_cursor_forwards );
	CPPUNIT_TEST( get_integer_column_attribute_forwards );
	CPPUNIT_TEST( get_string_column_attribute_forwards );
	CPPUNIT_TEST( row_count_forwards );
	CPPUNIT_TEST( describe_column_forwards );

CPPUNIT_TEST_SUITE_END();

public:

	void is_suitable_as_base_class();
	void get_integer_attribute_forwards();
	void set_integer_attribute_forwards();
	void set_pointer_attribute_forwards();
	void execute_forwards();
	void prepare_forwards();
	void bind_input_parameter_forwards();
	void execute_prepared_forwards();
	void number_of_columns_forwards();
	void bind_column_forwards();
	void fetch_next_forwards();
	void close_cursor_forwards();
	void get_integer_column_attribute_forwards();
	void get_string_column_attribute_forwards();
	void row_count_forwards();
	void describe_column_forwards();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( statement_test );


using cpp_odbc_test::mock_statement;

void statement_test::is_suitable_as_base_class()
{
	bool const is_base = cppunit_toolbox::is_abstract_base_class<cpp_odbc::statement>::value;
	CPPUNIT_ASSERT( is_base );
}

void statement_test::get_integer_attribute_forwards()
{
	SQLINTEGER const attribute = 23;
	long const expected = 42;

	mock_statement statement;
	EXPECT_CALL( statement, do_get_integer_attribute(attribute))
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT_EQUAL( expected, statement.get_integer_attribute(attribute));
}

void statement_test::set_integer_attribute_forwards()
{
	SQLINTEGER const attribute = 23;
	long const value = 42;

	mock_statement statement;
	EXPECT_CALL( statement, do_set_attribute(attribute, value)).Times(1);

	statement.set_attribute(attribute, value);
}

void statement_test::set_pointer_attribute_forwards()
{
	SQLINTEGER const attribute = 23;
	SQLULEN value = 42;

	mock_statement statement;
	EXPECT_CALL( statement, do_set_attribute(attribute, &value)).Times(1);

	statement.set_attribute(attribute, &value);
}

void statement_test::execute_forwards()
{
	std::string const query = "SELECT * FROM dummy";

	mock_statement statement;
	EXPECT_CALL( statement, do_execute(query)).Times(1);

	statement.execute(query);
}

void statement_test::prepare_forwards()
{
	std::string const query = "SELECT * FROM dummy";

	mock_statement statement;
	EXPECT_CALL( statement, do_prepare(query)).Times(1);

	statement.prepare(query);
}

void statement_test::bind_input_parameter_forwards()
{
	SQLUSMALLINT const parameter = 17;
	SQLSMALLINT const value_type = 23;
	SQLSMALLINT const parameter_type = 42;
	cpp_odbc::multi_value_buffer values(2,3);

	mock_statement statement;
	EXPECT_CALL( statement, do_bind_input_parameter(parameter, value_type, parameter_type, testing::Ref(values))).Times(1);

	statement.bind_input_parameter(parameter, value_type, parameter_type, values);
}


void statement_test::execute_prepared_forwards()
{
	mock_statement statement;
	EXPECT_CALL( statement, do_execute_prepared()).Times(1);

	statement.execute_prepared();
}

void statement_test::number_of_columns_forwards()
{
	short int const expected = 42;

	mock_statement statement;
	EXPECT_CALL( statement, do_number_of_columns())
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT_EQUAL( expected, statement.number_of_columns() );
}

void statement_test::bind_column_forwards()
{
	SQLUSMALLINT const column = 17;
	SQLSMALLINT const column_type = 23;
	cpp_odbc::multi_value_buffer values(2,3);

	mock_statement statement;
	EXPECT_CALL( statement, do_bind_column(column, column_type, testing::Ref(values))).Times(1);

	statement.bind_column(column, column_type, values);
}

void statement_test::fetch_next_forwards()
{
	bool const expected = false;
	mock_statement statement;
	EXPECT_CALL( statement, do_fetch_next()).WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT_EQUAL(expected, statement.fetch_next());
}

void statement_test::close_cursor_forwards()
{
	mock_statement statement;
	EXPECT_CALL( statement, do_close_cursor()).Times(1);

	statement.close_cursor();
}

void statement_test::get_integer_column_attribute_forwards()
{
	SQLUSMALLINT const column = 23;
	SQLUSMALLINT const field_identifier = 42;
	long const expected = 17;

	mock_statement statement;
	EXPECT_CALL( statement, do_get_integer_column_attribute(column, field_identifier))
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT_EQUAL( expected, statement.get_integer_column_attribute(column, field_identifier));
}

void statement_test::get_string_column_attribute_forwards()
{
	SQLUSMALLINT const column = 23;
	SQLUSMALLINT const field_identifier = 42;
	std::string const expected = "test value";

	mock_statement statement;
	EXPECT_CALL( statement, do_get_string_column_attribute(column, field_identifier))
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT_EQUAL( expected, statement.get_string_column_attribute(column, field_identifier));
}

void statement_test::row_count_forwards()
{
	SQLLEN const expected = 42;

	mock_statement statement;
	EXPECT_CALL( statement, do_row_count())
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT_EQUAL( expected, statement.row_count() );
}

void statement_test::describe_column_forwards()
{
	SQLUSMALLINT const column_id = 23;
	cpp_odbc::column_description const expected = {"dummy", 1, 2, 3, false};

	mock_statement statement;
	EXPECT_CALL( statement, do_describe_column(column_id))
		.WillOnce(testing::Return(expected));

	CPPUNIT_ASSERT( expected == statement.describe_column(column_id) );
}
