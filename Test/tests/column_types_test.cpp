/**
 *  @file column_types_test.cpp
 *  @date 09.01.2015
 *  @author mkoenig
 */

#include "pydbc/column_types.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/helpers/is_abstract_base_class.h>

#include "mock_classes.h"
#include <sqlext.h>
#include <boost/variant/get.hpp>
#include <cstring>


class column_types_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( column_types_test );

	CPPUNIT_TEST( long_column_get_field );
	CPPUNIT_TEST( string_column_get_field );

CPPUNIT_TEST_SUITE_END();

public:

	void long_column_get_field();
	void string_column_get_field();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( column_types_test );

using pydbc_test::mock_statement;


namespace {
	/**
	* Change the address of the given target_pointer to point to the third argument of the mocked function
	*/
	ACTION_P(store_pointer_to_buffer_in, target_pointer) {
		*target_pointer = &arg2;
	}

	std::size_t const column_index = 42;
}


namespace {
	void fill_buffer_with_value(cpp_odbc::multi_value_buffer & buffer, long value)
	{
		auto element = buffer[0];
		memcpy(element.data_pointer, &value, sizeof(long));
		element.indicator = sizeof(long);
	}
}

void column_types_test::long_column_get_field()
{
	long const expected = 12345;
	auto const buffer_type = SQL_C_SBIGINT;
	mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_column(column_index, buffer_type, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	pydbc::long_column column(statement, column_index);
	CPPUNIT_ASSERT( buffer != nullptr);

	fill_buffer_with_value(*buffer, expected);
	CPPUNIT_ASSERT_EQUAL(expected, boost::get<long>(*column.get_field()));
}


namespace {
	void fill_buffer_with_value(cpp_odbc::multi_value_buffer & buffer, std::string const & value)
	{
		auto element = buffer[0];
		memcpy(element.data_pointer, value.data(), value.size() + 1);
		element.indicator = value.size();
	}
}

void column_types_test::string_column_get_field()
{
	std::string const expected("this is a test string");
	auto const buffer_type = SQL_CHAR;
	mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_column(column_index, buffer_type, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	pydbc::string_column column(statement, column_index);
	CPPUNIT_ASSERT( buffer != nullptr);

	fill_buffer_with_value(*buffer, expected);
	CPPUNIT_ASSERT_EQUAL(expected, boost::get<std::string>(*column.get_field()));
}

