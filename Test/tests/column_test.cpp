/**
 *  @file column_test.cpp
 *  @date 09.01.2015
 *  @author mkoenig
 */

#include "pydbc/column.h"

#include <cppunit/extensions/HelperMacros.h>

#include "mock_classes.h"

#include <pydbc/description_types.h>
#include <boost/variant/get.hpp>


class column_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( column_test );

	CPPUNIT_TEST( get_field_non_nullable );
	CPPUNIT_TEST( get_field_nullable );

CPPUNIT_TEST_SUITE_END();

public:

	void get_field_non_nullable();
	void get_field_nullable();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( column_test );

namespace {
	void fill_buffer_with_value(cpp_odbc::multi_value_buffer & buffer, std::size_t row_index, std::string const & value)
	{
		auto element = buffer[row_index];
		memcpy(element.data_pointer, value.data(), value.size() + 1);
		element.indicator = value.size();
	}

	void set_buffer_element_to_null(cpp_odbc::multi_value_buffer & buffer, std::size_t row_index)
	{
		auto element = buffer[row_index];
		element.indicator = SQL_NULL_DATA;
	}

	/**
	* Change the address of the given target_pointer to point to the third argument of the mocked function
	*/
	ACTION_P(store_pointer_to_buffer_in, target_pointer) {
		*target_pointer = &arg2;
	}

	std::size_t const column_index = 42;

}

void column_test::get_field_non_nullable()
{
	std::string const expected("this is a test string");
	std::unique_ptr<pydbc::string_description> description(new pydbc::string_description(128));

	auto const buffer_type = description->column_type();
	pydbc_test::mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_column(column_index, buffer_type, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	auto const buffered_rows = 100;
	pydbc::column column(statement, column_index, buffered_rows, std::move(description));
	CPPUNIT_ASSERT( buffer != nullptr);

	auto const row_index = 42;
	fill_buffer_with_value(*buffer, row_index, expected);
	CPPUNIT_ASSERT_EQUAL(expected, boost::get<std::string>(*column.get_field(row_index)));
}

void column_test::get_field_nullable()
{
	std::unique_ptr<pydbc::string_description> description(new pydbc::string_description(128));

	pydbc_test::mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_column(column_index, testing::_, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	auto const buffered_rows = 100;
	pydbc::column column(statement, column_index, buffered_rows, std::move(description));
	CPPUNIT_ASSERT( buffer != nullptr);

	auto const row_index = 42;
	set_buffer_element_to_null(*buffer, row_index);
	CPPUNIT_ASSERT(not static_cast<bool>(column.get_field(row_index)));
}
