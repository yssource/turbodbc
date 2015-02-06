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
	void fill_buffer_with_value(cpp_odbc::multi_value_buffer & buffer, std::string const & value)
	{
		auto element = buffer[0];
		memcpy(element.data_pointer, value.data(), value.size() + 1);
		element.indicator = value.size();
	}

	void set_buffer_element_to_null(cpp_odbc::multi_value_buffer & buffer)
	{
		auto element = buffer[0];
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

	pydbc::column column(statement, column_index, std::move(description));
	CPPUNIT_ASSERT( buffer != nullptr);

	fill_buffer_with_value(*buffer, expected);
	CPPUNIT_ASSERT_EQUAL(expected, boost::get<std::string>(*column.get_field()));
}

void column_test::get_field_nullable()
{
	std::unique_ptr<pydbc::string_description> description(new pydbc::string_description(128));

	pydbc_test::mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_column(column_index, testing::_, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	pydbc::column column(statement, column_index, std::move(description));
	CPPUNIT_ASSERT( buffer != nullptr);

	set_buffer_element_to_null(*buffer);
	CPPUNIT_ASSERT(not static_cast<bool>(column.get_field()));
}
