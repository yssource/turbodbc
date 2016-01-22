#include "turbodbc/parameter.h"

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit_toolbox/extensions/assert_equal_with_different_types.h>

#include "mock_classes.h"

#include <turbodbc/descriptions/integer_description.h>
#include <turbodbc/descriptions/string_description.h>
#include <boost/variant/get.hpp>


class parameter_test : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE( parameter_test );

	CPPUNIT_TEST( set_non_nullable );
	CPPUNIT_TEST( set_nullable );
	CPPUNIT_TEST( copy_to_first_row );

CPPUNIT_TEST_SUITE_END();

public:

	void set_non_nullable();
	void set_nullable();
	void copy_to_first_row();

};

// Registers the fixture with the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION( parameter_test );

namespace {

	void set_buffer_element_to_null(cpp_odbc::multi_value_buffer & buffer, std::size_t row_index)
	{
		auto element = buffer[row_index];
		element.indicator = SQL_NULL_DATA;
	}

	/**
	* Change the address of the given target_pointer to point to the third argument of the mocked function
	*/
	ACTION_P(store_pointer_to_buffer_in, target_pointer) {
		*target_pointer = &arg3;
	}

	std::size_t const parameter_index = 42;

}

void parameter_test::set_non_nullable()
{
	std::unique_ptr<turbodbc::integer_description> description(new turbodbc::integer_description());

	auto const buffer_c_type = description->column_c_type();
	auto const buffer_sql_type = description->column_sql_type();
	turbodbc_test::mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_input_parameter(parameter_index, buffer_c_type, buffer_sql_type, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	auto const buffered_rows = 100;
	turbodbc::parameter parameter(statement, parameter_index, buffered_rows, std::move(description));
	CPPUNIT_ASSERT( buffer != nullptr);

	auto const row_index = 42;
	long const expected = 123;
	parameter.set(row_index, turbodbc::field{expected});

	auto const actual = *reinterpret_cast<long *>((*buffer)[row_index].data_pointer);
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}

void parameter_test::set_nullable()
{
	std::unique_ptr<turbodbc::integer_description> description(new turbodbc::integer_description());

	auto const buffer_c_type = description->column_c_type();
	auto const buffer_sql_type = description->column_sql_type();
	turbodbc_test::mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_input_parameter(parameter_index, buffer_c_type, buffer_sql_type, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	auto const buffered_rows = 100;
	turbodbc::parameter parameter(statement, parameter_index, buffered_rows, std::move(description));
	CPPUNIT_ASSERT( buffer != nullptr);

	auto const row_index = 42;
	turbodbc::nullable_field null;
	parameter.set(row_index, null);

	CPPUNIT_ASSERT_EQUAL(SQL_NULL_DATA, (*buffer)[row_index].indicator);
}

void parameter_test::copy_to_first_row()
{
	std::unique_ptr<turbodbc::string_description> description(new turbodbc::string_description(10));

	auto const buffer_c_type = description->column_c_type();
	auto const buffer_sql_type = description->column_sql_type();
	turbodbc_test::mock_statement statement;

	cpp_odbc::multi_value_buffer * buffer = nullptr;
	EXPECT_CALL(statement, do_bind_input_parameter(parameter_index, buffer_c_type, buffer_sql_type, testing::_))
		.WillOnce(store_pointer_to_buffer_in(&buffer));

	auto const buffered_rows = 100;
	turbodbc::parameter parameter(statement, parameter_index, buffered_rows, std::move(description));
	CPPUNIT_ASSERT( buffer != nullptr);

	auto const row_index = 42;
	std::string const expected("hi there!");
	parameter.set(row_index, turbodbc::field{expected});

	parameter.copy_to_first_row(row_index);

	std::string const actual_content((*buffer)[0].data_pointer);
	auto const actual_indicator((*buffer)[0].indicator);
	CPPUNIT_ASSERT_EQUAL(expected, actual_content);
	CPPUNIT_ASSERT_EQUAL(expected.size(), actual_indicator);
}
